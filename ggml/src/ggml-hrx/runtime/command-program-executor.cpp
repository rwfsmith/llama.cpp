#include "command-program-executor.h"

#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "dispatch_registration/dispatch-copy.h"
#include "dispatch_registration/dispatch-gather-add.h"
#include "dispatch_registration/dispatch-moe-router.h"
#include "dispatch_registration/dispatch-qwen-preamble.h"
#include "dispatch_registration/dispatch-routed-ffn.h"
#include "ggml-impl.h"
#include "hrx-interop-utils.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/transient-arena.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ggml::hrx {

struct MtpHostTrace {
    const char * phase = nullptr;
    uint64_t call = 0;
    size_t program = 0;
    size_t events = 0;
    size_t bytes = 0;
};

static thread_local MtpHostTrace mtp_host_trace;

void set_mtp_host_trace_scope(const char * phase, uint64_t call) {
    mtp_host_trace = {};
    mtp_host_trace.phase = phase;
    mtp_host_trace.call = call;
    if (phase != nullptr) {
        GGML_LOG_INFO("HRX MTP host trace: phase=%s call=%llu; existing transfer boundaries only, "
                      "512 events/64 MiB per call, 1 MiB per tensor; no callback cuts or extra device sync\n",
                      phase, static_cast<unsigned long long>(call));
    }
}

static bool hrx_environment_flag(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool trace_launch_enabled() {
    static const bool enabled = hrx_environment_flag("HRX_TRACE_LAUNCH");
    return enabled;
}

// Separate direct dispatch and graph replay wall time, including staging. Readbacks may wait for queued work.
static bool hrx_time_compute_enabled() {
    static const bool enabled = hrx_environment_flag("HRX_TIME_COMPUTE");
    return enabled;
}

// Forces individual dispatches through this per-kernel path (see the matching flags in
// graph-executor.cpp/prepared-command-program-cache.cpp) and, when set, brackets each kernel launch
// with a real HIP-event device-time measurement via context.device_timing. Diagnostic only: this
// synchronizes the GPU once per kernel and is far slower than normal operation.
static bool hrx_profile_dispatches_enabled() {
    static const bool enabled = [] {
        const bool value = hrx_environment_flag("HRX_PROFILE_DISPATCHES");
        GGML_LOG_INFO("HRX dispatch profiling (command-program-executor.cpp): %s\n", value ? "enabled" : "disabled");
        return value;
    }();
    return enabled;
}

struct DispatchProfileEntry {
    uint64_t count    = 0;
    double   total_ms = 0.0;
    double   max_ms   = 0.0;
};

static std::mutex g_dispatch_profile_mutex;
static std::unordered_map<uint64_t, DispatchProfileEntry> g_dispatch_profile_by_kernel_id;
static uint64_t g_dispatch_profile_samples = 0;
// Kept only so an atexit-time final summary (registered once, below) can still resolve kernel
// names for short runs that never reach the periodic sample threshold. The pointed-to context
// members live as long as the owning ggml_backend_hrx_context, i.e. for the process lifetime.
static const CommandProgramExecutionContext * g_dispatch_profile_last_context = nullptr;

static void log_dispatch_profile_summary_locked(const CommandProgramExecutionContext & context);

static void flush_dispatch_profile_summary_at_exit() {
    std::lock_guard<std::mutex> lock(g_dispatch_profile_mutex);
    if (g_dispatch_profile_samples == 0 || g_dispatch_profile_last_context == nullptr) {
        return;
    }
    GGML_LOG_INFO("HRX dispatch profile: final summary at process exit\n");
    log_dispatch_profile_summary_locked(*g_dispatch_profile_last_context);
}

static void record_dispatch_profile_sample(uint64_t kernel_id, double device_ms,
                                           const CommandProgramExecutionContext & context) {
    std::lock_guard<std::mutex> lock(g_dispatch_profile_mutex);
    static const bool registered_atexit = [] {
        std::atexit(flush_dispatch_profile_summary_at_exit);
        return true;
    }();
    (void) registered_atexit;
    g_dispatch_profile_last_context = &context;
    DispatchProfileEntry & entry = g_dispatch_profile_by_kernel_id[kernel_id];
    entry.count += 1;
    entry.total_ms += device_ms;
    if (device_ms > entry.max_ms) {
        entry.max_ms = device_ms;
    }
    ++g_dispatch_profile_samples;
}

static void log_dispatch_profile_summary_locked(const CommandProgramExecutionContext & context) {
    std::vector<std::pair<uint64_t, DispatchProfileEntry>> ranked(g_dispatch_profile_by_kernel_id.begin(),
                                                                   g_dispatch_profile_by_kernel_id.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
        return a.second.total_ms > b.second.total_ms;
    });
    double grand_total_ms = 0.0;
    for (const auto & entry : ranked) {
        grand_total_ms += entry.second.total_ms;
    }
    GGML_LOG_INFO("HRX dispatch profile: samples=%llu unique_kernels=%zu total_device_ms=%.1f\n",
                  static_cast<unsigned long long>(g_dispatch_profile_samples), ranked.size(), grand_total_ms);
    const size_t top_n = ranked.size() < 25 ? ranked.size() : 25;
    for (size_t i = 0; i < top_n; ++i) {
        const uint64_t              kernel_id = ranked[i].first;
        const DispatchProfileEntry & entry     = ranked[i].second;
        const KernelResolveResult    resolved  =
            context.corpus != nullptr ? resolve_kernel_definition(*context.corpus, context.target, kernel_id) :
                                         KernelResolveResult{};
        const std::string name = kernel_definition_name_or_id(resolved.definition, kernel_id);
        GGML_LOG_INFO("HRX dispatch profile #%zu: %s calls=%llu total_ms=%.3f avg_ms=%.4f max_ms=%.4f pct=%.1f%%\n",
                      i + 1, name.c_str(), static_cast<unsigned long long>(entry.count), entry.total_ms,
                      entry.total_ms / static_cast<double>(entry.count), entry.max_ms,
                      grand_total_ms > 0.0 ? 100.0 * entry.total_ms / grand_total_ms : 0.0);
    }
}

static void trace_kernel_preparation(const KernelDefinition & definition, const Dispatch & dispatch) {
    if (!trace_launch_enabled()) {
        return;
    }
    std::ostringstream out;
    out << "HRX launch prepare " << definition.family << '/' << definition.name << " id=" << dispatch.kernel.kernel_id;
    for (const auto & entry : dispatch.kernel.integer_parameters) {
        out << ' ' << entry.first << '=' << entry.second;
    }
    for (const auto & entry : dispatch.kernel.compile_parameters) {
        out << ' ' << entry.first << '=' << entry.second;
    }
    for (const DispatchBinding & binding : dispatch.bindings) {
        out << " bind[v" << binding.value.value << " +" << binding.offset << " " << binding.length << "B]";
    }
    GGML_LOG_ERROR("%s\n", out.str().c_str());
}

static void trace_kernel_record(const PreparedCommand & command, const std::vector<hrx_buffer_ref_t> & refs) {
    if (!trace_launch_enabled() || command.kernel.executable == nullptr) {
        return;
    }
    const ggml_hrx_loom_jit_launch_config & launch = command.kernel.executable->launch;
    std::ostringstream                      out;
    out << "HRX launch record ord=" << command.ordinal << " id=" << command.kernel.specialization.kernel_id << " grid="
        << launch.workgroup_count[0] << 'x' << launch.workgroup_count[1] << 'x' << launch.workgroup_count[2]
        << " wg=" << launch.workgroup_size[0] << 'x' << launch.workgroup_size[1] << 'x' << launch.workgroup_size[2]
        << " sg=" << launch.subgroup_size << " constants=" << command.kernel.constants.size() << "B";
    for (size_t i = 0; i + sizeof(uint32_t) <= command.kernel.constants.size(); i += sizeof(uint32_t)) {
        uint32_t word = 0;
        std::memcpy(&word, command.kernel.constants.data() + i, sizeof(word));
        out << ' ' << word;
    }
    for (const hrx_buffer_ref_t & ref : refs) {
        out << " ref[" << ref.buffer << " +" << ref.offset << " " << ref.length << "B]";
    }
    GGML_LOG_ERROR("%s\n", out.str().c_str());
}

PreparedProgramConstantBuffer::~PreparedProgramConstantBuffer() {
    if (buffer != nullptr) {
        hrx_buffer_release(buffer);
    }
}

PreparedProgramConstantBuffer::PreparedProgramConstantBuffer(PreparedProgramConstantBuffer && other) noexcept :
    value(other.value),
    name(std::move(other.name)),
    buffer(std::exchange(other.buffer, nullptr)),
    size(other.size) {
    other.size = 0;
}

PreparedProgramConstantBuffer &
PreparedProgramConstantBuffer::operator=(PreparedProgramConstantBuffer && other) noexcept {
    if (this != &other) {
        if (buffer != nullptr) {
            hrx_buffer_release(buffer);
        }
        value        = other.value;
        name         = std::move(other.name);
        buffer       = std::exchange(other.buffer, nullptr);
        size         = other.size;
        other.size   = 0;
    }
    return *this;
}

RecordedCommandGraph::~RecordedCommandGraph() {
    if (exec != nullptr) {
        hrx_graph_exec_release(exec);
    }
    if (graph != nullptr) {
        hrx_graph_release(graph);
    }
}

RecordedCommandGraph::RecordedCommandGraph(RecordedCommandGraph && other) noexcept :
    graph(std::exchange(other.graph, nullptr)),
    exec(std::exchange(other.exec, nullptr)),
    bound_transient_arena_allocation_id(other.bound_transient_arena_allocation_id),
    dispatch_count(other.dispatch_count),
    status(std::move(other.status)) {
    other.bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;
    other.dispatch_count                      = 0;
}

RecordedCommandGraph & RecordedCommandGraph::operator=(RecordedCommandGraph && other) noexcept {
    if (this != &other) {
        if (exec != nullptr) {
            hrx_graph_exec_release(exec);
        }
        if (graph != nullptr) {
            hrx_graph_release(graph);
        }
        graph                                = std::exchange(other.graph, nullptr);
        exec                                 = std::exchange(other.exec, nullptr);
        bound_transient_arena_allocation_id  = other.bound_transient_arena_allocation_id;
        dispatch_count                       = other.dispatch_count;
        status                               = std::move(other.status);
        other.bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;
        other.dispatch_count                      = 0;
    }
    return *this;
}

namespace {

static const char * status_first_error(const Status & status) {
    return status.errors().empty() ? "" : status.errors().front().c_str();
}

static bool resource_access_writes(ResourceAccess access) {
    return access == ResourceAccess::Write || access == ResourceAccess::ReadWrite;
}

static Status command_program_metadata_context_valid(const CommandProgramExecutionContext & context) {
    Status status;
    if (context.target == nullptr) {
        status.log("missing HRX target");
        return status;
    }
    if (context.corpus == nullptr) {
        status.log("missing HRX kernel corpus");
        return status;
    }
    return status;
}

static Status command_program_preparation_context_valid(const CommandProgramExecutionContext & context) {
    Status status;
    if (context.device == nullptr) {
        status.log("missing HRX device");
        return status;
    }
    if (context.kernel_executables == nullptr) {
        status.log("missing HRX kernel executable cache");
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager");
        return status;
    }
    if (context.host_weights == nullptr) {
        status.log("missing HRX host weight cache");
        return status;
    }
    return status;
}

static Status command_program_transient_context_valid(const CommandProgramExecutionContext & context,
                                                      const CommandProgram &                 commands) {
    Status status;
    if (commands.transients.arena_size == 0) {
        return status;
    }
    if (context.transient_arena == nullptr) {
        status.log("missing HRX transient arena");
        return status;
    }
    if (context.stream == nullptr) {
        status.log("missing HRX stream for transient arena");
        return status;
    }
    return status;
}

static bool prepared_program_has_constant(const PreparedCommandProgram & prepared, ValueId value) {
    for (const PreparedProgramConstantBuffer & constant : prepared.program_constants) {
        if (constant.value == value) {
            return true;
        }
    }
    return false;
}

static bool prepared_execution_context_valid(const CommandProgramExecutionContext & context) {
    if (context.stream == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX stream\n", __func__);
        return false;
    }
    return true;
}

static Status ensure_transient_arena(const CommandProgramExecutionContext & context,
                                     const CommandProgram &                 commands,
                                     TransientArenaAllocationRef &          allocation) {
    allocation    = {};
    Status status = command_program_transient_context_valid(context, commands);
    if (!status.success()) {
        return status;
    }
    if (commands.transients.arena_size == 0) {
        return status;
    }
    status = context.transient_arena->ensure_capacity(context.device, context.stream, commands.transients.arena_size);
    if (!status.success()) {
        return status;
    }
    allocation = context.transient_arena->current_allocation();
    return status;
}

static Status initialize_command_program_constants(const CommandProgramExecutionContext & context,
                                                   const CommandProgram &                 commands,
                                                   const TransientArenaAllocationRef &    allocation,
                                                   const PreparedCommandProgram &         prepared) {
    Status status;
    if (commands.constant_initializations.empty()) {
        return status;
    }
    for (const ConstantInitialization & initialization : commands.constant_initializations) {
        if (prepared_program_has_constant(prepared, initialization.value)) {
            continue;
        }
        if (allocation.buffer == nullptr) {
            status.log("command program has constant initialization %s without a transient arena allocation",
                       initialization.name.c_str());
            continue;
        }
        const TransientAllocation * transient = find_transient_allocation(commands.transients, initialization.value);
        if (transient == nullptr) {
            status.log("constant initialization %s references missing transient value %d", initialization.name.c_str(),
                       initialization.value.value);
            continue;
        }
        if (initialization.offset > transient->size ||
            initialization.data.size() > transient->size - initialization.offset) {
            status.log("constant initialization %s is outside transient allocation length %zu",
                       initialization.name.c_str(), transient->size);
            continue;
        }
        // TODO: Track initialized transient arena allocation ids so constants are not transferred every invocation.
        if (context.host_transfers == nullptr) {
            status.log("constant initialization %s requires a host transfer manager", initialization.name.c_str());
            continue;
        }
        Status upload_status = context.host_transfers->upload_synchronous(
            context.stream, initialization.data.data(), allocation.buffer,
            transient->arena_offset + initialization.offset, initialization.data.size());
        if (!upload_status.success()) {
            status.log("failed to upload constant initialization %s", initialization.name.c_str());
            status.append(upload_status);
        }
    }
    return status;
}

static Status initialize_command_program_completion_counters(const CommandProgramExecutionContext & context,
                                                             const CommandProgram &                 commands,
                                                             const TransientArenaAllocationRef &    allocation) {
    Status status;
    if (commands.completion_counters.byte_count == 0) {
        return status;
    }
    if (allocation.buffer == nullptr) {
        status.log("command program has completion counters without a transient arena allocation");
        return status;
    }
    if (commands.completion_counters.arena_offset > commands.transients.arena_size ||
        commands.completion_counters.byte_count >
            commands.transients.arena_size - commands.completion_counters.arena_offset) {
        status.log("completion counter initialization is outside transient arena length %zu",
                   commands.transients.arena_size);
        return status;
    }
    const uint32_t zero_pattern = 0;
    if (ErrorResult error = take_status(
            hrx_stream_fill_buffer(context.stream, allocation.buffer, commands.completion_counters.arena_offset,
                                   commands.completion_counters.byte_count, &zero_pattern, sizeof(zero_pattern)))) {
        status.log("failed to initialize completion counters: %s", error->c_str());
    }
    return status;
}

static std::string format_resolved_command_context(const ResolvedCommand & command) {
    std::ostringstream out;
    out << "command " << command.ordinal << " kind=" << command_kind_name(command.kind)
        << " kernel_id=" << command.kernel.kernel_id << " bindings=" << command.bindings.size();
    return out.str();
}

static std::string format_prepared_command_context(const PreparedCommand & command) {
    std::ostringstream out;
    out << "command " << command.ordinal << " kind=" << command_kind_name(command.kind);
    if (command.kind == CommandKind::Kernel) {
        out << " kernel_id=" << command.kernel.specialization.kernel_id
            << " bindings=" << command.kernel.bindings.size();
    }
    return out.str();
}

static Dispatch build_dispatch(const ResolvedCommand & command) {
    Dispatch dispatch;
    dispatch.kernel = command.kernel;
    dispatch.bindings.reserve(command.bindings.size());
    for (const ResolvedCommandBinding & binding : command.bindings) {
        dispatch.bindings.push_back({ binding.binding.value, binding.binding.offset, binding.binding.length });
    }
    return dispatch;
}

struct GraphValueAccess {
    bool read  = false;
    bool write = false;
};

static std::unordered_map<int32_t, GraphValueAccess> collect_graph_value_access(const CommandProgram & commands) {
    std::unordered_map<int32_t, GraphValueAccess> access_by_value;
    auto append_command_list_access = [&](const std::vector<Command> & command_list) {
        for (const Command & command : command_list) {
            for (const CommandBinding & binding : command.bindings) {
                if (binding.origin != CommandBindingOrigin::GraphValue) {
                    continue;
                }
                GraphValueAccess & access = access_by_value[binding.value.value];
                switch (binding.access) {
                    case ResourceAccess::Read:
                        access.read = true;
                        break;
                    case ResourceAccess::Write:
                        access.write = true;
                        break;
                    case ResourceAccess::ReadWrite:
                        access.read  = true;
                        access.write = true;
                        break;
                }
            }
        }
    };
    append_command_list_access(commands.initialization_commands);
    append_command_list_access(commands.commands);
    return access_by_value;
}

}  // namespace

Status plan_host_staging_groups(std::vector<HostStagingSlice> slices, std::vector<HostStagingGroup> & groups) {
    groups.clear();
    Status status;
    std::unordered_set<int32_t> values;
    for (const auto & slice : slices) {
        if (slice.address == 0 || slice.length == 0 ||
            slice.length > std::numeric_limits<uintptr_t>::max() - slice.address ||
            !values.insert(slice.value).second) {
            status.log("invalid or duplicate host staging range for value %d", slice.value);
            return status;
        }
    }
    std::sort(slices.begin(), slices.end(), [](const auto & a, const auto & b) {
        return a.address != b.address ? a.address < b.address : a.value < b.value;
    });
    uintptr_t end = 0;
    for (const auto & slice : slices) {
        const uintptr_t slice_end = slice.address + slice.length;
        if (groups.empty() || slice.address >= end) {
            // Preserve the original host address's alignment without copying the
            // padding prefix. Adjacent, nonoverlapping allocations stay separate.
            const uintptr_t base = slice.address & ~uintptr_t{255};
            groups.push_back({ base, static_cast<size_t>(slice_end - base), { slice } });
            end = slice_end;
        } else {
            auto & group = groups.back();
            end = std::max(end, slice_end);
            group.length = static_cast<size_t>(end - group.base);
            group.slices.push_back(slice);
        }
    }
    return status;
}

namespace {

static const HostStagingBuffer * find_host_staging(const std::vector<HostStagingBuffer> & staging, int32_t value) {
    for (const auto & entry : staging) {
        if (entry.value == value) {
            return &entry;
        }
    }
    return nullptr;
}

static Status allocate_host_staging_groups(hrx_device_t device, const std::vector<HostStagingBuffer> & sources,
                                           const std::vector<HostStagingGroup> & groups,
                                           std::vector<HostStagingBuffer> & output) {
    Status status;
    for (const auto & group : groups) {
        HostStagingBuffer allocation;
        status.append(allocate_host_staging_buffer(device, group.length, allocation));
        if (!status.success()) {
            return status;
        }
        for (const auto & slice : group.slices) {
            const auto * source = find_host_staging(sources, slice.value);
            if (source == nullptr) {
                status.log("missing host staging source for value %d", slice.value);
                return status;
            }
            HostStagingBuffer view;
            hrx_buffer_retain(allocation.buffer);
            view.buffer = allocation.buffer;
            view.host_data = reinterpret_cast<void *>(slice.address);
            view.value = slice.value;
            view.length = slice.length;
            view.offset = static_cast<size_t>(slice.address - group.base);
            view.upload = source->upload;
            view.download = source->download;
            output.push_back(std::move(view));
        }
    }
    return status;
}

static CommandProgramBindings materialize_host_bindings(const CommandProgramExecutionContext & context,
                                                        const CommandProgram &                 commands,
                                                        const CommandProgramBindings &         bindings,
                                                        PreparedCommandProgram &               prepared) {
    std::vector<CommandProgramBinding> materialized;
    Status                             status;
    materialized.reserve(bindings.bindings().size());
    const std::unordered_map<int32_t, GraphValueAccess> access_by_value = collect_graph_value_access(commands);
    std::vector<HostStagingBuffer> staging_sources;
    std::vector<HostStagingSlice> slices;
    for (const CommandProgramBinding & binding : bindings.bindings()) {
        if (!binding.requires_materialization()) {
            materialized.push_back(binding);
            continue;
        }
        const auto             found_access = access_by_value.find(binding.value.value);
        const GraphValueAccess access =
            found_access != access_by_value.end() ? found_access->second : GraphValueAccess{};
        const uintptr_t host_base = reinterpret_cast<uintptr_t>(binding.host_data);
        if (binding.offset > binding.capacity || binding.length > binding.capacity - binding.offset ||
            binding.offset > std::numeric_limits<uintptr_t>::max() - host_base ||
            binding.length > std::numeric_limits<uintptr_t>::max() - (host_base + binding.offset)) {
            status.log("invalid host binding range for value %d", binding.value.value);
            materialized.push_back(binding);
            continue;
        }
        const uintptr_t address = host_base + binding.offset;
        bool overlaps_writer = false;
        if (binding.weight && access.read && !access.write) {
            for (const auto & other : bindings.bindings()) {
                const auto writer = access_by_value.find(other.value.value);
                if (other.host_data == nullptr || writer == access_by_value.end() || !writer->second.write) {
                    continue;
                }
                const uintptr_t base = reinterpret_cast<uintptr_t>(other.host_data);
                if (other.offset > std::numeric_limits<uintptr_t>::max() - base) {
                    continue;
                }
                const uintptr_t at = base + other.offset;
                overlaps_writer |= address <= at ? at - address < binding.length : address - at < other.length;
            }
        }
        if (binding.weight && access.read && !access.write && !overlaps_writer) {
            HostWeightSource source;
            source.host_data  = binding.host_data;
            source.identity   = binding.identity;
            source.generation = binding.generation;
            source.capacity   = binding.capacity;
            source.offset     = binding.offset;
            source.length     = binding.length;
            HostWeightAcquireResult resident =
                context.host_weights->acquire(context.device, context.stream, *context.host_transfers, source);
            if (!resident.valid()) {
                status.log("materialize host weight value %d failed", binding.value.value);
                status.append(resident.status);
                materialized.push_back(binding);
                continue;
            }
            CommandProgramBinding device_binding = binding;
            device_binding.buffer                = resident.lease.buffer();
            device_binding.host_data             = nullptr;
            device_binding.offset                = 0;
            device_binding.capacity              = binding.length;
            materialized.push_back(device_binding);
            prepared.resident_host_weights.push_back(std::move(resident.lease));
            continue;
        }

        HostStagingBuffer staging;
        staging.value                        = binding.value.value;
        staging.host_data                    = reinterpret_cast<void *>(address);
        staging.length                       = binding.length;
        staging.upload                       = access.read;
        staging.download                     = access.write;
        if (mtp_host_trace.phase != nullptr && binding.tensor != nullptr) {
            prepared.host_trace_tensors.push_back({ staging.value, binding.tensor });
        }
        slices.push_back({ staging.value, address, binding.length });
        staging_sources.push_back(std::move(staging));
        materialized.push_back(binding);
    }
    std::vector<HostStagingGroup> groups;
    status.append(plan_host_staging_groups(std::move(slices), groups));
    if (status.success()) {
        status.append(allocate_host_staging_groups(context.device, staging_sources, groups, prepared.host_staging));
    }
    if (status.success()) {
        for (auto & binding : materialized) {
            const auto * staging = find_host_staging(prepared.host_staging, binding.value.value);
            if (staging != nullptr) {
                binding.buffer = staging->buffer;
                binding.host_data = nullptr;
                binding.offset = staging->offset;
                binding.capacity = staging->offset + staging->length;
            }
        }
    }
    return CommandProgramBindings::from_bindings(std::move(materialized), status);
}

struct ProgramConstantImage {
    ValueId              value;
    std::string          name;
    std::vector<uint8_t> data;
    bool                 read = false;
};

static Status collect_program_constant_images(const CommandProgram & commands,
                                              std::vector<ProgramConstantImage> & images) {
    Status                                  status;
    std::unordered_map<int32_t, size_t>     image_by_value;
    for (const ConstantInitialization & initialization : commands.constant_initializations) {
        const TransientAllocation * allocation = find_transient_allocation(commands.transients, initialization.value);
        if (allocation == nullptr) {
            status.log("constant initialization %s references missing transient value %d", initialization.name.c_str(),
                       initialization.value.value);
            continue;
        }
        if (initialization.offset > allocation->size ||
            initialization.data.size() > allocation->size - initialization.offset) {
            status.log("constant initialization %s is outside transient allocation length %zu",
                       initialization.name.c_str(), allocation->size);
            continue;
        }

        ProgramConstantImage * image = nullptr;
        const auto             found = image_by_value.find(initialization.value.value);
        if (found == image_by_value.end()) {
            ProgramConstantImage next;
            next.value = initialization.value;
            next.name  = initialization.name;
            next.data.resize(allocation->size);
            image_by_value.emplace(initialization.value.value, images.size());
            images.push_back(std::move(next));
            image = &images.back();
        } else {
            image = &images[found->second];
        }

        std::copy(initialization.data.begin(), initialization.data.end(),
                  image->data.begin() + initialization.offset);
    }
    return status;
}

static Status validate_program_constant_access(const CommandProgram &           commands,
                                               std::vector<ProgramConstantImage> & images) {
    Status                              status;
    std::unordered_map<int32_t, size_t> image_by_value;
    for (size_t i = 0; i < images.size(); ++i) {
        image_by_value.emplace(images[i].value.value, i);
    }

    auto validate_command_list = [&](const std::vector<Command> & command_list) {
        for (const Command & command : command_list) {
            for (const CommandBinding & binding : command.bindings) {
                const auto found = image_by_value.find(binding.value.value);
                if (found == image_by_value.end()) {
                    continue;
                }
                ProgramConstantImage & image = images[found->second];
                if (resource_access_writes(binding.access)) {
                    status.log("constant initialization %s is written by command %u; prepared constant buffer copy "
                               "support is required",
                               image.name.c_str(), command.ordinal);
                    continue;
                }
                image.read = true;
            }
        }
    };
    validate_command_list(commands.initialization_commands);
    validate_command_list(commands.commands);
    return status;
}

static PreparedProgramConstantBuffer make_program_constant_buffer(ValueId      value,
                                                                  std::string  name,
                                                                  hrx_buffer_t buffer,
                                                                  size_t       size) {
    PreparedProgramConstantBuffer result;
    result.value  = value;
    result.name   = std::move(name);
    result.buffer = buffer;
    result.size   = size;
    return result;
}

static Status bind_prepared_command_list_program_constants(
    const PreparedCommandProgram &              prepared,
    std::vector<PreparedCommand> &              prepared_commands) {
    Status status;
    for (PreparedCommand & command : prepared_commands) {
        for (PreparedCommandBinding & binding : command.kernel.bindings) {
            for (const PreparedProgramConstantBuffer & constant : prepared.program_constants) {
                if (binding.binding.origin != CommandBindingOrigin::Transient ||
                    binding.binding.value != constant.value) {
                    continue;
                }
                if (binding.binding.offset > constant.size ||
                    binding.binding.length > constant.size - binding.binding.offset) {
                    status.log("%s is outside prepared constant %s length %zu",
                               format_command_binding(binding.binding).c_str(), constant.name.c_str(), constant.size);
                    continue;
                }
                binding.ref = { constant.buffer, binding.binding.offset, binding.binding.length };
                binding.binding.origin = CommandBindingOrigin::ProgramConstant;
            }
        }
    }
    return status;
}

static Status prepare_program_constant_buffers(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               PreparedCommandProgram &               prepared) {
    Status status;
    if (commands.constant_initializations.empty()) {
        return status;
    }

    std::vector<ProgramConstantImage> images;
    status.append(collect_program_constant_images(commands, images));
    status.append(validate_program_constant_access(commands, images));
    if (!status.success()) {
        return status;
    }
    if (images.empty()) {
        return status;
    }
    if (context.device == nullptr) {
        status.log("missing HRX device for prepared constants");
        return status;
    }
    if (context.stream == nullptr) {
        status.log("missing HRX stream for prepared constants");
        return status;
    }

    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT,
        0,
    };
    for (const ProgramConstantImage & image : images) {
        if (!image.read) {
            continue;
        }
        hrx_buffer_t buffer = nullptr;
        if (ErrorResult error = take_status(
                hrx_allocator_allocate_buffer(hrx_device_allocator(context.device), params, image.data.size(),
                                              &buffer))) {
            status.log("allocate prepared constant %s: %s", image.name.c_str(), error->c_str());
            continue;
        }
        if (context.host_transfers == nullptr) {
            hrx_buffer_release(buffer);
            status.log("prepared constant %s requires a host transfer manager", image.name.c_str());
            continue;
        }
        Status upload_status =
            context.host_transfers->upload_synchronous(context.stream, image.data.data(), buffer, 0, image.data.size());
        if (!upload_status.success()) {
            hrx_buffer_release(buffer);
            status.log("upload prepared constant %s", image.name.c_str());
            status.append(upload_status);
            continue;
        }
        prepared.program_constants.push_back(
            make_program_constant_buffer(image.value, image.name, buffer, image.data.size()));
    }
    if (!status.success()) {
        return status;
    }

    status.append(bind_prepared_command_list_program_constants(prepared, prepared.initialization_commands));
    status.append(bind_prepared_command_list_program_constants(prepared, prepared.commands));
    return status;
}

static Status rebind_prepared_host_staging(const CommandProgramExecutionContext & context,
                                           const CommandProgramBindings & bindings, PreparedCommandProgram & prepared) {
    Status status;
    prepared.host_trace_tensors.clear();
    std::vector<HostStagingSlice> slices;
    for (HostStagingBuffer & staging : prepared.host_staging) {
        const CommandProgramBinding * binding = bindings.find(ValueId(staging.value));
        if (binding == nullptr || binding->host_data == nullptr || binding->length != staging.length ||
            binding->offset > binding->capacity || binding->length > binding->capacity - binding->offset) {
            status.log("live host binding does not match prepared value %d", staging.value);
            continue;
        }
        const uintptr_t base = reinterpret_cast<uintptr_t>(binding->host_data);
        if (binding->offset > std::numeric_limits<uintptr_t>::max() - base) {
            status.log("overflowed live host address for value %d", staging.value);
            continue;
        }
        slices.push_back({ staging.value, base + binding->offset, staging.length });
        if (mtp_host_trace.phase != nullptr && binding->tensor != nullptr) {
            prepared.host_trace_tensors.push_back({ staging.value, binding->tensor });
        }
    }
    std::vector<HostStagingGroup> groups;
    status.append(plan_host_staging_groups(std::move(slices), groups));
    if (!status.success()) {
        return status;
    }
    bool compatible = true;
    std::unordered_set<hrx_buffer_t> used;
    for (const auto & group : groups) {
        const auto * first = find_host_staging(prepared.host_staging, group.slices.front().value);
        compatible &= first != nullptr && first->buffer != nullptr && used.insert(first->buffer).second;
        for (const auto & slice : group.slices) {
            const auto * existing = find_host_staging(prepared.host_staging, slice.value);
            compatible &= existing != nullptr && first != nullptr && existing->buffer == first->buffer &&
                          existing->offset == slice.address - group.base;
        }
    }
    if (compatible) {
        for (auto & staging : prepared.host_staging) {
            const auto * binding = bindings.find(ValueId(staging.value));
            staging.host_data = static_cast<uint8_t *>(binding->host_data) + binding->offset;
        }
    } else {
        // A cached graph shape can acquire a different physical alias topology.
        // Finish prior uses before replacing their shared allocations; updated
        // command references below also invalidate any recorded graph.
        if (ErrorResult error = take_status(hrx_stream_synchronize(context.stream))) {
            status.log("synchronize changed host staging layout: %s", error->c_str());
            return status;
        }
        std::vector<HostStagingBuffer> rebound;
        status.append(allocate_host_staging_groups(context.device, prepared.host_staging, groups, rebound));
        if (status.success()) {
            prepared.host_staging = std::move(rebound);
            prepared.graph_bindings_dirty = true;
        }
    }
    return status;
}

// Prepared programs are cached per command-program shape, so a single prepared program is replayed for every
// split that shares that shape -- for example the residual ADD of all 48 layers. Host staging and transient
// arena bindings are refreshed per call, but GraphValue bindings point straight at ggml tensor memory, whose
// address differs for every one of those splits. Without this refresh the program keeps the addresses captured
// the first time it ran, so every later split reads and writes the first split's tensors and leaves its own
// destination untouched. Returns true through `changed` when any address moved, so a recorded graph that baked
// these references in can be re-recorded.
static Status rebind_prepared_graph_value_list(const CommandProgramBindings & bindings,
                                               const std::vector<HostStagingBuffer> & host_staging,
                                               std::vector<PreparedCommand> & commands,
                                               bool &                         changed) {
    Status status;
    for (PreparedCommand & command : commands) {
        for (PreparedCommandBinding & binding : command.kernel.bindings) {
            if (binding.binding.origin != CommandBindingOrigin::GraphValue) {
                continue;
            }
            const CommandProgramBinding * concrete = bindings.find(binding.binding.value);
            const auto * staged = find_host_staging(host_staging, binding.binding.value.value);
            // Immutable resident weights keep their cache lease. Mutable host
            // aliases and direct device values both need live buffer/offsets.
            if (concrete == nullptr || (concrete->buffer == nullptr && staged == nullptr)) {
                continue;
            }
            if (binding.binding.offset > concrete->length ||
                binding.binding.length > concrete->length - binding.binding.offset) {
                status.log("live graph binding for value %d is outside runtime binding length %zu",
                           binding.binding.value.value, concrete->length);
                continue;
            }
            const ResolvedBufferRef ref = {
                staged != nullptr ? staged->buffer : concrete->buffer,
                (staged != nullptr ? staged->offset : concrete->offset) + binding.binding.offset,
                binding.binding.length,
            };
            if (ref.buffer != binding.ref.buffer || ref.offset != binding.ref.offset ||
                ref.length != binding.ref.length) {
                binding.ref = ref;
                changed     = true;
            }
        }
    }
    return status;
}

static Status rebind_prepared_graph_values(const CommandProgramBindings & bindings,
                                           PreparedCommandProgram &       prepared,
                                           bool &                         changed) {
    Status status;
    status.append(rebind_prepared_graph_value_list(bindings, prepared.host_staging, prepared.initialization_commands, changed));
    status.append(rebind_prepared_graph_value_list(bindings, prepared.host_staging, prepared.commands, changed));
    prepared.graph_bindings_dirty |= changed;
    return status;
}

// HRX_TRACE_PROGRAM=1: dump the fully materialized command program right before launch -- every kernel
// binding with its origin, access, resolved buffer/offset, plus each host staging buffer. This is the only
// view that shows what the GPU is actually given, after value mapping, host materialization and transient
// arena assignment have all been applied.
static void trace_prepared_program(const char * label, const PreparedCommandProgram & prepared) {
    static const bool enabled  = hrx_environment_flag("HRX_TRACE_PROGRAM");
    static int        remaining = 6;
    if (!enabled || remaining <= 0) {
        return;
    }
    --remaining;
    GGML_LOG_ERROR("HRX program %s: init_commands=%zu commands=%zu staging=%zu weights=%zu\n", label,
                   prepared.initialization_commands.size(), prepared.commands.size(), prepared.host_staging.size(),
                   prepared.resident_host_weights.size());
    for (const PreparedCommand & command : prepared.commands) {
        GGML_LOG_ERROR("HRX program   command ord=%u kind=%d bindings=%zu exec=%p\n", command.ordinal,
                       static_cast<int>(command.kind), command.kernel.bindings.size(),
                       static_cast<const void *>(command.kernel.executable.get()));
        for (const PreparedCommandBinding & binding : command.kernel.bindings) {
            GGML_LOG_ERROR("HRX program     bind name=%s value=%d origin=%d access=%d buffer=%p offset=%zu len=%zu\n",
                           binding.binding.name.c_str(), binding.binding.value.value,
                           static_cast<int>(binding.binding.origin), static_cast<int>(binding.binding.access),
                           static_cast<const void *>(binding.ref.buffer), binding.ref.offset, binding.ref.length);
        }
    }
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        GGML_LOG_ERROR("HRX program   staging value=%d upload=%d download=%d buffer=%p host=%p offset=%zu len=%zu\n", staging.value,
                       static_cast<int>(staging.upload), static_cast<int>(staging.download),
                       static_cast<const void *>(staging.buffer), static_cast<const void *>(staging.host_data),
                       staging.offset, staging.length);
    }
}

static const ggml_tensor * host_trace_tensor(const PreparedCommandProgram & prepared, int32_t value) {
    for (const auto & entry : prepared.host_trace_tensors) {
        if (entry.value == value) {
            return entry.tensor;
        }
    }
    return nullptr;
}

static void trace_mtp_host_boundary(const PreparedCommandProgram & prepared, const HostStagingBuffer & staging,
                                    const char * boundary) {
    auto & trace = mtp_host_trace;
    if (trace.phase == nullptr || trace.events >= 512 || staging.host_data == nullptr) {
        return;
    }
    const auto * tensor = host_trace_tensor(prepared, staging.value);
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
        staging.length != ggml_nbytes(tensor) || staging.length > 1024 * 1024 ||
        staging.length > 64 * 1024 * 1024 - trace.bytes) {
        return;
    }
    ++trace.events;
    trace.bytes += staging.length;
    const size_t count = staging.length / sizeof(float);
    size_t nonfinite = 0, first_bad = count;
    uint32_t first_bits = 0;
    float minimum = 0.0f, maximum = 0.0f;
    size_t finite = 0;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; ++i) {
        float value;
        uint32_t bits;
        const auto * data = static_cast<const uint8_t *>(staging.host_data) + i * sizeof(value);
        std::memcpy(&value, data, sizeof(value));
        std::memcpy(&bits, data, sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
        if (std::isfinite(value)) {
            minimum = finite == 0 ? value : std::min(minimum, value);
            maximum = finite == 0 ? value : std::max(maximum, value);
            ++finite;
        } else {
            if (nonfinite++ == 0) {
                first_bad = i;
                first_bits = bits;
            }
        }
    }
    GGML_LOG_ERROR("HRX MTP host %s phase=%s call=%llu program=%zu event=%zu value=%d name=%s op=%s "
                   "host=%p device=%p device_offset=%zu bytes=%zu ne=[%lld,%lld,%lld,%lld] "
                   "nonfinite=%zu first_bad=%zu bits=%08x min=%g max=%g hash=%016llx\n",
                   boundary, trace.phase, static_cast<unsigned long long>(trace.call), trace.program, trace.events,
                   staging.value, tensor->name, ggml_op_name(tensor->op), staging.host_data,
                   static_cast<const void *>(staging.buffer), staging.offset, staging.length,
                   static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
                   static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]),
                   nonfinite, first_bad, first_bits, minimum, maximum, static_cast<unsigned long long>(hash));
    if (nonfinite != 0 && std::strcmp(boundary, "upload") == 0) {
        for (size_t i = 0; i < 6; ++i) {
            const auto * source = tensor->src[i];
            if (source != nullptr) {
                GGML_LOG_ERROR("HRX MTP host CPU-producer edge: output=%s src[%zu]=%s op=%s type=%s "
                               "data=%p ne=[%lld,%lld,%lld,%lld] (metadata only; source may be overwritten)\n",
                               tensor->name, i, source->name, ggml_op_name(source->op), ggml_type_name(source->type),
                               source->data, static_cast<long long>(source->ne[0]), static_cast<long long>(source->ne[1]),
                               static_cast<long long>(source->ne[2]), static_cast<long long>(source->ne[3]));
            }
        }
    }
    size_t printed = 0;
    for (const auto & command : prepared.commands) {
        for (size_t i = 0; i < command.kernel.bindings.size() && printed < 8; ++i) {
            const auto & binding = command.kernel.bindings[i];
            if (binding.binding.origin == CommandBindingOrigin::GraphValue &&
                binding.binding.value.value == staging.value) {
                ++printed;
                GGML_LOG_ERROR("HRX MTP host use: value=%d ord=%u kernel=%llu binding=%zu access=%d "
                               "buffer=%p offset=%zu span=%zu\n",
                               staging.value, command.ordinal,
                               static_cast<unsigned long long>(command.kernel.specialization.kernel_id),
                               i, static_cast<int>(binding.binding.access), static_cast<const void *>(binding.ref.buffer),
                               binding.ref.offset, binding.ref.length);
            }
        }
    }
    printed = 0;
    for (const auto & other : prepared.host_staging) {
        if (other.value == staging.value || other.host_data == nullptr || (!other.upload && !other.download)) {
            continue;
        }
        const uintptr_t a = reinterpret_cast<uintptr_t>(staging.host_data);
        const uintptr_t b = reinterpret_cast<uintptr_t>(other.host_data);
        if ((a <= b ? b - a < staging.length : a - b < other.length) && printed++ < 8) {
            const auto * alias = host_trace_tensor(prepared, other.value);
            GGML_LOG_ERROR("HRX MTP host overlapping range: value=%d other=%d name=%s host=%p bytes=%zu "
                           "device=%p device_offset=%zu upload=%d download=%d (overlap alone does not establish a live alias)\n",
                           staging.value, other.value, alias == nullptr ? "?" : alias->name, other.host_data,
                           other.length, static_cast<const void *>(other.buffer), other.offset, static_cast<int>(other.upload),
                           static_cast<int>(other.download));
        }
    }
}

static Status upload_prepared_host_staging(const CommandProgramExecutionContext & context,
                                           const PreparedCommandProgram &         prepared) {
    Status status;
    if (mtp_host_trace.phase != nullptr) {
        ++mtp_host_trace.program;
    }
    if (prepared.host_staging.empty()) {
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager");
        return status;
    }
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        if (!staging.upload) {
            continue;
        }
        trace_mtp_host_boundary(prepared, staging, "upload");
        Status upload_status =
            context.host_transfers->upload_async(context.stream, staging.host_data, staging.buffer, staging.offset, staging.length);
        status.append(upload_status);
    }
    return status;
}

static Status download_prepared_host_staging(const CommandProgramExecutionContext & context,
                                             const PreparedCommandProgram &         prepared) {
    Status status;
    if (prepared.host_staging.empty()) {
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager");
        return status;
    }
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        if (!staging.download) {
            continue;
        }
        Status download_status = context.host_transfers->download_synchronous(
            context.stream, staging.buffer, staging.offset, staging.host_data, staging.length);
        status.append(download_status);
        if (download_status.success()) {
            trace_mtp_host_boundary(prepared, staging, "download");
        }
    }
    return status;
}

static PreparedCommand make_prepared_command_shape(const ResolvedCommand & command) {
    PreparedCommand prepared;
    prepared.ordinal               = command.ordinal;
    prepared.kind                  = command.kind;
    prepared.kernel.specialization = command.kernel;
    prepared.kernel.bindings.reserve(command.bindings.size());
    for (const ResolvedCommandBinding & binding : command.bindings) {
        prepared.kernel.bindings.push_back({
            binding.binding,
            { binding.ref.buffer, binding.ref.offset, binding.ref.length },
        });
    }
    return prepared;
}

static Status prepare_kernel_command(const CommandProgramExecutionContext & context,
                                     const ResolvedCommand &                command,
                                     PreparedCommand &                      prepared,
                                     KernelExecutableRef &                  executable_ref) {
    Status            status;
    const std::string command_context = format_resolved_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        status.log("unsupported command kind in %s", command_context.c_str());
        return status;
    }
    Dispatch dispatch = build_dispatch(command);

    KernelResolveResult resolved =
        resolve_kernel_definition(*context.corpus, context.target, dispatch.kernel.kernel_id);
    if (!resolved.found()) {
        status.log("%s: %s", command_context.c_str(),
                   format_kernel_resolve_error(resolved, dispatch.kernel.kernel_id).c_str());
        return status;
    }

    prepared       = make_prepared_command_shape(command);
    trace_kernel_preparation(*resolved.definition, dispatch);
    executable_ref = context.kernel_executables->get_or_compile(
        { context.device, context.target }, *resolved.definition, dispatch, prepared.kernel.constants);
    if (!executable_ref.valid()) {
        status.log("failed to prepare %s", command_context.c_str());
        return status;
    }
    return status;
}

static bool validate_q4_routed_ids(const CommandProgramExecutionContext & context, const PreparedCommand & command) {
    const auto & bindings = command.kernel.bindings;
    const auto & parameters = command.kernel.specialization.integer_parameters;
    const auto parameter = [&parameters](const char * name) -> int64_t {
        const auto found = parameters.find(name);
        return found == parameters.end() ? -1 : found->second;
    };
    const int64_t tokens = parameter("token_count");
    const int64_t routes = parameter("route_count");
    const int64_t stride = parameter("route_stride");
    const int64_t experts = parameter("expert_count");
    const int64_t outputs = parameter("output_size");
    const size_t required = q4_routed_gate_up_id_byte_count(tokens, routes, stride, experts);
    if (required == 0 || outputs < 1 || outputs > 4096 ||
        (bindings.size() != 5 && bindings.size() != 7) || context.host_transfers == nullptr) {
        GGML_LOG_ERROR("HRX Q4 routed gate/up: invalid validation layout at ord=%u\n", command.ordinal);
        return false;
    }
    const auto & config = command.kernel.specialization.compile_parameters;
    const auto input_config = config.find("qwen3_moe.routed_gate_up.input_size");
    char * end = nullptr;
    const int64_t width = input_config == config.end() ? 0 : std::strtoll(input_config->second.c_str(), &end, 10);
    if (width < 512 || width > 32768 || width % 512 != 0 || end == nullptr || *end != '\0') {
        GGML_LOG_ERROR("HRX Q4 routed gate/up: invalid input width at ord=%u\n", command.ordinal);
        return false;
    }
    const auto & ids = bindings[1].ref;
    const size_t weight_bytes = static_cast<size_t>(experts * outputs * (width / 256)) * 144;
    const size_t input_bytes = static_cast<size_t>(tokens * (width / 128)) * 144;
    const size_t output_bytes = static_cast<size_t>(tokens * routes * outputs) * sizeof(float);
    if (ids.offset % sizeof(int32_t) != 0 || ids.length < required ||
        bindings[0].ref.length < input_bytes || bindings[2].ref.length < weight_bytes ||
        bindings[3].ref.length < weight_bytes || bindings[4].ref.length < output_bytes ||
        bindings[0].ref.offset % 2 != 0 || bindings[2].ref.offset % 2 != 0 ||
        bindings[3].ref.offset % 2 != 0 || bindings[4].ref.offset % sizeof(float) != 0) {
        GGML_LOG_ERROR("HRX Q4 routed gate/up: misaligned or undersized binding at ord=%u\n", command.ordinal);
        return false;
    }
    if (bindings.size() == 7) {
        const size_t groups = static_cast<size_t>(tokens * routes * (outputs / 128));
        if (tokens != 1 || outputs % 128 != 0 || bindings[5].ref.offset % 16 != 0 ||
            bindings[5].ref.length < groups * sizeof(int32_t) || bindings[6].ref.offset % 2 != 0 ||
            bindings[6].ref.length < groups * 144) {
            GGML_LOG_ERROR("HRX Q4 routed gate/up: invalid next-Q8/counter span at ord=%u\n", command.ordinal);
            return false;
        }
    }
    // Check actual bound ranges, not GGML storage identities: gallocr can reuse
    // one physical allocation for distinct values inside a fused graph.
    for (size_t write = 4; write < bindings.size(); ++write) {
        const auto & output = bindings[write].ref;
        for (size_t other = 0; other < write; ++other) {
            const auto & input = bindings[other].ref;
            if (input.buffer == nullptr || output.buffer == nullptr ||
                (input.buffer == output.buffer &&
                 (input.offset <= output.offset ? output.offset - input.offset < input.length :
                                                  input.offset - output.offset < output.length))) {
                GGML_LOG_ERROR("HRX Q4 routed gate/up: overlapping/missing bindings %zu/%zu at ord=%u\n",
                               other, write, command.ordinal);
                return false;
            }
        }
    }
    std::vector<uint8_t> bytes(required);
    // Commands are submitted in producer order. This readback waits for router
    // writes on the same stream, on every invocation (including cached plans).
    const Status status = context.host_transfers->download_synchronous(
        context.stream, ids.buffer, ids.offset, bytes.data(), bytes.size());
    if (!status.success()) {
        GGML_LOG_ERROR("HRX Q4 routed gate/up: ordered ID readback failed at ord=%u: %s\n",
                       command.ordinal, status_first_error(status));
        return false;
    }
    const auto checked = validate_q4_routed_gate_up_id_bytes(bytes.data(), bytes.size(), tokens, routes, stride, experts);
    if (!checked.valid) {
        GGML_LOG_ERROR("HRX Q4 routed gate/up: invalid expert ID=%lld token=%lld route=%lld "
                       "experts=%lld stride=%lld ord=%u IDs buffer=%p offset=%zu span=%zu\n",
                       static_cast<long long>(checked.invalid_id), static_cast<long long>(checked.token),
                       static_cast<long long>(checked.route), static_cast<long long>(experts),
                       static_cast<long long>(stride), command.ordinal,
                       static_cast<const void *>(ids.buffer), ids.offset, bytes.size());
        return false;
    }
    if (hrx_environment_flag("HRX_TRACE_PROGRAM")) {
        GGML_LOG_ERROR("HRX Q4 routed gate/up: validated ord=%u experts=%lld stride=%lld token0 IDs:",
                       command.ordinal, static_cast<long long>(experts), static_cast<long long>(stride));
        for (int64_t route = 0; route < routes; ++route) {
            int32_t id;
            std::memcpy(&id, bytes.data() + route * sizeof(id), sizeof(id));
            GGML_LOG_ERROR(" %d", id);
        }
        GGML_LOG_ERROR("\n");
    }
    return true;
}

static bool validate_row_ids(const CommandProgramExecutionContext & context, const PreparedCommand & command) {
    if (is_q4_routed_gate_up_kernel(command.kernel.specialization.kernel_id)) {
        return validate_q4_routed_ids(context, command);
    }
    const bool set_rows = is_set_rows_kernel(command.kernel.specialization.kernel_id);
    const bool f32_get_rows = is_f32_get_rows_kernel(command.kernel.specialization.kernel_id);
    const bool f16_get_rows = is_qsa_f16_gather_kernel(command.kernel.specialization.kernel_id);
    const bool q4_embedding = is_q4_embedding_kernel(command.kernel.specialization.kernel_id);
    const bool embedding = q4_embedding || is_q8_embedding_kernel(command.kernel.specialization.kernel_id);
    if (!set_rows && !embedding && !f32_get_rows && !f16_get_rows) {
        return true;
    }
    const char * operation = set_rows ? "SET_ROWS" :
                            f16_get_rows ? "QSA F16 GET_ROWS" :
                            f32_get_rows ? "F32 GET_ROWS" : (q4_embedding ? "Q4 GET_ROWS" : "Q8 GET_ROWS");
    if (command.kernel.bindings.size() != 3 || context.host_transfers == nullptr) {
        GGML_LOG_ERROR("HRX %s: missing validation bindings\n", operation);
        return false;
    }
    const auto & source = command.kernel.bindings[0].ref;
    const auto & ids = command.kernel.bindings[1].ref;
    const auto & output = command.kernel.bindings[2].ref;
    const auto overlaps_output = [&output](const auto & input) {
        return input.buffer == output.buffer &&
               (input.offset <= output.offset ? output.offset - input.offset < input.length :
                                                input.offset - output.offset < output.length);
    };
    // Allocator reuse can overlap distinct GGML storage roots. Check the bound ranges on every replay.
    if (overlaps_output(source) || overlaps_output(ids)) {
        GGML_LOG_ERROR("HRX %s: source or IDs overlap the destination\n", operation);
        return false;
    }
    const auto & parameters = command.kernel.specialization.integer_parameters;
    const size_t count = static_cast<size_t>(parameters.at(set_rows ? "row_count" : "token_count"));
    const size_t index_size = set_rows && parameters.at("index_i64") ? sizeof(int64_t) : sizeof(int32_t);
    const size_t stride = set_rows ? static_cast<size_t>(parameters.at("index_stride")) * index_size : index_size;
    const int64_t capacity = parameters.at(set_rows ? "cache_rows" : "vocabulary_count");
    if (f16_get_rows &&
        (parameters.at("hidden_size") != 128 || capacity < 1 || capacity > 4096 || count < 1 || count > 4096)) {
        GGML_LOG_ERROR("HRX %s: unsupported gather geometry\n", operation);
        return false;
    }
    if (embedding || f32_get_rows || f16_get_rows) {
        const size_t width = static_cast<size_t>(parameters.at("hidden_size"));
        const size_t row_bytes = f16_get_rows ? width * sizeof(ggml_fp16_t) :
                                f32_get_rows ? width * sizeof(float) :
                                q4_embedding ? width / 256 * 144 : width / 32 * 34;
        const size_t source_alignment = f32_get_rows ? sizeof(float) : sizeof(uint16_t);
        if (source.offset % source_alignment != 0 || ids.offset % sizeof(int32_t) != 0 ||
            output.offset % sizeof(float) != 0 ||
            source.length < row_bytes * static_cast<size_t>(capacity) ||
            output.length < width * count * sizeof(float)) {
            GGML_LOG_ERROR("HRX %s: misaligned or undersized binding\n", operation);
            return false;
        }
    }
    if (count == 0 || stride == 0 || ids.length < index_size ||
        (count - 1) > (ids.length - index_size) / stride) {
        GGML_LOG_ERROR("HRX %s: index binding is too small\n", operation);
        return false;
    }
    // Trusted producers (see DispatchBinding::trusted) are host-populated leaves with no
    // GPU-side producer in this graph, so their values cannot depend on anything this
    // graph computed. The allocator/geometry checks above still apply unconditionally;
    // only the synchronous GPU readback and per-row bounds/uniqueness scan below, which
    // exist to catch GPU-computed IDs, are unnecessary for them.
    if (command.kernel.bindings[1].binding.trusted) {
        return true;
    }
    std::vector<uint8_t> bytes((count - 1) * stride + index_size);
    // This waits on preceding producers, including GPU-generated top-k IDs.
    Status status = context.host_transfers->download_synchronous(
        context.stream, ids.buffer, ids.offset, bytes.data(), bytes.size());
    if (!status.success()) {
        GGML_LOG_ERROR("HRX %s: index validation readback failed: %s\n", operation, status_first_error(status));
        return false;
    }
    std::unordered_set<int64_t> seen;
    for (size_t row = 0; row < count; ++row) {
        int64_t id = 0;
        if (index_size == sizeof(int64_t)) {
            std::memcpy(&id, bytes.data() + row * stride, sizeof(id));
        } else {
            int32_t narrow = 0;
            std::memcpy(&narrow, bytes.data() + row * stride, sizeof(narrow));
            id = narrow;
        }
        if (id < 0 || id >= capacity || (set_rows && !seen.insert(id).second)) {
            GGML_LOG_ERROR("HRX %s: invalid%s row ID %lld at source row %zu (capacity %lld)\n",
                           operation, set_rows ? " or duplicate" : "",
                           static_cast<long long>(id), row, static_cast<long long>(capacity));
            return false;
        }
    }
    return true;
}

static bool execute_prepared_kernel_command(const CommandProgramExecutionContext & context,
                                            const PreparedCommand &                command) {
    const bool timing = hrx_time_compute_enabled();
    using clock       = std::chrono::steady_clock;
    const clock::time_point t_begin = timing ? clock::now() : clock::time_point{};
    const std::string command_context = format_prepared_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        GGML_LOG_ERROR("%s: unsupported command kind in %s\n", __func__, command_context.c_str());
        return false;
    }
    if (command.kernel.executable == nullptr) {
        GGML_LOG_ERROR("%s: missing kernel executable for %s\n", __func__, command_context.c_str());
        return false;
    }
    if (!validate_row_ids(context, command)) {
        return false;
    }

    std::vector<hrx_buffer_ref_t> refs;
    refs.reserve(command.kernel.bindings.size());
    for (const PreparedCommandBinding & binding : command.kernel.bindings) {
        refs.push_back({ binding.ref.buffer, binding.ref.offset, binding.ref.length });
    }

    trace_kernel_record(command, refs);

    const KernelExecutable & executable = *command.kernel.executable;
    hrx_dispatch_config_t    config     = {
        { executable.launch.workgroup_count[0], executable.launch.workgroup_count[1],
         executable.launch.workgroup_count[2] },
        { executable.launch.workgroup_size[0],  executable.launch.workgroup_size[1],
         executable.launch.workgroup_size[2]  },
        executable.launch.subgroup_size,
    };
    const clock::time_point t_dispatch = timing ? clock::now() : clock::time_point{};
    const bool profiling = hrx_profile_dispatches_enabled() && context.device_timing != nullptr;
    DeviceTimingManager::GraphMeasurement dispatch_measurement;
    if (profiling) {
        dispatch_measurement = context.device_timing->begin_graph_measurement(context.stream);
    }
    if (ErrorResult error = take_status(hrx_stream_dispatch(
            context.stream, executable.executable, executable.export_ordinal, &config, command.kernel.constants.data(),
            command.kernel.constants.size(), refs.data(), refs.size(), 0))) {
        if (profiling) {
            context.device_timing->cancel_graph_measurement(dispatch_measurement);
        }
        GGML_LOG_ERROR("%s: failed to execute %s: %s\n", __func__, command_context.c_str(), error->c_str());
        return false;
    }
    if (profiling) {
        // Bracketing the single dispatch above with begin/finish measurement forces a
        // synchronize per kernel, which is why this path is diagnostic-only (see the flag doc
        // comment); it is what makes the resulting per-kernel-id total_ms a real device time
        // rather than a host submission/queueing estimate.
        const std::optional<double> device_ms = context.device_timing->finish_graph_measurement(dispatch_measurement);
        if (device_ms.has_value()) {
            record_dispatch_profile_sample(command.kernel.specialization.kernel_id, *device_ms, context);
            uint64_t samples_snapshot;
            {
                std::lock_guard<std::mutex> lock(g_dispatch_profile_mutex);
                samples_snapshot = g_dispatch_profile_samples;
            }
            if (samples_snapshot % 20 == 0) {
                std::lock_guard<std::mutex> lock(g_dispatch_profile_mutex);
                log_dispatch_profile_summary_locked(context);
            }
        }
    }
    if (timing) {
        const clock::time_point t_end   = clock::now();
        const auto              elapsed = [](clock::time_point a, clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        static uint64_t dispatches   = 0;
        static double   glue_ms      = 0.0;
        static double   dispatch_ms  = 0.0;
        dispatches += 1;
        glue_ms += elapsed(t_begin, t_dispatch);
        dispatch_ms += elapsed(t_dispatch, t_end);
        if (dispatches % 50000 == 0) {
            const double total = glue_ms + dispatch_ms;
            GGML_LOG_INFO("HRX record: dispatches=%llu glue=%.1fms(%.0f%%) hrx_stream_dispatch=%.1fms(%.0f%%)\n",
                          static_cast<unsigned long long>(dispatches), glue_ms, 100.0 * glue_ms / total, dispatch_ms,
                          100.0 * dispatch_ms / total);
        }
    }
    return true;
}

static void prepare_command_list(const CommandProgramExecutionContext & context,
                                 const std::vector<ResolvedCommand> &   commands,
                                 std::vector<PreparedCommand> &         prepared_commands,
                                 std::vector<KernelExecutableRef> &     executable_refs,
                                 Status &                               status) {
    prepared_commands.reserve(commands.size());
    executable_refs.reserve(commands.size());
    for (const ResolvedCommand & command : commands) {
        PreparedCommand     prepared_command;
        KernelExecutableRef executable_ref;
        Status              command_status = prepare_kernel_command(context, command, prepared_command, executable_ref);
        if (command_status.success()) {
            prepared_commands.push_back(std::move(prepared_command));
            executable_refs.push_back(std::move(executable_ref));
        } else {
            status.append(command_status);
        }
    }
}

static void materialize_command_list_executables(const CommandProgramExecutionContext &   context,
                                                 std::vector<PreparedCommand> &           prepared_commands,
                                                 const std::vector<KernelExecutableRef> & executable_refs,
                                                 Status &                                 status) {
    for (size_t i = 0; i < prepared_commands.size(); ++i) {
        PreparedCommand & command = prepared_commands[i];
        command.kernel.executable = context.kernel_executables->materialize(
            { context.device, context.target }, executable_refs[i], command.kernel.constants);
        if (command.kernel.executable == nullptr) {
            status.log("failed to prepare %s", format_prepared_command_context(command).c_str());
        }
    }
}

static bool bind_prepared_command_list_transients(const CommandProgram &              commands,
                                                  const TransientArenaAllocationRef & transient_allocation,
                                                  std::vector<PreparedCommand> &      prepared_commands) {
    for (PreparedCommand & command : prepared_commands) {
        for (PreparedCommandBinding & binding : command.kernel.bindings) {
            if (binding.binding.origin != CommandBindingOrigin::Transient) {
                continue;
            }
            const TransientAllocation * allocation =
                find_transient_allocation(commands.transients, binding.binding.value);
            if (allocation == nullptr) {
                GGML_LOG_ERROR("%s: %s has no transient allocation\n", __func__,
                               format_command_binding(binding.binding).c_str());
                return false;
            }
            if (binding.binding.offset > allocation->size ||
                binding.binding.length > allocation->size - binding.binding.offset ||
                commands.transients.arena_size > transient_allocation.capacity) {
                GGML_LOG_ERROR("%s: %s is outside transient arena\n", __func__,
                               format_command_binding(binding.binding).c_str());
                return false;
            }
            binding.ref = {
                transient_allocation.buffer,
                allocation->arena_offset + binding.binding.offset,
                binding.binding.length,
            };
        }
    }
    return true;
}

static void report_router_logits(const char * label, const void * data, size_t count) {
    size_t finite = 0, nan = 0, positive_inf = 0, negative_inf = 0;
    size_t first_nonfinite = count;
    float minimum = 0.0f, maximum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float value;
        std::memcpy(&value, static_cast<const uint8_t *>(data) + i * sizeof(value), sizeof(value));
        if (std::isfinite(value)) {
            minimum = finite == 0 ? value : std::min(minimum, value);
            maximum = finite == 0 ? value : std::max(maximum, value);
            ++finite;
        } else if (std::isnan(value)) {
            ++nan;
        } else if (value > 0.0f) {
            ++positive_inf;
        } else {
            ++negative_inf;
        }
        if (!std::isfinite(value) && first_nonfinite == count) {
            first_nonfinite = i;
        }
    }
    GGML_LOG_ERROR("HRX router failure %s: count=%zu finite=%zu nan=%zu +inf=%zu -inf=%zu min=%g max=%g\n",
                   label, count, finite, nan, positive_inf, negative_inf, minimum, maximum);
    if (first_nonfinite < count) {
        uint32_t bits;
        std::memcpy(&bits, static_cast<const uint8_t *>(data) + first_nonfinite * sizeof(float), sizeof(bits));
        GGML_LOG_ERROR("HRX router failure %s: first_nonfinite=%zu bits=%08x\n", label, first_nonfinite, bits);
    }
    for (size_t i = 0; i < std::min(count, size_t{16}); ++i) {
        float value;
        uint32_t bits;
        const auto * at = static_cast<const uint8_t *>(data) + i * sizeof(value);
        std::memcpy(&value, at, sizeof(value));
        std::memcpy(&bits, at, sizeof(bits));
        GGML_LOG_ERROR("HRX router failure %s[%zu]=%.9g bits=%08x\n", label, i, value, bits);
    }
}

static void diagnose_router_input(const CommandProgramExecutionContext & context, const char * label,
                                  const PreparedCommandBinding & binding,
                                  const std::vector<HostStagingBuffer> & staging, size_t max_floats) {
    const auto & input = binding.ref;
    const size_t bytes = std::min(input.length, max_floats * sizeof(float));
    if (input.buffer == nullptr || bytes == 0 || bytes % sizeof(float) != 0) {
        return;
    }
    GGML_LOG_ERROR("HRX router failure %s: value=%d buffer=%p offset=%zu span=%zu\n",
                   label, binding.binding.value.value, static_cast<const void *>(input.buffer),
                   input.offset, input.length);
    std::vector<uint8_t> device(bytes);
    const Status status = context.host_transfers->download_synchronous(
        context.stream, input.buffer, input.offset, device.data(), bytes);
    if (!status.success()) {
        GGML_LOG_ERROR("HRX router failure %s readback failed: %s\n", label, status_first_error(status));
        return;
    }
    report_router_logits(label, device.data(), bytes / sizeof(float));
    for (const auto & host : staging) {
        if (host.buffer != input.buffer || host.host_data == nullptr || input.offset < host.offset ||
            input.offset - host.offset > host.length || bytes > host.length - (input.offset - host.offset)) {
            continue;
        }
        const auto * data = static_cast<const uint8_t *>(host.host_data) + (input.offset - host.offset);
        GGML_LOG_ERROR("HRX router failure CPU mirror (%s): value=%d host=%p upload=%d download=%d equal=%d\n",
                       label, host.value, static_cast<const void *>(data), static_cast<int>(host.upload),
                       static_cast<int>(host.download), std::memcmp(data, device.data(), bytes) == 0);
        report_router_logits("CPU staging mirror", data, bytes / sizeof(float));
    }
}

static void diagnose_q4_router_failure(const CommandProgramExecutionContext & context,
                                      const std::vector<PreparedCommand> & commands, size_t failed,
                                      const std::vector<HostStagingBuffer> & staging) {
    const PreparedCommand & consumer = commands[failed];
    if (!is_q4_routed_gate_up_kernel(consumer.kernel.specialization.kernel_id) ||
        consumer.kernel.bindings.size() < 2 || context.host_transfers == nullptr) {
        return;
    }
    const auto & ids = consumer.kernel.bindings[1].ref;
    for (size_t i = failed; i-- > 0;) {
        const PreparedCommand & producer = commands[i];
        if (!is_moe_router_top8_kernel(producer.kernel.specialization.kernel_id) ||
            producer.kernel.bindings.size() != 3) {
            continue;
        }
        const auto & output_ids = producer.kernel.bindings[1].ref;
        if (output_ids.buffer != ids.buffer || output_ids.offset != ids.offset) {
            continue;
        }
        GGML_LOG_ERROR("HRX router failure producer: %s\n", format_prepared_command_context(producer).c_str());
        for (size_t binding = 0; binding < producer.kernel.bindings.size(); ++binding) {
            const auto & bound = producer.kernel.bindings[binding];
            GGML_LOG_ERROR("HRX router failure binding[%zu]: value=%d buffer=%p offset=%zu span=%zu\n",
                           binding, bound.binding.value.value, static_cast<const void *>(bound.ref.buffer),
                           bound.ref.offset, bound.ref.length);
        }
        // Failure-only: no extra synchronization or partition boundary before
        // the error. These are post-router bytes, not a pre-launch snapshot.
        diagnose_router_input(context, "post-router GPU logits", producer.kernel.bindings[0], staging, 512);
        static constexpr KernelCatalogRef pack_kernel =
            GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_quantize_q8_1_x4_f32");
        for (size_t pack = i + 1; pack < failed; ++pack) {
            const auto & command = commands[pack];
            if (command.kernel.specialization.kernel_id != pack_kernel.id ||
                command.kernel.bindings.size() != 2 || consumer.kernel.bindings.empty()) {
                continue;
            }
            const auto & packed = command.kernel.bindings[1].ref;
            const auto & gate_input = consumer.kernel.bindings[0].ref;
            if (packed.buffer == gate_input.buffer && packed.offset == gate_input.offset) {
                diagnose_router_input(context, "post-pack GPU FFN input", command.kernel.bindings[0], staging, 4096);
            }
        }
        return;
    }
}

static bool execute_prepared_command_list(const CommandProgramExecutionContext & context,
                                          const std::vector<PreparedCommand> &   commands,
                                          const std::vector<HostStagingBuffer> & staging) {
    for (size_t i = 0; i < commands.size(); ++i) {
        const PreparedCommand & command = commands[i];
        if (!execute_prepared_kernel_command(context, command)) {
            diagnose_q4_router_failure(context, commands, i, staging);
            return false;
        }
    }
    return true;
}

struct GraphDependencyChain {
    hrx_graph_node_t last = nullptr;

    const hrx_graph_node_t * deps() const { return last == nullptr ? nullptr : &last; }
    size_t dep_count() const { return last == nullptr ? 0 : 1; }
    void update(hrx_graph_node_t node) { last = node; }
};

static Status record_completion_counter_fill(hrx_graph_t                         graph,
                                             GraphDependencyChain &              chain,
                                             const CommandProgram &              commands,
                                             const TransientArenaAllocationRef & allocation) {
    Status status;
    if (commands.completion_counters.byte_count == 0) {
        return status;
    }
    if (allocation.buffer == nullptr) {
        status.log("command program has completion counters without a transient arena allocation");
        return status;
    }
    if (commands.completion_counters.arena_offset > commands.transients.arena_size ||
        commands.completion_counters.byte_count >
            commands.transients.arena_size - commands.completion_counters.arena_offset) {
        status.log("completion counter graph fill is outside transient arena length %zu",
                   commands.transients.arena_size);
        return status;
    }

    hrx_graph_fill_buffer_node_attrs_t attrs = {
        { allocation.buffer, commands.completion_counters.arena_offset, commands.completion_counters.byte_count },
        0,
        sizeof(uint32_t),
    };
    hrx_graph_node_t node = nullptr;
    if (ErrorResult error =
            take_status(hrx_graph_add_fill_buffer_node(graph, chain.deps(), chain.dep_count(), &attrs, &node))) {
        status.log("record completion counter fill: %s", error->c_str());
        return status;
    }
    chain.update(node);
    return status;
}

static Status record_prepared_kernel_command(hrx_graph_t                  graph,
                                             GraphDependencyChain &       chain,
                                             const PreparedCommand &      command) {
    Status status;
    const std::string command_context = format_prepared_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        status.log("unsupported command kind in %s", command_context.c_str());
        return status;
    }
    if (command.kernel.executable == nullptr) {
        status.log("missing kernel executable for %s", command_context.c_str());
        return status;
    }

    std::vector<hrx_buffer_ref_t> refs;
    refs.reserve(command.kernel.bindings.size());
    for (const PreparedCommandBinding & binding : command.kernel.bindings) {
        if (binding.ref.buffer == nullptr) {
            status.log("%s has unbound buffer in %s", format_command_binding(binding.binding).c_str(),
                       command_context.c_str());
            continue;
        }
        refs.push_back({ binding.ref.buffer, binding.ref.offset, binding.ref.length });
    }
    if (!status.success()) {
        return status;
    }
    trace_kernel_record(command, refs);

    const KernelExecutable & executable = *command.kernel.executable;
    hrx_graph_kernel_node_attrs_t attrs = {
        executable.executable,
        executable.export_ordinal,
        {
            { executable.launch.workgroup_count[0], executable.launch.workgroup_count[1],
              executable.launch.workgroup_count[2] },
            { executable.launch.workgroup_size[0], executable.launch.workgroup_size[1],
              executable.launch.workgroup_size[2] },
            executable.launch.subgroup_size,
        },
        command.kernel.constants.data(),
        command.kernel.constants.size(),
        refs.data(),
        refs.size(),
        0,
    };
    hrx_graph_node_t node = nullptr;
    if (ErrorResult error =
            take_status(hrx_graph_add_kernel_node(graph, chain.deps(), chain.dep_count(), &attrs, &node))) {
        status.log("record %s: %s", command_context.c_str(), error->c_str());
        return status;
    }
    chain.update(node);
    return status;
}

static Status record_prepared_command_list(hrx_graph_t                        graph,
                                           GraphDependencyChain &             chain,
                                           const std::vector<PreparedCommand> & commands,
                                           size_t &                           dispatch_count) {
    Status status;
    for (const PreparedCommand & command : commands) {
        Status command_status = record_prepared_kernel_command(graph, chain, command);
        if (!command_status.success()) {
            status.append(command_status);
            return status;
        }
        ++dispatch_count;
    }
    return status;
}

static RecordedCommandGraph record_prepared_command_graph(const CommandProgramExecutionContext & context,
                                                          const CommandProgram &                 commands,
                                                          const PreparedCommandProgram &         prepared,
                                                          const TransientArenaAllocationRef &    allocation) {
    RecordedCommandGraph recorded;
    if (context.device == nullptr) {
        recorded.status.log("missing HRX device for graph replay");
        return recorded;
    }

    hrx_graph_t graph = nullptr;
    if (ErrorResult error = take_status(hrx_graph_create(context.device, 0, &graph))) {
        recorded.status.log("create HRX graph replay: %s", error->c_str());
        return recorded;
    }
    recorded.graph = graph;

    GraphDependencyChain chain;
    recorded.status.append(record_completion_counter_fill(recorded.graph, chain, commands, allocation));
    if (!recorded.status.success()) {
        return recorded;
    }
    recorded.status.append(record_prepared_command_list(recorded.graph, chain, prepared.initialization_commands,
                                                        recorded.dispatch_count));
    if (!recorded.status.success()) {
        return recorded;
    }
    recorded.status.append(record_prepared_command_list(recorded.graph, chain, prepared.commands,
                                                        recorded.dispatch_count));
    if (!recorded.status.success()) {
        return recorded;
    }

    hrx_graph_exec_t exec = nullptr;
    if (ErrorResult error = take_status(hrx_graph_instantiate(recorded.graph, 0, &exec))) {
        recorded.status.log("instantiate HRX graph replay: %s", error->c_str());
        return recorded;
    }
    recorded.exec = exec;
    recorded.bound_transient_arena_allocation_id = prepared.bound_transient_arena_allocation_id;
    return recorded;
}

}  // namespace

PreparedCommandProgram prepare_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings) {
    PreparedCommandProgram prepared;
    prepared.status = command_program_metadata_context_valid(context);
    if (!prepared.status.success()) {
        return prepared;
    }

    const VerificationResult verification = verify_command_program(commands, *context.corpus, context.target);
    if (!verification.valid()) {
        prepared.status.append(verification.status);
        return prepared;
    }
    if (!bindings.valid()) {
        prepared.status.append(bindings.status);
        return prepared;
    }

    TransientArenaAllocationRef transient_allocation;
    prepared.status = ensure_transient_arena(context, commands, transient_allocation);
    if (!prepared.status.success()) {
        return prepared;
    }

    prepared.status = command_program_preparation_context_valid(context);
    if (!prepared.status.success()) {
        return prepared;
    }

    const CommandProgramBindings materialized_bindings =
        materialize_host_bindings(context, commands, bindings, prepared);
    if (!materialized_bindings.valid()) {
        prepared.status.append(materialized_bindings.status);
        return prepared;
    }

    const TransientArenaAllocationRef * transient_allocation_ptr =
        commands.transients.arena_size == 0 ? nullptr : &transient_allocation;
    const ResolvedCommandProgram resolved =
        resolve_command_program_bindings(commands, materialized_bindings, transient_allocation_ptr);
    if (!resolved.valid()) {
        prepared.status.append(resolved.status);
        return prepared;
    }

    std::vector<KernelExecutableRef> initialization_executable_refs;
    std::vector<KernelExecutableRef> command_executable_refs;
    prepare_command_list(context, resolved.initialization_commands, prepared.initialization_commands,
                         initialization_executable_refs, prepared.status);
    prepare_command_list(context, resolved.commands, prepared.commands, command_executable_refs, prepared.status);
    materialize_command_list_executables(context, prepared.initialization_commands, initialization_executable_refs,
                                         prepared.status);
    materialize_command_list_executables(context, prepared.commands, command_executable_refs, prepared.status);
    prepared.bound_transient_arena_allocation_id = transient_allocation.allocation_id;
    if (prepared.status.success()) {
        prepared.status.append(prepare_program_constant_buffers(context, commands, prepared));
    }
    return prepared;
}

bool bind_prepared_command_program_transients(const CommandProgram &              commands,
                                              const TransientArenaAllocationRef & transient_allocation,
                                              PreparedCommandProgram &            prepared) {
    if (!prepared.valid()) {
        return false;
    }
    if (commands.transients.arena_size == 0) {
        prepared.bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;
        return true;
    }
    if (transient_allocation.buffer == nullptr ||
        transient_allocation.allocation_id == kInvalidTransientArenaAllocationId) {
        GGML_LOG_ERROR("%s: missing transient arena allocation\n", __func__);
        return false;
    }
    if (prepared.bound_transient_arena_allocation_id == transient_allocation.allocation_id) {
        return true;
    }
    if (!bind_prepared_command_list_transients(commands, transient_allocation, prepared.initialization_commands) ||
        !bind_prepared_command_list_transients(commands, transient_allocation, prepared.commands)) {
        return false;
    }
    prepared.bound_transient_arena_allocation_id = transient_allocation.allocation_id;
    return true;
}

bool bind_and_execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings,
                                               PreparedCommandProgram &               prepared) {
    if (!prepared.valid()) {
        return execute_prepared_command_program(context, prepared);
    }
    Status rebind_status = rebind_prepared_host_staging(context, bindings, prepared);
    if (!rebind_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(rebind_status));
        return false;
    }
    bool   graph_values_changed = false;
    Status graph_value_status   = rebind_prepared_graph_values(bindings, prepared, graph_values_changed);
    if (!graph_value_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(graph_value_status));
        return false;
    }
    if (commands.transients.arena_size == 0) {
        Status status = initialize_command_program_constants(context, commands, {}, prepared);
        if (!status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
            return false;
        }
        status = initialize_command_program_completion_counters(context, commands, {});
        if (!status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
            return false;
        }
        return bind_prepared_command_program_transients(commands, {}, prepared) &&
               execute_prepared_command_program(context, prepared);
    }

    Status status = command_program_transient_context_valid(context, commands);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }

    TransientArena::AllocationLease lease = context.transient_arena->acquire_allocation_lease();
    status = lease.ensure_capacity(context.device, context.stream, commands.transients.arena_size);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    if (!bind_prepared_command_program_transients(commands, lease.current_allocation(), prepared)) {
        return false;
    }
    status = initialize_command_program_constants(context, commands, lease.current_allocation(), prepared);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    status = initialize_command_program_completion_counters(context, commands, lease.current_allocation());
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    return execute_prepared_command_program(context, prepared);
}

RecordedCommandGraphExecutionResult bind_and_launch_recorded_command_graph(
    const CommandProgramExecutionContext & context,
    const CommandProgram &                 commands,
    const CommandProgramBindings &         bindings,
    PreparedCommandProgram &               prepared,
    RecordedCommandGraph &                 recorded) {
    RecordedCommandGraphExecutionResult result;
    result.event = HrxGraphReplayEvent::Ineligible;
    // Host validation is a dispatch boundary. A recorded graph must not bypass it or run consumers on failure.
    // SET_ROWS/GET_ROWS/embedding kernels whose IDs come from a trusted host-populated producer
    // (DispatchBinding::trusted, set only for provable leaves such as the KV-cache k_idxs/v_idxs,
    // build_inp_out_ids()'s output-position selection, or a token-embedding inp_tokens) already
    // skip the validation readback in validate_row_ids(), and may also safely reuse a previously
    // recorded command sequence: upload_prepared_host_staging() below refreshes its actual
    // buffer contents on every replay regardless of whether the graph is rebuilt or hit, so
    // replay eligibility never risks stale index data. Q4 routed gate/up IDs are always
    // GPU-computed top-k routing output and always require validation; no matcher marks them
    // trusted.
    const auto requires_validation = [](const PreparedCommand & command) {
        if (is_set_rows_kernel(command.kernel.specialization.kernel_id) ||
            is_f32_get_rows_kernel(command.kernel.specialization.kernel_id) ||
            is_qsa_f16_gather_kernel(command.kernel.specialization.kernel_id) ||
            is_q8_embedding_kernel(command.kernel.specialization.kernel_id) ||
            is_q4_embedding_kernel(command.kernel.specialization.kernel_id)) {
            return command.kernel.bindings.size() < 2 || !command.kernel.bindings[1].binding.trusted;
        }
        return is_q4_routed_gate_up_kernel(command.kernel.specialization.kernel_id);
    };
    if (std::any_of(prepared.commands.begin(), prepared.commands.end(), requires_validation) ||
        std::any_of(prepared.initialization_commands.begin(), prepared.initialization_commands.end(), requires_validation)) {
        const auto is_embedding = [](const PreparedCommand & command) {
            return is_q8_embedding_kernel(command.kernel.specialization.kernel_id);
        };
        result.ineligible_reason =
            std::any_of(prepared.commands.begin(), prepared.commands.end(), is_embedding) ||
            std::any_of(prepared.initialization_commands.begin(), prepared.initialization_commands.end(), is_embedding) ?
                "q8_embedding_requires_index_validation" : "set_rows_requires_index_validation";
        const auto is_q4_embedding = [](const PreparedCommand & command) {
            return is_q4_embedding_kernel(command.kernel.specialization.kernel_id);
        };
        if (std::any_of(prepared.commands.begin(), prepared.commands.end(), is_q4_embedding) ||
            std::any_of(prepared.initialization_commands.begin(), prepared.initialization_commands.end(), is_q4_embedding)) {
            result.ineligible_reason = "q4_embedding_requires_index_validation";
        }
        const auto is_f32_gather = [](const PreparedCommand & command) {
            return is_f32_get_rows_kernel(command.kernel.specialization.kernel_id);
        };
        if (std::any_of(prepared.commands.begin(), prepared.commands.end(), is_f32_gather) ||
            std::any_of(prepared.initialization_commands.begin(), prepared.initialization_commands.end(), is_f32_gather)) {
            result.ineligible_reason = "f32_get_rows_requires_index_validation";
        }
        const auto is_f16_gather = [](const PreparedCommand & command) {
            return is_qsa_f16_gather_kernel(command.kernel.specialization.kernel_id);
        };
        if (std::any_of(prepared.commands.begin(), prepared.commands.end(), is_f16_gather) ||
            std::any_of(prepared.initialization_commands.begin(), prepared.initialization_commands.end(), is_f16_gather)) {
            result.ineligible_reason = "qsa_f16_get_rows_requires_index_validation";
        }
        const auto is_q4_gate_up = [](const PreparedCommand & command) {
            return is_q4_routed_gate_up_kernel(command.kernel.specialization.kernel_id);
        };
        if (std::any_of(prepared.commands.begin(), prepared.commands.end(), is_q4_gate_up) ||
            std::any_of(prepared.initialization_commands.begin(), prepared.initialization_commands.end(), is_q4_gate_up)) {
            result.ineligible_reason = "q4_routed_gate_up_requires_index_validation";
        }
        return result;
    }
    const bool timing = hrx_time_compute_enabled();
    using clock       = std::chrono::steady_clock;
    const clock::time_point t_entry = timing ? clock::now() : clock::time_point{};

    if (!prepared.valid()) {
        result.status.append(prepared.status);
        if (result.status.success()) {
            result.status.log("invalid prepared command program");
        }
        return result;
    }
    if (!prepared_execution_context_valid(context)) {
        result.status.log("missing HRX stream");
        result.event = HrxGraphReplayEvent::BuildFailed;
        return result;
    }

    Status rebind_status = rebind_prepared_host_staging(context, bindings, prepared);
    if (!rebind_status.success()) {
        result.status.append(rebind_status);
        result.event = HrxGraphReplayEvent::BuildFailed;
        return result;
    }

    bool   graph_values_changed = false;
    Status graph_value_status   = rebind_prepared_graph_values(bindings, prepared, graph_values_changed);
    if (!graph_value_status.success()) {
        result.status.append(graph_value_status);
        result.event = HrxGraphReplayEvent::BuildFailed;
        return result;
    }

    TransientArenaAllocationRef transient_allocation;
    TransientArena::AllocationLease lease;
    const clock::time_point t_rebound = timing ? clock::now() : clock::time_point{};
    if (commands.transients.arena_size == 0) {
        if (!bind_prepared_command_program_transients(commands, {}, prepared)) {
            result.status.log("bind transient-free prepared command program failed");
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
    } else {
        Status status = command_program_transient_context_valid(context, commands);
        if (!status.success()) {
            result.status.append(status);
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
        lease  = context.transient_arena->acquire_allocation_lease();
        status = lease.ensure_capacity(context.device, context.stream, commands.transients.arena_size);
        if (!status.success()) {
            result.status.append(status);
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
        transient_allocation = lease.current_allocation();
        if (!bind_prepared_command_program_transients(commands, transient_allocation, prepared)) {
            result.status.log("bind prepared command program transients for graph replay failed");
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
    }

    const bool had_recorded = recorded.valid();
    const clock::time_point t_transient = timing ? clock::now() : clock::time_point{};
    result.transient_allocation_changed =
        had_recorded && recorded.bound_transient_arena_allocation_id != prepared.bound_transient_arena_allocation_id;
    if (!had_recorded || result.transient_allocation_changed || graph_values_changed || prepared.graph_bindings_dirty) {
        result.event =
            result.transient_allocation_changed ? HrxGraphReplayEvent::RebuildTransient : HrxGraphReplayEvent::MissBuild;
        const uint64_t build_start_ns = hrx_graph_replay_now_ns();
        RecordedCommandGraph rebuilt = record_prepared_command_graph(context, commands, prepared, transient_allocation);
        result.build_ns              = hrx_graph_replay_now_ns() - build_start_ns;
        if (!rebuilt.valid()) {
            result.status.append(rebuilt.status);
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
        recorded = std::move(rebuilt);
        prepared.graph_bindings_dirty = false;
    } else {
        result.event = HrxGraphReplayEvent::Hit;
    }

    const clock::time_point t_built = timing ? clock::now() : clock::time_point{};
    const uint64_t launch_start_ns = hrx_graph_replay_now_ns();
    trace_prepared_program("replay", prepared);
    Status upload_status = upload_prepared_host_staging(context, prepared);
    if (!upload_status.success()) {
        result.launch_ns = hrx_graph_replay_now_ns() - launch_start_ns;
        result.status.append(upload_status);
        result.event = HrxGraphReplayEvent::LaunchFailed;
        return result;
    }
    const clock::time_point t_uploaded = timing ? clock::now() : clock::time_point{};
    if (ErrorResult error = take_status(hrx_graph_exec_launch(recorded.exec, context.stream))) {
        result.launch_ns = hrx_graph_replay_now_ns() - launch_start_ns;
        result.status.log("launch HRX graph replay: %s", error->c_str());
        result.event = HrxGraphReplayEvent::LaunchFailed;
        return result;
    }
    const clock::time_point t_launched = timing ? clock::now() : clock::time_point{};
    Status download_status = download_prepared_host_staging(context, prepared);
    if (!download_status.success()) {
        result.launch_ns = hrx_graph_replay_now_ns() - launch_start_ns;
        result.status.append(download_status);
        result.event = HrxGraphReplayEvent::LaunchFailed;
        return result;
    }
    result.launch_ns      = hrx_graph_replay_now_ns() - launch_start_ns;
    result.dispatch_count = recorded.dispatch_count;
    result.success        = true;
    if (timing) {
        const clock::time_point t_end   = clock::now();
        const auto              elapsed = [](clock::time_point a, clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        static uint64_t calls        = 0;
        static double   rebind_ms    = 0.0;
        static double   transient_ms = 0.0;
        static double   build_ms     = 0.0;
        static double   launch_ms    = 0.0;
        static double   upload_ms      = 0.0;
        static double   execlaunch_ms  = 0.0;
        static double   download_ms    = 0.0;
        calls += 1;
        rebind_ms += elapsed(t_entry, t_rebound);
        transient_ms += elapsed(t_rebound, t_transient);
        build_ms += elapsed(t_transient, t_built);
        launch_ms += elapsed(t_built, t_end);
        upload_ms += elapsed(t_built, t_uploaded);
        execlaunch_ms += elapsed(t_uploaded, t_launched);
        download_ms += elapsed(t_launched, t_end);
        static const bool registered_replaypath_atexit = [] {
            std::atexit([] {
                if (calls == 0) {
                    return;
                }
                GGML_LOG_INFO("HRX launchphase (final): upload=%.1fms exec_launch=%.1fms download=%.1fms\n",
                              upload_ms, execlaunch_ms, download_ms);
                const double total = rebind_ms + transient_ms + build_ms + launch_ms;
                GGML_LOG_INFO(
                    "HRX replaypath (final): calls=%llu rebind=%.1fms(%.0f%%) transient=%.1fms(%.0f%%) "
                    "build=%.1fms(%.0f%%) launch=%.1fms(%.0f%%) total=%.1fms\n",
                    static_cast<unsigned long long>(calls), rebind_ms, 100.0 * rebind_ms / total, transient_ms,
                    100.0 * transient_ms / total, build_ms, 100.0 * build_ms / total, launch_ms,
                    100.0 * launch_ms / total, total);
            });
            return true;
        }();
        (void) registered_replaypath_atexit;
        if (calls % 5000 == 0) {
            GGML_LOG_INFO("HRX launchphase: upload=%.1fms exec_launch=%.1fms download=%.1fms\n", upload_ms,
                          execlaunch_ms, download_ms);
        }
        if (calls % 5000 == 0) {
            const double total = rebind_ms + transient_ms + build_ms + launch_ms;
            GGML_LOG_INFO(
                "HRX replaypath: calls=%llu rebind=%.1fms(%.0f%%) transient=%.1fms(%.0f%%) build=%.1fms(%.0f%%) "
                "launch=%.1fms(%.0f%%) total=%.1fms\n",
                static_cast<unsigned long long>(calls), rebind_ms, 100.0 * rebind_ms / total, transient_ms,
                100.0 * transient_ms / total, build_ms, 100.0 * build_ms / total, launch_ms, 100.0 * launch_ms / total,
                total);
        }
    }
    return result;
}

bool execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                      const PreparedCommandProgram &         commands) {
    if (!commands.valid()) {
        GGML_LOG_ERROR("%s: invalid HRX prepared command program: %s\n", __func__, status_first_error(commands.status));
        return false;
    }
    if (!prepared_execution_context_valid(context)) {
        return false;
    }
    trace_prepared_program("direct", commands);
    Status upload_status = upload_prepared_host_staging(context, commands);
    if (!upload_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(upload_status));
        return false;
    }
    if (!execute_prepared_command_list(context, commands.initialization_commands, commands.host_staging) ||
        !execute_prepared_command_list(context, commands.commands, commands.host_staging)) {
        return false;
    }
    Status download_status = download_prepared_host_staging(context, commands);
    if (!download_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(download_status));
        return false;
    }
    return true;
}

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings) {
    PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
    return bind_and_execute_prepared_command_program(context, commands, bindings, prepared);
}

}  // namespace ggml::hrx
