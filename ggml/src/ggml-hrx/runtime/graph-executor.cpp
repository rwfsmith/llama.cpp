#include "graph-executor.h"

#include "backend-buffer-binding.h"
#include "ggml-impl.h"
#include "runtime/graph-replay.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <chrono>
#include <cstdlib>
#include <utility>
#include <vector>

namespace ggml::hrx {

static bool hrx_environment_flag(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// Bounded so a long generation does not drown the log; the first splits are the interesting ones.
static bool hrx_trace_bindings_enabled() {
    static const bool enabled = hrx_environment_flag("HRX_TRACE_BINDINGS");
    if (!enabled) {
        return false;
    }
    static int remaining = 60;
    if (remaining <= 0) {
        return false;
    }
    --remaining;
    return true;
}

static bool hrx_prepared_fast_path_disabled() {
    static const bool disabled = hrx_environment_flag("HRX_DISABLE_PREPARED_FAST_PATH");
    return disabled;
}

// Companion to the HRX_TIME_COMPUTE accounting in graph_compute(): that shows submission dominates GPU
// occupancy, this attributes the submission cost to the four phases it is actually made of, so the
// expensive one can be identified instead of guessed at.
static bool hrx_time_compute_enabled() {
    static const bool enabled = hrx_environment_flag("HRX_TIME_COMPUTE");
    return enabled;
}

GraphExecutor::GraphExecutor(ggml_backend_hrx_context & context) : context_(context) {}

Status GraphExecutor::context_valid_for_graph_programs() const {
    Status status;
    if (context_.device == nullptr) {
        status.log("missing HRX device context");
    } else if (context_.device->architecture.empty()) {
        status.log("missing HRX target");
    }
    return status;
}

Status GraphExecutor::context_valid_for_execution() const {
    return context_valid_for_graph_programs();
}

GraphSupportResult GraphExecutor::can_execute(const ggml_cgraph & graph) const {
    GraphSupportResult result;
    if (graph.n_nodes == 0) {
        result.supported = true;
        return result;
    }
    result.status = context_valid_for_graph_programs();
    if (!result.status.success()) {
        return result;
    }
    const KernelCorpus &            corpus = get_qwen_kernel_corpus();
    const GraphProgramSupportResult support =
        context_.graph_programs.check_support(graph, corpus, context_.device->architecture);
    result.supported = support.supported;
    result.status.append(support.status);
    return result;
}

CommandProgramBindings GraphExecutor::bind_external_value_buffers(const GraphProgramMatch & match) const {
    std::vector<CommandProgramBinding> bindings;
    Status                             status;
    bindings.reserve(match.external_bindings.size());
    for (const GraphProgramExternalBinding & external : match.external_bindings) {
        ValueBufferBinding    value_binding;
        CommandProgramBinding binding;
        binding.value = external.value;
        if (ggml_backend_hrx_resolve_value_buffer(external.tensor, value_binding)) {
            binding.buffer     = value_binding.buffer;
            binding.host_data  = value_binding.host_data;
            binding.offset     = value_binding.offset;
            binding.length     = value_binding.length;
            binding.identity   = value_binding.identity;
            binding.generation = value_binding.generation;
            binding.capacity   = value_binding.capacity;
            binding.weight     = value_binding.weight;
            if (hrx_trace_bindings_enabled()) {
                GGML_LOG_ERROR("HRX bind value=%d tensor=%s op=%s buffer=%p offset=%zu length=%zu identity=%llu\n",
                               external.value.value, external.tensor->name, ggml_op_name(external.tensor->op),
                               static_cast<const void *>(binding.buffer), binding.offset, binding.length,
                               static_cast<unsigned long long>(binding.identity));
            }
        } else {
            status.log("external value %d is not bound", external.value.value);
        }
        bindings.push_back(binding);
    }
    return CommandProgramBindings::from_bindings(std::move(bindings), status);
}

GraphExecutionResult GraphExecutor::execute(const ggml_cgraph & graph) const {
    GraphExecutionResult result;
    if (graph.n_nodes == 0) {
        result.code = GGML_STATUS_SUCCESS;
        return result;
    }
    result.status = context_valid_for_execution();
    if (!result.status.success()) {
        return result;
    }

    const KernelCorpus & corpus = get_qwen_kernel_corpus();
    const bool           timing = hrx_time_compute_enabled();
    using clock                 = std::chrono::steady_clock;
    const clock::time_point t_begin = timing ? clock::now() : clock::time_point{};
    GraphProgramLookup   lookup = context_.graph_programs.get_or_build(graph, corpus, context_.device->architecture);
    if (!lookup.valid()) {
        result.status.append(lookup.status);
        result.status.append(lookup.match.status);
        if (result.status.success()) {
            result.status.log("build HRX graph program failed");
        }
        return result;
    }

    const bool use_graph_prepared = !hrx_prepared_fast_path_disabled() &&
                                    (!lookup.program->has_prepared_program() ||
                                     lookup.program->can_use_prepared_fast_path(graph));
    const clock::time_point t_looked_up = timing ? clock::now() : clock::time_point{};
    GraphProgramMatch binding_match = std::move(lookup.match);
    if (use_graph_prepared && lookup.program->has_prepared_program()) {
        binding_match = lookup.program->match_host_staging_graph(graph);
        if (!binding_match.valid()) {
            result.status.append(binding_match.status);
            return result;
        }
    }
    const clock::time_point t_matched = timing ? clock::now() : clock::time_point{};

    CommandProgramBindings bindings = bind_external_value_buffers(binding_match);
    if (!bindings.valid()) {
        result.status.append(bindings.status);
        return result;
    }
    const clock::time_point t_bound = timing ? clock::now() : clock::time_point{};
    const CommandProgramExecutionContext execution_context = {
        context_.device->device,
        context_.stream,
        context_.device->architecture.c_str(),
        &corpus,
        &context_.kernel_executables,
        &context_.transient_arena,
        &context_.host_transfers,
        &context_.host_weights,
    };
    const PreparedCommandProgramCacheExecutionResult execution =
        use_graph_prepared ? lookup.program->execute_with_result(execution_context, bindings) :
                             context_.prepared_programs.execute_with_result(execution_context, lookup.program->uid(),
                                                                            lookup.program->command_shape(),
                                                                            lookup.program->commands(), bindings);
    if (!execution.success) {
        result.status.append(execution.status);
        if (result.status.success()) {
            result.status.log("execute HRX command program failed");
        }
        return result;
    }

    result.code = GGML_STATUS_SUCCESS;
    if (timing) {
        const clock::time_point t_recorded = clock::now();
        const auto              elapsed    = [](clock::time_point a, clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        static uint64_t calls      = 0;
        static double   lookup_ms  = 0.0;
        static double   match_ms   = 0.0;
        static double   bind_ms    = 0.0;
        static double   record_ms  = 0.0;
        calls += 1;
        lookup_ms += elapsed(t_begin, t_looked_up);
        match_ms += elapsed(t_looked_up, t_matched);
        bind_ms += elapsed(t_matched, t_bound);
        record_ms += elapsed(t_bound, t_recorded);
        if (calls % 1000 == 0) {
            const double total = lookup_ms + match_ms + bind_ms + record_ms;
            GGML_LOG_INFO(
                "HRX submit: calls=%llu lookup=%.1fms(%.0f%%) match=%.1fms(%.0f%%) bind=%.1fms(%.0f%%) "
                "record=%.1fms(%.0f%%)\n",
                static_cast<unsigned long long>(calls), lookup_ms, 100.0 * lookup_ms / total, match_ms,
                100.0 * match_ms / total, bind_ms, 100.0 * bind_ms / total, record_ms, 100.0 * record_ms / total);
            GGML_LOG_INFO("HRX replay: last=%s reason=%s dispatches=%zu build=%.3fms launch=%.3fms\n",
                          hrx_graph_replay_event_name(execution.graph_replay_event),
                          execution.graph_replay_ineligible_reason.empty() ?
                              "-" :
                              execution.graph_replay_ineligible_reason.c_str(),
                          execution.graph_replay_dispatches, execution.graph_replay_build_ns / 1e6,
                          execution.graph_replay_launch_ns / 1e6);
        }
    }
    return result;
}

}  // namespace ggml::hrx
