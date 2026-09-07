#include "ggml-hrx.h"

#include "backend-buffer-binding.h"
#include "backend-context.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "hrx_runtime.h"
#include "kernel-corpus/kernel-corpus.h"
#include "loom-jit.h"
#include "runtime/graph-executor.h"
#include "runtime/graph-program-cache.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

static constexpr size_t      GGML_HRX_ALIGNMENT     = 256;
// GGML represents tensor locations as host pointers and derives view/arena offsets with ordinary pointer arithmetic.
// Device-local HRX buffers have no host address to return, so expose a non-null sentinel base as an offset coordinate.
static constexpr uintptr_t   GGML_HRX_FAKE_PTR_BASE = 0x1000;
static std::atomic<uint64_t> g_allocation_generation{ 1 };

static bool environment_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool hrx_check(hrx_status_t status, const char * expression, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }
    char * message = nullptr;
    size_t length  = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expression,
                   message != nullptr ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

#define HRX_CHECK(expression) hrx_check((expression), #expression, __FILE__, __LINE__)

}  // namespace

ggml_backend_hrx_reg_context::~ggml_backend_hrx_reg_context() {
    for (auto & context : device_contexts) {
        if (context->buffer_stream != nullptr) {
            hrx_stream_release(context->buffer_stream);
        }
        if (context->device != nullptr) {
            hrx_device_release(context->device);
        }
    }
    if (initialized) {
        hrx_status_t status = hrx_gpu_shutdown();
        if (!hrx_status_is_ok(status)) {
            hrx_status_ignore(status);
        }
    }
}

namespace {

static std::optional<std::string> device_string_property(hrx_device_t          device,
                                                         hrx_device_property_t property,
                                                         const char *          property_name) {
    std::vector<char> buffer(64);
    while (buffer.size() <= 4096) {
        hrx_status_t status = hrx_device_get_property(device, property, buffer.data(), buffer.size());
        if (hrx_status_is_ok(status)) {
            return std::string(buffer.data());
        }
        if (hrx_status_code(status) != HRX_STATUS_OUT_OF_RANGE) {
            hrx_check(status, property_name, __FILE__, __LINE__);
            return std::nullopt;
        }
        hrx_status_ignore(status);
        buffer.resize(buffer.size() * 2);
    }
    GGML_LOG_ERROR("%s exceeds the maximum supported property string length\n", property_name);
    return std::nullopt;
}

static ggml_guid_t ggml_backend_hrx_guid() {
    static ggml_guid guid = {
        0xd2, 0x3d, 0x72, 0x83, 0xb2, 0x82, 0x4d, 0xe0, 0x8a, 0x3e, 0x21, 0x1d, 0x68, 0x87, 0x2f, 0x4b,
    };
    return &guid;
}

static ggml_backend_hrx_device_context * device_context(ggml_backend_dev_t device) {
    return static_cast<ggml_backend_hrx_device_context *>(device->context);
}

static ggml_backend_hrx_buffer_context * buffer_context(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx_buffer_context_from_buffer(buffer);
}

static size_t tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return ggml_backend_hrx_tensor_offset(context, tensor);
}

static const char * buffer_type_name(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->name.c_str();
}

static bool buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->host_visible;
}

static bool buffer_submit_and_wait(ggml_backend_hrx_device_context * device,
                                   hrx_status_t (*submit)(hrx_stream_t, void *),
                                   void * user_data) {
    std::lock_guard<std::mutex> lock(device->buffer_stream_mutex);
    if (!HRX_CHECK(submit(device->buffer_stream, user_data))) {
        return false;
    }
    return HRX_CHECK(hrx_stream_synchronize(device->buffer_stream));
}

struct FillBufferArgs {
    hrx_buffer_t buffer;
    size_t       offset;
    size_t       size;
    uint8_t      value;
};

static hrx_status_t submit_fill_buffer(hrx_stream_t stream, void * user_data) {
    auto * args = static_cast<FillBufferArgs *>(user_data);
    return hrx_stream_fill_buffer(stream, args->buffer, args->offset, args->size, &args->value, sizeof(args->value));
}

struct CopyBufferArgs {
    hrx_buffer_t source;
    size_t       source_offset;
    hrx_buffer_t destination;
    size_t       destination_offset;
    size_t       size;
};

static hrx_status_t submit_copy_buffer(hrx_stream_t stream, void * user_data) {
    auto * args = static_cast<CopyBufferArgs *>(user_data);
    return hrx_stream_copy_buffer(stream, args->source, args->source_offset, args->destination,
                                  args->destination_offset, args->size);
}

static void buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = buffer_context(buffer);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        context->device->host_buffers.remove(context->buffer);
    }
    if (context->buffer != nullptr) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void buffer_memset(ggml_backend_buffer_t buffer,
                          ggml_tensor *         tensor,
                          uint8_t               value,
                          size_t                offset,
                          size_t                size) {
    if (size == 0) {
        return;
    }
    auto *       context            = buffer_context(buffer);
    const size_t destination_offset = tensor_offset(context, tensor) + offset;
    GGML_ASSERT(destination_offset <= buffer->size && size <= buffer->size - destination_offset);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        std::memset(context->base + destination_offset, value, size);
        return;
    }
    FillBufferArgs args{ context->buffer, destination_offset, size, value };
    if (!buffer_submit_and_wait(context->device, submit_fill_buffer, &args)) {
        GGML_LOG_ERROR("%s: HRX buffer fill failed\n", __func__);
    }
}

static void buffer_set(ggml_backend_buffer_t buffer,
                       ggml_tensor *         tensor,
                       const void *          data,
                       size_t                offset,
                       size_t                size) {
    if (size == 0) {
        return;
    }
    auto *       context            = buffer_context(buffer);
    const size_t destination_offset = tensor_offset(context, tensor) + offset;
    GGML_ASSERT(destination_offset <= buffer->size && size <= buffer->size - destination_offset);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        std::memcpy(context->base + destination_offset, data, size);
        return;
    }
    if (!HRX_CHECK(hrx_synchronous_h2d(context->device->device, data, context->buffer, destination_offset, size))) {
        GGML_LOG_ERROR("%s: HRX buffer upload failed\n", __func__);
    }
}

static void buffer_get(ggml_backend_buffer_t buffer,
                       const ggml_tensor *   tensor,
                       void *                data,
                       size_t                offset,
                       size_t                size) {
    if (size == 0) {
        return;
    }
    auto *       context       = buffer_context(buffer);
    const size_t source_offset = tensor_offset(context, tensor) + offset;
    GGML_ASSERT(source_offset <= buffer->size && size <= buffer->size - source_offset);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        std::memcpy(data, context->base + source_offset, size);
        return;
    }
    if (!HRX_CHECK(hrx_synchronous_d2h(context->device->device, context->buffer, source_offset, data, size))) {
        GGML_LOG_ERROR("%s: HRX buffer download failed\n", __func__);
    }
}

static bool buffer_copy(ggml_backend_buffer_t buffer, const ggml_tensor * source, ggml_tensor * destination) {
    ggml_backend_buffer_t source_buffer = source->view_src != nullptr ? source->view_src->buffer : source->buffer;
    if (source_buffer == nullptr || source_buffer->iface.get_base != ggml_backend_hrx_buffer_base) {
        return false;
    }
    auto * source_context      = buffer_context(source_buffer);
    auto * destination_context = buffer_context(buffer);
    if (source_context->device != destination_context->device) {
        return false;
    }
    const size_t source_offset      = tensor_offset(source_context, source);
    const size_t destination_offset = tensor_offset(destination_context, destination);
    const size_t size               = ggml_nbytes(source);
    if (source_offset > source_buffer->size || size > source_buffer->size - source_offset ||
        destination_offset > buffer->size || size > buffer->size - destination_offset) {
        return false;
    }
    CopyBufferArgs args{ source_context->buffer, source_offset, destination_context->buffer, destination_offset, size };
    return buffer_submit_and_wait(destination_context->device, submit_copy_buffer, &args);
}

static void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (buffer->size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        std::memset(context->base, value, buffer->size);
        return;
    }
    FillBufferArgs args{ context->buffer, 0, buffer->size, value };
    if (!buffer_submit_and_wait(context->device, submit_fill_buffer, &args)) {
        GGML_LOG_ERROR("%s: HRX buffer clear failed\n", __func__);
    }
}

static const ggml_backend_buffer_i buffer_i = {
    buffer_free, ggml_backend_hrx_buffer_base,
    nullptr,     buffer_memset,
    buffer_set,  buffer_get,
    nullptr,     nullptr,
    buffer_copy, buffer_clear,
    nullptr,
};

static ggml_backend_buffer_t buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto *              type_context = static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context);
    const bool          host_visible = type_context->host_visible;
    const bool          direct_host_binding = host_visible && type_context->device->use_direct_host_bindings;
    hrx_memory_type_t   memory_type          = HRX_MEMORY_TYPE_DEVICE_LOCAL;
    // Direct command-program bindings require coherent CPU/GPU visibility. Otherwise HRX host buffers are pinned
    // transfer memory: DEVICE_VISIBLE permits handle-based stream copies without implying direct device access.
    if (host_visible) {
        memory_type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
        if (direct_host_binding) {
            memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT;
        }
    }
    hrx_buffer_params_t params       = {
        memory_type,
        HRX_MEMORY_ACCESS_ALL,
        host_visible ?
            HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT :
            HRX_BUFFER_USAGE_DEFAULT,
        0,
    };
    hrx_buffer_t allocation = nullptr;
    if (size > 0 && !HRX_CHECK(hrx_allocator_allocate_buffer(hrx_device_allocator(type_context->device->device), params,
                                                             size, &allocation))) {
        return nullptr;
    }
    uint8_t * base = reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE);
    if (host_visible && size > 0) {
        void * mapped = nullptr;
        if (!HRX_CHECK(hrx_buffer_map(allocation, HRX_MAP_READ | HRX_MAP_WRITE, 0, size, &mapped))) {
            hrx_buffer_release(allocation);
            return nullptr;
        }
        base = static_cast<uint8_t *>(mapped);
    }
    const uint64_t generation = g_allocation_generation.fetch_add(1);
    auto *         context    = new (std::nothrow) ggml_backend_hrx_buffer_context{
        type_context->device, allocation, base, generation, generation, direct_host_binding,
    };
    if (context == nullptr) {
        if (allocation != nullptr) {
            hrx_buffer_release(allocation);
        }
        return nullptr;
    }
    if (host_visible && allocation != nullptr) {
        type_context->device->host_buffers.add(allocation, base, size);
    }
    return ggml_backend_buffer_init(buft, buffer_i, context, size);
}

static size_t buffer_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t buffer_max_size(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->device->memory_total;
}

static const ggml_backend_buffer_type_i buffer_type_i = {
    buffer_type_name, buffer_alloc, buffer_alignment, buffer_max_size, nullptr, buffer_type_is_host,
};

static const char * backend_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx_context *>(backend->context)->name.c_str();
}

static void backend_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    HRX_CHECK(hrx_stream_synchronize(context->stream));
    context->prepared_programs.clear();
    context->graph_programs.clear();
    context->kernel_executables.clear();
    context->transient_arena.clear();
    context->host_weights.clear();
    context->host_transfers.clear();
    hrx_stream_release(context->stream);
    delete context;
    delete backend;
}

static bool synchronous_upload_fallback(ggml_backend_hrx_context * backend,
                                        const void *               source,
                                        hrx_buffer_t               destination,
                                        size_t                     destination_offset,
                                        size_t                     size) {
    const uint64_t fallback =
        backend->device->synchronous_upload_fallbacks.fetch_add(1, std::memory_order_relaxed);
    if (fallback == 0) {
        GGML_LOG_WARN("ggml_hrx: synchronous upload fallback for an unregistered host pointer; use the HRX host "
                      "buffer type for asynchronous transfers\n");
    }
    // Compatibility path for arbitrary GGML pointers. Keep the synchronization explicit until a bounded staging ring
    // with transfer retirement is available.
    const ggml::hrx::Status status =
        backend->host_transfers.upload_synchronous(backend->stream, source, destination, destination_offset, size);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status.errors().front().c_str());
        return false;
    }
    return true;
}

static bool synchronous_download_fallback(ggml_backend_hrx_context * backend,
                                          hrx_buffer_t               source,
                                          size_t                     source_offset,
                                          void *                     destination,
                                          size_t                     size) {
    const uint64_t fallback =
        backend->device->synchronous_download_fallbacks.fetch_add(1, std::memory_order_relaxed);
    if (fallback == 0) {
        GGML_LOG_WARN("ggml_hrx: synchronous download fallback for an unregistered host pointer; use the HRX host "
                      "buffer type for asynchronous transfers\n");
    }
    // Compatibility path for arbitrary GGML pointers. Keep the synchronization explicit until a bounded staging ring
    // with transfer retirement is available.
    const ggml::hrx::Status status =
        backend->host_transfers.download_synchronous(backend->stream, source, source_offset, destination, size);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status.errors().front().c_str());
        return false;
    }
    return true;
}

static void backend_set_tensor_async(ggml_backend_t backend,
                                     ggml_tensor *  tensor,
                                     const void *   data,
                                     size_t         offset,
                                     size_t         size) {
    auto *                            backend_context = static_cast<ggml_backend_hrx_context *>(backend->context);
    ggml_backend_hrx_buffer_context * context         = nullptr;
    size_t                            tensor_base     = 0;
    if (!ggml_backend_hrx_tensor_binding(tensor, &context, &tensor_base) || offset > ggml_nbytes(tensor) ||
        size > ggml_nbytes(tensor) - offset) {
        GGML_LOG_ERROR("%s: invalid HRX tensor upload\n", __func__);
        return;
    }
    ggml::hrx::HostBufferRef source = backend_context->device->host_buffers.find(data, size);
    if (source.valid()) {
        // Registered host buffers can participate directly in the stream command buffer.
        HRX_CHECK(hrx_stream_copy_buffer(backend_context->stream, source.buffer(), source.offset(), context->buffer,
                                         tensor_base + offset, size));
    } else {
        synchronous_upload_fallback(backend_context, data, context->buffer, tensor_base + offset, size);
    }
}

static void backend_get_tensor_async(ggml_backend_t      backend,
                                     const ggml_tensor * tensor,
                                     void *              data,
                                     size_t              offset,
                                     size_t              size) {
    auto *                            backend_context = static_cast<ggml_backend_hrx_context *>(backend->context);
    ggml_backend_hrx_buffer_context * context         = nullptr;
    size_t                            tensor_base     = 0;
    if (!ggml_backend_hrx_tensor_binding(tensor, &context, &tensor_base) || offset > ggml_nbytes(tensor) ||
        size > ggml_nbytes(tensor) - offset) {
        GGML_LOG_ERROR("%s: invalid HRX tensor download\n", __func__);
        return;
    }
    ggml::hrx::HostBufferRef destination = backend_context->device->host_buffers.find(data, size);
    if (destination.valid()) {
        // Registered host buffers can participate directly in the stream command buffer.
        HRX_CHECK(hrx_stream_copy_buffer(backend_context->stream, context->buffer, tensor_base + offset,
                                         destination.buffer(), destination.offset(), size));
    } else {
        synchronous_download_fallback(backend_context, context->buffer, tensor_base + offset, data, size);
    }
}

static bool backend_copy_tensor_async(ggml_backend_t      backend_src,
                                      ggml_backend_t      backend_dst,
                                      const ggml_tensor * source,
                                      ggml_tensor *       destination) {
    GGML_UNUSED(backend_src);
    auto * destination_backend = static_cast<ggml_backend_hrx_context *>(backend_dst->context);
    ggml_backend_hrx_buffer_context * destination_context = nullptr;
    size_t                            destination_offset  = 0;
    if (!ggml_backend_hrx_tensor_binding(destination, &destination_context, &destination_offset)) {
        return false;
    }
    ggml_backend_hrx_buffer_context * source_context = nullptr;
    size_t                            source_offset  = 0;
    const size_t                      size           = ggml_nbytes(source);
    if (ggml_backend_hrx_tensor_binding(source, &source_context, &source_offset)) {
        if (source_context->device != destination_context->device) {
            return false;
        }
        return HRX_CHECK(hrx_stream_copy_buffer(destination_backend->stream, source_context->buffer, source_offset,
                                                destination_context->buffer, destination_offset, size));
    }
    ggml_backend_buffer_t source_buffer = source->view_src != nullptr ? source->view_src->buffer : source->buffer;
    if (source_buffer != nullptr && ggml_backend_buffer_is_host(source_buffer)) {
        return synchronous_upload_fallback(
            destination_backend, source->data, destination_context->buffer, destination_offset, size);
    }
    return false;
}

static void backend_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    HRX_CHECK(hrx_stream_synchronize(context->stream));
}

static const char * status_first_error(const ggml::hrx::Status & status) {
    return status.errors().empty() ? "" : status.errors().front().c_str();
}

// Companion to HRX_TRACE_DISPATCH (dispatch-scheduler.cpp): that one names the matchers that fire,
// this one shows the *splits* ggml_backend_sched actually hands to HRX. The two answer different
// questions -- a matcher can look right while the scheduler is routing a node here that should have
// stayed on the CPU -- and a wrong split is invisible from inside the dispatch registry. Deduplicated
// by the split's op/shape signature, so a 48-layer decode prints one line per distinct split shape.
static void hrx_trace_split(const ggml_cgraph & graph) {
    static const bool enabled = [] {
        const char * value = std::getenv("HRX_TRACE_DISPATCH");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    if (!enabled) {
        return;
    }
    std::string signature;
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        signature += ggml_op_name(node->op);
        signature += "[";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            signature += std::to_string(node->ne[d]);
            signature += d + 1 < GGML_MAX_DIMS ? "," : "";
        }
        signature += "] ";
    }
    static std::set<std::string> seen;
    if (seen.insert(signature).second) {
        GGML_LOG_ERROR("HRX split: %d nodes: %s\n", graph.n_nodes, signature.c_str());
    }
}

// HRX_VERIFY_NODES=1: snapshot the inputs of every f32 same-shape ADD in a split *before* the split
// runs, then read the destination back *after* it runs and compare against a host-computed a+b.
// Capturing before execution matters: ggml's in-place ADD makes dst a view of src0, so a purely
// post-hoc comparison reads the same memory twice and reports a bogus error equal to |b|.
// This distinguishes "the kernel wrote wrong values" from "the values are right but the scheduler
// consumes them from somewhere else" -- which no amount of output-text bisection can separate.
// Deliberately slow and synchronous: it is a diagnostic, not a fast path.
static bool hrx_read_tensor_f32(ggml_backend_hrx_context * context,
                                const ggml_tensor *        tensor,
                                std::vector<float> &       out,
                                bool *                     from_device = nullptr) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    const size_t bytes = ggml_nbytes(tensor);
    out.resize(bytes / sizeof(float));
    ggml_backend_hrx_buffer_context * tensor_context = nullptr;
    size_t                            tensor_offset  = 0;
    if (ggml_backend_hrx_tensor_binding(tensor, &tensor_context, &tensor_offset)) {
        if (from_device != nullptr) {
            *from_device = true;
        }
        return HRX_CHECK(hrx_synchronous_d2h(context->device->device, tensor_context->buffer, tensor_offset, out.data(),
                                             bytes));
    }
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == nullptr || !ggml_backend_buffer_is_host(buffer) || tensor->data == nullptr) {
        return false;
    }
    if (from_device != nullptr) {
        *from_device = false;
    }
    std::memcpy(out.data(), tensor->data, bytes);
    return true;
}

struct hrx_verify_capture {
    const ggml_tensor * node        = nullptr;
    std::vector<float>  a;
    std::vector<float>  b;
    bool                a_device    = false;
    bool                b_device    = false;
    bool                dst_aliases_a = false;
    bool                dst_aliases_b = false;
    int                 n_nodes       = 0;
    int                 node_index    = 0;
};

static bool hrx_verify_enabled() {
    static const bool enabled = environment_flag_enabled("HRX_VERIFY_NODES");
    return enabled;
}

static std::vector<hrx_verify_capture> hrx_verify_before(ggml_backend_hrx_context * context, const ggml_cgraph & graph) {
    std::vector<hrx_verify_capture> captures;
    if (!hrx_verify_enabled()) {
        return captures;
    }
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        if (node == nullptr || node->op != GGML_OP_ADD || node->type != GGML_TYPE_F32) {
            continue;
        }
        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];
        if (a == nullptr || b == nullptr || !ggml_are_same_shape(a, node) || !ggml_are_same_shape(b, node)) {
            continue;
        }
        hrx_verify_capture capture;
        capture.node          = node;
        capture.n_nodes       = graph.n_nodes;
        capture.node_index    = i;
        capture.dst_aliases_a = node->data == a->data;
        capture.dst_aliases_b = node->data == b->data;
        if (!hrx_read_tensor_f32(context, a, capture.a, &capture.a_device) ||
            !hrx_read_tensor_f32(context, b, capture.b, &capture.b_device)) {
            continue;
        }
        captures.push_back(std::move(capture));
    }
    return captures;
}

static void hrx_verify_after(ggml_backend_hrx_context * context, const std::vector<hrx_verify_capture> & captures) {
    if (!hrx_verify_enabled()) {
        return;
    }
    static std::set<std::string> reported;
    for (const hrx_verify_capture & capture : captures) {
        std::vector<float> host_out;
        bool               out_device = false;
        if (!hrx_read_tensor_f32(context, capture.node, host_out, &out_device)) {
            continue;
        }
        double worst       = 0.0;
        size_t worst_index = 0;
        for (size_t e = 0; e < host_out.size() && e < capture.a.size() && e < capture.b.size(); ++e) {
            const double diff = std::fabs(static_cast<double>(host_out[e]) -
                                          (static_cast<double>(capture.a[e]) + static_cast<double>(capture.b[e])));
            if (diff > worst) {
                worst       = diff;
                worst_index = e;
            }
        }
        char key[128];
        std::snprintf(key, sizeof(key), "ADD:%lld,%lld:nodes=%d:idx=%d", static_cast<long long>(capture.node->ne[0]),
                      static_cast<long long>(capture.node->ne[1]), capture.n_nodes, capture.node_index);
        if (!reported.insert(key).second) {
            continue;
        }
        ggml_backend_hrx_buffer_context * dst_context = nullptr;
        size_t                            dst_offset  = 0;
        ggml_backend_hrx_tensor_binding(capture.node, &dst_context, &dst_offset);
        GGML_LOG_ERROR(
            "HRX verify %s n=%zu worst=%.6g at %zu (out=%.6g a=%.6g b=%.6g) dev(a=%d b=%d out=%d) alias(a=%d b=%d) "
            "view_src=%d dst_buffer=%p dst_offset=%zu\n",
            key, host_out.size(), worst, worst_index, worst_index < host_out.size() ? host_out[worst_index] : 0.0f,
            worst_index < capture.a.size() ? capture.a[worst_index] : 0.0f,
            worst_index < capture.b.size() ? capture.b[worst_index] : 0.0f, static_cast<int>(capture.a_device),
            static_cast<int>(capture.b_device), static_cast<int>(out_device), static_cast<int>(capture.dst_aliases_a),
            static_cast<int>(capture.dst_aliases_b), capture.node->view_src != nullptr ? 1 : 0,
            static_cast<const void *>(dst_context != nullptr ? dst_context->buffer : nullptr), dst_offset);
    }
}

static enum ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    hrx_trace_split(*graph);
    const std::vector<hrx_verify_capture> captures = hrx_verify_before(context, *graph);
    const ggml::hrx::GraphExecutor        executor = ggml::hrx::GraphExecutor(*context);
    const ggml::hrx::GraphExecutionResult result   = executor.execute(*graph);
    if (!result.success()) {
        // HRX_SURVEY_UNSUPPORTED exists so that one run enumerates every node missing a dispatch, but the
        // scheduler can only report that list through the status it returns here. Printing just the first
        // message would throw the survey away and put us back to one run per missing kernel.
        static const bool survey = [] {
            const char * value = std::getenv("HRX_SURVEY_UNSUPPORTED");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        if (survey) {
            for (const std::string & error : result.status.errors()) {
                GGML_LOG_ERROR("%s: %s\n", __func__, error.c_str());
            }
        } else {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(result.status));
        }
    }
    // Dispatches are recorded on the backend stream, but the buffer interface moves tensor data on the
    // device's separate buffer_stream (see buffer_submit_and_wait). Nothing orders those two streams, so
    // a caller that computes a split here and then reads the result with ggml_backend_tensor_get() would
    // synchronize only buffer_stream and observe memory the kernel has not written yet. That is exactly
    // what ggml_backend_sched does between splits, and it silently corrupts every model that offloads any
    // op at all -- while ggml_backend_graph_compute() (used by test-backend-ops) hides the bug because it
    // calls ggml_backend_synchronize() itself. Retire the compute stream here so the buffer interface and
    // any host readback are guaranteed to see completed work.
    HRX_CHECK(hrx_stream_synchronize(context->stream));
    hrx_verify_after(context, captures);
    return result.code;
}

static const ggml_backend_i backend_i = {
    backend_name,
    backend_free,
    backend_set_tensor_async,
    backend_get_tensor_async,
    nullptr,
    nullptr,
    backend_copy_tensor_async,
    backend_synchronize,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    graph_compute,
    nullptr,
    nullptr,
    nullptr,
};

static const char * device_name(ggml_backend_dev_t device) {
    return device_context(device)->name.c_str();
}

static const char * device_description(ggml_backend_dev_t device) {
    return device_context(device)->description.c_str();
}

static void device_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    *free  = device_context(device)->memory_total;
    *total = device_context(device)->memory_total;
}

static enum ggml_backend_dev_type device_type(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void device_props(ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    props->name        = device_name(device);
    props->description = device_description(device);
    device_memory(device, &props->memory_free, &props->memory_total);
    props->type      = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    props->caps      = { true, true, false, false };
}

static ggml_backend_t device_init(ggml_backend_dev_t device, const char * parameters) {
    GGML_UNUSED(parameters);
    auto *       device_ctx = device_context(device);
    hrx_stream_t stream     = nullptr;
    if (!HRX_CHECK(hrx_stream_create(device_ctx->device, 0, &stream))) {
        return nullptr;
    }
    auto * context = new (std::nothrow) ggml_backend_hrx_context;
    if (context != nullptr) {
        context->device = device_ctx;
        context->stream = stream;
        context->name   = device_ctx->name;
    }
    auto * backend = context != nullptr ? new (std::nothrow)
                                              ggml_backend{ ggml_backend_hrx_guid(), backend_i, device, context } :
                                          nullptr;
    if (backend == nullptr) {
        delete context;
        hrx_stream_release(stream);
    }
    return backend;
}

static ggml_backend_buffer_type_t device_buffer_type(ggml_backend_dev_t device) {
    return &device_context(device)->buft;
}

static ggml_backend_buffer_type_t device_host_buffer_type(ggml_backend_dev_t device) {
    return &device_context(device)->host_buft;
}

// HRX_DISABLE_DISPATCH holds a comma-separated list of dispatch groups to force onto the CPU:
//
//     HRX_DISABLE_DISPATCH=gdn,hc,attn,matmul,embed
//
// Groups are hierarchical: "gdn" covers "gdnconv"/"gdncore"/"gdngate", and "misc" covers
// "mnorm"/"mrope"/"msoftmax"/"msetrows"/"mlayout" plus "melem", which in turn covers
// "eadd"/"emul"/"eother".
//
// A model that runs but produces wrong tokens has to be bisected one subsystem at a time, and a full
// rebuild-and-reload cycle per step is minutes of wall clock each. This makes each step an env var
// instead. Declining is always safe -- it is the same path an unsupported shape already takes -- so
// this only ever moves work to the CPU, never changes what HRX computes.
static bool hrx_dispatch_group_disabled(const char * group) {
    const char * disabled = std::getenv("HRX_DISABLE_DISPATCH");
    if (disabled == nullptr) {
        return false;
    }
    const size_t length = std::strlen(group);
    for (const char * match = std::strstr(disabled, group); match != nullptr;
         match             = std::strstr(match + 1, group)) {
        const bool starts_token = match == disabled || match[-1] == ',';
        const bool ends_token   = match[length] == '\0' || match[length] == ',';
        if (starts_token && ends_token) {
            return true;
        }
    }
    return false;
}

// qwen4exp.gdn_conv_prepare_decode used to be numerically wrong, and measurably so: generating with
// it enabled emitted garbage ("[Start thinking] V边容量2..."), while declining just this one prelude
// reproduced the CPU reference token for token. The root cause is now understood and fixed on the
// dispatch side: the matcher covered the q/k/v VIEWs and both L2_NORMs while the kernel only ever
// wrote `qkv_silu` plus a split-local `qk_inverse_norm` transient, so q_conv_predelta and
// k_conv_predelta were left permanently unwritten. That is only sound if
// qwen4exp.gdn_recurrent_decode consumes the transient in the SAME split, and GGML_SCHED_DEBUG=2
// proves it never can here: ggml orders the CPU beta/gate chain (MUL_MAT(ssm_alpha) -> ADD(ssm_dt)
// -> SOFTPLUS -> MUL(ssm_a) -> MUL_MAT(ssm_beta) -> SIGMOID) between the two, forcing a split
// boundary. The dispatch now covers only SSM_CONV and its SILU -- the two nodes it actually writes --
// so the prelude is self-contained and correct, with L2_NORM left to run normally.
// Still gated behind HRX_ENABLE_GDN_CONV_PREPARE=1 until the corrected form is verified end to end.
static bool hrx_gdn_conv_prepare_enabled() {
    const char * enabled = std::getenv("HRX_ENABLE_GDN_CONV_PREPARE");
    return enabled != nullptr && enabled[0] != '\0' && enabled[0] != '0';
}

// Mirrors dispatch-copy.cpp: HRX has kernels for the same-length contiguous f32 copy and for the 2D
// row-strided f32 copy (rows element-contiguous). Keeping the two in sync matters more than usual here,
// because a CPY on a pre-allocated cache tensor that this declines aborts ggml_backend_sched outright.
static bool hrx_copy_f32_supported(const ggml_tensor * op) {
    const ggml_tensor * source = op == nullptr ? nullptr : op->src[0];
    if (source == nullptr || op->type != GGML_TYPE_F32 || source->type != GGML_TYPE_F32 ||
        ggml_nelements(op) != ggml_nelements(source) || ggml_nelements(op) <= 0) {
        return false;
    }
    if (ggml_is_contiguous(op) && ggml_is_contiguous(source)) {
        return true;
    }
    const auto rows_copyable = [](const ggml_tensor * t) {
        return t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float) && t->nb[1] % sizeof(float) == 0 &&
               static_cast<int64_t>(t->nb[1] / sizeof(float)) >= t->ne[0];
    };
    return rows_copyable(op) && rows_copyable(source) && op->ne[0] == source->ne[0] && op->ne[1] == source->ne[1];
}

static bool eager_capability_declared(enum ggml_op op) {
    switch (op) {
        // The scheduler probes preallocated weight tensors as NONE operations when deciding whether their buffer type is
        // usable by this backend. Fused ops are declared here so graph-claim can validate the full dispatch pattern.
        // TODO: split this into placement capability and exact graph execution capability once graph claiming owns the
        // full decision.
        case GGML_OP_NONE:
        case GGML_OP_ADD:
        case GGML_OP_ARGSORT:
        case GGML_OP_CLAMP:
        // GGML_OP_CONCAT is deliberately NOT declared. HRX has no CONCAT kernel and no dispatch
        // matcher roots at one: qwen4exp.gdn_conv_prepare_decode roots at SSM_CONV and binds the
        // CONCAT's output as an already-materialized external value (see the header comment on
        // ConvPrepareMatch in dispatch_registration/dispatch-gated-delta-net.cpp). Declaring CONCAT
        // would let the scheduler place a node on HRX that nothing can execute, stranding the split
        // it lands in with "unsupported HRX node ... CONCAT".
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DIV:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_GET_ROWS:
        case GGML_OP_GLU:
        case GGML_OP_L2_NORM:
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
        case GGML_OP_SCALE:
        case GGML_OP_SET_ROWS:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_UNARY:
        case GGML_OP_VIEW:
            return true;
        default:
            return false;
    }
}

// The HRX kernel corpus only provides quantized matmul/embedding kernels for a small set of weight
// quantizations (Q4_K / Q6_K, plus unquantized F32/F16/BF16 operands). Weights stored in any other
// quantization (q8_0, iq3_xxs, iq4_nl, iq4_xs, Q3_K, Q5_K, ...) have no HRX kernel, so HRX must NOT
// claim them: doing so either pins the weight into an HRX buffer that cannot run its consuming op
// (hard "unsupported HRX node" failure) or forces per-token weight copies back to the CPU. Declining
// them lets the scheduler keep those weights on the CPU and run the op there.
static bool hrx_weight_quant_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// Dense MUL_MAT is narrower than hrx_weight_quant_supported() above: every registered MUL_MAT
// matcher (dispatch-llm-matmul.cpp's llm.matmul.dense_{q4k,q6k,q8_0}_f16_wmma, and
// dispatch-qwen-matmul.cpp's qwen.* projection fusions) hard-requires a Q4_K, Q6_K or Q8_0 weight and
// returns "no match" for any other weight type. An *unquantized* (F32/F16/BF16) weight therefore has
// no dense-matmul kernel at all, so claiming it here would pin the node onto HRX with no dispatch
// able to run it -- which DispatchScheduler reports as a hard "unsupported HRX node ...: MUL_MAT"
// failure that fails the entire split, rather than degrading to CPU for just that node.
//
// qwen4exp hits exactly this with its two small per-head GDN projections, ssm_beta and ssm_alpha
// ([n_embd, num_v_heads] == [2560,48], see build_layer_attn_linear() in src/models/qwen4exp.cpp),
// which stay F32 even in the Q4_K_XL quantization because they are far too small to be worth
// quantizing. Declining them leaves them on the CPU where they cost ~123K MACs each -- negligible
// next to the layer's quantized projections.
//
// Q8_0 is included because Unsloth's UD ("dynamic") quantizations keep most dense projection weights
// at Q8_0 rather than Q4_K -- for the qwen4exp UD-Q4_K_XL GGUF that is 503 of 1224 tensors, and
// without a Q8_0 dense route the loader pushes 1123 tensors to CPU and ~77% of the per-token graph
// runs there. @qwen3_moe_dense_linear_q8_0_f16_wmma covers it.
//
// Q8_0 is deliberately NOT folded into hrx_weight_quant_supported() because that helper is
// op-agnostic and also gates GET_ROWS (token_embd.weight is Q8_0 and there is no Q8_0 embed kernel)
// and MUL_MAT_ID. Instead this narrower helper is OR'd into the GGML_OP_NONE placement probe so a
// Q8_0 weight may live in an HRX buffer, exactly as hrx_moe_down_weight_supported() already is.
//
// Note this deliberately does NOT narrow the GGML_OP_NONE placement probe or GET_ROWS, which must
// keep accepting F32/F16/BF16: unquantized tensors like norm weights are consumed on HRX by
// RMS_NORM/MUL/etc. and do need to be placeable in an HRX buffer.
static bool hrx_dense_matmul_weight_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

// device_supports_op() must not claim a Q8_0 MUL_MAT that match_llm_dense_matmul()
// (dispatch-llm-matmul.cpp) would decline: an op HRX claims but that no matcher roots at strands the
// whole split with a hard "unsupported HRX node" abort instead of degrading to the CPU for that node.
// Unlike Q4_K/Q6_K -- which additionally have the fused qwen.* projection matchers in
// dispatch-qwen-matmul.cpp as a second chance -- Q8_0 is served only by
// "llm.matmul.dense_q8_0_f16_wmma", so these guards mirror that matcher's exactly: 2D contiguous
// weight/activation/result, f32 activation and result, a 256-aligned contraction width, and a token
// count inside the kernel's token_capacity bound.
static constexpr int64_t kHrxDenseMatmulMaxTokenCount = 2048;

static bool hrx_dense_matmul_q8_0_shape_supported(const ggml_tensor * op) {
    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * input  = op->src[1];
    if (weight == nullptr || input == nullptr) {
        return false;
    }
    const auto is_2d = [](const ggml_tensor * tensor) { return tensor->ne[2] == 1 && tensor->ne[3] == 1; };
    if (!is_2d(weight) || !is_2d(input) || !is_2d(op) || !ggml_is_contiguous(weight) ||
        !ggml_is_contiguous(input) || !ggml_is_contiguous(op) || input->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    return input->ne[0] == input_size && op->ne[0] == output_size && op->ne[1] == token_count &&
           input_size >= 256 && input_size <= 32768 && input_size % 256 == 0 && output_size >= 1 &&
           output_size <= 262144 && token_count >= 1 && token_count <= kHrxDenseMatmulMaxTokenCount;
}

// qwen4exp (Q4_K_XL / Q3_K_XL GGUF quantizations) stores its MoE down-projection expert weights
// (ffn_down_exps) in 32-block quantizations -- Q5_1, Q8_0, IQ4_NL -- because its contraction width
// (expert_hidden_size == kLlmMoeQwen4ExpDispatchProfile.expert_hidden_size == 640) is NOT a multiple
// of 256, so the existing Q4_K/Q6_K (256-superblock) HRX down kernels cannot read it (see
// hrx-moe-down-kernel-spec.md §4). A dedicated 32-block down kernel is authored against exactly that
// (op == MUL_MAT_ID, quant, contraction-width) triple.
//
// Crucially, Q8_0 is ALSO the quantization used for this model family's attention/lm_head projection
// weights, which are plain (dense) GGML_OP_MUL_MAT nodes, not GGML_OP_MUL_MAT_ID. Those are now
// covered by hrx_dense_matmul_weight_supported() above (@qwen3_moe_dense_linear_q8_0_f16_wmma), but
// this check must still stay separate from hrx_weight_quant_supported() (which is op-agnostic):
// folding Q5_1/Q8_0/IQ4_NL into hrx_weight_quant_supported() would make HRX also claim Q8_0
// GET_ROWS (token_embd.weight) and unmatched MUL_MAT_ID nodes, and crash with no kernel able to run
// them. Instead this is
// consulted only from the GGML_OP_MUL_MAT_ID arm of device_supports_op(), and only for weights whose
// shape matches the down-projection contraction width.
static bool hrx_moe_down_quant_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

// Contraction width (ne[0]) of a qwen4exp MUL_MAT_ID down-projection weight (ffn_down_exps):
// expert_hidden_size == 640 per kLlmMoeQwen4ExpDispatchProfile. 640 is not a multiple of 256 -- unlike
// qwen30b's expert_hidden_size == 768, which the existing Q4_K/Q6_K down kernels already cover -- so
// checking this exact width is what distinguishes a qwen4exp down weight from any other MUL_MAT_ID
// operand purely from the tensor itself, with no model/profile context available at this call site.
static constexpr int64_t kHrxMoeDownExpertHiddenSizeQwen4Exp = 640;

// Same quant+shape test the GGML_OP_MUL_MAT_ID arm below applies to a down-projection weight's
// src[0], factored out so the GGML_OP_NONE placement probe (where the weight tensor itself is
// passed as `op`, with no src[0] indirection) can apply the identical test. See the comment on
// GGML_OP_NONE in device_supports_op() for why the probe must agree with this decision: the model
// loader picks a weight's buffer type by probing its real consuming op (MUL_MAT_ID here, which
// accepts this weight), so the later NONE-probe on the pre-allocated weight tensor must accept the
// same weight or ggml-backend's scheduler aborts with "pre-allocated tensor ... that cannot run the
// operation (NONE)" -- the weight is stuck in an HRX buffer neither check will let it leave.
//
// Currently forced off: HRX's routed FFN is an all-or-nothing fused chain. The down dispatch
// ("llm.routed_ffn.decode_down_qwen4exp") binds match.input_alternate->alternate_value -- the Q8
// hidden state *produced by* the fused gate/up dispatch ("qwen.decode.moe.hidden_q8") -- so it can
// only match when gate/up ran on HRX in the same graph. Since qwen4exp's gate/up has no matcher (see
// hrx_moe_gate_up_weight_qwen4exp() below), that alternate never exists and the down node would be
// claimed with no dispatch able to run it. Keeping the quant/shape test intact but gating it here
// means restoring the whole path is a one-line change once a qwen4exp gate/up kernel exists.
static constexpr bool kHrxMoeDownQwen4ExpEnabled = false;

static bool hrx_moe_down_weight_supported(const ggml_tensor * weight) {
    return kHrxMoeDownQwen4ExpEnabled && weight != nullptr && hrx_moe_down_quant_supported(weight->type) &&
           weight->ne[0] == kHrxMoeDownExpertHiddenSizeQwen4Exp;
}

// qwen4exp hidden_size (n_embd). Shared by the HC residual-stream checks and the routed-FFN checks
// below, both of which key off the full-width row.
static constexpr int64_t kHrxHiddenSizeQwen4Exp = 2560;

// qwen4exp routed gate/up expert weights (ffn_gate_exps / ffn_up_exps): Q4_K
// [hidden_size=2560, expert_hidden_size=640, expert_count=512]. Note this is the transpose of the
// down-projection geometry checked above ([640, 2560, 512]), so the two never collide.
//
// Every registered routed-FFN gate/up matcher in dispatch-routed-ffn.cpp
// ("llm.routed_ffn.decode_gate_up_swiglu_q4k_q8" and "llm.routed_ffn.gate_up_swiglu_q4k_f16_wmma")
// is written against the kRoutedFfn* constants, which are pinned to the *qwen30b* profile
// (hidden_size 2048, expert_hidden_size 768, expert_count 128, route_count 8) and must stay pinned --
// see dispatch-llm-profiles.h. Only the *down* projection ever got a qwen4exp-specific matcher
// ("llm.routed_ffn.decode_down_qwen4exp"). So a qwen4exp gate/up MUL_MAT_ID matches nothing, and
// claiming it strands the node with no dispatch and fails its whole split.
//
// Declining leaves the routed gate/up on the CPU. At decode that is route_count(10) x 2560 x 640 x 2
// projections for a single token, which the CPU absorbs comfortably; a dedicated qwen4exp gate/up
// kernel + matcher is the natural follow-up once the GDN/QSA decode path is validated.
static constexpr int64_t kHrxMoeGateUpExpertCountQwen4Exp = 512;
static constexpr int64_t kHrxMoeRouteCountQwen4Exp        = 10;

static bool hrx_moe_gate_up_weight_qwen4exp(const ggml_tensor * weight) {
    return weight != nullptr && weight->ne[0] == kHrxHiddenSizeQwen4Exp &&
           weight->ne[1] == kHrxMoeDownExpertHiddenSizeQwen4Exp &&
           weight->ne[2] == kHrxMoeGateUpExpertCountQwen4Exp;
}

// The SwiGLU between qwen4exp's routed gate/up and down projections. HRX only ever runs this
// activation *inside* the fused "llm.routed_ffn.*gate_up_swiglu*" dispatches -- there is no matcher
// rooted at a standalone GGML_OP_GLU. Since hrx_moe_gate_up_weight_qwen4exp() above keeps qwen4exp's
// gate/up on the CPU, that fused dispatch never forms here and the GLU is left exposed, so it has to
// follow its producers to the CPU. Scoped to the routed-expert decode shape
// [expert_hidden_size=640, route_count=10] so other models' GLU nodes keep the permissive default.
static bool hrx_moe_glu_qwen4exp_decode(const ggml_tensor * op) {
    return op != nullptr && op->type == GGML_TYPE_F32 && op->ne[0] == kHrxMoeDownExpertHiddenSizeQwen4Exp &&
           op->ne[1] == kHrxMoeRouteCountQwen4Exp && op->ne[2] == 1 && op->ne[3] == 1;
}

// qwen4exp GDN (Gated DeltaNet) decode-only shape profile: head_k_dim == head_v_dim == 128
// (hparams.ssm_d_state), num_k_heads == 16 (hparams.ssm_n_group), num_v_heads == 48
// (hparams.ssm_dt_rank), single token / single sequence decode (K == 1). These constants are
// duplicated (rather than shared) from dispatch-gated-delta-net.cpp's matchers, mirroring how
// kHrxMoeDownExpertHiddenSizeQwen4Exp above is likewise a local, self-contained shape constant: this
// file has no dependency on dispatch_registration/.
static constexpr int64_t kHrxGdnHeadDimQwen4Exp       = 128;
static constexpr int64_t kHrxGdnKeyHeadCountQwen4Exp   = 16;
static constexpr int64_t kHrxGdnValueHeadCountQwen4Exp = 48;
static constexpr int64_t kHrxGdnConvChannelsQwen4Exp =
    kHrxGdnHeadDimQwen4Exp * kHrxGdnKeyHeadCountQwen4Exp * 2 + kHrxGdnHeadDimQwen4Exp * kHrxGdnValueHeadCountQwen4Exp;
static constexpr int64_t kHrxGdnConvKernelSizeQwen4Exp = 4;  // hparams.ssm_d_conv

// GGML_OP_GATED_DELTA_NET, GGML_OP_SSM_CONV, GGML_OP_L2_NORM, and GGML_OP_UNARY are all real compute
// ops used by many models beyond qwen4exp -- e.g. every other delta-net-family model built on
// delta-net-base.cpp (qwen3next, kimi-linear, ...) also emits GATED_DELTA_NET/L2_NORM/SSM_CONV with
// their own, different head-dim/head-count profiles; and GGML_OP_UNARY is how virtually every model's
// SiLU/sigmoid activations are represented (there is no separate GGML_OP_SILU/GGML_OP_SIGMOID -- both
// are GGML_OP_UNARY with a different ggml_unary_op sub-code). Unlike VIEW/RESHAPE/PERMUTE/TRANSPOSE
// (pure layout-alias ops with zero real-compute risk, handled by the `default: return true;`
// fallthrough below), declaring these four in eager_capability_declared() without narrowing them here
// would make HRX unconditionally claim every model's use of these ops regardless of shape -- the same
// "wall #3" over-claim risk documented above hrx_moe_down_quant_supported(). So each is checked below
// against qwen4exp's exact decode profile and declined for anything else.
//
// GGML_OP_CONCAT used to be narrowed here too, by a hrx_gdn_conv_concat_decode_supported() that
// matched build_conv_state_at()'s CONCAT(conv_history[3,C,1,1], transpose(qkv_mixed)[1,C,1,1]) ->
// [4,C,1,1] and returned true for it. That was always unsafe and is now removed along with the
// GGML_OP_CONCAT entry in eager_capability_declared(): no HRX kernel or dispatch matcher covers a
// CONCAT node, so *claiming* one can only ever strand a split. It went unnoticed because ggml's
// cross-backend scheduler happened to assign this CONCAT to CPU anyway (it prefers to run an op on
// the same backend as its inputs, and the CONCAT's sources root through build_rs()'s ggml_get_rows
// gather of the recurrent conv-state buffer, which was CPU-resident). Once the recurrent-state
// SCALE/CPY/CONT ops became HRX-supported that input locality flipped, ggml routed the CONCAT to HRX,
// and the claim turned into a hard "unsupported HRX node 0: CONCAT" failure.
static bool hrx_gdn_ssm_conv_decode_supported(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * sx = op->src[0];  // conv_input: [d_conv, d_inner, n_s]
    const ggml_tensor * c  = op->src[1];  // conv kernel: [d_conv, d_inner]
    return sx->ne[0] == kHrxGdnConvKernelSizeQwen4Exp && sx->ne[1] == kHrxGdnConvChannelsQwen4Exp && sx->ne[2] == 1 &&
           c->ne[0] == kHrxGdnConvKernelSizeQwen4Exp && c->ne[1] == kHrxGdnConvChannelsQwen4Exp &&
           op->ne[0] == kHrxGdnConvChannelsQwen4Exp && op->ne[1] == 1 && op->ne[2] == 1;
}

static bool hrx_gdn_gated_delta_net_decode_supported(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr ||
        op->src[3] == nullptr || op->src[4] == nullptr || op->src[5] == nullptr) {
        return false;
    }
    const ggml_tensor * q     = op->src[0];
    const ggml_tensor * k     = op->src[1];
    const ggml_tensor * v     = op->src[2];
    const ggml_tensor * gate  = op->src[3];
    const ggml_tensor * beta  = op->src[4];
    const ggml_tensor * state = op->src[5];
    return ggml_get_op_params_i32(op, 0) == 1 &&  // K == 1: single-chunk decode; chunked prefill (K>1) declines.
           q->ne[0] == kHrxGdnHeadDimQwen4Exp && q->ne[1] == kHrxGdnKeyHeadCountQwen4Exp && q->ne[2] == 1 &&
           q->ne[3] == 1 && k->ne[0] == kHrxGdnHeadDimQwen4Exp && k->ne[1] == kHrxGdnKeyHeadCountQwen4Exp &&
           k->ne[2] == 1 && k->ne[3] == 1 && v->ne[0] == kHrxGdnHeadDimQwen4Exp &&
           v->ne[1] == kHrxGdnValueHeadCountQwen4Exp && v->ne[2] == 1 && v->ne[3] == 1 && gate->ne[0] == 1 &&
           gate->ne[1] == kHrxGdnValueHeadCountQwen4Exp && gate->ne[2] == 1 && gate->ne[3] == 1 &&
           beta->ne[0] == 1 && beta->ne[1] == kHrxGdnValueHeadCountQwen4Exp && beta->ne[2] == 1 &&
           beta->ne[3] == 1 && state->ne[0] == kHrxGdnHeadDimQwen4Exp && state->ne[1] == kHrxGdnHeadDimQwen4Exp &&
           state->ne[2] == kHrxGdnValueHeadCountQwen4Exp && state->ne[3] == 1;
}

static bool hrx_gdn_unary_decode_supported(const ggml_tensor * op) {
    if (op == nullptr) {
        return false;
    }
    switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_SILU:
            // qwen4exp.gdn_conv_prepare_decode's post-SSM_CONV activation: [conv_channels,1,1,1].
            // The SSM_CONV producer check is load-bearing, not decoration: qwen4exp's GDN conv width
            // (2*key_dim + value_dim == 10240) collides exactly with hc_dim == 4*n_embd, so build_ple()'s
            // own depthwise-conv SILU has an identical shape but is fed by the ADD that sums its taps.
            // Only the SSM_CONV-rooted one has a dispatch; claiming PLE's would strand it.
            return op->ne[0] == kHrxGdnConvChannelsQwen4Exp && op->ne[1] == 1 && op->ne[2] == 1 &&
                   op->ne[3] == 1 && op->src[0] != nullptr && op->src[0]->op == GGML_OP_SSM_CONV;
        case GGML_UNARY_OP_SIGMOID:
            // qwen4exp.gdn_norm_gate_decode's z-gate: [head_v_dim,num_v_heads,1,1] == [128,48,1,1].
            // Deliberately narrower than the [1,48,1,1] shape of this same layer's *other* sigmoid
            // (build_layer_attn_linear's `beta = ggml_sigmoid(ctx0, beta)`), which is left
            // CPU-fallback: no dispatch matcher roots at a standalone SIGMOID, so claiming that shape
            // too would strand it with no matcher (hard scheduling failure).
            return op->ne[0] == kHrxGdnHeadDimQwen4Exp && op->ne[1] == kHrxGdnValueHeadCountQwen4Exp &&
                   op->ne[2] == 1 && op->ne[3] == 1;
        default:
            return false;
    }
}

// qwen4exp's GDN prelude (build_layer_attn_linear() in src/models/qwen4exp.cpp) derives two per-head
// scalar vectors of length num_v_heads == 48 from the ssm_alpha/ssm_beta projections:
//
//     alpha_biased   = ADD(alpha, ssm_dt)              -> f32[48,1,1,1]
//     alpha_softplus = SOFTPLUS(alpha_biased)          -> declined by hrx_gdn_unary_decode_supported()
//     gate           = MUL(alpha_softplus, ssm_a)      -> f32[48,1,1,1]
//
// No dispatch matcher roots at either of these -- the GDN kernels take `gate` and `beta` as already
// -computed inputs -- so, exactly like the sibling SIGMOID documented above, claiming them would
// strand them with no dispatch and fail the whole split. Their projections already run on the CPU
// (their ssm_alpha/ssm_beta weights are F32, declined by hrx_dense_matmul_weight_supported()), so
// keeping these two 48-element elementwise ops there as well costs nothing and avoids a round trip.
static bool hrx_gdn_per_head_scalar_decode_op(const ggml_tensor * op) {
    return op != nullptr && op->type == GGML_TYPE_F32 && op->ne[0] == kHrxGdnValueHeadCountQwen4Exp &&
           op->ne[1] == 1 && op->ne[2] == 1 && op->ne[3] == 1;
}

// qwen4exp attention geometry, from the GGUF: n_embd_head_k == n_embd_head_v == 256, n_head_kv == 2,
// plus a lightning-indexer key cache of indexer.key_length == 128 with a single head. These mirror
// kQwen4ExpAttentionHeadSize / kQwen4ExpKeyValueHeadCount in
// dispatch_registration/dispatch-qwen4exp-flash-attention.cpp.
static constexpr int64_t kHrxAttentionHeadSizeQwen4Exp = 256;
static constexpr int64_t kHrxIndexerKeyLengthQwen4Exp  = 128;

// qwen4exp Hyper-Connections geometry: hparams.dsv4_hc_mult == 4 parallel [n_embd] residual streams,
// so the flattened stream layout is hc_dim == 4*2560 == 10240 wide.
static constexpr int64_t kHrxHcMultiplierQwen4Exp = 4;
static constexpr int64_t kHrxHcDimQwen4Exp        = kHrxHcMultiplierQwen4Exp * kHrxHiddenSizeQwen4Exp;

// HRX's only GET_ROWS kernels are embedding lookups: they read a leaf weight tensor. qwen4exp's
// build_qsa_top_k() emits two gathers that are not lookups at all -- ggml_get_rows(k_all, blk_cells)
// over an F16 view of the indexer key cache (src/models/qwen4exp.cpp line 593) and
// ggml_get_rows(score, cell_blk) expanding per-block scores to per-token ones (line 651) -- neither of
// which any matcher roots at. Requiring a leaf source keeps every embedding lookup claimed, including
// F16 ones, while declining both.
static bool hrx_non_leaf_gather_qwen4exp(const ggml_tensor * op) {
    const ggml_tensor * source = op == nullptr ? nullptr : op->src[0];
    return source != nullptr && source->op != GGML_OP_NONE;
}

// qwen4exp's lightning indexer (build_qsa_top_k() in src/models/qwen4exp.cpp) works entirely in
// indexer.key_length == 128 wide rows: the pooled block keys [128, n_blocks], the four query heads
// [128, 4, T], their RMS norms and mropes, and the slice sums that collapse both. It has no HRX
// dispatch of any kind -- register_* in dispatch_registration/ never mentions it -- so every one of
// those nodes has to stay on the CPU.
//
// GDN is the only other qwen4exp subsystem that works in 128-wide rows (head_v_dim == 128), and its
// kernels do need to be claimed, so its three shapes are carved out: [128, num_k_heads == 16] for q/k,
// [128, num_v_heads == 48] for v and the norm/gate pair, and [128, 128, 48] for the recurrent state.
// The carve-out is by shape rather than provenance, so an indexer tensor whose block count happened to
// equal 16, 48 or 128 would be claimed and strand its split; n_blocks is n_kv/compress_ratio, and n_kv
// is padded to a multiple of 256, so that needs an unusual ratio to occur.
static bool hrx_qsa_indexer_row_qwen4exp(const ggml_tensor * op) {
    if (op == nullptr || op->ne[0] != kHrxIndexerKeyLengthQwen4Exp) {
        return false;
    }
    const bool gdn_shape = op->ne[1] == kHrxGdnKeyHeadCountQwen4Exp || op->ne[1] == kHrxGdnValueHeadCountQwen4Exp ||
                           op->ne[1] == kHrxGdnHeadDimQwen4Exp;
    return !gdn_shape || hrx_dispatch_group_disabled("gdn");
}

// The GDN pipeline is split into two independently disableable bisect groups: "gdnconv" (the
// CONCAT/SSM_CONV/SILU/L2_NORM conv-prepare prelude) and "gdncore" (the recurrent GATED_DELTA_NET
// step). "gdn" remains an umbrella that disables both, so existing invocations keep working.
static bool hrx_gdn_group_disabled(const char * half) {
    if (std::strcmp(half, "gdnconv") == 0 && !hrx_gdn_conv_prepare_enabled()) {
        return true;  // see hrx_gdn_conv_prepare_enabled()
    }
    // "gdncore" (qwen4exp.gdn_recurrent_decode) is unconditionally disabled. Its matcher binds the
    // shared `qkv_silu` value plus the `qk_inverse_norm` transient published by
    // qwen4exp.gdn_conv_prepare_decode, and additionally requires that dispatch's L2_NORM/VIEW nodes
    // to be covered in the *same* split. Now that conv-prepare correctly covers only SSM_CONV+SILU
    // (and no longer hides unwritten L2_NORM outputs), that precondition can never be satisfied --
    // and GGML_SCHED_DEBUG=2 shows ggml always separates the two with the CPU beta/gate chain
    // anyway. Claiming GATED_DELTA_NET while the matcher cannot fire would strand its split, so the
    // recurrent step stays on the CPU reference implementation until the kernel is decoupled from
    // the fused conv prelude (needs a variant taking pre-normalized q/k/v as separate buffers).
    if (std::strcmp(half, "gdncore") == 0) {
        return true;
    }
    return hrx_dispatch_group_disabled("gdn") || hrx_dispatch_group_disabled(half);
}

// "misc" is an umbrella over the per-op-class groups; see the switch in device_supports_op().
// "melem" is in turn an umbrella over the elementwise sub-classes "eadd"/"emul"/"eother".
static bool hrx_misc_group_disabled(const char * op_class) {
    if (hrx_dispatch_group_disabled("misc") || hrx_dispatch_group_disabled(op_class)) {
        return true;
    }
    const bool elementwise = std::strcmp(op_class, "eadd") == 0 || std::strcmp(op_class, "emul") == 0 ||
                             std::strcmp(op_class, "eother") == 0;
    return elementwise && hrx_dispatch_group_disabled("melem");
}

// qwen4exp.gdn_norm_gate_decode (dispatch-gated-delta-net.cpp) is a *third* GDN dispatch, rooted at
// the [head_dim, value_head_count] RMS_NORM of the delta-net output and swallowing the gamma MUL,
// the SIGMOID and the gated MUL. Because it roots at RMS_NORM rather than at a GDN-specific op, the
// CONCAT/SSM_CONV/GATED_DELTA_NET arms do not cover it, so it needs its own decline hook to be
// separable from the rest of GDN. Only the RMS_NORM root and the MUL shapes are keyed here; the
// SIGMOID rides along on the UNARY arm.
static bool hrx_gdn_norm_gate_row(const ggml_tensor * op) {
    if (op == nullptr || !hrx_gdn_group_disabled("gdngate")) {
        return false;
    }
    return op->ne[0] == kHrxGdnHeadDimQwen4Exp && op->ne[1] == kHrxGdnValueHeadCountQwen4Exp;
}

// The "hc" bisect group: the grouped RMS_NORM registered in dispatch-rmsnorm.cpp roots at the
// stream-major [n_embd, hc, T] norm and fuses the gamma MUL over its flattened [hc_dim, T] view, so
// both row widths have to be declined to push the whole hyper-connection norm back onto the CPU.
static bool hrx_hc_grouped_norm_row_qwen4exp(const ggml_tensor * op) {
    if (op == nullptr || !hrx_dispatch_group_disabled("hc")) {
        return false;
    }
    return op->ne[0] == kHrxHcDimQwen4Exp ||
           (op->ne[0] == kHrxHiddenSizeQwen4Exp && op->ne[1] == kHrxHcMultiplierQwen4Exp);
}

// qwen4exp's attention preamble -- the per-head RMS_NORM, its gamma MUL, and the ROPE for Q and K --
// operates on head-major [256, n_head, T] tensors. Every matcher in dispatch-rmsnorm.cpp and
// dispatch-qwen-attention-postprocess.cpp is written for either the flat hidden width or
// kQwenAttentionHeadSize == 128 heads, so none of them can root at a 256-wide head. The qwen4exp flash
// attention dispatch does not need them: it roots at FLASH_ATTN_EXT and takes Q/K/V as plain inputs
// (see match_qwen4exp_flash_attention_gate in dispatch-qwen4exp-flash-attention.cpp), so leaving the
// preamble on the CPU costs one small transfer per attention layer and nothing else.
static bool hrx_attention_head_major_qwen4exp(const ggml_tensor * op) {
    return op != nullptr && op->ne[0] == kHrxAttentionHeadSizeQwen4Exp;
}

// Walk back through pure-layout nodes looking for a tensor of the given row width. A cache publish
// source arrives flattened to [n_embd_gqa, T], but the head-major [head_size, head_count, T] shape it
// was reshaped from is still one or two hops up: ggml records the immediate parent in src[0] for both
// views and reshapes. Every hop is tested, not just the terminal one, because the value projection
// reshapes head-major and then straight back to flat before the store.
static bool hrx_layout_chain_has_row_width(const ggml_tensor * t, int64_t row_width) {
    for (int hop = 0; hop < 5 && t != nullptr; ++hop) {
        if (t->ne[0] == row_width) {
            return true;
        }
        switch (t->op) {
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONT:
                t = t->src[0];
                break;
            default:
                return false;
        }
    }
    return false;
}

// The KV-cache publish dispatches in dispatch_registration/dispatch-qwen-attention-postprocess.cpp are
// written entirely against kQwenAttentionHeadSize == 128 with several key/value heads. qwen4exp has
// neither geometry and publishes two caches that no matcher can accept:
//
//   build_attn_qsa() K/V:      head-major [256,2,T] -- 256-wide heads, so every head-size check fails
//   build_qsa_top_k() indexer: head-major [128,1,T] -- one head, and the key is cached *raw* (line 581
//                              of src/models/qwen4exp.cpp: pooling precedes norm and rotation), so the
//                              key-publish chain's mandatory ROPE producer is absent
//
// Claiming either strands the SET_ROWS with no dispatch. This only ever fires on qwen4exp shapes; any
// geometry it does not positively recognise stays claimed, so the qwen30b profile is untouched.
static bool hrx_cache_publish_unmatched_qwen4exp(const ggml_tensor * op) {
    const ggml_tensor * source = op == nullptr ? nullptr : op->src[0];
    if (source == nullptr) {
        return false;
    }
    if (hrx_layout_chain_has_row_width(source, kHrxAttentionHeadSizeQwen4Exp)) {
        return true;
    }
    return source->ne[0] == kHrxIndexerKeyLengthQwen4Exp && source->ne[1] == 1;
}

// HRX's only hc_dim-wide MUL kernel is the gamma scale fused into the grouped RMS_NORM registered in
// dispatch-rmsnorm.cpp, whose second operand is always a leaf norm weight. qwen4exp emits two other
// MULs at exactly the same f32[10240,T] shape, both multiplying two *computed* activations and neither
// rooted at by any matcher:
//
//   build_hc_mix():            MUL(hc_norm_out, SIGMOID(...))  -- gates the streams before collapsing
//   build_ple() depthwise conv: MUL(CONT(view), RESHAPE(wk))   -- one shifted tap times its per-channel
//                                                                 weight column, summed over the kernel
//
// Requiring the second operand to be a leaf keeps the fused grouped norm claimed while declining both.
static bool hrx_hc_dim_activation_mul_qwen4exp(const ggml_tensor * op) {
    if (op == nullptr || op->type != GGML_TYPE_F32 || op->src[1] == nullptr) {
        return false;
    }
    return op->ne[0] == kHrxHcDimQwen4Exp && op->src[1]->op != GGML_OP_NONE;
}

// The only CLAMP the HRX corpus can execute is the epsilon floor folded into the MoE router and
// routed-FFN dispatches (dispatch-moe-router.cpp, dispatch-routed-ffn.cpp), which always sits directly
// on a router-weight SUM_ROWS. qwen4exp's build_ple() emits a second, unrelated CLAMP -- the magnitude
// floor of its signed-sqrt gate, which sits on an ABS -- and a MUL that re-applies the sign, neither of
// which any matcher roots at. Keying off the producing op rather than a shape keeps this exact: the
// router clamp never reads an ABS, and nothing but PLE multiplies by an SGN.
static bool hrx_ple_signed_sqrt_gate_qwen4exp(const ggml_tensor * op) {
    const ggml_tensor * src = op == nullptr ? nullptr : op->src[0];
    return src != nullptr && src->op == GGML_OP_UNARY &&
           (ggml_get_unary_op(src) == GGML_UNARY_OP_ABS || ggml_get_unary_op(src) == GGML_UNARY_OP_SGN);
}

// An elementwise MUL over qwen4exp's stream-major [n_embd, streams] layout, where `streams` is the
// hyper-connection count (4) or the routed-expert count (10). qwen4exp emits four of these, none of
// which any dispatch matcher roots at:
//
//   build_hc_combine():         f32[2560,4]  * f32[1,4]     -- scatters a block output back across the
//                                                              hc residual streams, consumed by an ADD
//   build_ple() gate scaling:   f32[2560,4]  * f32[1,4]     -- broadcasts the PLE gate over the value
//   build_ple() dot product:    f32[2560,4]  * f32[2560,4]  -- per-stream key.query, consumed by SUM_ROWS
//   routed-FFN weighted reduce: f32[2560,10] * f32[1,10]    -- scales each expert's contribution by its
//                                                              router weight
//
// The HC kernels registered in dispatch-rmsnorm.cpp cover the grouped *norm*, a different pattern that
// MULs by a full hc_dim-wide (10240) weight; there is no PLE dispatch at all; and the routed-FFN reduce
// only ever runs fused inside "llm.routed_ffn.*down*" -- which cannot match here (see
// hrx_moe_down_weight_supported()). So claiming any of them would strand the node with no dispatch.
//
// The `op->ne[1] > 1` guard is load-bearing: it keeps the ordinary RMS_NORM-times-weight fusion
// (f32[2560,T] * f32[2560,1]) out of this predicate, which would otherwise collide during single-token
// decode where T == 1. Every shape here is at most n_embd*10 == 25600 elements per layer, so the CPU
// handles them for free.
static bool hrx_stream_major_mul_qwen4exp(const ggml_tensor * op) {
    if (op == nullptr || op->type != GGML_TYPE_F32 || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * rows    = op->src[0];
    const ggml_tensor * scalars = op->src[1];
    return op->ne[0] == kHrxHiddenSizeQwen4Exp && op->ne[1] > 1 && op->ne[3] == 1 &&
           rows->ne[0] == kHrxHiddenSizeQwen4Exp && rows->ne[1] == op->ne[1] &&
           (scalars->ne[0] == 1 || scalars->ne[0] == kHrxHiddenSizeQwen4Exp) &&
           scalars->ne[1] == op->ne[1];
}

// The only SUM_ROWS the HRX corpus can execute are the two folded into the MoE router and routed-FFN
// dispatches (dispatch-moe-router.cpp, dispatch-routed-ffn.cpp), which reduce a [n_expert_used, T]
// router-weight row. qwen4exp's build_ple() emits a second, unrelated SUM_ROWS that reduces the
// [n_embd, hc] per-stream key.query product; it has no matcher, so decline it by its input width.
static bool hrx_ple_stream_reduce_qwen4exp(const ggml_tensor * op) {
    return op != nullptr && op->src[0] != nullptr && op->src[0]->ne[0] == kHrxHiddenSizeQwen4Exp;
}

static bool device_supports_op(ggml_backend_dev_t device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    if (op == nullptr || !eager_capability_declared(op->op)) {
        return false;
    }
    // Zero-token graphs are real: llama.cpp builds worst-case reserve graphs and, for qwen4exp, an MTP
    // draft layer whose ubatch can carry no tokens, which reaches here as e.g. f32[2560,4,0,1]. The
    // recurrent-state clear in build_rs() is likewise a zero-sized view whenever no sequence slot needs
    // resetting. These must be claimed rather than declined: ggml_backend_sched aborts outright when a
    // pre-allocated tensor -- the recurrent state cache is one -- lands in a buffer whose backend
    // refuses its op, so declining is not a fallback but a crash. The dispatch scheduler elides empty
    // nodes instead of matching them, so claiming one costs nothing and strands no split.
    if (ggml_is_empty(op)) {
        return true;
    }
    // The "misc" bisect group covers every op HRX claims by default rather than through one of the
    // named subsystem groups below -- the elementwise, normalisation, layout and residual ops.
    // Disabling it alongside the named groups leaves nothing at all on HRX, which is the known-good
    // starting point a numerical bisection needs. It is further subdivided by op class ("mnorm",
    // "mrope", "msoftmax", "msetrows", "melem", "mlayout") so a bisection can narrow to one class
    // without giving up the rest; "misc" is the umbrella that disables all six.
    switch (op->op) {
        // Owned by a named subsystem group, so never part of "misc".
        case GGML_OP_NONE:
        case GGML_OP_GET_ROWS:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_SSM_CONV:
        case GGML_OP_L2_NORM:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_UNARY:
        case GGML_OP_FLASH_ATTN_EXT:
            break;
        case GGML_OP_RMS_NORM:
            if (hrx_misc_group_disabled("mnorm")) {
                return false;
            }
            break;
        case GGML_OP_ROPE:
            if (hrx_misc_group_disabled("mrope")) {
                return false;
            }
            break;
        case GGML_OP_SOFT_MAX:
            if (hrx_misc_group_disabled("msoftmax")) {
                return false;
            }
            break;
        case GGML_OP_SET_ROWS:
            if (hrx_misc_group_disabled("msetrows")) {
                return false;
            }
            break;
        case GGML_OP_ADD:
            if (hrx_misc_group_disabled("eadd")) {
                return false;
            }
            break;
        case GGML_OP_MUL:
            if (hrx_misc_group_disabled("emul")) {
                return false;
            }
            break;
        case GGML_OP_DIV:
        case GGML_OP_CLAMP:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_GLU:
            if (hrx_misc_group_disabled("eother")) {
                return false;
            }
            break;
        default:
            if (hrx_misc_group_disabled("mlayout")) {
                return false;
            }
            break;
    }

    switch (op->op) {
        // Placement probe: the scheduler asks whether an HRX buffer may hold a pre-allocated tensor by
        // presenting it as a NONE op (see ggml_backend_sched_backend_id_from_cur() in
        // ggml-backend.cpp, which calls device_supports_op() with the weight tensor itself, whose
        // ->op field is GGML_OP_NONE for any leaf/weight tensor). Decline weights whose quantization
        // has no HRX kernel so they stay on the CPU instead of being pinned into an HRX buffer we
        // cannot compute against. This must accept exactly the same weights the real consuming-op
        // arms below accept -- the model loader chooses a weight's buffer type by probing its real
        // consuming op (e.g. MUL_MAT_ID for hrx_moe_down_weight_supported() below), so if this probe
        // disagreed with that choice, ggml-backend's scheduler would abort with "pre-allocated tensor
        // ... that cannot run the operation (NONE)" for a weight already placed on HRX0.
        case GGML_OP_NONE:
            return hrx_weight_quant_supported(op->type) || hrx_dense_matmul_weight_supported(op->type) ||
                   hrx_moe_down_weight_supported(op);
        // Weight-consuming ops: the first source is the weight. Decline unsupported weight quantizations.
        case GGML_OP_GET_ROWS:
            return op->src[0] == nullptr ||
                   (!hrx_dispatch_group_disabled("embed") && hrx_weight_quant_supported(op->src[0]->type) &&
                    !hrx_non_leaf_gather_qwen4exp(op));
        case GGML_OP_MUL_MAT:
            if (op->src[0] == nullptr) {
                return true;
            }
            if (hrx_dispatch_group_disabled("matmul")) {
                return false;
            }
            // Q8_0 has exactly one dense matcher, so its shape guards must be replicated here to
            // avoid stranding shapes that matcher declines. See hrx_dense_matmul_q8_0_shape_supported().
            if (op->src[0]->type == GGML_TYPE_Q8_0) {
                return hrx_dense_matmul_q8_0_shape_supported(op);
            }
            return hrx_dense_matmul_weight_supported(op->src[0]->type);
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0] == nullptr) {
                return true;
            }
            if (hrx_dispatch_group_disabled("matmul")) {
                return false;
            }
            // Checked before the quantization arm below: the qwen4exp routed gate/up weights are Q4_K,
            // which hrx_weight_quant_supported() accepts, but no gate/up matcher covers qwen4exp's
            // geometry. See hrx_moe_gate_up_weight_qwen4exp().
            if (hrx_moe_gate_up_weight_qwen4exp(op->src[0])) {
                return false;
            }
            if (hrx_weight_quant_supported(op->src[0]->type)) {
                return true;
            }
            // qwen4exp routed (MoE) down-projection: 32-block quant + down-projection contraction
            // shape. See hrx_moe_down_quant_supported() above for why this is scoped to MUL_MAT_ID
            // and to this specific shape, and hrx-moe-down-kernel-spec.md §4 for the full rationale.
            // dispatch-routed-ffn.cpp registers "llm.routed_ffn.decode_down_qwen4exp", which matches
            // Q5_1/Q8_0/IQ4_NL MUL_MAT_ID nodes fitting the decode-time weighted-reduce+residual-add
            // topology. NOTE: this device_supports_op() check is still shape-only (op-level), so it
            // will also claim MUL_MAT_ID nodes of this quant/shape whose surrounding graph topology
            // does NOT match that dispatch (e.g. prefill, or an unexpected fusion) -- those nodes have
            // no other registered matcher and will hard-abort in DispatchScheduler rather than falling
            // back to CPU. This is the same general class of issue as the broader compute-op
            // over-claim in the `default: return true;` case below (RMS_NORM/ADD/ROPE/etc.), just
            // narrower in scope since it is gated to one specific weight quant+shape combination.
            return hrx_moe_down_weight_supported(op->src[0]);
        // qwen4exp GDN (Gated DeltaNet): all four of these are real compute ops with the over-claim
        // risk described in the comment above hrx_gdn_ssm_conv_decode_supported(), so each is
        // scoped to the exact qwen4exp decode-time shape profile and declined otherwise (falls back to
        // CPU, which still implements the general-shape/prefill case correctly).
        // "gdnconv" (the SSM_CONV/SILU conv-prepare prelude) and "gdncore" (the
        // recurrent GATED_DELTA_NET step itself) are separately disableable so a numerical bisection
        // can tell the two halves of the GDN pipeline apart; "gdn" disables both.
        // The prelude's feeding CONCAT stays on CPU by design -- see eager_capability_declared().
        case GGML_OP_SSM_CONV:
            return !hrx_gdn_group_disabled("gdnconv") && hrx_gdn_ssm_conv_decode_supported(op);
        // L2_NORM is never a registered dispatch root: it only ever executed as part of the fused
        // qwen4exp.gdn_conv_prepare_decode match, which deliberately no longer covers it (covering it
        // left q_conv_predelta/k_conv_predelta unwritten -- see that matcher's comment). With no
        // matcher able to claim an L2_NORM node, claiming one here would strand its split, so the
        // per-head q/k normalization runs on the CPU. These are [128,16] tensors; the cost is noise.
        case GGML_OP_L2_NORM:
            return false;
        case GGML_OP_GATED_DELTA_NET:
            return !hrx_gdn_group_disabled("gdncore") && hrx_gdn_gated_delta_net_decode_supported(op);
        case GGML_OP_UNARY:
            return !hrx_gdn_group_disabled("gdnconv") && hrx_gdn_unary_decode_supported(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return !hrx_dispatch_group_disabled("attn");
        // llama.cpp clears a recurrent state slot with ggml_scale_inplace(s, 0) and carries surviving
        // slots forward with ggml_cpy(), both writing straight into the pre-allocated recurrent state
        // cache. ggml_backend_sched aborts instead of falling back when a pre-allocated tensor sits in a
        // buffer whose backend declines its op, so these two arms are what allow the GDN state to stay
        // resident on HRX at all. Each is scoped to the forms that actually have a kernel: the zero-fill
        // scale (common.zero_f32), the same-length contiguous f32 copy (common.copy_f32), and the 2D
        // row-strided f32 copy (common.copy_rows_f32) that the GDN conv-state writeback needs. Any other
        // scale factor, or a type-converting or permuted copy, is declined and runs on the CPU.
        case GGML_OP_SCALE:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
                   ggml_get_op_params_f32(op, 0) == 0.0f && ggml_get_op_params_f32(op, 1) == 0.0f;
        case GGML_OP_CPY:
        // GGML_OP_CONT is the same copy with an implicit contiguous destination. qwen4exp's GDN
        // conv-state writeback is ggml_cpy(ggml_cont(tail), dst), and that CONT sits between the CONCAT
        // and the SSM_CONV, so leaving it on the CPU splits the conv-prepare chain across backends and
        // strands the CONCAT in a split of its own where no matcher can root at it.
        case GGML_OP_CONT:
            return hrx_copy_f32_supported(op);
        // See hrx_gdn_per_head_scalar_decode_op(): qwen4exp's GDN alpha/gate prelude emits a 48-element
        // ADD and MUL that no matcher roots at. Everything else keeps the permissive default.
        case GGML_OP_ADD:
            return !hrx_gdn_per_head_scalar_decode_op(op) && !hrx_qsa_indexer_row_qwen4exp(op);
        case GGML_OP_MUL:
            return !hrx_gdn_norm_gate_row(op) && !hrx_gdn_per_head_scalar_decode_op(op) &&
                   !hrx_stream_major_mul_qwen4exp(op) &&
                   !hrx_ple_signed_sqrt_gate_qwen4exp(op) && !hrx_hc_dim_activation_mul_qwen4exp(op) &&
                   !hrx_attention_head_major_qwen4exp(op) && !hrx_qsa_indexer_row_qwen4exp(op) &&
                   !hrx_hc_grouped_norm_row_qwen4exp(op);
        case GGML_OP_RMS_NORM:
            return !hrx_gdn_norm_gate_row(op) && !hrx_attention_head_major_qwen4exp(op) &&
                   !hrx_qsa_indexer_row_qwen4exp(op) && !hrx_hc_grouped_norm_row_qwen4exp(op);
        // VIEW is deliberately absent from these shape guards. It is a pure layout alias that the
        // dispatch scheduler elides rather than dispatching, so HRX can always "run" one -- declining it
        // states a capability HRX does have. It also cannot be declined safely: the KV and QSA caches are
        // pre-allocated in HRX buffers, and ggml_backend_sched aborts instead of falling back when a
        // pre-allocated tensor's backend refuses its op, so refusing a view of cache_k_l* is fatal.
        case GGML_OP_ROPE:
            return !hrx_attention_head_major_qwen4exp(op) && !hrx_qsa_indexer_row_qwen4exp(op) &&
                   !hrx_hc_grouped_norm_row_qwen4exp(op);
        case GGML_OP_CLAMP:
            return !hrx_ple_signed_sqrt_gate_qwen4exp(op);
        case GGML_OP_SUM_ROWS:
            return !hrx_ple_stream_reduce_qwen4exp(op);
        case GGML_OP_SET_ROWS:
            return !hrx_cache_publish_unmatched_qwen4exp(op);
        case GGML_OP_GLU:
            return !hrx_moe_glu_qwen4exp_decode(op);
        default:
            return true;
    }
}

static bool device_supports_buffer_type(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    auto * context = device_context(device);
    return buft == &context->buft || buft == &context->host_buft || ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i device_i = {
    device_name,
    device_description,
    device_memory,
    device_type,
    device_props,
    device_init,
    device_buffer_type,
    device_host_buffer_type,
    nullptr,
    device_supports_op,
    device_supports_buffer_type,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

static const char * registry_name(ggml_backend_reg_t registry) {
    GGML_UNUSED(registry);
    return "HRX";
}

static size_t registry_device_count(ggml_backend_reg_t registry) {
    return static_cast<ggml_backend_hrx_reg_context *>(registry->context)->devices.size();
}

static ggml_backend_dev_t registry_device(ggml_backend_reg_t registry, size_t index) {
    auto * context = static_cast<ggml_backend_hrx_reg_context *>(registry->context);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * registry_proc(ggml_backend_reg_t registry, const char * name) {
    GGML_UNUSED(registry);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i registry_i = { registry_name, registry_device_count, registry_device, registry_proc };

static std::unique_ptr<ggml_backend_hrx_reg_context> create_registry_context() {
    auto         context = std::make_unique<ggml_backend_hrx_reg_context>();
    hrx_status_t status  = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        hrx_status_ignore(status);
        return context;
    }
    int count = 0;
    if (!HRX_CHECK(hrx_gpu_device_count(&count))) {
        return context;
    }
    context->device_contexts.reserve(count);
    context->devices.reserve(count);
    for (int i = 0; i < count; ++i) {
        hrx_device_t hrx_device = nullptr;
        if (!HRX_CHECK(hrx_gpu_device_get(i, &hrx_device)) || hrx_device == nullptr) {
            continue;
        }
        hrx_device_retain(hrx_device);
        auto device_ctx                      = std::make_unique<ggml_backend_hrx_device_context>();
        device_ctx->device                   = hrx_device;
        device_ctx->name                     = "HRX" + std::to_string(i);
        device_ctx->use_direct_host_bindings = environment_flag_enabled("GGML_HRX_USE_UNIFIED_MEMORY");
        if (device_ctx->use_direct_host_bindings) {
            GGML_LOG_INFO("ggml_hrx: direct coherent host bindings enabled by GGML_HRX_USE_UNIFIED_MEMORY\n");
        }
        const std::optional<std::string> name =
            device_string_property(hrx_device, HRX_DEVICE_PROPERTY_NAME, "query HRX device name");
        const std::optional<std::string> architecture =
            device_string_property(hrx_device, HRX_DEVICE_PROPERTY_ARCHITECTURE, "query HRX device architecture");
        if (!name || !architecture) {
            hrx_device_release(hrx_device);
            continue;
        }
        uint64_t memory = 0;
        if (!HRX_CHECK(
                hrx_device_get_property(hrx_device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &memory, sizeof(memory)))) {
            hrx_device_release(hrx_device);
            continue;
        }
        if (!HRX_CHECK(hrx_stream_create(hrx_device, 0, &device_ctx->buffer_stream))) {
            hrx_device_release(hrx_device);
            continue;
        }
        device_ctx->memory_total      = static_cast<size_t>(memory);
        device_ctx->description       = *name + " (" + *architecture + ")";
        device_ctx->architecture      = *architecture;
        device_ctx->buft_context      = { device_ctx.get(), device_ctx->name, false };
        device_ctx->buft              = { buffer_type_i, nullptr, &device_ctx->buft_context };
        device_ctx->host_buft_context = { device_ctx.get(), device_ctx->name + "_HOST", true };
        device_ctx->host_buft         = { buffer_type_i, nullptr, &device_ctx->host_buft_context };
        context->device_contexts.emplace_back(std::move(device_ctx));
        context->devices.push_back({ device_i, nullptr, context->device_contexts.back().get() });
        context->device_contexts.back()->buft.device      = &context->devices.back();
        context->device_contexts.back()->host_buft.device = &context->devices.back();
    }
    return context;
}

}  // namespace

ggml_backend_reg_t ggml_backend_hrx_reg() {
    static std::unique_ptr<ggml_backend_hrx_reg_context> context = create_registry_context();
    static ggml_backend_reg registry = { GGML_BACKEND_API_VERSION, registry_i, context.get() };
    for (auto & device : context->devices) {
        device.reg = &registry;
    }
    return &registry;
}

ggml_backend_t ggml_backend_hrx_init(size_t device) {
    ggml_backend_reg_t registry = ggml_backend_hrx_reg();
    if (device >= ggml_backend_reg_dev_count(registry)) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, device), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

bool ggml_backend_hrx_get_cache_stats(ggml_backend_t backend, ggml_backend_hrx_cache_stats * stats) {
    if (!ggml_backend_is_hrx(backend) || stats == nullptr) {
        return false;
    }
    auto *                                  context     = static_cast<ggml_backend_hrx_context *>(backend->context);
    const ggml::hrx::GraphProgramCacheStats graph_stats = context->graph_programs.stats();
    const ggml::hrx::PreparedCommandProgramCacheStats prepared_stats = context->prepared_programs.stats();
    stats->graph_program_builds                                      = graph_stats.builds;
    stats->graph_program_hits                                        = graph_stats.hits;
    stats->prepared_program_builds = graph_stats.prepared_program_builds + prepared_stats.builds;
    stats->prepared_program_hits   = graph_stats.prepared_program_hits + prepared_stats.hits;
    return true;
}

int ggml_backend_hrx_get_device_count() {
    return static_cast<int>(ggml_backend_reg_dev_count(ggml_backend_hrx_reg()));
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t device) {
    ggml_backend_reg_t registry = ggml_backend_hrx_reg();
    return device < ggml_backend_reg_dev_count(registry) ?
               ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(registry, device)) :
               nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)
