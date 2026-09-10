#include "backend-buffer-binding.h"
#include "backend-context.h"
#include "dispatch_registration/dispatch-copy.h"
#include "dispatch_registration/dispatch-gather-add.h"
#include "dispatch_registration/dispatch-qwen4exp-flash-attention.h"
#include "dispatch/dispatch-scheduler.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "hrx-interop-utils.h"
#include "kernel-corpus/kernel-corpus-catalog.h"
#include "runtime/host-memory.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static void require_hrx_status(hrx_status_t status) {
    if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(status)) {
        std::fprintf(stderr, "HRX status failed: %s\n", error->c_str());
        std::abort();
    }
}

static ggml_backend_hrx_context * backend_context(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    REQUIRE(context != nullptr);
    REQUIRE(context->device != nullptr);
    REQUIRE(context->device->device != nullptr);
    REQUIRE(context->stream != nullptr);
    return context;
}

static void run_backend_buffer_checks(ggml_backend_t backend) {
    ggml_backend_hrx_context * hrx = backend_context(backend);

    ggml_init_params params = {};
    params.mem_size         = 16 * 1024;
    params.no_alloc         = true;
    ggml_context * context  = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor *         tensor      = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_tensor *         copy        = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_backend_buffer_t buffer      = ggml_backend_alloc_buffer(backend, 4096);
    ggml_backend_buffer_t copy_buffer = ggml_backend_alloc_buffer(backend, 4096);
    REQUIRE(buffer != nullptr);
    REQUIRE(copy_buffer != nullptr);
    tensor->buffer = buffer;
    tensor->data   = ggml_backend_buffer_get_base(buffer);
    copy->buffer   = copy_buffer;
    copy->data     = ggml_backend_buffer_get_base(copy_buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(buffer, tensor) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_buffer_init_tensor(copy_buffer, copy) == GGML_STATUS_SUCCESS);

    std::array<uint32_t, 64> input = {};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint32_t>(i * 17 + 3);
    }
    ggml_backend_tensor_set(tensor, input.data(), 0, sizeof(input));
    std::array<uint32_t, 64> output = {};
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    REQUIRE(output == input);
    ggml_backend_tensor_copy(tensor, copy);
    output.fill(0);
    ggml_backend_tensor_get(copy, output.data(), 0, sizeof(output));
    REQUIRE(output == input);

    input[0] = 0x12345678;
    ggml_backend_tensor_set_async(backend, tensor, input.data(), 0, sizeof(input));
    ggml_backend_synchronize(backend);
    output.fill(0);
    ggml_backend_tensor_get_async(backend, tensor, output.data(), 0, sizeof(output));
    ggml_backend_synchronize(backend);
    REQUIRE(output == input);
    REQUIRE(hrx->device->synchronous_upload_fallbacks.load(std::memory_order_relaxed) == 1);
    REQUIRE(hrx->device->synchronous_download_fallbacks.load(std::memory_order_relaxed) == 1);

    ggml_backend_tensor_memset(tensor, 0x5a, 16, 32);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(output.data());
    for (size_t i = 16; i < 48; ++i) {
        REQUIRE(bytes[i] == 0x5a);
    }

    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    for (uint32_t value : output) {
        REQUIRE(value == 0);
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(copy_buffer);
    ggml_free(context);
    ggml_backend_synchronize(backend);
    REQUIRE(hrx->device->device != nullptr);
}

static void run_host_buffer_checks(ggml_backend_t backend) {
    ggml_backend_hrx_context * context = backend_context(backend);
    ggml_backend_buffer_type_t buft    = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    REQUIRE(buft != nullptr);
    REQUIRE(ggml_backend_buft_is_host(buft));

    const bool original_direct_host_bindings = context->device->use_direct_host_bindings;
    context->device->use_direct_host_bindings = false;
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 4096);
    context->device->use_direct_host_bindings = original_direct_host_bindings;
    REQUIRE(buffer != nullptr);
    REQUIRE(ggml_backend_buffer_is_host(buffer));
    auto * buffer_context = ggml_backend_hrx_buffer_context_from_buffer(buffer);
    REQUIRE(buffer_context != nullptr);
    REQUIRE(buffer_context->buffer != nullptr);
    REQUIRE(buffer_context->base == ggml_backend_buffer_get_base(buffer));
    REQUIRE(!buffer_context->direct_host_binding);

    const uint32_t pattern = 0x12345678;
    require_hrx_status(
        hrx_stream_fill_buffer(context->stream, buffer_context->buffer, 0, 4096, &pattern, sizeof(pattern)));
    require_hrx_status(hrx_stream_synchronize(context->stream));
    const auto * words = static_cast<const uint32_t *>(ggml_backend_buffer_get_base(buffer));
    for (size_t i = 0; i < 4096 / sizeof(uint32_t); ++i) {
        REQUIRE(words[i] == pattern);
    }

    ggml_init_params params = {};
    params.mem_size         = 4096;
    params.no_alloc         = true;
    ggml_context * ggml     = ggml_init(params);
    REQUIRE(ggml != nullptr);
    ggml_tensor * host_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 64);
    host_tensor->buffer       = buffer;
    host_tensor->data         = ggml_backend_buffer_get_base(buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(buffer, host_tensor) == GGML_STATUS_SUCCESS);
    ggml::hrx::ValueBufferBinding staged_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(host_tensor, staged_binding));
    REQUIRE(staged_binding.buffer == nullptr);
    REQUIRE(staged_binding.host_data == ggml_backend_buffer_get_base(buffer));
    REQUIRE(staged_binding.offset == 0);
    REQUIRE(staged_binding.length == ggml_nbytes(host_tensor));

    context->device->use_direct_host_bindings = true;
    ggml_backend_buffer_t direct_buffer = ggml_backend_buft_alloc_buffer(buft, 4096);
    context->device->use_direct_host_bindings = original_direct_host_bindings;
    REQUIRE(direct_buffer != nullptr);
    auto * direct_buffer_context = ggml_backend_hrx_buffer_context_from_buffer(direct_buffer);
    REQUIRE(direct_buffer_context->direct_host_binding);
    ggml_tensor * direct_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 64);
    direct_tensor->buffer       = direct_buffer;
    direct_tensor->data         = ggml_backend_buffer_get_base(direct_buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(direct_buffer, direct_tensor) == GGML_STATUS_SUCCESS);
    ggml::hrx::ValueBufferBinding direct_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(direct_tensor, direct_binding));
    REQUIRE(direct_binding.buffer == direct_buffer_context->buffer);
    REQUIRE(direct_binding.host_data == nullptr);
    REQUIRE(direct_binding.offset == 0);
    REQUIRE(direct_binding.length == ggml_nbytes(direct_tensor));

    ggml_tensor *         tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 64);
    ggml_backend_buffer_t local  = ggml_backend_alloc_buffer(backend, 4096);
    REQUIRE(local != nullptr);
    tensor->buffer = local;
    tensor->data   = ggml_backend_buffer_get_base(local);
    REQUIRE(ggml_backend_buffer_init_tensor(local, tensor) == GGML_STATUS_SUCCESS);

    const uint64_t upload_fallbacks =
        context->device->synchronous_upload_fallbacks.load(std::memory_order_relaxed);
    const uint64_t download_fallbacks =
        context->device->synchronous_download_fallbacks.load(std::memory_order_relaxed);
    auto * host_words = static_cast<uint32_t *>(ggml_backend_buffer_get_base(buffer));
    for (size_t i = 0; i < 64; ++i) {
        host_words[i] = static_cast<uint32_t>(i * 13 + 7);
    }
    ggml_backend_tensor_set_async(backend, tensor, host_words, 0, 64 * sizeof(uint32_t));
    ggml_backend_synchronize(backend);
    std::memset(host_words, 0, 64 * sizeof(uint32_t));
    ggml_backend_tensor_get_async(backend, tensor, host_words, 0, 64 * sizeof(uint32_t));
    ggml_backend_synchronize(backend);
    for (size_t i = 0; i < 64; ++i) {
        REQUIRE(host_words[i] == static_cast<uint32_t>(i * 13 + 7));
    }
    REQUIRE(context->device->synchronous_upload_fallbacks.load(std::memory_order_relaxed) == upload_fallbacks);
    REQUIRE(context->device->synchronous_download_fallbacks.load(std::memory_order_relaxed) == download_fallbacks);

    ggml_backend_buffer_free(local);
    ggml_backend_buffer_free(direct_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(ggml);
}

static void run_host_transfer_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostStagingBuffer   staging;
    REQUIRE(ggml::hrx::allocate_host_staging_buffer(context->device->device, 64, staging).success());

    const std::array<uint8_t, 64> zero = {};
    require_hrx_status(hrx_synchronous_h2d(context->device->device, zero.data(), staging.buffer, 0, zero.size()));

    std::array<uint8_t, 64> host = {};
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>(i + 1);
    }

    REQUIRE(transfers.upload_synchronous(context->stream, host.data(), staging.buffer, 0, 0).success());
    REQUIRE(transfers.upload_synchronous(context->stream, host.data() + 8, staging.buffer, 16, 24).success());
    ggml::hrx::HostTransferStats stats = transfers.stats();
    REQUIRE(stats.uploads == 1);
    REQUIRE(stats.upload_bytes == 24);
    require_hrx_status(hrx_stream_synchronize(context->stream));

    std::array<uint8_t, 64> upload_result = {};
    require_hrx_status(
        hrx_synchronous_d2h(context->device->device, staging.buffer, 0, upload_result.data(), upload_result.size()));
    for (size_t i = 0; i < upload_result.size(); ++i) {
        const uint8_t expected = i >= 16 && i < 40 ? host[i - 8] : 0;
        REQUIRE(upload_result[i] == expected);
    }

    std::array<uint8_t, 64> device_values = {};
    for (size_t i = 0; i < device_values.size(); ++i) {
        device_values[i] = static_cast<uint8_t>(0xa0 + i);
    }
    require_hrx_status(
        hrx_synchronous_h2d(context->device->device, device_values.data(), staging.buffer, 0, device_values.size()));

    std::array<uint8_t, 48> download_result = {};
    REQUIRE(transfers.download_synchronous(
        context->stream, staging.buffer, 12, download_result.data() + 4, 20).success());
    stats = transfers.stats();
    REQUIRE(stats.downloads == 1);
    REQUIRE(stats.download_bytes == 20);
    require_hrx_status(hrx_stream_synchronize(context->stream));
    for (size_t i = 0; i < download_result.size(); ++i) {
        const uint8_t expected = i >= 4 && i < 24 ? device_values[i + 8] : 0;
        REQUIRE(download_result[i] == expected);
    }

    REQUIRE(!transfers.upload_synchronous(nullptr, host.data(), staging.buffer, 0, 4).success());
    REQUIRE(!transfers.upload_synchronous(context->stream, nullptr, staging.buffer, 0, 4).success());
    REQUIRE(!transfers.upload_synchronous(context->stream, host.data(), nullptr, 0, 4).success());
    REQUIRE(!transfers.download_synchronous(nullptr, staging.buffer, 0, download_result.data(), 4).success());
    REQUIRE(!transfers.download_synchronous(context->stream, nullptr, 0, download_result.data(), 4).success());
    REQUIRE(!transfers.download_synchronous(context->stream, staging.buffer, 0, nullptr, 4).success());

    transfers.clear();
    stats = transfers.stats();
    REQUIRE(stats.uploads == 0);
    REQUIRE(stats.downloads == 0);
    REQUIRE(stats.upload_bytes == 0);
    REQUIRE(stats.download_bytes == 0);
}

static void run_host_staging_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostStagingBuffer staging;
    REQUIRE(ggml::hrx::allocate_host_staging_buffer(context->device->device, 32, staging).success());
    REQUIRE(staging.buffer != nullptr);
    REQUIRE(staging.length == 32);

    hrx_buffer_t                 original = staging.buffer;
    ggml::hrx::HostStagingBuffer moved(std::move(staging));
    REQUIRE(moved.buffer == original);
    REQUIRE(moved.length == 32);
    REQUIRE(staging.buffer == nullptr);
    REQUIRE(staging.length == 0);

    ggml::hrx::HostStagingBuffer assigned;
    assigned = std::move(moved);
    REQUIRE(assigned.buffer == original);
    REQUIRE(assigned.length == 32);
    REQUIRE(moved.buffer == nullptr);
    REQUIRE(moved.length == 0);

    assigned.clear();
    REQUIRE(assigned.buffer == nullptr);
    REQUIRE(assigned.length == 0);
    assigned.clear();
    REQUIRE(assigned.buffer == nullptr);
}

static void run_host_weight_cache_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostWeightCache     weights;

    std::array<uint8_t, 128> host = {};
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>(i);
    }

    ggml::hrx::HostWeightSource source;
    source.host_data  = host.data();
    source.identity   = 0x1234;
    source.generation = 1;
    source.capacity   = host.size();
    source.offset     = 16;
    source.length     = 32;
    source.layout     = "ggml-native";

    ggml::hrx::HostWeightAcquireResult first =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(first.valid());

    std::array<uint8_t, 32> first_bytes = {};
    require_hrx_status(
        hrx_synchronous_d2h(context->device->device, first.lease.buffer(), 0, first_bytes.data(), first_bytes.size()));
    for (size_t i = 0; i < first_bytes.size(); ++i) {
        REQUIRE(first_bytes[i] == host[source.offset + i]);
    }

    ggml::hrx::HostWeightAcquireResult second =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(second.valid());
    REQUIRE(second.lease.buffer() == first.lease.buffer());

    source.offset = 32;
    ggml::hrx::HostWeightAcquireResult slice =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(slice.valid());
    REQUIRE(slice.lease.buffer() != first.lease.buffer());

    source.offset     = 16;
    source.generation = 2;
    ggml::hrx::HostWeightAcquireResult next_generation =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(next_generation.valid());
    REQUIRE(next_generation.lease.buffer() != first.lease.buffer());

    source.generation = 1;
    source.layout     = "alternate-layout";
    ggml::hrx::HostWeightAcquireResult conflict =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(!conflict.valid());

    ggml::hrx::HostWeightCacheStats weight_stats = weights.stats();
    REQUIRE(weight_stats.hits == 1);
    REQUIRE(weight_stats.misses == 3);
    REQUIRE(weight_stats.layout_conflicts == 1);
    REQUIRE(weight_stats.allocation_count == 3);
    REQUIRE(weight_stats.resident_bytes == 96);

    const ggml::hrx::HostTransferStats transfer_stats = transfers.stats();
    REQUIRE(transfer_stats.uploads == 3);
    REQUIRE(transfer_stats.upload_bytes == 96);

    weights.clear();
    weight_stats = weights.stats();
    REQUIRE(weight_stats.hits == 0);
    REQUIRE(weight_stats.misses == 0);
    REQUIRE(weight_stats.layout_conflicts == 0);
    REQUIRE(weight_stats.allocation_count == 0);
    REQUIRE(weight_stats.resident_bytes == 0);
}

static void run_uniform_outer_copy_checks(ggml_backend_t backend, int64_t heads, int64_t tokens, int64_t batches) {
    using namespace ggml::hrx;
    const int64_t rows = heads * tokens * batches;
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    auto * context = ggml_init(params);
    REQUIRE(context != nullptr);
    auto * projection = ggml_new_tensor_4d(context, GGML_TYPE_F32, 512, heads, tokens, batches);
    auto * query_view = ggml_view_4d(context, projection, 256, heads, tokens, batches,
                                    512 * sizeof(float), 512 * heads * sizeof(float),
                                    512 * heads * tokens * sizeof(float), 0);
    auto * gate_view = ggml_view_4d(context, projection, 256, heads, tokens, batches,
                                   512 * sizeof(float), 512 * heads * sizeof(float),
                                   512 * heads * tokens * sizeof(float), 256 * sizeof(float));
    auto * query = ggml_cont(context, query_view);
    auto * gate_flat = ggml_cont_2d(context, gate_view, 256 * heads, tokens * batches);
    auto * destination = ggml_new_tensor_1d(context, GGML_TYPE_F32, 256 * rows);
    auto * gate_copy = ggml_cpy(context, gate_view, destination);
    auto * padded = ggml_new_tensor_4d(context, GGML_TYPE_F32, 272, heads, tokens, batches);
    auto * padded_view = ggml_view_4d(context, padded, 256, heads, tokens, batches,
                                     272 * sizeof(float), 272 * heads * sizeof(float),
                                     272 * heads * tokens * sizeof(float), 0);
    auto * padded_copy = ggml_cpy(context, gate_view, padded_view);
    auto * graph = ggml_new_graph(context);
    for (auto * op : { query, gate_flat, gate_copy, padded_copy }) {
        REQUIRE(supports_copy_f32_dispatch(op));
        REQUIRE(ggml_backend_supports_op(backend, op));
        CopyF32Geometry geometry;
        REQUIRE(copy_f32_geometry(*op->src[0], *op, geometry));
        REQUIRE(!geometry.contiguous && geometry.row_length == 256 && geometry.row_count == rows);
        REQUIRE(geometry.element_count == 256 * rows && geometry.source_row_stride == 512);
        REQUIRE(geometry.output_row_stride == (op == padded_copy ? 272 : 256));
        ggml_build_forward_expand(graph, op);
    }
    auto imported = import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    for (auto * op : { query, gate_flat, gate_copy, padded_copy }) {
        const auto * source_value = imported.graph.values().find_tensor(op->src[0]);
        const auto * output_value = imported.graph.values().find_tensor(op);
        REQUIRE(source_value != nullptr && output_value != nullptr);
        CopyF32Geometry geometry;
        REQUIRE(copy_f32_geometry(*source_value, *output_value, geometry));
        REQUIRE(geometry.row_count == rows && geometry.source_row_stride == 512);
        REQUIRE(geometry.output_row_stride == (op == padded_copy ? 272 : 256));
        auto too_short = *source_value;
        too_short.byte_count = static_cast<size_t>((rows - 1) * 512 + 256) * sizeof(float) - 1;
        REQUIRE(!copy_f32_geometry(too_short, *output_value, geometry));
        too_short = *output_value;
        too_short.byte_count = static_cast<size_t>((rows - 1) * (op == padded_copy ? 272 : 256) + 256) *
                              sizeof(float) - 1;
        REQUIRE(!copy_f32_geometry(*source_value, too_short, geometry));
    }
    DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));

    auto * buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    REQUIRE(buffer != nullptr);
    std::vector<float> input(static_cast<size_t>(512 * rows));
    std::vector<float> expected_query(static_cast<size_t>(256 * rows)), expected_gate(expected_query.size());
    std::vector<float> expected_padded(static_cast<size_t>(272 * rows), -99999.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) - 20000.0f;
    }
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < 256; ++col) {
            expected_query[row * 256 + col] = input[row * 512 + col];
            expected_gate[row * 256 + col] = input[row * 512 + 256 + col];
            expected_padded[row * 272 + col] = expected_gate[row * 256 + col];
        }
    }
    std::vector<float> initial_padding(expected_padded.size(), -99999.0f);
    ggml_backend_tensor_set(projection, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(padded, initial_padding.data(), 0, initial_padding.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    for (auto * result : { query, gate_flat, gate_copy, padded }) {
        const auto & expected = result == query ? expected_query :
                                result == padded ? expected_padded : expected_gate;
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
        REQUIRE(actual == expected);
    }
    ggml_backend_buffer_free(buffer);

    const auto decline = [&](ggml_tensor * op) {
        REQUIRE(!supports_copy_f32_dispatch(op));
        REQUIRE(!ggml_backend_supports_op(backend, op));
    };
    // Enough backing storage for valid ggml views, but a gap between planes/batches
    // cannot be addressed by the uniform-row kernel.
    auto * gapped_storage = ggml_new_tensor_1d(context, GGML_TYPE_F32, 512 * 25 * 3 * 3);
    auto * plane_gap = ggml_view_3d(context, gapped_storage, 256, 24, 3,
                                   512 * sizeof(float), 512 * 25 * sizeof(float), 256 * sizeof(float));
    decline(ggml_cont(context, plane_gap));
    auto * batch_gap = ggml_view_4d(context, gapped_storage, 256, 4, 3, 2,
                                   512 * sizeof(float), 512 * 4 * sizeof(float),
                                   512 * 13 * sizeof(float), 256 * sizeof(float));
    decline(ggml_cont(context, batch_gap));
    decline(ggml_cont(context, ggml_transpose(context, query_view)));
    auto * wrong_rows = ggml_view_2d(context, projection, 128, 2 * rows, 256 * sizeof(float), 0);
    decline(ggml_cpy(context, gate_view, wrong_rows));
    auto * gap_destination = ggml_view_3d(context, gapped_storage, 256, heads, tokens * batches,
                                         512 * sizeof(float), 512 * (heads + 1) * sizeof(float), 0);
    if (tokens * batches > 1) {
        decline(ggml_cpy(context, gate_view, gap_destination));
    }
    auto * f16_destination = ggml_new_tensor_1d(context, GGML_TYPE_F16, 256 * rows);
    decline(ggml_cpy(context, gate_view, f16_destination));
    auto * too_large = ggml_new_tensor_1d(context, GGML_TYPE_F32, kCopyF32MaxElements + 1);
    decline(ggml_cont(context, too_large));
    ggml_free(context);
    std::fprintf(stderr, "uniform-row COPY [%lld,%lld,%lld] checks passed\n",
                 (long long) heads, (long long) tokens, (long long) batches);
}

static void run_strided_flatten_copy_checks(ggml_backend_t backend) {
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor * projection = ggml_new_tensor_2d(context, GGML_TYPE_F32, 512, 24);
    ggml_tensor * gate = ggml_view_3d(context, projection, 256, 24, 1,
                                    512 * sizeof(float), 512 * 24 * sizeof(float), 256 * sizeof(float));
    ggml_tensor * flat = ggml_cont_2d(context, gate, 6144, 1);
    ggml_tensor * destination = ggml_new_tensor_1d(context, GGML_TYPE_F32, 6144);
    ggml_tensor * copy = ggml_cpy(context, gate, destination);
    REQUIRE(ggml_backend_supports_op(backend, flat));
    REQUIRE(ggml_backend_supports_op(backend, copy));
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, flat);
    ggml_build_forward_expand(graph, copy);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    REQUIRE(buffer != nullptr);
    std::vector<float> input(512 * 24);
    std::vector<float> expected(6144);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) - 6000.0f;
    }
    for (size_t row = 0; row < 24; ++row) {
        for (size_t col = 0; col < 256; ++col) {
            expected[row * 256 + col] = input[row * 512 + 256 + col];
        }
    }
    ggml_backend_tensor_set(projection, input.data(), 0, input.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(expected.size());
    for (ggml_tensor * result : {flat, copy}) {
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
        REQUIRE(actual == expected);
    }
    ggml_backend_buffer_free(buffer);

    ggml_tensor * other_rows = ggml_view_2d(context, projection, 128, 48, 256 * sizeof(float), 0);
    REQUIRE(!ggml_backend_supports_op(backend, ggml_cpy(context, gate, other_rows)));
    ggml_tensor * transposed = ggml_transpose(context, projection);
    REQUIRE(!ggml_backend_supports_op(backend, ggml_cont(context, transposed)));
    ggml_tensor * large = ggml_new_tensor_1d(context, GGML_TYPE_F32, ggml::hrx::kCopyF32MaxElements + 1);
    REQUIRE(!ggml_backend_supports_op(backend, ggml_cont(context, large)));
    ggml_free(context);
    run_uniform_outer_copy_checks(backend, 24, 3, 1);
    run_uniform_outer_copy_checks(backend, 4, 3, 2);
}

static void run_shutdown_checks() {
    REQUIRE(ggml_backend_hrx_shutdown());
    auto * reg = ggml_backend_hrx_reg();
    const int count = ggml_backend_hrx_get_device_count();
    REQUIRE(ggml_backend_hrx_shutdown());
    REQUIRE(ggml_backend_hrx_shutdown());
    if (count == 0) {
        std::fprintf(stderr, "shutdown device checks skipped: no HRX devices\n");
        return;
    }
    auto * dev = ggml_backend_reg_dev_get(reg, 0);
    auto * buft = ggml_backend_dev_buffer_type(dev);
    auto shutdown = reinterpret_cast<ggml_backend_reg_shutdown_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_reg_shutdown"));
    REQUIRE(shutdown != nullptr);
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    auto * second_backend = ggml_backend_dev_init(dev, nullptr);
    REQUIRE(backend != nullptr && second_backend != nullptr);
    REQUIRE(!shutdown(reg));
    auto * buffer = ggml_backend_buft_alloc_buffer(buft, 256);
    REQUIRE(buffer != nullptr);
    ggml_backend_free(backend);
    REQUIRE(!shutdown(reg));
    ggml_backend_free(second_backend);
    REQUIRE(!shutdown(reg));
    ggml_backend_buffer_free(buffer);
    REQUIRE(shutdown(reg));
    REQUIRE(shutdown(reg));
    REQUIRE(ggml_backend_hrx_get_device_count() == count);

    buffer = ggml_backend_buft_alloc_buffer(buft, 256);
    REQUIRE(buffer != nullptr);
    REQUIRE(!shutdown(reg));
    ggml_backend_buffer_free(buffer);
    REQUIRE(shutdown(reg));
}

static void set_rows_flag(const char * value) {
#ifdef _WIN32
    REQUIRE(_putenv_s("HRX_ENABLE_SET_ROWS", value == nullptr ? "" : value) == 0);
#else
    REQUIRE((value == nullptr ? unsetenv("HRX_ENABLE_SET_ROWS") : setenv("HRX_ENABLE_SET_ROWS", value, 1)) == 0);
#endif
}

static void run_set_rows_numeric(ggml_backend_t backend, int64_t width, ggml_type output_type, ggml_type index_type,
                                 bool strided) {
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    REQUIRE(context != nullptr);
    const int64_t rows = 3;
    const int64_t cache_rows = 7;
    const size_t output_size = ggml_type_size(output_type);
    const size_t index_size = ggml_type_size(index_type);
    const int64_t source_stride = width + (strided ? 3 : 0);
    const int64_t output_stride = width + (strided ? 5 : 0);
    const int64_t index_stride = strided ? 2 : 1;
    const size_t source_offset = strided ? 1 : 0;
    const size_t output_offset = strided ? 2 : 0;
    const size_t index_offset = strided ? 1 : 0;
    ggml_tensor * source_storage = ggml_new_tensor_2d(context, GGML_TYPE_F32, source_stride, rows + 1);
    ggml_tensor * destination_storage = ggml_new_tensor_2d(context, output_type, output_stride, cache_rows + 1);
    ggml_tensor * index_storage = ggml_new_tensor_1d(context, index_type, index_stride * rows + 1);
    ggml_tensor * source = ggml_view_2d(context, source_storage, width, rows,
                                       source_stride * sizeof(float), source_offset * sizeof(float));
    ggml_tensor * destination = ggml_view_2d(context, destination_storage, width, cache_rows,
                                            output_stride * output_size, output_offset * output_size);
    ggml_tensor * ids = ggml_transpose(context, ggml_view_2d(context, index_storage, 1, rows,
                                      index_stride * index_size, index_offset * index_size));
    ggml_tensor * write = ggml_set_rows(context, destination, source, ids);
    REQUIRE(ggml::hrx::supports_set_rows_dispatch(write));
    REQUIRE(ggml_backend_supports_op(backend, write));
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, write);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    REQUIRE(buffer != nullptr);
    REQUIRE(write->data == destination->data);

    std::vector<float> input(ggml_nelements(source_storage));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = (static_cast<int>(i % 257) - 128) * 0.03127f;
    }
    ggml_backend_tensor_set(source_storage, input.data(), 0, input.size() * sizeof(float));
    const float sentinel = -71.5f;
    std::vector<uint8_t> initial(ggml_nbytes(destination_storage));
    for (size_t i = 0; i < initial.size(); i += output_size) {
        if (output_type == GGML_TYPE_F16) {
            const ggml_fp16_t value = ggml_fp32_to_fp16(sentinel);
            std::memcpy(initial.data() + i, &value, sizeof(value));
        } else {
            std::memcpy(initial.data() + i, &sentinel, sizeof(sentinel));
        }
    }
    const auto upload_ids = [&](const std::array<int64_t, 3> & values) {
        std::vector<uint8_t> bytes(ggml_nbytes(index_storage), 0xff);
        for (size_t row = 0; row < values.size(); ++row) {
            const size_t offset = (index_offset + row * index_stride) * index_size;
            if (index_type == GGML_TYPE_I64) {
                std::memcpy(bytes.data() + offset, &values[row], sizeof(int64_t));
            } else {
                const int32_t value = static_cast<int32_t>(values[row]);
                std::memcpy(bytes.data() + offset, &value, sizeof(value));
            }
        }
        ggml_backend_tensor_set(index_storage, bytes.data(), 0, bytes.size());
    };
    const std::array<int64_t, 3> valid_ids = { cache_rows - 1, 0, 3 };
    std::vector<uint8_t> expected = initial;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < width; ++col) {
            const float value = input[source_offset + row * source_stride + col];
            const size_t offset = (output_offset + valid_ids[row] * output_stride + col) * output_size;
            if (output_type == GGML_TYPE_F16) {
                const ggml_fp16_t half = ggml_fp32_to_fp16(value);
                std::memcpy(expected.data() + offset, &half, sizeof(half));
            } else {
                std::memcpy(expected.data() + offset, &value, sizeof(value));
            }
        }
    }
    std::vector<uint8_t> actual(initial.size());
    // Reuse the cached program, changing IDs between successful and failing invocations.
    for (int pass = 0; pass < 2; ++pass) {
        ggml_backend_tensor_set(destination_storage, initial.data(), 0, initial.size());
        upload_ids(valid_ids);
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(destination_storage, actual.data(), 0, actual.size());
        REQUIRE(actual == expected);
        std::vector<int64_t> invalid_ids = { -1, cache_rows, valid_ids[0] };
        if (index_type == GGML_TYPE_I64) {
            invalid_ids.push_back(int64_t{1} << 32);
            invalid_ids.push_back(std::numeric_limits<int64_t>::max());
            invalid_ids.push_back(std::numeric_limits<int64_t>::min());
        } else {
            invalid_ids.push_back(std::numeric_limits<int32_t>::max());
        }
        for (const int64_t bad : invalid_ids) {
            auto invalid = valid_ids;
            invalid[2] = bad;
            upload_ids(invalid);
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
            ggml_backend_tensor_get(destination_storage, actual.data(), 0, actual.size());
            REQUIRE(actual == expected);
        }
    }
    std::vector<float> source_after(input.size());
    ggml_backend_tensor_get(source_storage, source_after.data(), 0, source_after.size() * sizeof(float));
    REQUIRE(source_after == input);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
}

static void run_set_rows_capabilities(ggml_backend_t backend) {
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor * source = ggml_new_tensor_2d(context, GGML_TYPE_F32, 256, 1);
    ggml_tensor * destination = ggml_new_tensor_2d(context, GGML_TYPE_F16, 256, 8);
    ggml_tensor * ids = ggml_new_tensor_1d(context, GGML_TYPE_I64, 1);
    ggml_tensor * write = ggml_set_rows(context, destination, source, ids);
    set_rows_flag("0");
    REQUIRE(!ggml_backend_supports_op(backend, write));
    set_rows_flag("1");
    REQUIRE(ggml_backend_supports_op(backend, write));
    const auto reject = [&](ggml_tensor * op) {
        REQUIRE(!ggml::hrx::supports_set_rows_dispatch(op));
        REQUIRE(!ggml_backend_supports_op(backend, op));
    };
    source->type = GGML_TYPE_F16;
    reject(write);
    source->type = GGML_TYPE_F32;
    ids->type = GGML_TYPE_F32;
    reject(write);
    ids->type = GGML_TYPE_I64;
    ids->nb[0] = 9;
    reject(write);
    ids->nb[0] = 8;
    source->nb[0] = 8;
    reject(write);
    source->nb[0] = 4;
    source->nb[1] = 255 * sizeof(float);
    reject(write);
    source->nb[1] = 256 * sizeof(float);
    source->ne[2] = 2;
    reject(write);
    source->ne[2] = 1;
    write->nb[1] += 2;
    reject(write);
    write->nb[1] -= 2;
    write->view_offs = 2;
    reject(write);
    write->view_offs = 0;
    write->view_src = nullptr;
    reject(write);
    write->view_src = destination;
    source->view_src = destination;
    reject(write);
    source->view_src = nullptr;
    source->nb[1] = (ggml::hrx::kCopyF32MaxElements + 1) * sizeof(float);
    reject(write);
    source->nb[1] = 256 * sizeof(float);
    REQUIRE(ggml_backend_supports_op(backend, write));
    ggml_free(context);
}

static uint64_t next_host_alias_graph_uid();

static void run_set_rows_adjacent_mask_checks(ggml_backend_t backend) {
    constexpr int64_t rows = 512;
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
        REQUIRE(ctx != nullptr);
        auto * mask = ggml_new_tensor_2d(ctx, type, 1, rows);
        auto * zeros = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, rows);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
        ggml_set_input(ids);
        auto * output = ggml_set_rows(ctx, mask, zeros, ids);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        graph->uid = next_host_alias_graph_uid();
        REQUIRE(ggml_backend_supports_op(backend, output));
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer != nullptr);
        const std::vector<float> source(rows, 0.0f);
        const std::vector<float> initial_f32(rows, -INFINITY);
        const std::vector<ggml_fp16_t> initial_f16(rows, ggml_fp32_to_fp16(-INFINITY));
        std::vector<int32_t> indices(rows);
        std::vector<uint8_t> actual(ggml_nbytes(output));
        ggml_backend_tensor_set(zeros, source.data(), 0, source.size() * sizeof(float));
        for (int repeat = 0; repeat < 64; ++repeat) {
            for (int64_t row = 0; row < rows; ++row) {
                indices[row] = static_cast<int32_t>((row * 73 + repeat * 17) % rows);
            }
            ggml_backend_tensor_set(ids, indices.data(), 0, indices.size() * sizeof(int32_t));
            const void * initial = type == GGML_TYPE_F16 ?
                static_cast<const void *>(initial_f16.data()) : static_cast<const void *>(initial_f32.data());
            ggml_backend_tensor_set(mask, initial, 0, ggml_nbytes(mask));
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size());
            const auto nonzero = std::count_if(actual.begin(), actual.end(), [](uint8_t byte) { return byte != 0; });
            if (nonzero != 0) {
                std::fprintf(stderr, "Adjacent scalar SET_ROWS type=%s repeat=%d nonzero_bytes=%zu\n",
                             ggml_type_name(type), repeat, static_cast<size_t>(nonzero));
            }
            REQUIRE(nonzero == 0);
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    std::fprintf(stderr, "Adjacent scalar SET_ROWS mask checks passed (F16/F32, shuffled full permutation)\n");
}

static void run_set_rows_checks(ggml_backend_t backend) {
    const char * flag = std::getenv("HRX_ENABLE_SET_ROWS");
    const bool had_flag = flag != nullptr;
    const std::string previous = flag == nullptr ? "" : flag;
    run_set_rows_capabilities(backend);
    run_set_rows_adjacent_mask_checks(backend);
    for (const int64_t width : { 1, 128, 256, 512 }) {
        for (const ggml_type output_type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            for (const ggml_type index_type : { GGML_TYPE_I32, GGML_TYPE_I64 }) {
                for (const bool strided : { false, true }) {
                    run_set_rows_numeric(backend, width, output_type, index_type, strided);
                }
            }
        }
    }
    set_rows_flag(had_flag ? previous.c_str() : nullptr);
}

static void set_f32_get_rows_flag(const char * value) {
#ifdef _WIN32
    REQUIRE(_putenv_s("HRX_ENABLE_F32_GET_ROWS", value == nullptr ? "" : value) == 0);
#else
    REQUIRE((value == nullptr ? unsetenv("HRX_ENABLE_F32_GET_ROWS") :
                               setenv("HRX_ENABLE_F32_GET_ROWS", value, 1)) == 0);
#endif
}

struct f32_get_rows_fixture {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_buffer_t guarded_output = nullptr;
    ggml_tensor * storage = nullptr;
    ggml_tensor * source = nullptr;
    ggml_tensor * ids_storage = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    static constexpr size_t guard_words = 16;
    static constexpr int64_t capacity = 3;
    static constexpr int64_t count = 4;

    f32_get_rows_fixture(ggml_backend_t backend, int64_t width) {
        context = ggml_init({1024 * 1024, nullptr, true});
        REQUIRE(context != nullptr);
        storage = ggml_new_tensor_1d(context, GGML_TYPE_F32, width * capacity + 2 * guard_words);
        source = ggml_view_2d(context, storage, width, capacity, width * sizeof(float), guard_words * sizeof(float));
        ids_storage = ggml_new_tensor_1d(context, GGML_TYPE_I32, count + 2 * guard_words);
        ids = ggml_view_1d(context, ids_storage, count, guard_words * sizeof(int32_t));
        output = ggml_get_rows(context, source, ids);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(context, 32, false);
        ggml_build_forward_expand(graph, output);
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        REQUIRE(buffer != nullptr);
        guarded_output = ggml_backend_alloc_buffer(backend, ggml_nbytes(output) + 2 * guard_words * sizeof(uint32_t));
        REQUIRE(guarded_output != nullptr);
        output->buffer = guarded_output;
        output->data = static_cast<uint8_t *>(ggml_backend_buffer_get_base(guarded_output)) +
                       guard_words * sizeof(uint32_t);
        REQUIRE(ggml_backend_buffer_init_tensor(guarded_output, output) == GGML_STATUS_SUCCESS);
        ggml_backend_buffer_clear(guarded_output, 0xa5);
    }

    ~f32_get_rows_fixture() {
        ggml_backend_buffer_free(guarded_output);
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }

    std::vector<uint32_t> output_bits() const {
        std::vector<uint32_t> bits(static_cast<size_t>(ggml_nelements(output)));
        ggml_backend_tensor_get(output, bits.data(), 0, ggml_nbytes(output));
        return bits;
    }

    void check_guards() const {
        ggml_tensor all = *output;
        all.data = ggml_backend_buffer_get_base(guarded_output);
        all.ne[0] = static_cast<int64_t>(ggml_nbytes(output) / sizeof(uint32_t) + 2 * guard_words);
        all.ne[1] = all.ne[2] = all.ne[3] = 1;
        all.nb[1] = all.nb[2] = all.nb[3] = all.ne[0] * sizeof(uint32_t);
        std::array<uint32_t, guard_words> guard;
        ggml_backend_tensor_get(&all, guard.data(), 0, sizeof(guard));
        REQUIRE(std::all_of(guard.begin(), guard.end(), [](uint32_t v) { return v == 0xa5a5a5a5; }));
        ggml_backend_tensor_get(&all, guard.data(), guard_words * sizeof(uint32_t) + ggml_nbytes(output), sizeof(guard));
        REQUIRE(std::all_of(guard.begin(), guard.end(), [](uint32_t v) { return v == 0xa5a5a5a5; }));
    }
};

static std::vector<uint32_t> f32_state_bits(size_t count, int pass) {
    const std::array<uint32_t, 10> patterns = {
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x3f800001,
    };
    std::vector<uint32_t> bits(count);
    for (size_t i = 0; i < count; ++i) {
        bits[i] = patterns[(i + i / 257 + pass) % patterns.size()];
    }
    return bits;
}

static void run_f32_get_rows_numeric(ggml_backend_t backend, ggml_backend_t cpu, int64_t width) {
    f32_get_rows_fixture gpu(backend, width);
    f32_get_rows_fixture reference(cpu, width);
    REQUIRE(ggml::hrx::supports_f32_get_rows_dispatch(gpu.output));
    REQUIRE(ggml_backend_supports_op(backend, gpu.output));
    for (int pass = 0; pass < 3; ++pass) {
        const auto bits = f32_state_bits(static_cast<size_t>(ggml_nelements(gpu.storage)), pass);
        const std::array<int32_t, 4> ids = { (2 + pass) % 3, pass % 3, (2 + pass) % 3, (1 + pass) % 3 };
        for (auto * f : { &gpu, &reference }) {
            ggml_backend_tensor_set(f->storage, bits.data(), 0, ggml_nbytes(f->storage));
            ggml_backend_tensor_memset(f->ids_storage, 0xff, 0, ggml_nbytes(f->ids_storage));
            ggml_backend_tensor_set(f->ids, ids.data(), 0, sizeof(ids));
            ggml_backend_tensor_memset(f->output, 0x5a, 0, ggml_nbytes(f->output));
        }
        REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(backend, gpu.graph) == GGML_STATUS_SUCCESS);
        const auto expected = reference.output_bits();
        REQUIRE(gpu.output_bits() == expected);
        for (size_t row = 0; row < ids.size(); ++row) {
            REQUIRE(std::memcmp(expected.data() + row * width,
                                bits.data() + f32_get_rows_fixture::guard_words + ids[row] * width,
                                width * sizeof(uint32_t)) == 0);
        }
        for (const int32_t bad : { -1, 3, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max() }) {
            auto invalid = ids;
            invalid[1] = bad;
            ggml_backend_tensor_set(gpu.ids, invalid.data(), 0, sizeof(invalid));
            REQUIRE(ggml_backend_graph_compute(backend, gpu.graph) == GGML_STATUS_FAILED);
            REQUIRE(gpu.output_bits() == expected);
        }
        ggml_backend_tensor_set(gpu.ids, ids.data(), 0, sizeof(ids));
        REQUIRE(ggml_backend_graph_compute(backend, gpu.graph) == GGML_STATUS_SUCCESS);
        REQUIRE(gpu.output_bits() == expected);
        gpu.check_guards();
        reference.check_guards();
        std::vector<uint32_t> source_after(bits.size());
        ggml_backend_tensor_get(gpu.storage, source_after.data(), 0, ggml_nbytes(gpu.storage));
        REQUIRE(source_after == bits);
        std::array<int32_t, 4 + 2 * f32_get_rows_fixture::guard_words> ids_after;
        ggml_backend_tensor_get(gpu.ids_storage, ids_after.data(), 0, sizeof(ids_after));
        for (size_t i = 0; i < ids_after.size(); ++i) {
            const size_t guard = f32_get_rows_fixture::guard_words;
            REQUIRE(ids_after[i] == (i >= guard && i < guard + ids.size() ? ids[i - guard] : -1));
        }
    }
    std::fprintf(stderr, "F32 GET_ROWS width=%lld bitwise/guards/invalid/recovery passed\n",
                 static_cast<long long>(width));
}

static void run_f32_get_rows_capabilities(ggml_backend_t backend) {
    auto * context = ggml_init({1024 * 1024, nullptr, true});
    REQUIRE(context != nullptr);
    auto * storage = ggml_new_tensor_2d(context, GGML_TYPE_F32, 30720, 3);
    auto * source = ggml_view_tensor(context, storage);
    REQUIRE(source->op == GGML_OP_NONE && source->view_src == storage);
    auto * ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, 2);
    auto * output = ggml_get_rows(context, source, ids);
    set_f32_get_rows_flag("0");
    REQUIRE(!ggml::hrx::supports_f32_get_rows_dispatch(output));
    REQUIRE(!ggml_backend_supports_op(backend, output));
    for (const int64_t width : { 257, 92160, 786432 }) {
        auto * leaf = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, 3);
        REQUIRE(!ggml_backend_supports_op(backend, ggml_get_rows(context, leaf, ids)));
    }
    set_f32_get_rows_flag("1");
    REQUIRE(ggml::hrx::supports_f32_get_rows_dispatch(output));
    REQUIRE(ggml_backend_supports_op(backend, output));
    const auto reject = [&] {
        REQUIRE(!ggml::hrx::supports_f32_get_rows_dispatch(output));
        REQUIRE(!ggml_backend_supports_op(backend, output));
    };
    source->type = GGML_TYPE_BF16;
    reject();
    source->type = GGML_TYPE_F16;
    reject();
    source->type = GGML_TYPE_F32;
    ids->type = GGML_TYPE_I64;
    reject();
    ids->type = GGML_TYPE_I32;
    ids->nb[0] = 8;
    reject();
    ids->nb[0] = 4;
    source->nb[1] += 4;
    reject();
    source->nb[1] -= 4;
    source->ne[2] = 2;
    reject();
    source->ne[2] = 1;
    source->view_offs = 1;
    reject();
    source->view_offs = 0;
    output->view_src = source;
    reject();
    output->view_src = nullptr;
    output->nb[1] += 4;
    reject();
    output->nb[1] -= 4;
    output->ne[1] = 1;
    reject();
    output->ne[1] = 2;
    auto * wide = ggml_new_tensor_2d(context, GGML_TYPE_F32, ggml::hrx::kF32GetRowsMaxWidth + 1, 1);
    REQUIRE(!ggml::hrx::supports_f32_get_rows_dispatch(ggml_get_rows(context, wide, ids)));
    REQUIRE(!ggml_backend_supports_op(backend, ggml_get_rows(context, wide, ids)));
    auto * many = ggml_new_tensor_2d(context, GGML_TYPE_F32, 1, ggml::hrx::kF32GetRowsMaxRows + 1);
    REQUIRE(!ggml::hrx::supports_f32_get_rows_dispatch(ggml_get_rows(context, many, ids)));
    REQUIRE(!ggml_backend_supports_op(backend, ggml_get_rows(context, many, ids)));
    auto * huge = ggml_new_tensor_2d(context, GGML_TYPE_F32, ggml::hrx::kF32GetRowsMaxWidth, 129);
    REQUIRE(!ggml::hrx::supports_f32_get_rows_dispatch(ggml_get_rows(context, huge, ids)));
    auto * many_ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, ggml::hrx::kF32GetRowsMaxRows + 1);
    REQUIRE(!ggml::hrx::supports_f32_get_rows_dispatch(ggml_get_rows(context, source, many_ids)));
    auto * wide_row = ggml_new_tensor_2d(context, GGML_TYPE_F32, ggml::hrx::kF32GetRowsMaxWidth, 1);
    auto * large_ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, 129);
    REQUIRE(!ggml::hrx::supports_f32_get_rows_dispatch(ggml_get_rows(context, wide_row, large_ids)));
    ggml_free(context);
}

static void run_f32_get_rows_legacy_fusion_checks(ggml_backend_t backend) {
    using namespace ggml::hrx;
    auto * context = ggml_init({1024 * 1024, nullptr, true});
    REQUIRE(context != nullptr);
    for (const int64_t rows : { 1, 11 }) {
        const int64_t width = rows == 1 ? 2048 : 256;
        auto * a = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, rows);
        auto * b = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, rows);
        auto * ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, rows == 1 ? 1 : 5);
        auto * selected_a = ggml_get_rows(context, a, ids);
        auto * selected_b = ggml_get_rows(context, b, ids);
        auto * result = ggml_add(context, selected_a, selected_b);
        auto * graph = ggml_new_graph_custom(context, 32, false);
        ggml_build_forward_expand(graph, result);
        for (const char * flag : { "0", "1" }) {
            set_f32_get_rows_flag(flag);
            REQUIRE(ggml_backend_supports_op(backend, selected_a));
            REQUIRE(ggml_backend_supports_op(backend, selected_b));
            auto imported = import_ggml_graph(*graph);
            REQUIRE(imported.valid());
            DispatchScheduler scheduler;
            REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
            REQUIRE(scheduler.dispatches().size() == 1);
            REQUIRE(scheduler.dispatches()[0].kernel.kernel_id ==
                    kernel_catalog_id("qwen3_moe", "ggml_gather_add_f32"));
        }
    }
    ggml_free(context);
}

static void run_f32_get_rows_misaligned_output(ggml_backend_t backend) {
    f32_get_rows_fixture f(backend, 257);
    const std::array<int32_t, 4> ids = { 2, 0, 2, 1 };
    ggml_backend_tensor_set(f.ids, ids.data(), 0, sizeof(ids));
    // Metadata remains contiguous; only the actual allocation offset is invalid.
    f.output->data = static_cast<uint8_t *>(f.output->data) + 1;
    REQUIRE(ggml_backend_buffer_init_tensor(f.output->buffer, f.output) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(backend, f.graph) == GGML_STATUS_FAILED);
    const auto after = f.output_bits();
    REQUIRE(std::all_of(after.begin(), after.end(), [](uint32_t v) { return v == 0xa5a5a5a5; }));
    f.check_guards();
}

static void run_f32_get_rows_alias(ggml_backend_t backend, bool alias_ids) {
    f32_get_rows_fixture f(backend, 1);
    const auto original = f32_state_bits(static_cast<size_t>(ggml_nelements(f.storage)), 0);
    ggml_backend_tensor_set(f.storage, original.data(), 0, ggml_nbytes(f.storage));
    const std::array<int32_t, 4> ids = { 2, 0, 2, 1 };
    ggml_backend_tensor_set(f.ids, ids.data(), 0, sizeof(ids));
    auto * overlap = alias_ids ? f.ids : f.source;
    f.output->buffer = overlap->buffer;
    f.output->data = overlap->data;
    REQUIRE(ggml_backend_buffer_init_tensor(f.output->buffer, f.output) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(backend, f.graph) == GGML_STATUS_FAILED);
    std::vector<uint32_t> after(original.size());
    ggml_backend_tensor_get(f.storage, after.data(), 0, ggml_nbytes(f.storage));
    REQUIRE(after == original);
    std::array<int32_t, 4> ids_after;
    ggml_backend_tensor_get(f.ids, ids_after.data(), 0, sizeof(ids_after));
    REQUIRE(ids_after == ids);
}

static void run_f32_get_rows_scheduler(ggml_backend_t backend, ggml_backend_t cpu, bool enable) {
    constexpr int64_t width = 786432;
    auto * context = ggml_init({1024 * 1024, nullptr, true});
    auto * ids_context = ggml_init({1024 * 1024, nullptr, true});
    REQUIRE(context != nullptr && ids_context != nullptr);
    auto * storage = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, 3);
    // Match build_rs(): a real layout node exposes the storage root to gallocr.
    // ggml_view_tensor() is a NONE leaf and hides that extra allocator hash key.
    auto * source = ggml_reshape_2d(context, storage, width, 3);
    auto * state_buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    REQUIRE(state_buffer != nullptr);
    auto * ids = ggml_new_tensor_1d(ids_context, GGML_TYPE_I32, 2);
    ggml_set_input(ids);
    auto * ids_buffer = ggml_backend_alloc_ctx_tensors(ids_context, cpu);
    REQUIRE(ids_buffer != nullptr);
    auto * output = ggml_get_rows(context, source, ids);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(context, 32, false);
    ggml_build_forward_expand(graph, output);
    REQUIRE(ggml_graph_n_nodes(graph) == 2);
    set_f32_get_rows_flag(enable ? "1" : "0");
    ggml_backend_t backends[] = { backend, cpu };
    auto * scheduler = ggml_backend_sched_new(backends, nullptr, 2, 32, false, true);
    REQUIRE(scheduler != nullptr);
    std::fprintf(stderr, "F32 GET_ROWS scheduler enabled=%d\n", enable ? 1 : 0);
    REQUIRE(ggml_backend_sched_alloc_graph(scheduler, graph));
    REQUIRE(ggml_backend_sched_get_tensor_backend(scheduler, output) == (enable ? backend : cpu));
    for (int pass = 0; pass < 2; ++pass) {
        const auto bits = f32_state_bits(static_cast<size_t>(width * 3), pass);
        const std::array<int32_t, 2> selected = { 2 - pass, pass };
        ggml_backend_tensor_set(storage, bits.data(), 0, ggml_nbytes(storage));
        ggml_backend_tensor_set(ids, selected.data(), 0, sizeof(selected));
        REQUIRE(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(scheduler);
        std::vector<uint32_t> actual(static_cast<size_t>(width * 2));
        ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
        for (size_t row = 0; row < selected.size(); ++row) {
            REQUIRE(std::memcmp(actual.data() + row * width, bits.data() + selected[row] * width,
                                width * sizeof(uint32_t)) == 0);
        }
    }
    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(ids_buffer);
    ggml_backend_buffer_free(state_buffer);
    ggml_free(ids_context);
    ggml_free(context);
    set_f32_get_rows_flag("1");
}

static void run_f32_get_rows_checks(ggml_backend_t backend) {
    const char * flag = std::getenv("HRX_ENABLE_F32_GET_ROWS");
    const bool had_flag = flag != nullptr;
    const std::string previous = flag == nullptr ? "" : flag;
    run_f32_get_rows_capabilities(backend);
    run_f32_get_rows_legacy_fusion_checks(backend);
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(cpu != nullptr);
    for (const int64_t width : { 1, 257, 30720, 92160, 786432 }) {
        run_f32_get_rows_numeric(backend, cpu, width);
    }
    run_f32_get_rows_alias(backend, false);
    run_f32_get_rows_alias(backend, true);
    run_f32_get_rows_misaligned_output(backend);
    run_f32_get_rows_scheduler(backend, cpu, false);
    run_f32_get_rows_scheduler(backend, cpu, true);
    ggml_backend_free(cpu);
    set_f32_get_rows_flag(had_flag ? previous.c_str() : nullptr);
}

static void set_repeat_flag(const char * value) {
#ifdef _WIN32
    REQUIRE(_putenv_s("HRX_ENABLE_REPEAT", value == nullptr ? "" : value) == 0);
#else
    REQUIRE((value == nullptr ? unsetenv("HRX_ENABLE_REPEAT") : setenv("HRX_ENABLE_REPEAT", value, 1)) == 0);
#endif
}

static void require_repeat_bits(ggml_tensor * tensor, const std::vector<float> & expected,
                                const char * allocation, int pass) {
    std::vector<uint32_t> actual(expected.size());
    REQUIRE(ggml_nbytes(tensor) == actual.size() * sizeof(uint32_t));
    ggml_backend_tensor_get(tensor, actual.data(), 0, ggml_nbytes(tensor));
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &expected[i], sizeof(bits));
        if (actual[i] != bits) {
            if (mismatches++ < 4) {
                std::fprintf(stderr, "REPEAT %s pass=%d %s index=%zu expected=%08x actual=%08x\n",
                             allocation, pass, tensor->name, i, static_cast<unsigned>(bits),
                             static_cast<unsigned>(actual[i]));
            }
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "REPEAT mismatches: %zu/%zu\n", mismatches, expected.size());
    }
    REQUIRE(mismatches == 0);
}

enum class repeat_allocation { context, gallocr, scheduler };

static void run_repeat_numeric(ggml_backend_t backend, ggml_backend_t cpu, int64_t width, int64_t tokens,
                               repeat_allocation allocation, bool mixed, bool cpu_repeat = false) {
    const char * label = allocation == repeat_allocation::context ? "ctx" :
                        allocation == repeat_allocation::gallocr ? "gallocr" : "scheduler";
    std::fprintf(stderr, "REPEAT %s width=%lld T=%lld mixed=%d placement=%s\n", label,
                 static_cast<long long>(width), static_cast<long long>(tokens), mixed,
                 cpu_repeat ? "CPU" : "HRX");
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor * input = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, 1, tokens);
    ggml_set_name(input, "repeat_input");
    ggml_set_input(input);
    ggml_tensor * source = mixed ? ggml_scale(context, input, -2.0f) : input;
    ggml_set_name(source, mixed ? "repeat_producer" : "repeat_input");
    ggml_tensor * shape = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, 4, tokens);
    ggml_tensor * repeated = ggml_repeat(context, source, shape);
    ggml_set_name(repeated, "repeat_tile");
    REQUIRE(repeated->op == GGML_OP_REPEAT);
    REQUIRE(ggml_backend_supports_op(backend, repeated));
    ggml_tensor * residual = nullptr;
    ggml_tensor * combined = nullptr;
    ggml_tensor * output = repeated;
    if (mixed) {
        residual = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, 4, tokens);
        ggml_set_name(residual, "repeat_residual");
        ggml_set_input(residual);
        combined = ggml_add(context, repeated, residual);
        ggml_set_name(combined, "repeat_combine");
        output = ggml_scale(context, combined, 0.5f);
        ggml_set_name(output, "repeat_output");
        REQUIRE(ggml_backend_supports_op(backend, source));
        REQUIRE(ggml_backend_supports_op(backend, combined));
        REQUIRE(ggml_backend_supports_op(backend, output));
    }
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(context, 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    if (allocation == repeat_allocation::context) {
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        REQUIRE(buffer != nullptr);
    } else if (allocation == repeat_allocation::gallocr) {
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        REQUIRE(allocator != nullptr);
        REQUIRE(ggml_gallocr_reserve(allocator, graph));
        REQUIRE(ggml_gallocr_alloc_graph(allocator, graph));
    } else {
        ggml_backend_t backends[] = { backend, cpu };
        scheduler = ggml_backend_sched_new(backends, nullptr, 2, 64, false, true);
        REQUIRE(scheduler != nullptr);
        ggml_backend_sched_set_tensor_backend(scheduler, repeated, cpu_repeat ? cpu : backend);
        if (mixed) {
            // Keep the neighbors fixed while moving only REPEAT across the CPU/HRX boundary.
            ggml_backend_sched_set_tensor_backend(scheduler, source, backend);
            ggml_backend_sched_set_tensor_backend(scheduler, combined, backend);
            ggml_backend_sched_set_tensor_backend(scheduler, output, cpu);
        }
        REQUIRE(ggml_backend_sched_alloc_graph(scheduler, graph));
        REQUIRE(ggml_backend_sched_get_tensor_backend(scheduler, repeated) == (cpu_repeat ? cpu : backend));
        std::fprintf(stderr, "REPEAT scheduler splits=%d\n", ggml_backend_sched_get_n_splits(scheduler));
    }

    const uintptr_t source_begin = reinterpret_cast<uintptr_t>(source->data);
    const uintptr_t repeat_begin = reinterpret_cast<uintptr_t>(repeated->data);
    std::fprintf(stderr, "REPEAT binding src=%p/%p dst=%p/%p output=%p/%p\n",
                 static_cast<void *>(source->buffer), source->data, static_cast<void *>(repeated->buffer),
                 repeated->data, static_cast<void *>(output->buffer), output->data);
    REQUIRE(source_begin != 0 && repeat_begin != 0);
    REQUIRE(source_begin + ggml_nbytes(source) <= repeat_begin ||
            repeat_begin + ggml_nbytes(repeated) <= source_begin);

    std::vector<float> values(static_cast<size_t>(ggml_nelements(input)));
    std::vector<float> residual_values(static_cast<size_t>(ggml_nelements(repeated)));
    std::vector<float> expected(residual_values.size());
    // Same graph UID and allocation, but new data and dirty outputs on every replay.
    for (int pass = 0; pass < 3; ++pass) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(static_cast<int>((i * 37 + pass * 101) % 4093) - 2046) / 16.0f;
        }
        for (size_t i = 0; i < residual_values.size(); ++i) {
            residual_values[i] = static_cast<float>(static_cast<int>((i * 19 + pass * 53) % 1021) - 510) / 8.0f;
        }
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t mid = 0; mid < 4; ++mid) {
                for (int64_t col = 0; col < width; ++col) {
                    const size_t dst = static_cast<size_t>((t * 4 + mid) * width + col);
                    const float value = values[static_cast<size_t>(t * width + col)];
                    // Dyadic inputs keep the mixed graph's arithmetic exact too.
                    expected[dst] = mixed ? (-2.0f * value + residual_values[dst]) * 0.5f : value;
                }
            }
        }
        ggml_backend_tensor_memset(repeated, 0xa5 + pass, 0, ggml_nbytes(repeated));
        ggml_backend_tensor_memset(output, 0x5a + pass, 0, ggml_nbytes(output));
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        if (mixed) {
            ggml_backend_tensor_set(residual, residual_values.data(), 0, ggml_nbytes(residual));
        }
        if (scheduler != nullptr) {
            REQUIRE(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_sched_synchronize(scheduler);
        } else {
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);
        }
        require_repeat_bits(output, expected, label, pass);
        // gallocr may legally recycle mixed-graph inputs after their last consumer.
        if (!mixed || allocation == repeat_allocation::context) {
            require_repeat_bits(input, values, label, pass);
            if (mixed) {
                require_repeat_bits(residual, residual_values, label, pass);
            }
        }
    }
    if (scheduler != nullptr) {
        ggml_backend_sched_free(scheduler);
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    std::fprintf(stderr, "REPEAT %s passed\n", label);
}

static void run_repeat_checks(ggml_backend_t backend) {
    const char * flag = std::getenv("HRX_ENABLE_REPEAT");
    const bool had_flag = flag != nullptr;
    const std::string previous = flag == nullptr ? "" : flag;
    set_repeat_flag("1");
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(cpu != nullptr);
    for (const repeat_allocation allocation :
         { repeat_allocation::context, repeat_allocation::gallocr, repeat_allocation::scheduler }) {
        for (const int64_t tokens : { 1, 2, 3, 5 }) {
            run_repeat_numeric(backend, cpu, 2560, tokens, allocation, false);
        }
        // Non-multiple of 256 with z > 1 distinguishes column tails from outer-axis addressing.
        run_repeat_numeric(backend, cpu, 257, 3, allocation, false);
        run_repeat_numeric(backend, cpu, 2560, 1, allocation, true);
        run_repeat_numeric(backend, cpu, 2560, 3, allocation, true);
    }
    for (const int64_t tokens : { 1, 3 }) {
        run_repeat_numeric(backend, cpu, 2560, tokens, repeat_allocation::scheduler, true, true);
    }
    ggml_backend_free(cpu);
    set_repeat_flag(had_flag ? previous.c_str() : nullptr);
}

struct qsa_fixture {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * projection = nullptr;
    ggml_tensor * query = nullptr;
    ggml_tensor * key_cache = nullptr;
    ggml_tensor * value_cache = nullptr;
    ggml_tensor * key_new = nullptr;
    ggml_tensor * value_new = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * flash = nullptr;
    ggml_tensor * gate_copy = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * output = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;

    qsa_fixture(int64_t tokens, int64_t cells, bool gate_first = false) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        context = ggml_init(params);
        REQUIRE(context != nullptr);
        graph = ggml_new_graph(context);
        input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 12288, tokens);
        projection = ggml_scale(context, input, 0.75f);
        ggml_set_name(projection, "qsa_projection");
        auto * q_view = ggml_view_3d(context, projection, 256, 24, tokens,
                                    512 * sizeof(float), 12288 * sizeof(float), 0);
        // Boundary surrogate for the model's contiguous post-norm/post-RoPE query.
        auto * q_cur = ggml_cont(context, q_view);
        query = ggml_permute(context, q_cur, 0, 2, 1, 3);
        ggml_build_forward_expand(graph, query);
        key_cache = ggml_new_tensor_2d(context, GGML_TYPE_F16, 512, cells);
        value_cache = ggml_new_tensor_2d(context, GGML_TYPE_F16, 512, cells);
        key_new = ggml_new_tensor_2d(context, GGML_TYPE_F32, 512, tokens);
        value_new = ggml_new_tensor_2d(context, GGML_TYPE_F32, 512, tokens);
        ids = ggml_new_tensor_1d(context, GGML_TYPE_I64, tokens);
        // Like build_attn_qsa, cache publications are explicitly expanded BEFORE FLASH.
        ggml_build_forward_expand(graph, ggml_set_rows(context, key_cache, key_new, ids));
        ggml_build_forward_expand(graph, ggml_set_rows(context, value_cache, value_new, ids));
        auto * k = ggml_permute(context, ggml_reshape_3d(context, key_cache, 256, 2, cells), 0, 2, 1, 3);
        auto * v = ggml_permute(context, ggml_reshape_3d(context, value_cache, 256, 2, cells), 0, 2, 1, 3);
        // The materialized sparse+causal mask is this fixture's input boundary.
        // Index projection/top-k/FILL/SET_ROWS(width=1)/F16 ADD are NOT GPU-tested here.
        mask = ggml_new_tensor_2d(context, GGML_TYPE_F16, cells, std::max<int64_t>(32, tokens));
        flash = ggml_flash_attn_ext(context, query, k, v, mask, 0.0625f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        ggml_set_name(flash, "qsa_flash");
        auto * gate_view = ggml_view_3d(context, projection, 256, 24, tokens,
                                       512 * sizeof(float), 12288 * sizeof(float), 256 * sizeof(float));
        gate_copy = ggml_cont_2d(context, gate_view, 6144, tokens);
        gate = ggml_sigmoid(context, gate_copy);
        if (gate_first) {
            ggml_build_forward_expand(graph, gate);
        }
        output = ggml_mul(context, ggml_reshape_2d(context, flash, 6144, tokens), gate);
        output = ggml_reshape_3d(context, output, 256, 24, tokens);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        for (auto * tensor : { input, key_cache, value_cache, key_new, value_new, ids, mask }) {
            ggml_set_input(tensor);
        }
    }

    void allocate(ggml_backend_t backend, bool reuse) {
        if (reuse) {
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            REQUIRE(allocator != nullptr);
            REQUIRE(ggml_gallocr_alloc_graph(allocator, graph));
        } else {
            buffer = ggml_backend_alloc_ctx_tensors(context, backend);
            REQUIRE(buffer != nullptr);
        }
    }

    ~qsa_fixture() {
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }
};

static size_t qsa_node_index(const ggml::hrx::Graph & graph, const ggml_tensor * tensor) {
    const auto * value = graph.values().find_tensor(tensor);
    REQUIRE(value != nullptr);
    size_t index = 0;
    REQUIRE(graph.index().node_index(graph.index().producer(value->id), index));
    return index;
}

static void check_qsa_plan(qsa_fixture & fixture, bool gate_first) {
    using namespace ggml::hrx;
    auto imported = import_ggml_graph(*fixture.graph);
    REQUIRE(imported.valid());
    auto & graph = imported.graph;
    const size_t root = qsa_node_index(graph, fixture.flash);
    REQUIRE((qsa_node_index(graph, fixture.gate_copy) < root) == gate_first);
    REQUIRE((qsa_node_index(graph, fixture.gate) < root) == gate_first);
    std::vector<bool> covered(graph.nodes().size(), false);
    CommandPlan plan;
    DispatchRegistryBuilder builder;
    register_qwen4exp_flash_attention_dispatches(builder);
    const auto registry = builder.build();
    DispatchMatch match;
    const ValueId next(static_cast<int32_t>(graph.values().size()));
    REQUIRE(registry.match({ graph, &graph.nodes()[root], root, covered, plan, next }, match));
    REQUIRE(match.covered_nodes == std::vector<size_t>{ root });
    REQUIRE(match.value_aliases.empty());
    const auto flash_id = graph.nodes()[root].output;
    for (const auto & dispatch : match.dispatches) {
        for (const auto & binding : dispatch.bindings) {
            const auto * producer = graph.index().producer(binding.value);
            size_t index = 0;
            if (producer) {
                REQUIRE(graph.index().node_index(producer, index));
                REQUIRE(index <= root);
            }
        }
    }
    REQUIRE(match.dispatches.back().bindings.back().value == flash_id);
    // This covers the former failure mode: late gate remains late, but the complete
    // subgraph schedules without reading it early or writing future MUL storage.
    DispatchScheduler scheduler;
    if (!scheduler.schedule_graph(graph, { "gfx1151" })) {
        std::fprintf(stderr, "QSA plan failed: %s\n", scheduler.error().c_str());
        REQUIRE(false);
    }

    // A future *actual FLASH input* must still be rejected (global readiness is
    // unchanged). Build an intentionally invalid producer order without GPU launch.
    auto original = import_ggml_graph(*fixture.graph);
    REQUIRE(original.valid());
    Graph late;
    late.values() = original.graph.values();
    const auto & nodes = original.graph.nodes();
    auto & first = late.add_node(nodes[root].op, nodes[root].output, nodes[root].inputs);
    first.params = nodes[root].params;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (i != root) {
            auto & node = late.add_node(nodes[i].op, nodes[i].output, nodes[i].inputs);
            node.params = nodes[i].params;
        }
    }
    REQUIRE(late.build_index().success());
    DispatchMatch rejected;
    REQUIRE(!registry.match({ late, &late.nodes()[0], 0, covered, plan, next }, rejected));
}

static void check_qsa_shapes(qsa_fixture & fixture) {
    using ggml::hrx::supports_qwen4exp_flash_attention_dispatch;
    REQUIRE(supports_qwen4exp_flash_attention_dispatch(fixture.flash));
    auto op = *fixture.flash;
    auto query = *op.src[0];
    auto key = *op.src[1];
    auto mask = *op.src[3];
    op.src[0] = &query;
    op.src[1] = &key;
    op.src[3] = &mask;
    query.ne[0] = 128;
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    query = *fixture.query;
    query.ne[1] = 16;
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    query = *fixture.query;
    query.ne[2] = 48;
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    query = *fixture.query;
    query.nb[2] *= 2;
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    query = *fixture.query;
    key.type = GGML_TYPE_F32;
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    key = *fixture.flash->src[1];
    key.nb[1] += sizeof(ggml_fp16_t);
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    key = *fixture.flash->src[1];
    mask.ne[0] -= 1;
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    mask = *fixture.mask;
    op.src[4] = fixture.gate;
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    op.src[4] = nullptr;
    ggml_flash_attn_ext_set_n_kv_max(&op, 1);
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    ggml_flash_attn_ext_set_n_kv_max(&op, 0);
    float bias = 1.0f;
    std::memcpy(reinterpret_cast<uint8_t *>(op.op_params) + sizeof(float), &bias, sizeof(bias));
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    std::memcpy(op.op_params, fixture.flash->op_params, sizeof(op.op_params));
    float wrong_scale = 0.125f;
    std::memcpy(op.op_params, &wrong_scale, sizeof(wrong_scale));
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(&op));
    // Width 6144 by itself (not FLASH provenance) never identifies QSA rather than GDN.
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(fixture.gate));
    REQUIRE(!supports_qwen4exp_flash_attention_dispatch(fixture.output->src[0]));
}

static std::vector<float> upload_qsa_inputs(qsa_fixture & fixture, int pass) {
    const int64_t tokens = fixture.input->ne[1];
    const int64_t cells = fixture.key_cache->ne[1];
    std::vector<float> projection(static_cast<size_t>(12288 * tokens));
    std::vector<ggml_fp16_t> key(static_cast<size_t>(512 * cells)), value(key.size());
    std::vector<float> new_key(static_cast<size_t>(512 * tokens)), new_value(new_key.size());
    std::vector<int64_t> ids(static_cast<size_t>(tokens));
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(ggml_nelements(fixture.mask)),
                                ggml_fp32_to_fp16(-INFINITY));
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < 24; ++h) {
            for (int64_t d = 0; d < 256; ++d) {
                const size_t i = static_cast<size_t>(t * 12288 + h * 512 + d);
                projection[i] = static_cast<float>((d * 7 + h * 11 + t * 5 + pass * 3) % 41 - 20) / 32.0f;
                projection[i + 256] =
                    static_cast<float>((d * 3 + h * 13 + t * 19 + pass * 7) % 65 - 32) / 8.0f;
            }
        }
        ids[t] = (cells - 1 - t * 3 + cells * 3) % cells;
        for (int64_t c = 0; c < cells; ++c) {
            const bool visible = c == cells - 1 || (c + t + pass) % 3 == 1;
            mask[t * cells + c] = ggml_fp32_to_fp16(visible ? -static_cast<float>((c + t) % 5) / 8.0f : -INFINITY);
        }
    }
    for (size_t i = 0; i < key.size(); ++i) {
        const int64_t c = static_cast<int64_t>(i / 512);
        const int64_t h = static_cast<int64_t>(i / 256) % 2;
        const int64_t d = static_cast<int64_t>(i % 256);
        key[i] = ggml_fp32_to_fp16(static_cast<float>((c * 17 + h * 23 + d * 3 + pass * 5) % 53 - 26) / 32.0f);
        value[i] = ggml_fp32_to_fp16(static_cast<float>((c * 7 + h * 31 + d * 11 + pass * 13) % 71 - 35) / 16.0f);
    }
    for (size_t i = 0; i < new_key.size(); ++i) {
        new_key[i] = static_cast<float>((static_cast<int64_t>(i) * 13 + pass * 17) % 59 - 29) / 32.0f;
        new_value[i] = static_cast<float>((static_cast<int64_t>(i) * 19 + pass * 11) % 83 - 41) / 16.0f;
    }
    ggml_backend_tensor_set(fixture.input, projection.data(), 0, projection.size() * sizeof(float));
    ggml_backend_tensor_set(fixture.key_cache, key.data(), 0, key.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(fixture.value_cache, value.data(), 0, value.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(fixture.key_new, new_key.data(), 0, new_key.size() * sizeof(float));
    ggml_backend_tensor_set(fixture.value_new, new_value.data(), 0, new_value.size() * sizeof(float));
    ggml_backend_tensor_set(fixture.ids, ids.data(), 0, ids.size() * sizeof(int64_t));
    ggml_backend_tensor_set(fixture.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t d = 0; d < 512; ++d) {
            key[ids[t] * 512 + d] = ggml_fp32_to_fp16(new_key[t * 512 + d]);
            value[ids[t] * 512 + d] = ggml_fp32_to_fp16(new_value[t * 512 + d]);
        }
    }
    std::vector<float> expected(static_cast<size_t>(6144 * tokens));
    std::vector<double> scores(static_cast<size_t>(cells));
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < 24; ++h) {
            double maximum = -INFINITY;
            for (int64_t c = 0; c < cells; ++c) {
                double dot = 0.0;
                for (int64_t d = 0; d < 256; ++d) {
                    dot += (0.75 * projection[t * 12288 + h * 512 + d]) *
                           ggml_fp16_to_fp32(key[c * 512 + (h / 12) * 256 + d]);
                }
                scores[c] = dot / 16.0 + ggml_fp16_to_fp32(mask[t * cells + c]);
                maximum = std::max(maximum, scores[c]);
            }
            double sum = 0.0;
            for (double & score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            REQUIRE(sum > 0.0);
            for (int64_t d = 0; d < 256; ++d) {
                double weighted = 0.0;
                for (int64_t c = 0; c < cells; ++c) {
                    weighted += scores[c] * ggml_fp16_to_fp32(value[c * 512 + (h / 12) * 256 + d]);
                }
                const double gate = 1.0 / (1.0 + std::exp(-0.75 * projection[t * 12288 + h * 512 + 256 + d]));
                expected[t * 6144 + h * 256 + d] = static_cast<float>(weighted / sum * gate);
            }
        }
    }
    return expected;
}

static void check_qsa_output(qsa_fixture & fixture, const std::vector<float> & expected, const char * label) {
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(fixture.output, actual.data(), 0, actual.size() * sizeof(float));
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || error > 0.002f + 0.003f * std::fabs(expected[i])) {
            std::fprintf(stderr, "QSA %s mismatch at %zu: actual=%g expected=%g\n",
                         label, i, actual[i], expected[i]);
            REQUIRE(false);
        }
        max_error = std::max(max_error, error);
    }
    std::fprintf(stderr, "QSA %s max_abs_error=%g\n", label, max_error);
}

static void set_qsa_flag(const char * value) {
#ifdef _WIN32
    REQUIRE(_putenv_s("HRX_ENABLE_QSA_ATTN", value == nullptr ? "" : value) == 0);
#else
    REQUIRE((value == nullptr ? unsetenv("HRX_ENABLE_QSA_ATTN") : setenv("HRX_ENABLE_QSA_ATTN", value, 1)) == 0);
#endif
}

static void run_qsa_checks(ggml_backend_t backend) {
    const char * old_qsa = std::getenv("HRX_ENABLE_QSA_ATTN");
    const bool had_qsa = old_qsa != nullptr;
    const std::string saved_qsa = old_qsa ? old_qsa : "";
    const char * old_rows = std::getenv("HRX_ENABLE_SET_ROWS");
    const bool had_rows = old_rows != nullptr;
    const std::string saved_rows = old_rows ? old_rows : "";
    set_qsa_flag("1");
    set_rows_flag("1");
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(cpu != nullptr);
    for (bool reuse : { false, true }) {
        for (const auto shape : { std::array<int64_t, 2>{1, 1}, {1, 7}, {1, 256}, {3, 17}, {5, 33},
                                  {1, 2049}, {3, 2049} }) {
            for (bool gate_first : { false, true }) {
                const int64_t tokens = shape[0], cells = shape[1];
                std::fprintf(stderr, "QSA T=%lld cells=%lld alloc=%s gate=%s\n",
                             (long long) tokens, (long long) cells, reuse ? "gallocr" : "context",
                             gate_first ? "early" : "model-order");
                qsa_fixture gpu_fixture(tokens, cells, gate_first);
                qsa_fixture cpu_fixture(tokens, cells, gate_first);
                check_qsa_shapes(gpu_fixture);
                check_qsa_plan(gpu_fixture, gate_first);
                set_qsa_flag("0");
                REQUIRE(!ggml_backend_supports_op(backend, gpu_fixture.flash));
                set_qsa_flag("1");
                REQUIRE(ggml_backend_supports_op(backend, gpu_fixture.flash));
                gpu_fixture.allocate(backend, reuse);
                cpu_fixture.allocate(cpu, reuse);
                for (int pass = 0; pass < 3; ++pass) {
                    const auto expected = upload_qsa_inputs(gpu_fixture, pass);
                    REQUIRE(upload_qsa_inputs(cpu_fixture, pass) == expected);
                    REQUIRE(ggml_backend_graph_compute(cpu, cpu_fixture.graph) == GGML_STATUS_SUCCESS);
                    check_qsa_output(cpu_fixture, expected, "CPU/reference");
                    REQUIRE(ggml_backend_graph_compute(backend, gpu_fixture.graph) == GGML_STATUS_SUCCESS);
                    check_qsa_output(gpu_fixture, expected, "HRX/reference");
                    std::vector<float> cpu_output(expected.size());
                    ggml_backend_tensor_get(cpu_fixture.output, cpu_output.data(), 0, cpu_output.size() * sizeof(float));
                    check_qsa_output(gpu_fixture, cpu_output, "HRX/CPU");
                    for (int cache = 0; cache < 2; ++cache) {
                        auto * gpu_cache = cache == 0 ? gpu_fixture.key_cache : gpu_fixture.value_cache;
                        auto * cpu_cache = cache == 0 ? cpu_fixture.key_cache : cpu_fixture.value_cache;
                        std::vector<uint8_t> gpu_bytes(ggml_nbytes(gpu_cache)), cpu_bytes(gpu_bytes.size());
                        ggml_backend_tensor_get(gpu_cache, gpu_bytes.data(), 0, gpu_bytes.size());
                        ggml_backend_tensor_get(cpu_cache, cpu_bytes.data(), 0, cpu_bytes.size());
                        REQUIRE(gpu_bytes == cpu_bytes);
                    }
                }
            }
        }
    }
    ggml_backend_free(cpu);
    set_qsa_flag(had_qsa ? saved_qsa.c_str() : nullptr);
    set_rows_flag(had_rows ? saved_rows.c_str() : nullptr);
}

struct gdn_norm_fixture {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * packed = nullptr;
    ggml_tensor * raw = nullptr;
    ggml_tensor * gamma = nullptr;
    ggml_tensor * z_input = nullptr;
    ggml_tensor * z_producer = nullptr;
    ggml_tensor * z = nullptr;
    ggml_tensor * rms = nullptr;
    ggml_tensor * weighted = nullptr;
    ggml_tensor * activation = nullptr;
    ggml_tensor * gated = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v_storage = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * decay = nullptr;
    ggml_tensor * beta = nullptr;
    ggml_tensor * state = nullptr;
    ggml_tensor * new_state = nullptr;
    ggml_tensor * state_cache = nullptr;
    ggml_tensor * state_destination = nullptr;
    ggml_tensor * state_copy = nullptr;
    ggml_tensor * y_snapshot = nullptr;
    ggml_tensor * z_weight = nullptr;
    ggml_tensor * output_weight = nullptr;
    ggml_tensor * projected = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    size_t prefix;
    bool early;
    bool silu;
    bool retain_packed;
    bool recurrent;
    bool projections;

    gdn_norm_fixture(bool early_z, bool silu_control, size_t prefix_elements, bool retain,
                     bool with_recurrent = false, bool observe_y = false, bool with_projections = false) :
        prefix(prefix_elements), early(early_z), silu(silu_control), retain_packed(retain),
        recurrent(with_recurrent), projections(with_projections) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        context = ggml_init(params);
        REQUIRE(context != nullptr);
        graph = ggml_new_graph(context);
        if (recurrent) {
            REQUIRE(prefix == 0 && !silu);
            // Inputs start at the real recurrent split boundary: materialized
            // L2-normalized Q/K, activated g/beta, and build_rs's gathered state.
            q = ggml_new_tensor_4d(context, GGML_TYPE_F32, 128, 16, 1, 1);
            k = ggml_dup_tensor(context, q);
            v_storage = ggml_new_tensor_2d(context, GGML_TYPE_F32, 10240, 1);
            v = ggml_view_4d(context, v_storage, 128, 48, 1, 1,
                             128 * sizeof(float), 10240 * sizeof(float), 10240 * sizeof(float),
                             4096 * sizeof(float));
            decay = ggml_new_tensor_4d(context, GGML_TYPE_F32, 1, 48, 1, 1);
            beta = ggml_dup_tensor(context, decay);
            state = ggml_new_tensor_4d(context, GGML_TYPE_F32, 128, 128, 48, 1);
            packed = ggml_gated_delta_net(context, q, k, v, decay, beta, state, 1);
            ggml_set_name(packed, "gdn_recurrent_packed_y_state");
            for (auto * tensor : {q, k, v_storage, decay, beta, state}) {
                ggml_set_input(tensor);
            }
            if (retain) {
                ggml_set_output(state);
            }
        } else {
            // Leaf control isolates postnorm from the recurrent producer.
            packed = ggml_new_tensor_1d(context, GGML_TYPE_F32, prefix + 6144 + 128 * 128 * 48);
            ggml_set_input(packed);
        }
        raw = ggml_view_4d(context, packed, 128, 48, 1, 1,
                           128 * sizeof(float), 6144 * sizeof(float), 6144 * sizeof(float),
                           prefix * sizeof(float));
        if (recurrent) {
            const int64_t state_elements = 128 * 128 * 48;
            // delta-net-base.cpp build_delta_net_fused + build_recurrent_attn,
            // !keep: publish the packed tail into cache slot kv_head before RMS.
            new_state = ggml_view_4d(context, packed, 128, 128, 48, 1,
                                    128 * sizeof(float), 128 * 128 * sizeof(float),
                                    state_elements * sizeof(float), 6144 * sizeof(float));
            state_cache = ggml_new_tensor_2d(context, GGML_TYPE_F32, state_elements, 3);
            state_destination = ggml_view_2d(context, state_cache, state_elements, 1,
                                            state_cache->nb[1], state_elements * sizeof(float));
            state_copy = ggml_cpy(context, new_state, state_destination);
            ggml_set_name(state_copy, "gdn_recurrent_state_publish");
            ggml_set_input(state_cache);
            ggml_set_output(state_cache);
            ggml_build_forward_expand(graph, state_copy);
            if (observe_y) {
                // Observation-only control: snapshot before last-use RMS can
                // reuse Y. The unobserved case below keeps exact model consumers.
                y_snapshot = ggml_cont(context, raw);
                ggml_set_name(y_snapshot, "gdn_recurrent_y_snapshot_control");
                ggml_set_output(y_snapshot);
                ggml_build_forward_expand(graph, y_snapshot);
            }
        }
        gamma = ggml_new_tensor_1d(context, GGML_TYPE_F32, 128);
        z_input = ggml_new_tensor_2d(context, GGML_TYPE_F32, projections ? 2560 : 6144, 1);
        if (projections) {
            REQUIRE(recurrent && !early && !silu && !observe_y);
            z_weight = ggml_new_tensor_2d(context, GGML_TYPE_Q8_0, 2560, 6144);
            output_weight = ggml_new_tensor_2d(context, GGML_TYPE_Q8_0, 6144, 2560);
            z_producer = ggml_mul_mat(context, z_weight, z_input);
            ggml_set_name(z_input, "gdn_shared_hidden_2560");
            // Model weights are persistent; the shared hidden input also stays
            // live for the residual branch outside this focused subgraph.
            for (auto * tensor : {z_weight, output_weight, z_input}) {
                ggml_set_input(tensor);
                ggml_set_output(tensor);
            }
        } else {
            // SCALE is the explicit surrogate in the original isolated controls.
            z_producer = ggml_scale(context, z_input, 0.75f);
        }
        z = ggml_reshape_4d(context, z_producer, 128, 48, 1, 1);
        ggml_set_name(raw, "gdn_norm_raw_view");
        ggml_set_name(z_producer, "gdn_norm_z_producer");
        ggml_set_name(z, "gdn_norm_z_reshape");
        if (early) {
            ggml_build_forward_expand(graph, z);
        }
        rms = ggml_rms_norm(context, raw, 1.0e-6f);
        weighted = ggml_mul(context, rms, gamma);
        // qwen4exp.cpp:696 uses SIGMOID. SILU is an explicit non-model control.
        activation = silu ? ggml_silu(context, z) : ggml_sigmoid(context, z);
        gated = ggml_mul(context, weighted, activation);
        output = ggml_reshape_3d(context, gated, 6144, 1, 1);
        ggml_set_name(rms, "gdn_norm_rms");
        ggml_set_name(weighted, "gdn_norm_gamma_mul");
        ggml_set_name(activation, silu ? "gdn_norm_silu_control" : "gdn_norm_sigmoid_model");
        ggml_set_name(gated, "gdn_norm_gated_mul");
        if (projections) {
            projected = ggml_mul_mat(context, output_weight, output);
            ggml_set_name(projected, "gdn_output_projection_2560");
            ggml_set_output(projected);
            if (retain_packed) {
                for (auto * tensor : {weighted, z_producer, activation, gated, output}) {
                    ggml_set_output(tensor);
                }
            }
        } else {
            ggml_set_output(output);
        }
        for (auto * tensor : { gamma, z_input }) {
            ggml_set_input(tensor);
        }
        // INPUT requests early allocation, not post-compute retention. OUTPUT is
        // gallocr's no-reuse/no-free contract. The model's transient GDN result
        // may instead die after its state was published and RMS consumed raw.
        if (retain_packed) {
            ggml_set_output(packed);
        }
        ggml_build_forward_expand(graph, projections ? projected : output);
    }

    void allocate(ggml_backend_t backend, bool reuse) {
        if (reuse) {
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            REQUIRE(allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph));
        } else {
            buffer = ggml_backend_alloc_ctx_tensors(context, backend);
            REQUIRE(buffer != nullptr);
        }
    }

    ~gdn_norm_fixture() {
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }
};

static void check_gdn_norm_plan(gdn_norm_fixture & fixture) {
    using namespace ggml::hrx;
    auto imported = import_ggml_graph(*fixture.graph);
    REQUIRE(imported.valid());
    auto & graph = imported.graph;
    const auto index = [&](ggml_tensor * tensor) { return qsa_node_index(graph, tensor); };
    const size_t root = index(fixture.rms);
    const size_t gamma_mul = index(fixture.weighted);
    const size_t z_producer = index(fixture.z_producer);
    const size_t gate = index(fixture.activation);
    const size_t final_mul = index(fixture.gated);
    REQUIRE(root < gamma_mul && gamma_mul < gate && gate < final_mul);
    REQUIRE((z_producer < root) == fixture.early);
    REQUIRE((index(fixture.z) < root) == fixture.early);
    if (!fixture.early) {
        REQUIRE(gamma_mul < z_producer && z_producer < index(fixture.z) && index(fixture.z) < gate);
    }
    const auto * raw = graph.values().find_tensor(fixture.raw);
    REQUIRE(raw && raw->contiguous && raw->ne[0] == 128 && raw->ne[1] == 48);
    REQUIRE(raw->nb[0] == sizeof(float) && raw->nb[1] == 128 * sizeof(float));
    REQUIRE(raw->storage_offset == fixture.prefix * sizeof(float));
    const auto * registry = find_dispatch_registry({ "gfx1151" });
    REQUIRE(registry != nullptr);
    std::vector<bool> covered(graph.nodes().size(), false);
    CommandPlan empty_plan;
    const ValueId next(static_cast<int32_t>(graph.values().size()));
    DispatchMatch match;
    DispatchMatchDiagnostics diagnostics;
    REQUIRE(registry->match({ graph, &graph.nodes()[root], root, covered, empty_plan, next }, match, &diagnostics));
    const bool fused_gate = fixture.early && !fixture.silu;
    REQUIRE(!diagnostics.attempts.empty());
    REQUIRE(diagnostics.attempts.back().name ==
            (fused_gate ? "qwen4exp.gdn_norm_gate_decode" : "qwen.rmsnorm_f32.mul_weight"));
    std::fprintf(stderr, "GDN norm RMS registration=%s\n", diagnostics.attempts.back().name.c_str());
    REQUIRE(match.dispatches.size() == 1);
    const auto & dispatch = match.dispatches.front();
    if (fused_gate) {
        REQUIRE(dispatch.kernel.kernel_id == kernel_catalog_id("qwen4exp", "qwen38_gdn_norm_gate_decode"));
        REQUIRE(match.covered_nodes.size() == 4);
    } else {
        REQUIRE(std::any_of(diagnostics.attempts.begin(), diagnostics.attempts.end(), [](const auto & attempt) {
            return attempt.name == "qwen4exp.gdn_norm_gate_decode" && !attempt.matched;
        }));
        REQUIRE(match.covered_nodes == (std::vector<size_t>{root, gamma_mul}));
        REQUIRE(dispatch.kernel.kernel_id == kernel_catalog_id("qwen3_moe", "qwen3_moe_rmsnorm_f32"));
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == 48);
        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.model.hidden_size") == "128");
        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.workload.token_capacity") == "48");
        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.model.rms_epsilon") == "0.000001");
        REQUIRE(dispatch.bindings.size() == 3 && dispatch.bindings[0].value == raw->id);
        REQUIRE(dispatch.bindings[1].value == graph.values().find_tensor(fixture.gamma)->id);
        REQUIRE(dispatch.bindings[2].value == graph.values().find_tensor(fixture.weighted)->id);
        for (const auto & binding : dispatch.bindings) {
            REQUIRE(binding.value != graph.values().find_tensor(fixture.z)->id);
        }
        DispatchMatch mul;
        REQUIRE(registry->match({ graph, &graph.nodes()[final_mul], final_mul, covered, empty_plan, next },
                                mul, &diagnostics));
        REQUIRE(diagnostics.attempts.back().name == "common.mul_f32");
        REQUIRE(mul.dispatches.size() == 1 && mul.covered_nodes == std::vector<size_t>{final_mul});
        REQUIRE(mul.dispatches[0].kernel.kernel_id == kernel_catalog_id("qwen3_moe", "ggml_mul_f32"));
        REQUIRE(mul.dispatches[0].kernel.integer_parameters.at("period") == 6144);
        REQUIRE(mul.dispatches[0].kernel.integer_parameters.at("outer_count") == 1);
        REQUIRE(mul.dispatches[0].bindings[0].value == graph.values().find_tensor(fixture.weighted)->id);
        REQUIRE(mul.dispatches[0].bindings[1].value == graph.values().find_tensor(fixture.activation)->id);
    }
    DispatchScheduler scheduler;
    if (!scheduler.schedule_graph(graph, { "gfx1151" })) {
        std::fprintf(stderr, "GDN norm full-split scheduling failed: %s\n", scheduler.error().c_str());
        REQUIRE(false);
    }
    size_t rms_dispatches = 0, mul_dispatches = 0;
    for (const auto & command : scheduler.dispatches()) {
        rms_dispatches += command.kernel.kernel_id == dispatch.kernel.kernel_id;
        mul_dispatches += command.kernel.kernel_id == kernel_catalog_id("qwen3_moe", "ggml_mul_f32");
    }
    REQUIRE(rms_dispatches == 1 && mul_dispatches == (fused_gate ? 0 : 1));
}

struct gdn_norm_oracle {
    std::vector<float> packed, gamma, z_input, weighted, activation, output;
};

static gdn_norm_oracle gdn_norm_inputs(const gdn_norm_fixture & fixture, int pass) {
    gdn_norm_oracle oracle;
    oracle.packed.resize(static_cast<size_t>(ggml_nelements(fixture.packed)), -17.25f - pass);
    oracle.gamma.resize(128);
    oracle.z_input.resize(6144);
    oracle.weighted.resize(6144);
    oracle.activation.resize(6144);
    oracle.output.resize(6144);
    for (int d = 0; d < 128; ++d) {
        oracle.gamma[d] = static_cast<float>((d * 13 + pass * 11) % 47 - 23) / 16.0f;
    }
    for (int h = 0; h < 48; ++h) {
        double sum = 0.0;
        for (int d = 0; d < 128; ++d) {
            const int i = h * 128 + d;
            const float magnitude = h < 2 ? 0.00001f : (h % 7 + 1) * 0.03125f;
            const float x = static_cast<float>((d * 17 + h * 11 + pass * 7) % 67 - 33) * magnitude;
            oracle.packed[fixture.prefix + i] = x;
            oracle.z_input[i] = static_cast<float>((d * 7 + h * 19 + pass * 13) % 97 - 48) / 8.0f;
            sum += static_cast<double>(x) * x;
        }
        const double inverse = 1.0 / std::sqrt(sum / 128.0 + static_cast<double>(1.0e-6f));
        for (int d = 0; d < 128; ++d) {
            const int i = h * 128 + d;
            const double weighted = oracle.packed[fixture.prefix + i] * inverse * oracle.gamma[d];
            const double z = 0.75 * oracle.z_input[i];
            const double activation = (fixture.silu ? z : 1.0) / (1.0 + std::exp(-z));
            oracle.weighted[i] = static_cast<float>(weighted);
            oracle.activation[i] = static_cast<float>(activation);
            oracle.output[i] = static_cast<float>(weighted * activation);
        }
    }
    return oracle;
}

static void upload_gdn_norm(gdn_norm_fixture & fixture, const gdn_norm_oracle & oracle) {
    ggml_backend_tensor_set(fixture.packed, oracle.packed.data(), 0, oracle.packed.size() * sizeof(float));
    ggml_backend_tensor_set(fixture.gamma, oracle.gamma.data(), 0, oracle.gamma.size() * sizeof(float));
    ggml_backend_tensor_set(fixture.z_input, oracle.z_input.data(), 0, oracle.z_input.size() * sizeof(float));
}

static std::vector<bool> gdn_norm_reused_elements(gdn_norm_fixture & fixture) {
    using namespace ggml::hrx;
    REQUIRE((fixture.packed->flags & GGML_TENSOR_FLAG_INPUT) != 0);
    REQUIRE(((fixture.packed->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) == fixture.retain_packed);
    std::vector<bool> rewritten(static_cast<size_t>(ggml_nelements(fixture.packed)), false);
    auto imported = import_ggml_graph(*fixture.graph);
    REQUIRE(imported.valid());
    auto & graph = imported.graph;
    const size_t last_read = qsa_node_index(graph, fixture.rms);
    DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(graph, { "gfx1151" }));
    const uintptr_t packed_begin = reinterpret_cast<uintptr_t>(fixture.packed->data);
    const uintptr_t packed_end = packed_begin + ggml_nbytes(fixture.packed);
    size_t overlaps = 0;
    for (const auto & dispatch : scheduler.dispatches()) {
        // These fixture kernels each have precisely one output, the last binding.
        const uint64_t id = dispatch.kernel.kernel_id;
        REQUIRE(id == kernel_catalog_id("qwen3_moe", "qwen3_moe_rmsnorm_f32") ||
                id == kernel_catalog_id("qwen4exp", "qwen38_gdn_norm_gate_decode") ||
                id == kernel_catalog_id("qwen3_moe", "ggml_mul_f32") ||
                id == kernel_catalog_id("qwen3_moe", "ggml_sigmoid_f32") ||
                id == kernel_catalog_id("qwen3_moe", "ggml_silu_f32") ||
                id == kernel_catalog_id("qwen3_moe", "ggml_scale_f32"));
        const auto & binding = dispatch.bindings.back();
        const auto * value = graph.values().find(binding.value);
        REQUIRE(value && value->tensor && value->tensor->data && value->contiguous);
        const uintptr_t write_begin = reinterpret_cast<uintptr_t>(value->tensor->data) + binding.offset;
        const uintptr_t write_end = write_begin + binding.length;
        const uintptr_t begin = std::max(packed_begin, write_begin);
        const uintptr_t end = std::min(packed_end, write_end);
        if (begin >= end) {
            continue;
        }
        const size_t writer = qsa_node_index(graph, value->tensor);
        // RMS may legally reuse its last-use input in place. Later nodes may
        // also use the released block, but an earlier writer is not legitimate.
        REQUIRE(!fixture.retain_packed && writer >= last_read);
        REQUIRE((begin - packed_begin) % sizeof(float) == 0 && (end - begin) % sizeof(float) == 0);
        ++overlaps;
        std::fprintf(stderr, "GDN packed planned reuse writer=%s node=%zu last_read=%zu floats=[%zu,%zu)\n",
                     value->tensor->name, writer, last_read,
                     static_cast<size_t>((begin - packed_begin) / sizeof(float)),
                     static_cast<size_t>((end - packed_begin) / sizeof(float)));
        for (size_t i = (begin - packed_begin) / sizeof(float); i < (end - packed_begin) / sizeof(float); ++i) {
            rewritten[i] = true;
        }
    }
    std::fprintf(stderr, "GDN packed retention=%s planned_overlaps=%zu\n",
                 fixture.retain_packed ? "OUTPUT" : "transient", overlaps);
    return rewritten;
}

static void check_gdn_norm_storage(gdn_norm_fixture & fixture, const gdn_norm_oracle & oracle,
                                   const std::vector<bool> & rewritten) {
    std::vector<float> actual(oracle.packed.size());
    ggml_backend_tensor_get(fixture.packed, actual.data(), 0, actual.size() * sizeof(float));
    size_t changed = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &oracle.packed[i], sizeof(float)) != 0) {
            if (fixture.retain_packed || !rewritten[i]) {
                std::fprintf(stderr, "GDN packed unexpected write element=%zu retained=%d actual=%g expected=%g\n",
                             i, fixture.retain_packed, actual[i], oracle.packed[i]);
                REQUIRE(false);
            }
            ++changed;
        }
    }
    std::fprintf(stderr, "GDN packed changed_elements=%zu (all within planned last-use reuse)\n", changed);
}

static void check_gdn_norm_tensor(ggml_tensor * tensor, const std::vector<float> & expected, const char * stage) {
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));
    float max_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || error > 2.0e-5f + 2.0e-5f * std::fabs(expected[i])) {
            std::fprintf(stderr, "GDN norm %s mismatch head=%zu column=%zu actual=%.9g expected=%.9g\n",
                         stage, i / 128, i % 128, actual[i], expected[i]);
            REQUIRE(false);
        }
        max_error = std::max(max_error, error);
    }
    std::fprintf(stderr, "GDN norm %s max_abs_error=%g\n", stage, max_error);
}

struct gdn_norm_env {
    const char * name;
    bool existed;
    std::string previous;
    static void set(const char * name, const char * value) {
#ifdef _WIN32
        REQUIRE(_putenv_s(name, value ? value : "") == 0);
#else
        REQUIRE((value ? setenv(name, value, 1) : unsetenv(name)) == 0);
#endif
    }
    gdn_norm_env(const char * key, const char * value) : name(key), existed(std::getenv(key) != nullptr),
        previous(existed ? std::getenv(key) : "") { set(name, value); }
    ~gdn_norm_env() { set(name, existed ? previous.c_str() : nullptr); }
};

static void run_gdn_norm_checks(ggml_backend_t backend) {
    gdn_norm_env norm_flag("HRX_ENABLE_GDN_NORM_GATE", "1");
    gdn_norm_env mul_generic("HRX_MUL_CLAIM_GENERIC", "1");
    gdn_norm_env mul_mask("HRX_MUL_CLAIM_MASK", "0x7F");
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(cpu != nullptr);
    // Retained controls must preserve every input byte; the transient case keeps
    // the actual model lifetime and validates only allocator-authorized reuse.
    for (int allocation : {0, 1, 2}) {
        const bool reuse = allocation != 0;
        const bool retain = allocation != 2;
        for (bool early : {false, true}) {
            for (bool silu : {false, true}) {
                for (size_t prefix : {size_t(0), size_t(256)}) {
                    std::fprintf(stderr, "GDN norm alloc=%s z=%s gate=%s prefix=%zu retain=%d\n",
                                 reuse ? "gallocr" : "context", early ? "ready" : "late/model-order",
                                 silu ? "SILU-control" : "SIGMOID-model", prefix, retain);
                    gdn_norm_fixture gpu_fixture(early, silu, prefix, retain), cpu_fixture(early, silu, prefix, retain);
                    check_gdn_norm_plan(gpu_fixture);
                    for (auto * op : {gpu_fixture.rms, gpu_fixture.weighted, gpu_fixture.activation, gpu_fixture.gated}) {
                        REQUIRE(ggml_backend_supports_op(backend, op));
                    }
                    gpu_fixture.allocate(backend, reuse);
                    cpu_fixture.allocate(cpu, reuse);
                    // Validate the allocated original views too, not merely a shape-only graph.
                    check_gdn_norm_plan(gpu_fixture);
                    const auto rewritten = gdn_norm_reused_elements(gpu_fixture);
                    const std::vector<ggml_tensor *> original_order(
                        ggml_graph_nodes(gpu_fixture.graph),
                        ggml_graph_nodes(gpu_fixture.graph) + ggml_graph_n_nodes(gpu_fixture.graph));
                    for (int pass : {0, 1, 2, 0}) {
                        std::fprintf(stderr, "GDN norm replay input=%d\n", pass);
                        const auto oracle = gdn_norm_inputs(gpu_fixture, pass);
                        upload_gdn_norm(gpu_fixture, oracle);
                        upload_gdn_norm(cpu_fixture, oracle);
                        REQUIRE(ggml_backend_graph_compute(cpu, cpu_fixture.graph) == GGML_STATUS_SUCCESS);
                        check_gdn_norm_tensor(cpu_fixture.output, oracle.output, "CPU/double");
                        REQUIRE(ggml_backend_graph_compute(backend, gpu_fixture.graph) == GGML_STATUS_SUCCESS);
                        REQUIRE(ggml_graph_n_nodes(gpu_fixture.graph) == static_cast<int>(original_order.size()));
                        for (size_t i = 0; i < original_order.size(); ++i) {
                            REQUIRE(ggml_graph_node(gpu_fixture.graph, static_cast<int>(i)) == original_order[i]);
                        }
                        if (!reuse && (!early || silu)) {
                            check_gdn_norm_tensor(gpu_fixture.weighted, oracle.weighted, "HRX RMS*gamma/double");
                            check_gdn_norm_tensor(gpu_fixture.activation, oracle.activation, "HRX activation/double");
                        }
                        check_gdn_norm_tensor(gpu_fixture.output, oracle.output, "HRX/double");
                        std::vector<float> cpu_output(6144);
                        ggml_backend_tensor_get(cpu_fixture.output, cpu_output.data(), 0, cpu_output.size() * sizeof(float));
                        check_gdn_norm_tensor(gpu_fixture.output, cpu_output, "HRX/CPU");
                        check_gdn_norm_storage(gpu_fixture, oracle, rewritten);
                        if (retain) {
                            check_gdn_norm_storage(cpu_fixture, oracle, rewritten);
                        }
                    }
                }
            }
        }
    }
    ggml_backend_free(cpu);
}

static void check_gdn_recurrent_plan(gdn_norm_fixture & fixture) {
    using namespace ggml::hrx;
    REQUIRE(fixture.recurrent);
    check_gdn_norm_plan(fixture);
    auto imported = import_ggml_graph(*fixture.graph);
    REQUIRE(imported.valid());
    auto & graph = imported.graph;
    const size_t root = qsa_node_index(graph, fixture.packed);
    const size_t publish = qsa_node_index(graph, fixture.state_copy);
    const size_t rms = qsa_node_index(graph, fixture.rms);
    REQUIRE(root < qsa_node_index(graph, fixture.new_state));
    REQUIRE(qsa_node_index(graph, fixture.new_state) < publish && publish < rms);
    REQUIRE(publish < qsa_node_index(graph, fixture.raw));
    REQUIRE(fixture.raw->view_src == fixture.packed && fixture.raw->view_offs == 0);
    REQUIRE(fixture.new_state->view_src == fixture.packed &&
            fixture.new_state->view_offs == 6144 * sizeof(float));
    REQUIRE(fixture.state_copy->src[0] == fixture.new_state);
    REQUIRE(fixture.state_copy->src[1] == fixture.state_destination);
    REQUIRE(fixture.state_destination->view_src == fixture.state_cache);
    REQUIRE(fixture.state_destination->view_offs == ggml_nbytes(fixture.state));
    const auto * packed = graph.values().find_tensor(fixture.packed);
    const auto * raw = graph.values().find_tensor(fixture.raw);
    const auto * tail = graph.values().find_tensor(fixture.new_state);
    const auto * v = graph.values().find_tensor(fixture.v);
    REQUIRE(packed && raw && tail && v);
    REQUIRE(packed->byte_count == 6144 * sizeof(float) + ggml_nbytes(fixture.state));
    REQUIRE(raw->storage == packed->storage && tail->storage == packed->storage);
    REQUIRE(raw->storage_offset == 0 && tail->storage_offset == 6144 * sizeof(float));
    REQUIRE(v->storage_offset == 4096 * sizeof(float) && v->contiguous);
    const auto * registry = find_dispatch_registry({ "gfx1151" });
    REQUIRE(registry != nullptr);
    std::vector<bool> covered(graph.nodes().size(), false);
    CommandPlan empty_plan;
    DispatchMatch match;
    DispatchMatchDiagnostics diagnostics;
    const ValueId next(static_cast<int32_t>(graph.values().size()));
    REQUIRE(registry->match({graph, &graph.nodes()[root], root, covered, empty_plan, next},
                            match, &diagnostics));
    REQUIRE(!diagnostics.attempts.empty() &&
            diagnostics.attempts.back().name == "qwen4exp.gdn_recurrent_decode_split");
    REQUIRE(match.covered_nodes == std::vector<size_t>{root} && match.dispatches.size() == 1);
    const auto & core = match.dispatches.front();
    REQUIRE(core.kernel.kernel_id == kernel_catalog_id("qwen4exp", "qwen38_gdn_recurrent_decode_split"));
    REQUIRE(core.kernel.integer_parameters.empty());
    REQUIRE(core.bindings.size() == 8);
    const std::array<ggml_tensor *, 6> inputs = {
        fixture.q, fixture.k, fixture.v, fixture.decay, fixture.beta, fixture.state};
    for (size_t i = 0; i < inputs.size(); ++i) {
        REQUIRE(core.bindings[i].value == graph.values().find_tensor(inputs[i])->id);
        REQUIRE(core.bindings[i].offset == 0 && core.bindings[i].length == ggml_nbytes(inputs[i]));
    }
    REQUIRE(core.bindings[6].value == packed->id && core.bindings[6].offset == tail->storage_offset &&
            core.bindings[6].length == ggml_nbytes(fixture.state));
    REQUIRE(core.bindings[7].value == packed->id && core.bindings[7].offset == 0 &&
            core.bindings[7].length == ggml_nbytes(fixture.raw));
    DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(graph, {"gfx1151"}));
    size_t core_count = 0, publish_count = 0, norm_count = 0;
    for (const auto & dispatch : scheduler.dispatches()) {
        if (dispatch.kernel.kernel_id == core.kernel.kernel_id) {
            REQUIRE(publish_count == 0 && norm_count == 0);
            ++core_count;
        }
        if (!dispatch.bindings.empty() &&
            dispatch.bindings.back().value == graph.values().find_tensor(fixture.state_copy)->id) {
            REQUIRE(core_count == 1 && norm_count == 0);
            REQUIRE(dispatch.kernel.kernel_id == kernel_catalog_id("qwen3_moe", "ggml_copy_f32"));
            REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 128 * 128 * 48);
            REQUIRE(dispatch.bindings.front().value == tail->id);
            ++publish_count;
        }
        if (dispatch.kernel.kernel_id == kernel_catalog_id("qwen3_moe", "qwen3_moe_rmsnorm_f32") ||
            dispatch.kernel.kernel_id == kernel_catalog_id("qwen4exp", "qwen38_gdn_norm_gate_decode")) {
            REQUIRE(core_count == 1 && publish_count == 1);
            REQUIRE(dispatch.bindings.front().value == raw->id);
            ++norm_count;
        }
    }
    REQUIRE(core_count == 1 && publish_count == 1 && norm_count == 1);
    if (fixture.packed->data) {
        REQUIRE(fixture.raw->data == fixture.packed->data);
        REQUIRE(static_cast<char *>(fixture.new_state->data) ==
                static_cast<char *>(fixture.packed->data) + tail->storage_offset);
        REQUIRE(static_cast<char *>(fixture.state_copy->data) ==
                static_cast<char *>(fixture.state_cache->data) + ggml_nbytes(fixture.state));
        std::fprintf(stderr, "GDN recurrent allocated packed=%p Y=%p tail=%p state-in=%p publish=%p RMS-out=%p\n",
                     fixture.packed->data, fixture.raw->data, fixture.new_state->data,
                     fixture.state->data, fixture.state_copy->data, fixture.weighted->data);
    }
}

struct gdn_recurrent_oracle {
    gdn_norm_oracle norm;
    std::vector<float> q, k, v_storage, decay, beta, state, cache_input, y, new_state;
};

static gdn_recurrent_oracle gdn_recurrent_inputs(const gdn_norm_fixture & fixture, int pass) {
    gdn_recurrent_oracle oracle;
    oracle.norm = gdn_norm_inputs(fixture, pass);
    oracle.q.resize(2048);
    oracle.k.resize(2048);
    oracle.v_storage.resize(10240, -9.75f - pass);
    oracle.decay.resize(48);
    oracle.beta.resize(48);
    oracle.state.resize(128 * 128 * 48);
    oracle.new_state.resize(oracle.state.size());
    oracle.y.resize(6144);
    oracle.cache_input.resize(3 * oracle.state.size(), -23.5f - pass);
    for (int h = 0; h < 16; ++h) {
        double q_squared = 0.0, k_squared = 0.0;
        for (int d = 0; d < 128; ++d) {
            const size_t i = h * 128 + d;
            oracle.q[i] = static_cast<float>((d * 7 + h * 11 + pass * 13) % 43 - 21) / 32.0f;
            oracle.k[i] = static_cast<float>((d * 13 + h * 19 + pass * 7) % 53 - 26) / 32.0f;
            q_squared += static_cast<double>(oracle.q[i]) * oracle.q[i];
            k_squared += static_cast<double>(oracle.k[i]) * oracle.k[i];
        }
        for (int d = 0; d < 128; ++d) {
            oracle.q[h * 128 + d] = static_cast<float>(oracle.q[h * 128 + d] / std::sqrt(q_squared));
            oracle.k[h * 128 + d] = static_cast<float>(oracle.k[h * 128 + d] / std::sqrt(k_squared));
        }
    }
    for (int h = 0; h < 48; ++h) {
        oracle.decay[h] = -0.015625f * (1 + (h * 7 + pass * 11) % 63);
        oracle.beta[h] = static_cast<float>((h * 13 + pass * 17) % 65) / 64.0f;
        const double decay = std::exp(static_cast<double>(oracle.decay[h]));
        double y_squared = 0.0;
        std::array<double, 128> y = {};
        // ggml GDN repeats Q/K cyclically, h % 16, NOT consecutive groups h / 3.
        const int qk_head = h % 16;
        for (int j = 0; j < 128; ++j) {
            const size_t row = (h * 128 + j) * 128;
            const size_t out = h * 128 + j;
            const float value = static_cast<float>((j * 17 + h * 23 + pass * 19) % 79 - 39) / 32.0f;
            oracle.v_storage[4096 + out] = value;
            double prediction = 0.0;
            for (int i = 0; i < 128; ++i) {
                oracle.state[row + i] =
                    static_cast<float>((i * 11 + j * 7 + h * 13 + pass * 17) % 101 - 50) / 512.0f;
                prediction += decay * oracle.state[row + i] * oracle.k[qk_head * 128 + i];
            }
            const double delta = (value - prediction) * oracle.beta[h];
            for (int i = 0; i < 128; ++i) {
                const double updated = decay * oracle.state[row + i] + delta * oracle.k[qk_head * 128 + i];
                oracle.new_state[row + i] = static_cast<float>(updated);
                y[j] += updated * oracle.q[qk_head * 128 + i];
            }
            y[j] /= std::sqrt(128.0);
            oracle.y[out] = static_cast<float>(y[j]);
            y_squared += y[j] * y[j];
        }
        const double inverse = 1.0 / std::sqrt(y_squared / 128.0 + static_cast<double>(1.0e-6f));
        for (int j = 0; j < 128; ++j) {
            const size_t out = h * 128 + j;
            const double weighted = y[j] * inverse * oracle.norm.gamma[j];
            const double activation = 1.0 / (1.0 + std::exp(-0.75 * oracle.norm.z_input[out]));
            oracle.norm.weighted[out] = static_cast<float>(weighted);
            oracle.norm.activation[out] = static_cast<float>(activation);
            oracle.norm.output[out] = static_cast<float>(weighted * activation);
        }
    }
    std::copy(oracle.y.begin(), oracle.y.end(), oracle.norm.packed.begin());
    std::copy(oracle.new_state.begin(), oracle.new_state.end(), oracle.norm.packed.begin() + 6144);
    return oracle;
}

static void upload_gdn_recurrent(gdn_norm_fixture & fixture, const gdn_recurrent_oracle & oracle) {
    const std::array<std::pair<ggml_tensor *, const std::vector<float> *>, 9> inputs = {{
        {fixture.q, &oracle.q}, {fixture.k, &oracle.k}, {fixture.v_storage, &oracle.v_storage},
        {fixture.decay, &oracle.decay}, {fixture.beta, &oracle.beta}, {fixture.state, &oracle.state},
        {fixture.state_cache, &oracle.cache_input}, {fixture.gamma, &oracle.norm.gamma},
        {fixture.z_input, &oracle.norm.z_input},
    }};
    for (const auto & input : inputs) {
        REQUIRE(ggml_nbytes(input.first) == input.second->size() * sizeof(float));
        ggml_backend_tensor_set(input.first, input.second->data(), 0, ggml_nbytes(input.first));
    }
    // Never upload packed Y/state: the recurrent dispatch must produce both.
}

static void check_gdn_recurrent_results(gdn_norm_fixture & fixture, const gdn_recurrent_oracle & oracle,
                                        bool check_norm = true) {
    if (fixture.retain_packed) {
        check_gdn_norm_tensor(fixture.raw, oracle.y, "recurrent Y/double");
        check_gdn_norm_tensor(fixture.new_state, oracle.new_state, "recurrent packed tail/double");
        std::vector<float> old_state(oracle.state.size());
        ggml_backend_tensor_get(fixture.state, old_state.data(), 0, ggml_nbytes(fixture.state));
        REQUIRE(old_state == oracle.state);
        std::vector<float> tail(oracle.state.size()), published(oracle.state.size());
        ggml_backend_tensor_get(fixture.new_state, tail.data(), 0, ggml_nbytes(fixture.new_state));
        ggml_backend_tensor_get(fixture.state_copy, published.data(), 0, ggml_nbytes(fixture.state_copy));
        REQUIRE(tail == published);
    } else if (fixture.y_snapshot) {
        check_gdn_norm_tensor(fixture.y_snapshot, oracle.y, "recurrent Y snapshot/double");
    }
    check_gdn_norm_tensor(fixture.state_copy, oracle.new_state, "recurrent published state/double");
    if (check_norm) {
        check_gdn_norm_tensor(fixture.output, oracle.norm.output, "recurrent norm+gate/double");
    }
    std::vector<float> cache(oracle.cache_input.size());
    ggml_backend_tensor_get(fixture.state_cache, cache.data(), 0, ggml_nbytes(fixture.state_cache));
    const size_t n = oracle.state.size();
    REQUIRE(std::memcmp(cache.data(), oracle.cache_input.data(), n * sizeof(float)) == 0);
    REQUIRE(std::memcmp(cache.data() + 2 * n, oracle.cache_input.data() + 2 * n, n * sizeof(float)) == 0);
}

static void run_gdn_recurrent_checks(ggml_backend_t backend) {
    gdn_norm_env norm_flag("HRX_ENABLE_GDN_NORM_GATE", "1");
    gdn_norm_env mul_generic("HRX_MUL_CLAIM_GENERIC", "1");
    gdn_norm_env mul_mask("HRX_MUL_CLAIM_MASK", "0x01");
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(cpu != nullptr);
    // Retained observations, last-use Y snapshot control, then exact unobserved
    // model consumers. No transient readback after RMS is asserted to preserve Y.
    for (int allocation : {0, 1, 2, 3}) {
        const bool reuse = allocation != 0, retain = allocation < 2, observe = allocation == 2;
        for (bool early : {false, true}) {
            std::fprintf(stderr, "GDN recurrent T1 K1 QK=16 V=48 alloc=%s retain=%d snapshot=%d z=%s\n",
                         reuse ? "gallocr" : "context", retain, observe, early ? "ready" : "late/model-order");
            gdn_norm_fixture gpu_fixture(early, false, 0, retain, true, observe);
            gdn_norm_fixture cpu_fixture(early, false, 0, retain, true, observe);
            check_gdn_recurrent_plan(gpu_fixture);
            for (auto * op : {gpu_fixture.packed, gpu_fixture.state_copy, gpu_fixture.rms,
                             gpu_fixture.weighted, gpu_fixture.activation, gpu_fixture.gated}) {
                REQUIRE(ggml_backend_supports_op(backend, op));
            }
            gpu_fixture.allocate(backend, reuse);
            cpu_fixture.allocate(cpu, reuse);
            check_gdn_recurrent_plan(gpu_fixture);
            const std::vector<ggml_tensor *> original_order(
                ggml_graph_nodes(gpu_fixture.graph),
                ggml_graph_nodes(gpu_fixture.graph) + ggml_graph_n_nodes(gpu_fixture.graph));
            for (int pass : {0, 1, 2, 0}) {
                std::fprintf(stderr, "GDN recurrent changed-input replay=%d CPU\n", pass);
                const auto oracle = gdn_recurrent_inputs(gpu_fixture, pass);
                upload_gdn_recurrent(cpu_fixture, oracle);
                upload_gdn_recurrent(gpu_fixture, oracle);
                REQUIRE(ggml_backend_graph_compute(cpu, cpu_fixture.graph) == GGML_STATUS_SUCCESS);
                check_gdn_recurrent_results(cpu_fixture, oracle);
                std::fprintf(stderr, "GDN recurrent changed-input replay=%d HRX\n", pass);
                REQUIRE(ggml_backend_graph_compute(backend, gpu_fixture.graph) == GGML_STATUS_SUCCESS);
                check_gdn_recurrent_results(gpu_fixture, oracle);
                for (const auto & pair : {
                    std::make_pair(cpu_fixture.output, gpu_fixture.output),
                    std::make_pair(cpu_fixture.state_copy, gpu_fixture.state_copy),
                    std::make_pair(retain ? cpu_fixture.raw : cpu_fixture.y_snapshot,
                                   retain ? gpu_fixture.raw : gpu_fixture.y_snapshot)}) {
                    if (!pair.first) {
                        continue;
                    }
                    std::vector<float> expected(static_cast<size_t>(ggml_nelements(pair.first)));
                    ggml_backend_tensor_get(pair.first, expected.data(), 0, expected.size() * sizeof(float));
                    check_gdn_norm_tensor(pair.second, expected, "recurrent HRX/CPU");
                }
                REQUIRE(ggml_graph_n_nodes(gpu_fixture.graph) == static_cast<int>(original_order.size()));
                for (size_t i = 0; i < original_order.size(); ++i) {
                    REQUIRE(ggml_graph_node(gpu_fixture.graph, static_cast<int>(i)) == original_order[i]);
                }
            }
        }
    }
    ggml_backend_free(cpu);
}

static std::vector<uint8_t> gdn_projection_weight(int width, int rows, uint32_t seed) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    REQUIRE(traits && traits->from_float_ref && traits->to_float);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, width);
    std::vector<uint8_t> bytes(row_bytes * rows);
    std::vector<float> row(width);
    const float magnitude = 0.75f / std::sqrt(static_cast<float>(width));
    for (int r = 0; r < rows; ++r) {
        for (float & value : row) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            value = (static_cast<float>(seed & 0xffff) / 32768.0f - 1.0f) * magnitude;
        }
        traits->from_float_ref(row.data(), bytes.data() + r * row_bytes, width);
    }
    return bytes;
}

struct gdn_projection_reference {
    std::vector<float> raw, cpu, wmma;
    std::vector<float> legacy_bound;
};

static gdn_projection_reference gdn_projection_dot(const std::vector<uint8_t> & weight,
                                                   const gdn_projection_reference & input, int rows) {
    const size_t width = input.raw.size();
    REQUIRE(width > 0 && width % 32 == 0 && input.cpu.size() == width && input.wmma.size() == width);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, width);
    REQUIRE(weight.size() == row_bytes * rows);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<uint8_t> quantized(row_bytes);
    std::vector<float> cpu_input(width), half_input(width), row(width);
    // CPU Q8_0 MUL_MAT quantizes its activation too. Keep that reference
    // separate from dequantized-weight/F32 math and F16-rounded WMMA operands.
    for (size_t i = 0; i < width; ++i) {
        REQUIRE(std::isfinite(input.raw[i]) && std::isfinite(input.cpu[i]) && std::isfinite(input.wmma[i]));
        half_input[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(input.wmma[i]));
    }
    traits->from_float_ref(input.cpu.data(), quantized.data(), width);
    traits->to_float(quantized.data(), cpu_input.data(), width);
    gdn_projection_reference result;
    result.raw.resize(rows);
    result.cpu.resize(rows);
    result.wmma.resize(rows);
    result.legacy_bound.resize(rows);
    for (int r = 0; r < rows; ++r) {
        traits->to_float(weight.data() + r * row_bytes, row.data(), width);
        double raw = 0.0, cpu = 0.0, wmma = 0.0, bound = 0.0, block_absolute = 0.0;
        for (size_t i = 0; i < width; ++i) {
            const float half_weight = ggml_fp16_to_fp32(ggml_fp32_to_fp16(row[i]));
            raw += static_cast<double>(row[i]) * input.raw[i];
            cpu += static_cast<double>(row[i]) * cpu_input[i];
            const double product = static_cast<double>(half_weight) * half_input[i];
            wmma += product;
            block_absolute += std::fabs(product);
            if ((i + 1) % 16 == 0) {
                // Legacy vector<8xf16> carries one rounded accumulator through
                // each 16-term MMA. Allow one half ULP (including directed
                // rounding), propagated carry error, FP32 reduction noise and
                // a half-subnormal ULP. This is a bound, not an ISA emulator.
                constexpr double half_ulp = 1.0 / 1024.0;
                constexpr double half_subnormal = 1.0 / 16777216.0;
                const double reduction = 16.0 * std::numeric_limits<float>::epsilon() *
                                         (block_absolute + std::fabs(wmma) + bound);
                bound = (1.0 + half_ulp) * (bound + reduction) +
                        half_ulp * std::fabs(wmma) + half_subnormal;
                block_absolute = 0.0;
            }
        }
        result.raw[r] = static_cast<float>(raw);
        result.cpu[r] = static_cast<float>(cpu);
        result.wmma[r] = static_cast<float>(wmma);
        result.legacy_bound[r] = static_cast<float>(bound);
    }
    return result;
}

static void check_gdn_projection_tensor(ggml_tensor * tensor, const std::vector<float> & expected,
                                         const char * stage, float abs_tolerance, float rel_tolerance,
                                         const std::vector<float> * bounds = nullptr, bool enforce = true) {
    REQUIRE(static_cast<size_t>(ggml_nelements(tensor)) == expected.size());
    REQUIRE(bounds == nullptr || bounds->size() == expected.size());
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, ggml_nbytes(tensor));
    double error_squared = 0.0, reference_squared = 0.0;
    float maximum = 0.0f, max_ratio = 0.0f, max_bound = 0.0f;
    size_t failures = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            std::fprintf(stderr, "GDN projection %s NONFINITE index=%zu actual=%g expected=%g\n",
                         stage, i, actual[i], expected[i]);
            REQUIRE(false);
        }
        const float error = std::fabs(actual[i] - expected[i]);
        const float bound = bounds ? (*bounds)[i] : 0.0f;
        REQUIRE(std::isfinite(bound) && bound >= 0.0f);
        const float allowed = abs_tolerance + rel_tolerance * std::fabs(expected[i]) + bound;
        max_bound = std::max(max_bound, bound);
        if (error > allowed) {
            if (failures == 0) {
                std::fprintf(stderr, "GDN projection %s first mismatch index=%zu actual=%.9g expected=%.9g limit=%g\n",
                             stage, i, actual[i], expected[i], allowed);
            }
            ++failures;
        }
        maximum = std::max(maximum, error);
        max_ratio = std::max(max_ratio, error / allowed);
        error_squared += static_cast<double>(error) * error;
        reference_squared += static_cast<double>(expected[i]) * expected[i];
    }
    std::fprintf(stderr, "GDN projection %s max_abs=%g rms_error=%g reference_rms=%g max_tolerance_ratio=%g max_bound=%g bad=%zu enforced=%d\n",
                 stage, maximum, std::sqrt(error_squared / actual.size()),
                 std::sqrt(reference_squared / actual.size()), max_ratio, max_bound, failures, enforce);
    REQUIRE(!enforce || failures == 0);
}

static std::vector<float> gdn_projection_read_finite(ggml_tensor * tensor, bool require_half = false) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            std::fprintf(stderr, "GDN projection %s NONFINITE index=%zu value=%g\n", tensor->name, i, values[i]);
            REQUIRE(false);
        }
        if (require_half) {
            REQUIRE(values[i] == ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[i])));
        }
    }
    return values;
}

static void check_gdn_projection_plan(gdn_norm_fixture & fixture, bool f32_accum) {
    using namespace ggml::hrx;
    REQUIRE(fixture.projections && !fixture.early);
    check_gdn_recurrent_plan(fixture);
    auto imported = import_ggml_graph(*fixture.graph);
    REQUIRE(imported.valid());
    auto & graph = imported.graph;
    const auto * registry = find_dispatch_registry({"gfx1151"});
    REQUIRE(registry != nullptr);
    const uint64_t expected_kernel = f32_accum ?
        kernel_catalog_id("hrx_owned", "ggml_dense_q8_0_f32_accum") :
        kernel_catalog_id("qwen3_moe", "qwen3_moe_dense_linear_q8_0_f16_wmma");
    std::vector<bool> covered(graph.nodes().size(), false);
    CommandPlan empty_plan;
    const ValueId next(static_cast<int32_t>(graph.values().size()));
    REQUIRE(qsa_node_index(graph, fixture.weighted) < qsa_node_index(graph, fixture.z_producer));
    REQUIRE(qsa_node_index(graph, fixture.gated) < qsa_node_index(graph, fixture.output));
    REQUIRE(qsa_node_index(graph, fixture.output) < qsa_node_index(graph, fixture.projected));
    for (auto * op : {fixture.z_producer, fixture.projected}) {
        const size_t root = qsa_node_index(graph, op);
        DispatchMatch match;
        DispatchMatchDiagnostics diagnostics;
        REQUIRE(registry->match({graph, &graph.nodes()[root], root, covered, empty_plan, next},
                                match, &diagnostics));
        REQUIRE(match.covered_nodes == std::vector<size_t>{root} && match.dispatches.size() == 1);
        REQUIRE(!diagnostics.attempts.empty() &&
                diagnostics.attempts.back().name == "llm.matmul.dense_q8_0_f16_wmma");
        const auto & dispatch = match.dispatches.front();
        REQUIRE(dispatch.kernel.kernel_id == expected_kernel);
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == 1);
        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.input_size") ==
                std::to_string(op->src[0]->ne[0]));
        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_size") ==
                std::to_string(op->ne[0]));
        if (f32_accum) {
            REQUIRE(dispatch.kernel.compile_parameters.count("qwen3_moe.dense_quantized.output_accumulation") == 0);
        } else {
            REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_accumulation") == "0");
        }
        REQUIRE(dispatch.bindings.size() == 3);
        const std::array<ggml_tensor *, 3> tensors = {op->src[1], op->src[0], op};
        for (size_t i = 0; i < tensors.size(); ++i) {
            REQUIRE(dispatch.bindings[i].value == graph.values().find_tensor(tensors[i])->id);
            REQUIRE(dispatch.bindings[i].offset == 0 && dispatch.bindings[i].length == ggml_nbytes(tensors[i]));
        }
        std::fprintf(stderr, "GDN projection accumulation=%s registration=%s %lld->%lld input=%p output=%p\n",
                     f32_accum ? "F32-strict" : "F16-legacy-bounded",
                     diagnostics.attempts.back().name.c_str(), static_cast<long long>(op->src[0]->ne[0]),
                     static_cast<long long>(op->ne[0]), op->src[1]->data, op->data);
    }
    DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(graph, {"gfx1151"}));
    size_t projections = 0;
    for (const auto & dispatch : scheduler.dispatches()) {
        projections += dispatch.kernel.kernel_id == expected_kernel;
    }
    REQUIRE(projections == 2);
}

static void run_gdn_projection_mode(ggml_backend_t backend, bool f32_accum) {
    gdn_norm_env norm_flag("HRX_ENABLE_GDN_NORM_GATE", "1");
    gdn_norm_env mul_generic("HRX_MUL_CLAIM_GENERIC", "1");
    gdn_norm_env mul_mask("HRX_MUL_CLAIM_MASK", "0x01");
    gdn_norm_env gemv("HRX_ENABLE_Q8_GEMV", "0");
    gdn_norm_env accumulation("HRX_ENABLE_DENSE_F32_ACCUM", f32_accum ? "1" : "0");
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(cpu != nullptr);
    const auto z_weight = gdn_projection_weight(2560, 6144, 0x1729u);
    const auto output_weight = gdn_projection_weight(6144, 2560, 0x314159u);
    // Context-retained legacy results anchor replay and unobserved/gallocr
    // comparisons without reading intermediates past their allocated lifetimes.
    std::array<std::vector<float>, 2> retained_legacy_outputs;
    for (bool reuse : {false, true}) {
        for (bool retain : {true, false}) {
            std::fprintf(stderr, "GDN Q8 projections accumulation=%s T1 Z=2560->6144 output=6144->2560 alloc=%s retained=%d Z=late\n",
                         f32_accum ? "F32-strict" : "F16-legacy-bounded",
                         reuse ? "gallocr" : "context", retain);
            gdn_norm_fixture gpu_fixture(false, false, 0, retain, true, false, true);
            gdn_norm_fixture cpu_fixture(false, false, 0, retain, true, false, true);
            check_gdn_projection_plan(gpu_fixture, f32_accum);
            for (auto * op : {gpu_fixture.packed, gpu_fixture.state_copy, gpu_fixture.rms, gpu_fixture.weighted,
                             gpu_fixture.z_producer, gpu_fixture.activation, gpu_fixture.gated, gpu_fixture.projected}) {
                REQUIRE(ggml_backend_supports_op(backend, op));
            }
            gpu_fixture.allocate(backend, reuse);
            cpu_fixture.allocate(cpu, reuse);
            check_gdn_projection_plan(gpu_fixture, f32_accum);
            for (auto * fixture : {&gpu_fixture, &cpu_fixture}) {
                ggml_backend_tensor_set(fixture->z_weight, z_weight.data(), 0, z_weight.size());
                ggml_backend_tensor_set(fixture->output_weight, output_weight.data(), 0, output_weight.size());
            }
            const std::vector<ggml_tensor *> original_order(
                ggml_graph_nodes(gpu_fixture.graph),
                ggml_graph_nodes(gpu_fixture.graph) + ggml_graph_n_nodes(gpu_fixture.graph));
            for (int pass : {0, 1, 0}) {
                std::fprintf(stderr, "GDN Q8 projection changed-input replay=%d\n", pass);
                auto oracle = gdn_recurrent_inputs(gpu_fixture, pass);
                oracle.norm.z_input.resize(2560);
                for (size_t i = 0; i < oracle.norm.z_input.size(); ++i) {
                    oracle.norm.z_input[i] =
                        static_cast<float>(static_cast<int>((i * 37 + i / 17 + pass * 197) % 1009) - 504) * 0.0017f;
                }
                const gdn_projection_reference hidden = {
                    oracle.norm.z_input, oracle.norm.z_input, oracle.norm.z_input, {}};
                const auto z = gdn_projection_dot(z_weight, hidden, 6144);
                gdn_projection_reference activation, gate;
                const std::array<const std::vector<float> *, 3> z_modes = {&z.raw, &z.cpu, &z.wmma};
                const std::array<std::vector<float> *, 3> activation_modes = {
                    &activation.raw, &activation.cpu, &activation.wmma};
                const std::array<std::vector<float> *, 3> gate_modes = {&gate.raw, &gate.cpu, &gate.wmma};
                for (size_t mode = 0; mode < 3; ++mode) {
                    activation_modes[mode]->resize(6144);
                    gate_modes[mode]->resize(6144);
                    for (size_t i = 0; i < 6144; ++i) {
                        const double sigmoid = 1.0 / (1.0 + std::exp(-static_cast<double>((*z_modes[mode])[i])));
                        (*activation_modes[mode])[i] = static_cast<float>(sigmoid);
                        (*gate_modes[mode])[i] = static_cast<float>(oracle.norm.weighted[i] * sigmoid);
                    }
                }
                const auto projected = gdn_projection_dot(output_weight, gate, 2560);
                upload_gdn_recurrent(cpu_fixture, oracle);
                upload_gdn_recurrent(gpu_fixture, oracle);
                REQUIRE(ggml_backend_graph_compute(cpu, cpu_fixture.graph) == GGML_STATUS_SUCCESS);
                std::fprintf(stderr, "GDN Q8 CPU state checks\n");
                check_gdn_recurrent_results(cpu_fixture, oracle, false);
                REQUIRE(ggml_backend_graph_compute(backend, gpu_fixture.graph) == GGML_STATUS_SUCCESS);
                std::fprintf(stderr, "GDN Q8 HRX state checks\n");
                check_gdn_recurrent_results(gpu_fixture, oracle, false);
                // F32 accumulation keeps the original strict operand-double
                // limits. Legacy F16 accumulation uses its explicit row bounds.
                if (retain) {
                    check_gdn_norm_tensor(gpu_fixture.weighted, oracle.norm.weighted, "Q8 boundary RMS*gamma/double");
                    check_gdn_norm_tensor(cpu_fixture.weighted, oracle.norm.weighted, "Q8 boundary CPU RMS*gamma/double");
                    check_gdn_projection_tensor(cpu_fixture.z_producer, z.cpu, "Z CPU/Q8-reference", 0.003f, 0.005f);
                    check_gdn_projection_tensor(gpu_fixture.z_producer, z.raw,
                        f32_accum ? "Z F32/raw-double" : "Z legacy/raw-double diagnostic",
                        0.01f, 0.02f, nullptr, f32_accum);
                    check_gdn_projection_tensor(cpu_fixture.activation, activation.cpu, "sigmoid CPU/Q8-reference", 0.001f, 0.002f);
                    check_gdn_projection_tensor(cpu_fixture.output, gate.cpu, "gate CPU/Q8-reference", 0.003f, 0.005f);
                    if (f32_accum) {
                        check_gdn_projection_tensor(gpu_fixture.z_producer, z.wmma, "Z F32/F16-operands", 0.002f, 0.003f);
                        check_gdn_projection_tensor(gpu_fixture.activation, activation.wmma, "sigmoid F32/F16-operands", 0.001f, 0.002f);
                        check_gdn_projection_tensor(gpu_fixture.output, gate.wmma, "gate F32/F16-operands", 0.003f, 0.005f);
                    } else {
                        check_gdn_projection_tensor(gpu_fixture.z_producer, z.wmma,
                            "Z legacy/F16-accumulation-bound", 2.0e-5f, 2.0e-5f, &z.legacy_bound);
                        const auto actual_z = gdn_projection_read_finite(gpu_fixture.z_producer, true);
                        std::vector<float> actual_z_sigmoid(6144), actual_z_gate(6144);
                        for (size_t i = 0; i < actual_z.size(); ++i) {
                            const double sigmoid = 1.0 / (1.0 + std::exp(-static_cast<double>(actual_z[i])));
                            actual_z_sigmoid[i] = static_cast<float>(sigmoid);
                            actual_z_gate[i] = static_cast<float>(oracle.norm.weighted[i] * sigmoid);
                        }
                        check_gdn_projection_tensor(gpu_fixture.activation, actual_z_sigmoid,
                            "sigmoid legacy/actual-Z strict", 2.0e-5f, 2.0e-5f);
                        check_gdn_projection_tensor(gpu_fixture.output, actual_z_gate,
                            "gate legacy/actual-Z strict", 2.0e-5f, 2.0e-5f);
                        const auto actual_gate = gdn_projection_read_finite(gpu_fixture.output);
                        const auto conditional = gdn_projection_dot(output_weight,
                            {actual_gate, actual_gate, actual_gate, {}}, 2560);
                        check_gdn_projection_tensor(gpu_fixture.projected, conditional.wmma,
                            "final legacy/actual-gate F16-accumulation-bound",
                            2.0e-5f, 2.0e-5f, &conditional.legacy_bound);
                    }
                } else {
                    std::fprintf(stderr, "GDN Q8 intermediate readbacks omitted: unobserved model lifetimes\n");
                }
                check_gdn_projection_tensor(cpu_fixture.projected, projected.cpu, "final CPU/Q8-reference", 0.003f, 0.005f);
                if (f32_accum) {
                    check_gdn_projection_tensor(gpu_fixture.projected, projected.wmma,
                        "final F32/F16-operands strict", 0.003f, 0.005f);
                } else {
                    const auto actual = gdn_projection_read_finite(gpu_fixture.projected, true);
                    if (retained_legacy_outputs[pass].empty()) {
                        REQUIRE(retain && !reuse);
                        retained_legacy_outputs[pass] = actual;
                    } else {
                        check_gdn_projection_tensor(gpu_fixture.projected, retained_legacy_outputs[pass],
                            "final legacy/retained-context replay strict", 2.0e-5f, 2.0e-5f);
                    }
                }
                check_gdn_projection_tensor(cpu_fixture.projected, projected.raw, "final CPU/raw-double", 0.01f, 0.02f);
                check_gdn_projection_tensor(gpu_fixture.projected, projected.raw,
                    f32_accum ? "final F32/raw-double" : "final legacy/raw-double diagnostic",
                    0.01f, 0.02f, nullptr, f32_accum);
                std::vector<float> expected(2560);
                ggml_backend_tensor_get(cpu_fixture.projected, expected.data(), 0, expected.size() * sizeof(float));
                check_gdn_projection_tensor(gpu_fixture.projected, expected,
                    f32_accum ? "final F32/CPU" : "final legacy/CPU diagnostic",
                    0.015f, 0.02f, nullptr, f32_accum);
                for (auto * fixture : {&cpu_fixture, &gpu_fixture}) {
                    ggml_backend_tensor_get(fixture->z_input, expected.data(), 0, expected.size() * sizeof(float));
                    REQUIRE(std::memcmp(expected.data(), oracle.norm.z_input.data(), expected.size() * sizeof(float)) == 0);
                }
                REQUIRE(ggml_graph_n_nodes(gpu_fixture.graph) == static_cast<int>(original_order.size()));
                for (size_t i = 0; i < original_order.size(); ++i) {
                    REQUIRE(ggml_graph_node(gpu_fixture.graph, static_cast<int>(i)) == original_order[i]);
                }
            }
        }
    }
    ggml_backend_free(cpu);
}

static void run_gdn_projection_checks(ggml_backend_t backend) {
    for (bool f32_accum : {true, false}) {
        run_gdn_projection_mode(backend, f32_accum);
    }
}

enum class host_alias_allocation { device_context, device_gallocr, host_context, host_gallocr, mixed_gallocr };

static const char * host_alias_label(host_alias_allocation allocation) {
    switch (allocation) {
        case host_alias_allocation::device_context: return "device-context";
        case host_alias_allocation::device_gallocr: return "device-gallocr";
        case host_alias_allocation::host_context: return "host-context";
        case host_alias_allocation::host_gallocr: return "host-gallocr";
        case host_alias_allocation::mixed_gallocr: return "mixed-gallocr";
    }
    return "?";
}

static bool host_alias_exact(ggml_tensor * tensor, const std::vector<float> & expected, const char * label) {
    std::vector<uint32_t> actual(expected.size());
    REQUIRE(ggml_nbytes(tensor) == actual.size() * sizeof(uint32_t));
    ggml_backend_tensor_get(tensor, actual.data(), 0, ggml_nbytes(tensor));
    size_t mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &expected[i], sizeof(bits));
        if (actual[i] != bits && mismatches++ < 3) {
            std::fprintf(stderr, "HOST alias %s %s index=%zu expected=%08x actual=%08x\n",
                         label, tensor->name, i, static_cast<unsigned>(bits), static_cast<unsigned>(actual[i]));
        }
    }
    std::fprintf(stderr, "HOST alias %s %s mismatches=%zu/%zu\n", label, tensor->name, mismatches, actual.size());
    return mismatches == 0;
}

static uint64_t next_host_alias_graph_uid() {
    static uint64_t next = 1;
    return next++;
}

static bool host_alias_same_storage(const ggml::hrx::ValueBufferBinding & first,
                                    const ggml::hrx::ValueBufferBinding & second) {
    if (first.length != second.length) {
        return false;
    }
    if (first.buffer != nullptr || second.buffer != nullptr) {
        return first.buffer != nullptr && first.buffer == second.buffer && first.offset == second.offset;
    }
    REQUIRE(first.host_data != nullptr && second.host_data != nullptr);
    return reinterpret_cast<uintptr_t>(first.host_data) + first.offset ==
           reinterpret_cast<uintptr_t>(second.host_data) + second.offset;
}

static void host_alias_require_distinct_allocations(ggml_tensor * first, ggml_tensor * second) {
    ggml::hrx::ValueBufferBinding first_binding, second_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(first, first_binding));
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(second, second_binding));
    REQUIRE((first_binding.buffer != nullptr) == (second_binding.buffer != nullptr));
    if (first_binding.buffer != nullptr) {
        // Device tensor->data is an opaque offset token, not an allocation address.
        REQUIRE(first_binding.buffer != second_binding.buffer);
    } else {
        REQUIRE(first_binding.host_data != nullptr && second_binding.host_data != nullptr);
        const uintptr_t first_address = reinterpret_cast<uintptr_t>(first_binding.host_data) + first_binding.offset;
        const uintptr_t second_address = reinterpret_cast<uintptr_t>(second_binding.host_data) + second_binding.offset;
        REQUIRE(first_address < second_address ?
                    second_address - first_address >= first_binding.length :
                    first_address - second_address >= second_binding.length);
    }
    REQUIRE(!host_alias_same_storage(first_binding, second_binding));
}

struct host_alias_fixture {
    ggml_context * context = nullptr;
    ggml_context * device_context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * gamma = nullptr;
    ggml_tensor * gate_input = nullptr;
    ggml_tensor * weighted = nullptr;
    ggml_tensor * gated = nullptr;
    ggml_tensor * flat = nullptr;
    ggml_tensor * output = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_buffer_t device_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;

    host_alias_fixture(ggml_backend_t backend, host_alias_allocation allocation, int padding) {
        const bool host = allocation != host_alias_allocation::device_context &&
                          allocation != host_alias_allocation::device_gallocr;
        const bool mixed = allocation == host_alias_allocation::mixed_gallocr;
        const bool reuse = allocation != host_alias_allocation::device_context &&
                           allocation != host_alias_allocation::host_context;
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        context = ggml_init(params);
        REQUIRE(context != nullptr);
        if (mixed) {
            device_context = ggml_init(params);
            REQUIRE(device_context != nullptr);
        }
        // Unused context padding changes offsets as well as allocation identity on
        // same-UID context rebinding. gallocr controls instead exercise real reuse.
        ggml_new_tensor_1d(context, GGML_TYPE_F32, padding);
        input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 128, 48);
        gamma = ggml_new_tensor_1d(context, GGML_TYPE_F32, 128);
        ggml_context * gate_context = mixed ? device_context : context;
        gate_input = ggml_new_tensor_2d(gate_context, GGML_TYPE_F32, 128, 48);
        weighted = ggml_mul(context, input, gamma);
        ggml_tensor * gate = ggml_scale(gate_context, gate_input, 0.5f);
        gated = ggml_mul(context, weighted, gate);
        flat = ggml_reshape_2d(context, gated, 6144, 1);
        output = ggml_scale(gate_context, flat, 2.0f);
        ggml_set_name(input, "host_alias_cpu_norm");
        ggml_set_name(gamma, "host_alias_gamma");
        ggml_set_name(gate_input, "host_alias_gate_input");
        ggml_set_name(weighted, "host_alias_gamma_mul");
        ggml_set_name(gate, "host_alias_gate");
        ggml_set_name(gated, "host_alias_gated_mul");
        ggml_set_name(flat, "host_alias_final_reshape");
        ggml_set_name(output, "host_alias_consumer");
        for (auto * tensor : { input, gamma, gate_input }) {
            ggml_set_input(tensor);
        }
        ggml_set_output(gamma);
        ggml_set_output(gate_input);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(context, 32, false);
        graph->uid = next_host_alias_graph_uid();
        ggml_build_forward_expand(graph, output);
        REQUIRE(graph->n_nodes == 5 && graph->nodes[3] == flat && graph->nodes[4] == output);
        for (auto * tensor : { weighted, gate, gated, output }) {
            REQUIRE(ggml_backend_supports_op(backend, tensor));
        }
        if (mixed) {
            device_buffer = ggml_backend_alloc_ctx_tensors(device_context, backend);
            REQUIRE(device_buffer != nullptr);
        }
        auto * hrx = backend_context(backend);
        const bool direct = hrx->device->use_direct_host_bindings;
        hrx->device->use_direct_host_bindings = false;
        ggml_backend_buffer_type_t buft = host ?
            ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend)) :
            ggml_backend_get_default_buffer_type(backend);
        REQUIRE(buft != nullptr);
        if (reuse) {
            allocator = ggml_gallocr_new(buft);
            REQUIRE(allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph));
        } else {
            buffer = ggml_backend_alloc_ctx_tensors_from_buft(context, buft);
            REQUIRE(buffer != nullptr);
        }
        hrx->device->use_direct_host_bindings = direct;
        ggml::hrx::ValueBufferBinding bindings[4];
        ggml_tensor * tensors[] = { input, weighted, gated, flat };
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(ggml_backend_hrx_resolve_value_buffer(tensors[i], bindings[i]));
            REQUIRE((bindings[i].host_data != nullptr) == host);
            REQUIRE((bindings[i].buffer == nullptr) == host);
            std::fprintf(stderr, "HOST alias %s tensor=%s data=%p host=%p device=%p offset=%zu len=%zu\n",
                         host_alias_label(allocation), tensors[i]->name, tensors[i]->data, bindings[i].host_data,
                         static_cast<void *>(bindings[i].buffer), bindings[i].offset, bindings[i].length);
        }
        REQUIRE(host_alias_same_storage(bindings[2], bindings[3]));
        REQUIRE(host_alias_same_storage(bindings[0], bindings[1]) == reuse);
        REQUIRE(host_alias_same_storage(bindings[1], bindings[2]) == reuse);
        if (mixed) {
            ggml::hrx::ValueBufferBinding device;
            REQUIRE(ggml_backend_hrx_resolve_value_buffer(output, device));
            REQUIRE(device.buffer != nullptr && device.host_data == nullptr);
        }
    }

    ~host_alias_fixture() {
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(buffer);
        ggml_backend_buffer_free(device_buffer);
        ggml_free(context);
        ggml_free(device_context);
    }
};

// ggml_graph_view is private to ggml-base and is not exported by its Windows DLL.
static ggml_cgraph host_alias_graph_view(ggml_cgraph * graph, int first, int last) {
    REQUIRE(first >= 0 && first <= last && last <= graph->n_nodes);
    ggml_cgraph view = {};
    view.n_nodes = last - first;
    view.nodes = graph->nodes + first;
    view.use_counts = graph->use_counts;
    view.visited_hash_set = graph->visited_hash_set;
    view.order = graph->order;
    return view;
}

static bool run_host_alias_replay(ggml_backend_t backend, host_alias_fixture & fixture, int pass, bool cut,
                                  bool direct) {
    std::vector<float> input(6144), gate(6144), gamma(128), expected(6144);
    for (size_t i = 0; i < gamma.size(); ++i) {
        gamma[i] = static_cast<float>(i % 9 + 1) * 0.125f;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 251) - 125 + pass) / 64.0f;
        gate[i] = static_cast<float>(static_cast<int>(i % 31) - 15 - pass) / 16.0f;
        // SCALE includes a positive-zero bias, so a negative-zero product
        // becomes positive zero. These dyadic products are otherwise exact.
        const float scaled_gate = std::fma(gate[i], 0.5f, 0.0f);
        expected[i] = std::fma((input[i] * gamma[i % 128]) * scaled_gate, 2.0f, 0.0f);
    }
    ggml_backend_tensor_memset(fixture.gated, 0xa5 + pass, 0, ggml_nbytes(fixture.gated));
    ggml_backend_tensor_memset(fixture.output, 0x5a + pass, 0, ggml_nbytes(fixture.output));
    ggml_backend_tensor_set(fixture.input, input.data(), 0, ggml_nbytes(fixture.input));
    ggml_backend_tensor_set(fixture.gamma, gamma.data(), 0, ggml_nbytes(fixture.gamma));
    ggml_backend_tensor_set(fixture.gate_input, gate.data(), 0, ggml_nbytes(fixture.gate_input));
    if (cut) {
        auto prefix = host_alias_graph_view(fixture.graph, 0, 4);
        auto suffix = host_alias_graph_view(fixture.graph, 4, 5);
        REQUIRE(ggml_backend_graph_compute(backend, &prefix) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(backend, &suffix) == GGML_STATUS_SUCCESS);
    } else if (direct) {
        auto graph = host_alias_graph_view(fixture.graph, 0, fixture.graph->n_nodes);
        REQUIRE(graph.uid == 0);
        REQUIRE(ggml_backend_graph_compute(backend, &graph) == GGML_STATUS_SUCCESS);
    } else {
        REQUIRE(ggml_backend_graph_compute(backend, fixture.graph) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_synchronize(backend);
    return host_alias_exact(fixture.output, expected, cut ? "cut" : "whole");
}

static bool run_host_alias_partial(ggml_backend_t backend, int64_t write_start, int64_t read_start, bool direct) {
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * destination = ggml_view_1d(ctx, storage, 64, write_start * sizeof(float));
    ggml_tensor * write = ggml_cpy(ctx, input, destination);
    ggml_tensor * read = ggml_view_1d(ctx, storage, 64, read_start * sizeof(float));
    ggml_tensor * output = ggml_scale(ctx, read, 2.0f);
    ggml_set_name(storage, "host_alias_partial_storage");
    ggml_set_name(output, "host_alias_partial_output");
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    graph->uid = next_host_alias_graph_uid();
    // Explicit order: update one subrange, then read an overlapping subrange.
    // CPY's source is disjoint; this does not assume memmove semantics.
    ggml_build_forward_expand(graph, write);
    ggml_build_forward_expand(graph, output);
    if (direct) {
        graph->uid = 0;
    }
    REQUIRE(ggml_backend_supports_op(backend, write) && ggml_backend_supports_op(backend, output));
    auto * hrx = backend_context(backend);
    const bool direct_host_bindings = hrx->device->use_direct_host_bindings;
    hrx->device->use_direct_host_bindings = false;
    auto buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    hrx->device->use_direct_host_bindings = direct_host_bindings;
    REQUIRE(buffer != nullptr);
    ggml::hrx::ValueBufferBinding storage_binding, write_binding, read_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(storage, storage_binding));
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(write, write_binding));
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(read, read_binding));
    for (const auto * binding : { &storage_binding, &write_binding, &read_binding }) {
        REQUIRE(binding->host_data != nullptr && binding->buffer == nullptr);
    }
    const uintptr_t storage_address = reinterpret_cast<uintptr_t>(storage_binding.host_data) + storage_binding.offset;
    REQUIRE(reinterpret_cast<uintptr_t>(write_binding.host_data) + write_binding.offset ==
            storage_address + static_cast<size_t>(write_start) * sizeof(float));
    REQUIRE(reinterpret_cast<uintptr_t>(read_binding.host_data) + read_binding.offset ==
            storage_address + static_cast<size_t>(read_start) * sizeof(float));
    host_alias_require_distinct_allocations(input, storage);
    host_alias_require_distinct_allocations(output, storage);
    bool passed = true;
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<float> initial(256), values(64), expected(256), result(64);
        for (size_t i = 0; i < initial.size(); ++i) {
            initial[i] = -100.0f - static_cast<float>(i + pass) * 0.25f;
        }
        expected = initial;
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(i + pass + 1) * 0.125f;
            expected[static_cast<size_t>(write_start) + i] = values[i];
        }
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = std::fma(expected[static_cast<size_t>(read_start) + i], 2.0f, 0.0f);
        }
        ggml_backend_tensor_set(storage, initial.data(), 0, ggml_nbytes(storage));
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_memset(output, 0xa5, 0, ggml_nbytes(output));
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        std::fprintf(stderr, "HOST alias partial write=%lld read=%lld pass=%d\n",
                     static_cast<long long>(write_start), static_cast<long long>(read_start), pass);
        passed = host_alias_exact(output, result, "partial read") && passed;
        passed = host_alias_exact(storage, expected, "partial guards") && passed;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

static void run_host_alias_checks(ggml_backend_t backend, bool direct) {
    gdn_norm_env mul_generic("HRX_MUL_CLAIM_GENERIC", "1");
    gdn_norm_env mul_mask("HRX_MUL_CLAIM_MASK", "0x01");
    // The setting is cached on first graph execution; each selector runs in its
    // own process. UID0 additionally selects the uncached direct executor.
    gdn_norm_env prepared("HRX_DISABLE_PREPARED_FAST_PATH", direct ? "1" : "0");
    std::fprintf(stderr, "HOST alias execution=%s\n", direct ? "direct" : "recorded");
    bool passed = true;
    for (auto allocation : { host_alias_allocation::device_context, host_alias_allocation::device_gallocr,
                             host_alias_allocation::host_context, host_alias_allocation::host_gallocr,
                             host_alias_allocation::mixed_gallocr }) {
        for (bool cut : { true, false }) {
            host_alias_fixture first(backend, allocation, 1);
            host_alias_fixture second(backend, allocation, 129);
            host_alias_require_distinct_allocations(first.input, second.input);
            REQUIRE(first.graph->uid != 0);
            second.graph->uid = first.graph->uid;
            for (int pass = 0; pass < 3; ++pass) {
                std::fprintf(stderr, "HOST alias %s %s pass=%d same-UID allocation=%d\n",
                             host_alias_label(allocation), cut ? "cut" : "whole", pass, pass % 2);
                passed = run_host_alias_replay(backend, pass % 2 ? second : first, pass, cut, direct) && passed;
            }
        }
    }
    {
        host_alias_fixture split(backend, host_alias_allocation::host_context, 1);
        host_alias_fixture merged(backend, host_alias_allocation::host_gallocr, 1);
        REQUIRE(split.graph->uid != 0);
        host_alias_require_distinct_allocations(split.input, merged.input);
        merged.graph->uid = split.graph->uid;
        // Same logical program, first split storage roots, then physical reuse,
        // then split again. Repeat each binding to also exercise unchanged replay.
        for (int pass = 0; pass < 6; ++pass) {
            const bool merge = pass == 2 || pass == 3;
            std::fprintf(stderr, "HOST alias topology=%s pass=%d same-UID\n", merge ? "merged" : "split", pass);
            passed = run_host_alias_replay(backend, merge ? merged : split, pass, false, direct) && passed;
        }
    }
    // Both overlap directions and a disjoint-range negative control, with guard
    // checks over the entire storage to detect accidental offset-zero merging.
    passed = run_host_alias_partial(backend, 16, 48, direct) && passed;
    passed = run_host_alias_partial(backend, 48, 16, direct) && passed;
    passed = run_host_alias_partial(backend, 16, 112, direct) && passed;
    REQUIRE(passed);
}

int main(int argc, char ** argv) {
    const bool copy_only = argc == 2 && std::strcmp(argv[1], "--copy-f32") == 0;
    const bool rows_only = argc == 2 && std::strcmp(argv[1], "--set-rows") == 0;
    const bool f32_rows_only = argc == 2 && std::strcmp(argv[1], "--f32-get-rows") == 0;
    const bool repeat_only = argc == 2 && std::strcmp(argv[1], "--repeat") == 0;
    const bool qsa_only = argc == 2 && std::strcmp(argv[1], "--qsa") == 0;
    const bool gdn_norm_only = argc == 2 && std::strcmp(argv[1], "--gdn-norm") == 0;
    const bool gdn_recurrent_only = argc == 2 && std::strcmp(argv[1], "--gdn-recurrent") == 0;
    const bool gdn_projections_only = argc == 2 && std::strcmp(argv[1], "--gdn-projections") == 0;
    const bool host_alias_direct = argc == 2 && std::strcmp(argv[1], "--host-alias-direct") == 0;
    const bool host_alias_only = host_alias_direct || (argc == 2 && std::strcmp(argv[1], "--host-alias") == 0);
    const bool gdn_only = gdn_norm_only || gdn_recurrent_only || gdn_projections_only;
    if (argc > 1 && !copy_only && !rows_only && !f32_rows_only && !repeat_only && !qsa_only && !gdn_only &&
        !host_alias_only) {
        std::fprintf(stderr, "usage: test-hrx-buffer [--copy-f32|--set-rows|--f32-get-rows|--repeat|--qsa|--gdn-norm|--gdn-recurrent|--gdn-projections|--host-alias|--host-alias-direct]\n");
        return 1;
    }
    run_shutdown_checks();
    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        REQUIRE(ggml_backend_hrx_shutdown());
        return 0;
    }

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    ggml_backend_hrx_context * context = backend_context(backend);
    if (host_alias_only) {
        run_host_alias_checks(backend, host_alias_direct);
        ggml_backend_free(backend);
        REQUIRE(ggml_backend_hrx_shutdown());
        return 0;
    }

    if (!copy_only && !rows_only && !f32_rows_only && !repeat_only && !qsa_only && !gdn_only) {
        std::fprintf(stderr, "running backend buffer checks\n");
        run_backend_buffer_checks(backend);
        std::fprintf(stderr, "running host buffer checks\n");
        run_host_buffer_checks(backend);
        std::fprintf(stderr, "running host transfer checks\n");
        run_host_transfer_checks(context);
        std::fprintf(stderr, "running host staging checks\n");
        run_host_staging_checks(context);
        std::fprintf(stderr, "running host weight cache checks\n");
        run_host_weight_cache_checks(context);
    }
    if (!rows_only && !f32_rows_only && !repeat_only && !qsa_only && !gdn_only) {
        std::fprintf(stderr, "running strided flatten copy checks\n");
        run_strided_flatten_copy_checks(backend);
        std::fprintf(stderr, "strided flatten copy checks passed\n");
    }
    if (!copy_only && !f32_rows_only && !repeat_only && !qsa_only && !gdn_only) {
        std::fprintf(stderr, "running SET_ROWS checks\n");
        run_set_rows_checks(backend);
        std::fprintf(stderr, "SET_ROWS checks passed\n");
    }
    if (f32_rows_only) {
        run_f32_get_rows_checks(backend);
        std::fprintf(stderr, "F32 GET_ROWS checks passed\n");
    }
    if (repeat_only) {
        std::fprintf(stderr, "running REPEAT checks\n");
        run_repeat_checks(backend);
        std::fprintf(stderr, "REPEAT checks passed\n");
    }
    if (qsa_only) {
        std::fprintf(stderr, "running QSA attention + gate subgraph checks (not fused)\n");
        run_qsa_checks(backend);
        std::fprintf(stderr, "QSA attention + gate subgraph checks passed\n");
    }
    if (gdn_norm_only) {
        run_gdn_norm_checks(backend);
        std::fprintf(stderr, "GDN norm subgraph checks passed\n");
    }
    if (gdn_norm_only || gdn_recurrent_only) {
        run_gdn_recurrent_checks(backend);
        std::fprintf(stderr, "GDN recurrent + norm subgraph checks passed\n");
    }
    if (gdn_norm_only || gdn_projections_only) {
        run_gdn_projection_checks(backend);
        std::fprintf(stderr, "GDN recurrent + Q8 projection subgraph checks passed\n");
    }

    ggml_backend_free(backend);
    std::fprintf(stderr, "HRX backend freed\n");
    REQUIRE(ggml_backend_hrx_shutdown());
    std::fprintf(stderr, "HRX runtime shut down\n");
    return 0;
}
