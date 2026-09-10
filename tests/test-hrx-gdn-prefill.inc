// Include after test-hrx-ops.cpp's helpers, before main.
#include "dispatch_registration/dispatch-gated-delta-net.h"

enum class GdnPrefillLayout { ModelNative, Packed, Padded };

struct GdnPrefillFixture {
    ggml_tensor * sources[6] = {};
    ggml_tensor * output = nullptr;
    ggml_tensor * attention = nullptr;
    ggml_tensor * snapshots = nullptr;
    ggml_cgraph * graph = nullptr;
};

static GdnPrefillFixture build_gdn_prefill_fixture(ggml_context * ctx, int64_t tokens,
                                                 int64_t slots, GdnPrefillLayout layout,
                                                 int64_t key_head_count = 16) {
    GdnPrefillFixture result;
    for (int i = 0; i < 3; ++i) {
        const int64_t heads = i == 2 ? 48 : key_head_count;
        if (layout == GdnPrefillLayout::ModelNative) {
            // qwen4exp.cpp: L2_NORM materializes dense Q/K, while V remains a
            // slice starting after Q+K in the 10240-channel convolution output.
            if (i < 2) {
                result.sources[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, heads, tokens, 1);
            } else {
                auto * qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10240, tokens);
                result.sources[i] = ggml_view_4d(ctx, qkv, 128, 48, tokens, 1,
                    128 * sizeof(float), 10240 * sizeof(float), 10240 * tokens * sizeof(float),
                    4096 * sizeof(float));
            }
            continue;
        }
        const bool padded = layout == GdnPrefillLayout::Padded;
        const size_t head_stride = (padded ? 132 : 128) * sizeof(float);
        // Nonzero VIEW offsets and independently padded head/token strides.
        // V also exercises the model's 10240-wide QKV convolution slice.
        const size_t token_stride = (padded ? (i == 2 ? 10240 : 2304 + i * 128) : heads * 128) *
                                    sizeof(float);
        const size_t bytes = (tokens - 1) * token_stride + (heads - 1) * head_stride + 128 * sizeof(float);
        auto * backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bytes / sizeof(float) + 7);
        result.sources[i] = ggml_view_4d(ctx, backing, 128, heads, tokens, 1,
                                        head_stride, token_stride, bytes, 7 * sizeof(float));
    }
    result.sources[3] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
    result.sources[4] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
    result.sources[5] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 48, 1);
    result.output = ggml_gated_delta_net(ctx, result.sources[0], result.sources[1], result.sources[2],
                                      result.sources[3], result.sources[4], result.sources[5], slots);
    // delta-net-base.cpp: K=1 for fused prefill without rollback, otherwise
    // K=n_rs_seq+1. K is independent of T; this is not the unfused chunk size.
    result.attention = ggml_view_4d(ctx, result.output, 128, 48, tokens, 1,
        128 * sizeof(float), 6144 * sizeof(float), 6144 * tokens * sizeof(float), 0);
    result.snapshots = ggml_view_3d(ctx, result.output, 786432, 1, std::min(tokens, slots),
        786432 * sizeof(float), 786432 * sizeof(float), 6144 * tokens * sizeof(float));
    ggml_set_output(result.output);
    ggml_set_output(result.attention);
    ggml_set_output(result.snapshots);
    result.graph = ggml_new_graph(ctx);
    result.graph->uid = next_hrx_test_graph_uid();
    ggml_build_forward_expand(result.graph, result.attention);
    ggml_build_forward_expand(result.graph, result.snapshots);
    return result;
}

static void run_gdn_prefill_scheduling_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_GDN_PREFILL", "1");
    for (int64_t tokens = 2; tokens <= 8; ++tokens) {
        for (int64_t slots = 1; slots <= tokens + 1; ++slots) {
            for (auto layout : { GdnPrefillLayout::ModelNative, GdnPrefillLayout::Packed,
                                 GdnPrefillLayout::Padded }) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                ggml_context * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                const auto fixture = build_gdn_prefill_fixture(ctx, tokens, slots, layout);
                if (slots > tokens) {
                    // A legal model rollback op, with correctly sized output, not a
                    // malformed synthetic tensor. Older unwritten slots remain CPU-only.
                    REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(fixture.output));
                    auto imported = ggml::hrx::import_ggml_graph(*fixture.graph);
                    REQUIRE(imported.valid());
                    ggml::hrx::DispatchScheduler scheduler;
                    REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
                    ggml_free(ctx);
                    continue;
                }
                REQUIRE(ggml::hrx::supports_gdn_prefill_dispatch(fixture.output));
                const auto names = scheduled_kernel_sequence(fixture.graph);
                REQUIRE(names.size() == 1);
                REQUIRE(names[0] == "qwen4exp:qwen38_gdn_recurrent_prefill");
                for (const char * flag : { static_cast<const char *>(nullptr), "0", "1x", "true" }) {
                    ScopedHrxEnvironment disabled("HRX_ENABLE_GDN_PREFILL", flag);
                    REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(fixture.output));
                    auto imported = ggml::hrx::import_ggml_graph(*fixture.graph);
                    REQUIRE(imported.valid());
                    ggml::hrx::DispatchScheduler scheduler;
                    REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
                }
                for (int32_t bad_slots : { 0, static_cast<int32_t>(tokens + 1) }) {
                    auto bad = *fixture.output;
                    ggml_set_op_params_i32(&bad, 0, bad_slots);
                    REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(&bad));
                }
                for (int source = 0; source < 6; ++source) {
                    auto bad = *fixture.output;
                    auto input = *bad.src[source];
                    bad.src[source] = &input;
                    input.type = GGML_TYPE_F16;
                    REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(&bad));
                    input = *fixture.sources[source];
                    input.ne[3] = 2;
                    REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(&bad));
                    input = *fixture.sources[source];
                    input.nb[0] = 2 * sizeof(float);
                    REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(&bad));
                    input = *fixture.sources[source];
                    input.view_offs += 1;
                    REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(&bad));
                }
                auto bad = *fixture.output;
                auto input = *bad.src[3];
                input.ne[0] = 128; // KDA is deliberately excluded.
                bad.src[3] = &input;
                REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(&bad));
                bad = *fixture.output;
                input = *bad.src[2];
                input.nb[2] = 128 * sizeof(float); // overlapping tokens
                bad.src[2] = &input;
                REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(&bad));
                ggml_free(ctx);
            }
        }
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        // qwen4exp repeats Q/K to 48 heads if either fused-GDN switch is off.
        // Rollback still emits GATED_DELTA_NET in that configuration; decline it.
        const auto repeated = build_gdn_prefill_fixture(ctx, tokens, tokens, GdnPrefillLayout::ModelNative, 48);
        REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(repeated.output));
        auto imported = ggml::hrx::import_ggml_graph(*repeated.graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
        ggml_free(ctx);
    }
    for (int64_t tokens : { 1, 9 }) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        const auto fixture = build_gdn_prefill_fixture(ctx, tokens, 1, GdnPrefillLayout::ModelNative);
        REQUIRE(!ggml::hrx::supports_gdn_prefill_dispatch(fixture.output));
        ggml_free(ctx);
    }
    std::fprintf(stderr, "GDN prefill scheduling: native/packed/padded T2..8, all K<=T, legal K>T fallback, one dispatch passed\n");
}

static void require_gdn_prefill_close(const std::vector<float> & actual, const std::vector<float> & expected) {
    for (float value : actual) {
        REQUIRE(std::isfinite(value));
    }
    for (float value : expected) {
        REQUIRE(std::isfinite(value));
    }
    require_close(actual, expected, 2.0e-5f, 2.0e-4f);
}

static void run_gdn_prefill_reference_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_GDN_PREFILL", "1");
    ggml_backend_t cpu = init_cpu_backend();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(hrx != nullptr);
    for (int64_t tokens = 2; tokens <= 8; ++tokens) {
        for (int64_t slots = 1; slots <= tokens; ++slots) {
            for (auto layout : { GdnPrefillLayout::ModelNative, GdnPrefillLayout::Packed,
                                 GdnPrefillLayout::Padded }) {
                if (layout != GdnPrefillLayout::ModelNative && slots != 1 &&
                    slots != tokens / 2 && slots != tokens) {
                    continue;
                }
                ggml_init_params params = {};
                params.mem_size = 4 * 1024 * 1024;
                params.no_alloc = true;
                ggml_context * cpu_ctx = ggml_init(params);
                ggml_context * hrx_ctx = ggml_init(params);
                REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
                const auto reference = build_gdn_prefill_fixture(cpu_ctx, tokens, slots, layout);
                const auto candidate = build_gdn_prefill_fixture(hrx_ctx, tokens, slots, layout);
                REQUIRE(ggml_backend_supports_op(hrx, candidate.output));
                const auto names = scheduled_kernel_sequence(candidate.graph);
                REQUIRE(names.size() == 1 && names[0] == "qwen4exp:qwen38_gdn_recurrent_prefill");

                // Independent CPU graph: T one-token ops linked by actual dst-tail state views.
                ggml_cgraph * serial_graph = ggml_new_graph(cpu_ctx);
                std::vector<ggml_tensor *> serial;
                ggml_tensor * state = reference.sources[5];
                for (int64_t t = 0; t < tokens; ++t) {
                    ggml_tensor * token_inputs[5];
                    for (int i = 0; i < 5; ++i) {
                        auto * src = reference.sources[i];
                        token_inputs[i] = ggml_view_4d(cpu_ctx, src, src->ne[0], src->ne[1], 1, 1,
                                                     src->nb[1], src->nb[2], src->nb[3], t * src->nb[2]);
                    }
                    auto * out = ggml_gated_delta_net(cpu_ctx, token_inputs[0], token_inputs[1],
                        token_inputs[2], token_inputs[3], token_inputs[4], state, 1);
                    serial.push_back(out);
                    state = ggml_view_4d(cpu_ctx, out, 128, 128, 48, 1,
                        128 * sizeof(float), 16384 * sizeof(float), 786432 * sizeof(float),
                        6144 * sizeof(float));
                    ggml_build_forward_expand(serial_graph, out);
                }
                ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
                ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
                REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
                std::vector<float> last_state;
                for (int replay = 0; replay < 3; ++replay) {
                    std::vector<float> input_data[6];
                    for (int i = 0; i < 6; ++i) {
                        const auto * src = reference.sources[i];
                        auto & data = input_data[i];
                        data.assign(ggml_nbytes(src) / sizeof(float), -123.0f);
                        for (int64_t z = 0; z < src->ne[2]; ++z) {
                            for (int64_t y = 0; y < src->ne[1]; ++y) {
                                for (int64_t x = 0; x < src->ne[0]; ++x) {
                                    const int64_t code = (x * 17 + y * 31 + z * 43 + replay * 59 + i * 71) % 127;
                                    float value = static_cast<float>(code - 63) * (i < 2 ? 0.002f : 0.004f);
                                    if (i == 3) {
                                        value = -0.01f - static_cast<float>(code) * 0.013f;
                                    } else if (i == 4) {
                                        value = static_cast<float>(code) / 126.0f;
                                    }
                                    data[(z * src->nb[2] + y * src->nb[1]) / sizeof(float) + x] = value;
                                }
                                if (layout == GdnPrefillLayout::ModelNative && i < 2) {
                                    const size_t base = (z * src->nb[2] + y * src->nb[1]) / sizeof(float);
                                    float squares = 0.0f;
                                    for (int64_t x = 0; x < 128; ++x) {
                                        squares += data[base + x] * data[base + x];
                                    }
                                    const float norm = std::max(std::sqrt(squares), kQwenRmsNormEps);
                                    for (int64_t x = 0; x < 128; ++x) {
                                        data[base + x] /= norm;
                                    }
                                }
                            }
                        }
                        // Third execution explicitly hands the previous GPU batch's newest
                        // snapshot to the next batch. Earlier executions reset all inputs.
                        if (i == 5 && replay == 2) {
                            data = last_state;
                        }
                        set_tensor_pair_bytes(cpu, reference.sources[i], hrx, candidate.sources[i],
                                              data.data(), data.size() * sizeof(float));
                    }
                    const std::vector<float> poison(ggml_nelements(candidate.output),
                                                    std::numeric_limits<float>::quiet_NaN());
                    set_tensor_pair_bytes(cpu, reference.output, hrx, candidate.output,
                                          poison.data(), poison.size() * sizeof(float));
                    REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
                    REQUIRE(ggml_backend_graph_compute(cpu, serial_graph) == GGML_STATUS_SUCCESS);
                    REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_SUCCESS);
                    const auto actual = get_f32_tensor(hrx, candidate.output);
                    require_gdn_prefill_close(actual, get_f32_tensor(cpu, reference.output));
                    require_gdn_prefill_close(get_f32_tensor(hrx, candidate.attention),
                                              get_f32_tensor(cpu, reference.attention));
                    require_gdn_prefill_close(get_f32_tensor(hrx, candidate.snapshots),
                                              get_f32_tensor(cpu, reference.snapshots));
                    for (int64_t t = 0; t < tokens; ++t) {
                        const auto serial_out = get_f32_tensor(cpu, serial[t]);
                        require_gdn_prefill_close(
                            std::vector<float>(actual.begin() + t * 6144, actual.begin() + (t + 1) * 6144),
                            std::vector<float>(serial_out.begin(), serial_out.begin() + 6144));
                        const int64_t slot = tokens - 1 - t;
                        if (slot < slots) {
                            const auto begin = actual.begin() + tokens * 6144 + slot * 786432;
                            require_gdn_prefill_close(std::vector<float>(begin, begin + 786432),
                                std::vector<float>(serial_out.begin() + 6144, serial_out.end()));
                        }
                    }
                    last_state.assign(actual.begin() + tokens * 6144, actual.begin() + tokens * 6144 + 786432);
                    // Read raw spans, including padding: every source must stay bit-identical.
                    for (int i = 0; i < 6; ++i) {
                        std::vector<float> unchanged(input_data[i].size());
                        ggml_backend_tensor_get(candidate.sources[i], unchanged.data(), 0,
                                                unchanged.size() * sizeof(float));
                        ggml_backend_synchronize(hrx);
                        REQUIRE(unchanged == input_data[i]);
                    }
                }
                std::fprintf(stderr, "GDN prefill T=%lld K=%lld layout=%s: CPU/serial/replay/state handoff passed\n",
                    static_cast<long long>(tokens), static_cast<long long>(slots),
                    layout == GdnPrefillLayout::ModelNative ? "model-native" :
                    layout == GdnPrefillLayout::Packed ? "packed" : "padded");
                ggml_backend_buffer_free(cpu_buffer);
                ggml_backend_buffer_free(hrx_buffer);
                ggml_free(cpu_ctx);
                ggml_free(hrx_ctx);
            }
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::fprintf(stderr, "GDN prefill: CPU/batched GPU/serial CPU snapshots, changed-input replay and state handoff passed\n");
}
