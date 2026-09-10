// Include after test-hrx-ops.cpp's helpers, before main.
#include "dispatch_registration/dispatch-gated-delta-net.h"

struct GdnConvPrefillFixture {
    ggml_tensor * input = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * conv = nullptr;
    ggml_tensor * silu = nullptr;
    ggml_cgraph * conv_graph = nullptr;
    ggml_cgraph * graph = nullptr;
};

static GdnConvPrefillFixture build_gdn_conv_prefill_fixture(
    ggml_context * ctx, int64_t tokens, bool offset_views = false,
    int64_t channels = 10240, int64_t taps = 4, int64_t sequences = 1) {
    GdnConvPrefillFixture result;
    const int64_t width = taps - 1 + tokens;
    if (offset_views) {
        auto * input_backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width * channels * sequences + 7);
        auto * weight_backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, taps * channels + 11);
        result.input = ggml_view_3d(ctx, input_backing, width, channels, sequences,
            width * sizeof(float), width * channels * sizeof(float), 7 * sizeof(float));
        result.weight = ggml_view_2d(ctx, weight_backing, taps, channels, taps * sizeof(float), 11 * sizeof(float));
    } else {
        result.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, channels, sequences);
        result.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, taps, channels);
    }
    result.conv = ggml_ssm_conv(ctx, result.input, result.weight);
    ggml_set_output(result.conv);
    result.silu = ggml_silu(ctx, result.conv);
    ggml_set_output(result.silu);
    result.conv_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(result.conv_graph, result.conv);
    result.graph = ggml_new_graph(ctx);
    result.graph->uid = next_hrx_test_graph_uid();
    ggml_build_forward_expand(result.graph, result.silu);
    return result;
}

static void run_gdn_conv_prefill_scheduling_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_GDN_CONV_PREFILL", "1");
    for (int64_t tokens = 2; tokens <= 8; ++tokens) {
        for (bool offset_views : { false, true }) {
            ggml_init_params params = {};
            params.mem_size = 4 * 1024 * 1024;
            params.no_alloc = true;
            ggml_context * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            const auto fixture = build_gdn_conv_prefill_fixture(ctx, tokens, offset_views);
            REQUIRE(ggml::hrx::supports_gdn_conv_prefill_dispatch(fixture.conv));
            const auto conv_names = scheduled_kernel_sequence(fixture.conv_graph);
            REQUIRE(conv_names.size() == 1 && conv_names[0] == "qwen4exp:qwen38_gdn_conv_prefill");
            const auto names = scheduled_kernel_sequence(fixture.graph);
            REQUIRE(names.size() == 2 && names[0] == conv_names[0] && names[1] == "qwen3_moe:ggml_silu_f32");
            for (const char * flag : { static_cast<const char *>(nullptr), "0", "1x", "true" }) {
                ScopedHrxEnvironment disabled("HRX_ENABLE_GDN_CONV_PREFILL", flag);
                REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(fixture.conv));
                auto imported = ggml::hrx::import_ggml_graph(*fixture.graph);
                REQUIRE(imported.valid());
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
            }
            for (int source = 0; source < 2; ++source) {
                auto bad = *fixture.conv;
                auto input = *bad.src[source];
                bad.src[source] = &input;
                input.type = GGML_TYPE_F16;
                REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(&bad));
                input = *fixture.conv->src[source];
                input.nb[0] *= 2;
                REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(&bad));
                input = *fixture.conv->src[source];
                input.nb[1] += sizeof(float);
                REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(&bad));
                input = *fixture.conv->src[source];
                input.view_offs += 1;
                REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(&bad));
                bad.src[source] = nullptr;
                REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(&bad));
            }
            auto bad = *fixture.conv;
            bad.type = GGML_TYPE_F16;
            REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(&bad));
            bad = *fixture.conv;
            bad.nb[1] += sizeof(float);
            REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(&bad));
            bad = *fixture.conv;
            bad.view_src = fixture.input;
            REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(&bad));
            ggml_free(ctx);
        }
    }
    ggml_init_params params = {};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    for (int64_t tokens : { 0, 1, 9 }) {
        const auto fixture = build_gdn_conv_prefill_fixture(ctx, tokens);
        REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(fixture.conv));
    }
    for (int64_t taps : { 3, 5 }) {
        const auto fixture = build_gdn_conv_prefill_fixture(ctx, 8, false, 10240, taps);
        REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(fixture.conv));
    }
    const auto channels = build_gdn_conv_prefill_fixture(ctx, 8, false, 10239);
    const auto sequences = build_gdn_conv_prefill_fixture(ctx, 8, false, 10240, 4, 2);
    REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(channels.conv));
    REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(sequences.conv));
    REQUIRE(!ggml::hrx::supports_gdn_conv_prefill_dispatch(nullptr));
    ggml_free(ctx);
    std::fprintf(stderr, "GDN convolution prefill scheduling: T2..8, one conv + separate SILU, opt-in/rejections passed\n");
}

static void require_gdn_conv_prefill_close(const std::vector<float> & actual,
                                           const std::vector<float> & expected) {
    for (float value : actual) {
        REQUIRE(std::isfinite(value));
    }
    for (float value : expected) {
        REQUIRE(std::isfinite(value));
    }
    require_close(actual, expected, 2.0e-6f, 2.0e-6f);
}

static void run_gdn_conv_prefill_reference_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_GDN_CONV_PREFILL", "1");
    ggml_backend_t cpu = init_cpu_backend();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(hrx != nullptr);
    for (int64_t tokens = 2; tokens <= 8; ++tokens) {
        for (bool offset_views : { false, true }) {
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            ggml_context * cpu_ctx = ggml_init(params);
            ggml_context * hrx_ctx = ggml_init(params);
            REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
            const auto reference = build_gdn_conv_prefill_fixture(cpu_ctx, tokens, offset_views);
            const auto candidate = build_gdn_conv_prefill_fixture(hrx_ctx, tokens, offset_views);
            REQUIRE(ggml_backend_supports_op(hrx, candidate.conv));
            REQUIRE(ggml_backend_supports_op(hrx, candidate.silu));
            const auto names = scheduled_kernel_sequence(candidate.graph);
            REQUIRE(names.size() == 2 && names[0] == "qwen4exp:qwen38_gdn_conv_prefill" &&
                    names[1] == "qwen3_moe:ggml_silu_f32");
            ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
            ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
            REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
            ggml_backend_hrx_cache_stats before = {};
            REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &before));
            for (int replay = 0; replay < 4; ++replay) {
                std::vector<float> input(10240 * (tokens + 3));
                std::vector<float> weight(10240 * 4);
                for (int64_t channel = 0; channel < 10240; ++channel) {
                    for (int64_t t = 0; t < tokens + 3; ++t) {
                        input[channel * (tokens + 3) + t] =
                            static_cast<float>((channel * 31 + t * 17 + replay * 43) % 127 - 63) * 0.013f;
                    }
                    for (int64_t tap = 0; tap < 4; ++tap) {
                        weight[channel * 4 + tap] =
                            static_cast<float>((channel * 11 + tap * 19 + replay * 29) % 61 - 30) * 0.017f;
                    }
                }
                if (replay == 2) {
                    // Distinguish the CPU's sequential four-tap sum (1) from
                    // decode's pairwise sum (0), with exactly representable products.
                    input[0] = 1.0e8f;
                    input[1] = 1.0f;
                    input[2] = -1.0e8f;
                    input[3] = 1.0f;
                    std::fill(weight.begin(), weight.begin() + 4, 1.0f);
                } else if (replay == 3) {
                    std::fill(input.begin(), input.end(), 0.0f);
                }
                set_tensor_pair_bytes(cpu, reference.input, hrx, candidate.input,
                                      input.data(), input.size() * sizeof(float));
                set_tensor_pair_bytes(cpu, reference.weight, hrx, candidate.weight,
                                      weight.data(), weight.size() * sizeof(float));
                const std::vector<float> poison(10240 * tokens, std::numeric_limits<float>::quiet_NaN());
                set_tensor_pair_bytes(cpu, reference.conv, hrx, candidate.conv,
                                      poison.data(), poison.size() * sizeof(float));
                set_tensor_pair_bytes(cpu, reference.silu, hrx, candidate.silu,
                                      poison.data(), poison.size() * sizeof(float));
                REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
                REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_SUCCESS);
                const auto expected = get_f32_tensor(cpu, reference.conv);
                if (replay == 2) {
                    REQUIRE(expected[0] == 1.0f);
                }
                require_gdn_conv_prefill_close(get_f32_tensor(hrx, candidate.conv), expected);
                require_gdn_conv_prefill_close(get_f32_tensor(hrx, candidate.silu),
                                              get_f32_tensor(cpu, reference.silu));
                REQUIRE(get_f32_tensor(hrx, candidate.input) == input);
                REQUIRE(get_f32_tensor(hrx, candidate.weight) == weight);
            }
            ggml_backend_hrx_cache_stats after = {};
            REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &after));
            REQUIRE(after.graph_program_hits >= before.graph_program_hits + 3);
            REQUIRE(after.prepared_program_hits >= before.prepared_program_hits + 3);
            ggml_backend_buffer_free(cpu_buffer);
            ggml_backend_buffer_free(hrx_buffer);
            ggml_free(cpu_ctx);
            ggml_free(hrx_ctx);
            std::fprintf(stderr, "GDN convolution prefill T=%lld offset_views=%d: CPU/SILU/changed-input+weight replay passed\n",
                         static_cast<long long>(tokens), offset_views ? 1 : 0);
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
}
