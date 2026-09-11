struct PleConvFusionFixture {
    ggml_tensor * input = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
};

static PleConvFusionFixture build_ple_conv_fusion_fixture(ggml_context * ctx, int64_t tokens) {
    PleConvFusionFixture f;
    f.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tokens + 9, 10240);
    f.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 10240);
    ggml_tensor * sum = nullptr;
    for (int64_t tap = 0; tap < 4; ++tap) {
        auto * shifted = ggml_cont(ctx, ggml_transpose(ctx,
            ggml_view_2d(ctx, f.input, tokens, 10240, f.input->nb[1],
                         static_cast<size_t>(tap * 3) * sizeof(float))));
        auto * weight = ggml_cont(ctx,
            ggml_view_2d(ctx, f.weight, 1, 10240, f.weight->nb[1],
                         static_cast<size_t>(tap) * sizeof(float)));
        weight = ggml_reshape_1d(ctx, weight, 10240);
        auto * term = ggml_mul(ctx, shifted, weight);
        sum = sum == nullptr ? term : ggml_add(ctx, sum, term);
    }
    f.output = ggml_silu(ctx, sum);
    ggml_set_input(f.input);
    ggml_set_input(f.weight);
    ggml_set_output(f.output);
    f.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(f.graph, f.output);
    f.graph->uid = next_hrx_test_graph_uid();
    return f;
}

static void run_ple_conv_fusion_reference_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_PLE_CONV_FUSION", "1");
    auto * cpu = init_cpu_backend();
    auto * hrx = ggml_backend_hrx_init(0);
    REQUIRE(hrx != nullptr);
    for (int64_t tokens : { 1, 2, 4, 8 }) {
        auto * cpu_ctx = ggml_init({ 8 * 1024 * 1024, nullptr, true });
        auto * hrx_ctx = ggml_init({ 8 * 1024 * 1024, nullptr, true });
        REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
        const auto c = build_ple_conv_fusion_fixture(cpu_ctx, tokens);
        const auto h = build_ple_conv_fusion_fixture(hrx_ctx, tokens);
        REQUIRE(scheduled_kernel_sequence(h.graph).size() == 1);
        auto cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
        auto hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
        REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
        std::vector<float> input((tokens + 9) * 10240);
        std::vector<float> weight(4 * 10240);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 251) - 125) / 64.0f;
        }
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = static_cast<float>(static_cast<int>((i * 7) % 53) - 26) / 128.0f;
        }
        ggml_backend_hrx_cache_stats before = {}, first = {};
        REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &before));
        for (int replay = 0; replay < 2; ++replay) {
            set_tensor_pair_bytes(cpu, c.input, hrx, h.input, input.data(), input.size() * sizeof(float));
            set_tensor_pair_bytes(cpu, c.weight, hrx, h.weight, weight.data(), weight.size() * sizeof(float));
            const auto input_before = qsa_mask_bytes(hrx, h.input);
            const auto weight_before = qsa_mask_bytes(hrx, h.weight);
            REQUIRE(ggml_backend_graph_compute(cpu, c.graph) == GGML_STATUS_SUCCESS);
            REQUIRE(ggml_backend_graph_compute(hrx, h.graph) == GGML_STATUS_SUCCESS);
            const auto expected = qsa_mask_bytes(cpu, c.output);
            const auto actual = qsa_mask_bytes(hrx, h.output);
            REQUIRE(expected.size() == actual.size());
            for (size_t i = 0; i < expected.size(); i += sizeof(float)) {
                float e = 0.0f;
                float value = 0.0f;
                std::memcpy(&e, expected.data() + i, sizeof(float));
                std::memcpy(&value, actual.data() + i, sizeof(float));
                REQUIRE(std::abs(value - e) <= 2e-6f);
            }
            REQUIRE(qsa_mask_bytes(hrx, h.input) == input_before);
            REQUIRE(qsa_mask_bytes(hrx, h.weight) == weight_before);
            ggml_backend_hrx_cache_stats stats = {};
            REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &stats));
            if (replay == 0) {
                REQUIRE(stats.graph_program_builds == before.graph_program_builds + 1);
                REQUIRE(stats.prepared_program_builds == before.prepared_program_builds + 1);
                first = stats;
            } else {
                REQUIRE(stats.graph_program_builds == first.graph_program_builds);
                REQUIRE(stats.prepared_program_builds == first.prepared_program_builds);
                REQUIRE(stats.graph_program_hits == first.graph_program_hits + 1);
                REQUIRE(stats.prepared_program_hits == first.prepared_program_hits + 1);
            }
            input[static_cast<size_t>(replay * 97)] += 0.5f;
        }
        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::fprintf(stderr, "PLE fused depthwise convolution checks passed (T1/2/4/8, exact graph replay)\n");
}
