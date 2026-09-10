static void run_qsa_projection_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_QSA_PROJECTIONS", "1");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(cpu != nullptr && hrx != nullptr);
    for (int64_t outputs : { 128, 512 }) {
        for (int64_t tokens : { 1, 2, 8 }) {
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            ggml_context * cpu_ctx = ggml_init(params);
            ggml_context * hrx_ctx = ggml_init(params);
            REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
            struct ProjectionGraph {
                ggml_tensor * weight;
                ggml_tensor * input;
                ggml_tensor * output;
                ggml_cgraph * graph;
            };
            const auto build = [=](ggml_context * ctx) {
                ProjectionGraph result = {};
                result.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 2560, outputs);
                result.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, tokens);
                result.output = ggml_mul_mat(ctx, result.weight, result.input);
                result.graph = ggml_new_graph(ctx);
                result.graph->uid = next_hrx_test_graph_uid();
                ggml_build_forward_expand(result.graph, result.output);
                return result;
            };
            const auto reference = build(cpu_ctx);
            const auto candidate = build(hrx_ctx);
            REQUIRE(ggml_backend_supports_op(hrx, candidate.output));
            {
                ScopedHrxEnvironment disabled("HRX_ENABLE_QSA_PROJECTIONS", "0");
                REQUIRE(!ggml::hrx::qsa_bf16_projection_supported(candidate.output));
            }
            auto malformed = *candidate.output;
            auto bad_input = *candidate.input;
            bad_input.ne[1] = 9;
            malformed.src[1] = &bad_input;
            REQUIRE(!ggml::hrx::qsa_bf16_projection_supported(&malformed));
            require_kernel_subsequence(scheduled_kernel_sequence(candidate.graph),
                { "hrx_owned:ggml_qsa_bf16_projection" });
            ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
            ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
            REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
            for (int replay = 0; replay < 3; ++replay) {
                const auto weights_f32 = make_pattern_f32(2560 * outputs, replay + 59);
                std::vector<ggml_bf16_t> weights(weights_f32.size());
                ggml_fp32_to_bf16_row(weights_f32.data(), weights.data(), weights.size());
                auto input = make_pattern_f32(2560 * tokens, replay + 73);
                // Exercise both round-to-even directions, including negative BF16 ties.
                input[0] = 1.00390625f;
                input[1] = 1.01171875f;
                input[2] = -1.00390625f;
                set_tensor_pair_bytes(cpu, reference.weight, hrx, candidate.weight,
                                      weights.data(), weights.size() * sizeof(ggml_bf16_t));
                set_tensor_pair_bytes(cpu, reference.input, hrx, candidate.input,
                                      input.data(), input.size() * sizeof(float));
                REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
                REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_SUCCESS);
                require_close(get_f32_tensor(hrx, candidate.output), get_f32_tensor(cpu, reference.output),
                              2.0e-4f, 2.0e-5f);
            }
            ggml_backend_buffer_free(cpu_buffer);
            ggml_backend_buffer_free(hrx_buffer);
            ggml_free(cpu_ctx);
            ggml_free(hrx_ctx);
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::fprintf(stderr, "QSA BF16 projection checks passed (CPU-compatible rounding, T1/2/8, replay)\n");
}
