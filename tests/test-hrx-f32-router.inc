static void run_f32_router_projection_checks() {
    ScopedHrxEnvironment router("HRX_ENABLE_QWEN4EXP_ROUTER", "1");
    for (const char * flag : { static_cast<const char *>(nullptr), "0", "1", "1x" }) {
        ScopedHrxEnvironment projection("HRX_ENABLE_F32_ROUTER", flag);
        const bool enabled = flag != nullptr && std::strcmp(flag, "1") == 0;
        REQUIRE(ggml::hrx::dense_f32_shape_supported(2560, 512, 1) == enabled);
        REQUIRE(ggml::hrx::dense_f32_shape_supported(2560, 512, 8) == enabled);
        REQUIRE(!ggml::hrx::dense_f32_shape_supported(2560, 512, 9));
        REQUIRE(!ggml::hrx::dense_f32_shape_supported(2048, 512, 1));
        REQUIRE(!ggml::hrx::dense_f32_shape_supported(2560, 513, 1));
        REQUIRE(ggml::hrx::dense_f32_shape_supported(2560, 48, 1));
    }
    ScopedHrxEnvironment projection("HRX_ENABLE_F32_ROUTER", "1");
    {
        ScopedHrxEnvironment disabled_router("HRX_ENABLE_QWEN4EXP_ROUTER", "0");
        REQUIRE(!ggml::hrx::dense_f32_shape_supported(2560, 512, 1));
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(cpu != nullptr && hrx != nullptr);
    for (int64_t tokens : { 1, 2, 8 }) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * cpu_ctx = ggml_init(params);
        ggml_context * hrx_ctx = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
        struct RouterGraph {
            ggml_tensor * weight;
            ggml_tensor * input;
            ggml_tensor * logits;
            ggml_tensor * ids;
            ggml_tensor * output;
            ggml_cgraph * graph;
        };
        const auto build = [tokens](ggml_context * ctx) {
            RouterGraph result = {};
            result.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 512);
            result.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, tokens);
            result.logits = ggml_mul_mat(ctx, result.weight, result.input);
            ggml_set_output(result.logits);
            result.output = build_qwen_router_top8_graph(ctx, result.logits, &result.ids, 512, 10);
            result.graph = ggml_new_graph(ctx);
            result.graph->uid = next_hrx_test_graph_uid();
            ggml_build_forward_expand(result.graph, result.output);
            return result;
        };
        const auto reference = build(cpu_ctx);
        const auto candidate = build(hrx_ctx);
        REQUIRE(ggml_backend_supports_op(hrx, candidate.logits));
        {
            ScopedHrxEnvironment disabled("HRX_ENABLE_F32_ROUTER", "0");
            REQUIRE(!ggml_backend_supports_op(hrx, candidate.logits));
        }
        require_kernel_subsequence(scheduled_kernel_sequence(candidate.graph),
            { "qwen3_moe:ggml_dense_matmul_f32", "qwen3_moe:qwen3_moe_router_top8_f32" });
        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
        ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
        REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
        for (int replay = 0; replay < 3; ++replay) {
            auto weights = make_pattern_f32(2560 * 512, replay + 31, 0.00001f);
            auto input = make_pattern_f32(2560 * tokens, replay + 7);
            // Separate route scores clearly; alternating signs reverses the selected experts.
            for (int expert = 0; expert < 512; ++expert) {
                weights[expert * 2560] = (expert - 256) * 0.01f;
            }
            for (int64_t token = 0; token < tokens; ++token) {
                input[token * 2560] = replay % 2 == 0 ? 1.0f : -1.0f;
            }
            set_tensor_pair_bytes(cpu, reference.weight, hrx, candidate.weight,
                                  weights.data(), weights.size() * sizeof(float));
            set_tensor_pair_bytes(cpu, reference.input, hrx, candidate.input,
                                  input.data(), input.size() * sizeof(float));
            REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
            REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_SUCCESS);
            require_close(get_f32_tensor(hrx, candidate.logits), get_f32_tensor(cpu, reference.logits),
                          5.0e-5f, 2.0e-5f);
            require_close(get_f32_tensor(hrx, candidate.output), get_f32_tensor(cpu, reference.output),
                          2.0e-5f, 2.0e-5f);
            for (int64_t token = 0; token < tokens; ++token) {
                std::vector<int32_t> expected_ids(10), actual_ids(10);
                ggml_backend_tensor_get(reference.ids, expected_ids.data(), token * reference.ids->nb[1],
                                        expected_ids.size() * sizeof(int32_t));
                ggml_backend_tensor_get(candidate.ids, actual_ids.data(), token * candidate.ids->nb[1],
                                        actual_ids.size() * sizeof(int32_t));
                REQUIRE(actual_ids == expected_ids);
            }
        }
        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::fprintf(stderr, "F32 router projection checks passed (logits, top-10 IDs/weights, changed-input replay)\n");
}
