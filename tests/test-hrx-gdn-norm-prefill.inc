static void run_gdn_norm_prefill_checks(bool run_gpu = true) {
    for (int64_t tokens = 2; tokens <= 8; ++tokens) {
        for (int64_t offset : { 0, 2048 }) {
            const float eps = offset == 0 ? 1.0e-6f : 0.125f;
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            auto * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            auto * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10240, tokens);
            ggml_set_input(source);
            auto * view = ggml_view_3d(ctx, source, 128, 16, tokens,
                                      128 * sizeof(float), 10240 * sizeof(float), offset * sizeof(float));
            auto * result = ggml_l2_norm(ctx, view, eps);
            auto * graph = ggml_new_graph(ctx);
            graph->uid = next_hrx_test_graph_uid();
            ggml_build_forward_expand(graph, result);
            for (const char * flag : { static_cast<const char *>(nullptr), "0", "true", "1" }) {
                ScopedHrxEnvironment enabled("HRX_ENABLE_GDN_NORM_PREFILL", flag);
                REQUIRE(ggml::hrx::gdn_norm_prefill_supported(result) ==
                        (flag != nullptr && std::strcmp(flag, "1") == 0));
            }
            ScopedHrxEnvironment enabled("HRX_ENABLE_GDN_NORM_PREFILL", "1");
            auto sequence = scheduled_kernel_sequence(graph);
            REQUIRE(sequence.size() == 2);
            REQUIRE(sequence[0] == "qwen3_moe:ggml_copy_rows_f32");
            REQUIRE(sequence[1] == "qwen3_moe:ggml_l2_norm_f32");
            const size_t stride = view->nb[2];
            view->nb[2] += sizeof(float);
            REQUIRE(!ggml::hrx::gdn_norm_prefill_supported(result));
            view->nb[2] = stride;
            for (int64_t unsupported_tokens : { 1, 9 }) {
                view->ne[2] = unsupported_tokens;
                REQUIRE(!ggml::hrx::gdn_norm_prefill_supported(result));
            }
            view->ne[2] = tokens;
            for (float invalid : { -1.0f, std::numeric_limits<float>::quiet_NaN() }) {
                std::memcpy(result->op_params, &invalid, sizeof(invalid));
                REQUIRE(!ggml::hrx::gdn_norm_prefill_supported(result));
            }
            std::memcpy(result->op_params, &eps, sizeof(eps));
            ggml_backend_t backend = run_gpu ? ggml_backend_hrx_init(0) : nullptr;
            ggml_backend_buffer_t buffer = nullptr;
            if (run_gpu) {
                REQUIRE(backend != nullptr && ggml_backend_supports_op(backend, result));
                buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                REQUIRE(buffer != nullptr);
            }
            for (int replay = 0; replay < 2; ++replay) {
                auto input = make_pattern_f32(10240 * tokens, 19 + replay * 13, 0.03125f);
                std::fill(input.begin() + offset, input.begin() + offset + 128, 0.0f);
                std::fill(input.begin() + offset + 128, input.begin() + offset + 256, 1.0e-8f);
                std::vector<float> expected(2048 * tokens);
                for (int64_t token = 0; token < tokens; ++token) {
                    for (int64_t head = 0; head < 16; ++head) {
                        const int64_t begin = token * 10240 + offset + head * 128;
                        double sum = 0.0;
                        for (int64_t k = 0; k < 128; ++k) {
                            sum += static_cast<double>(input[begin + k]) * input[begin + k];
                        }
                        const float scale = 1.0f / std::max(std::sqrt(static_cast<float>(sum)), eps);
                        for (int64_t k = 0; k < 128; ++k) {
                            expected[token * 2048 + head * 128 + k] = input[begin + k] * scale;
                        }
                    }
                }
                if (run_gpu) {
                    set_tensor_bytes(backend, source, input.data(), input.size() * sizeof(float));
                    const std::vector<float> dirty(expected.size(), std::numeric_limits<float>::quiet_NaN());
                    set_tensor_bytes(backend, result, dirty.data(), dirty.size() * sizeof(float));
                    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    auto actual = get_f32_tensor(backend, result);
                    for (float value : actual) {
                        REQUIRE(std::isfinite(value));
                    }
                    require_close(actual, expected, 1.0e-6f, 1.0e-5f);
                    require_close(get_f32_tensor(backend, source), input, 0.0f, 0.0f);
                    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    require_close(get_f32_tensor(backend, result), actual, 0.0f, 0.0f);
                }
            }
            if (run_gpu) {
                ggml_backend_hrx_cache_stats stats = {};
                REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &stats));
                REQUIRE(stats.graph_program_hits >= 3 && stats.prepared_program_hits >= 3);
                ggml_backend_buffer_free(buffer);
                ggml_backend_free(backend);
            }
            ggml_free(ctx);
        }
    }
}
