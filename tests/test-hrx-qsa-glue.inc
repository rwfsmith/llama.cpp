static void run_qsa_glue_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_QSA_GLUE", "1");
    ScopedHrxEnvironment glue("HRX_ENABLE_SMALL_BATCH_GLUE", "1");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(cpu != nullptr && hrx != nullptr);
    for (int64_t tokens : { 1, 2, 3, 4, 5, 6, 7, 8, 32, 128, 129, 512 }) {
        for (int layout : { 0, 1, 2 }) {
            if (tokens > 8 && layout != 0) {
                continue;
            }
            const bool broadcast = layout == 1;
            const bool strided = layout == 2;
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            ggml_context * cpu_ctx = ggml_init(params);
            ggml_context * hrx_ctx = ggml_init(params);
            REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
            struct AddGraph {
                ggml_tensor * a;
                ggml_tensor * b;
                ggml_tensor * output;
                ggml_cgraph * graph;
            };
            const auto build = [&](ggml_context * ctx) {
                AddGraph result = {};
                result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, tokens);
                result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, strided ? 256 : 128, broadcast ? 1 : tokens);
                ggml_tensor * b_view = strided ?
                    ggml_view_2d(ctx, result.b, 128, tokens, 256 * sizeof(float), 128 * sizeof(float)) : result.b;
                result.output = ggml_add(ctx, result.a, b_view);
                result.graph = ggml_new_graph(ctx);
                result.graph->uid = next_hrx_test_graph_uid();
                ggml_build_forward_expand(result.graph, result.output);
                return result;
            };
            const auto reference = build(cpu_ctx);
            const auto candidate = build(hrx_ctx);
            REQUIRE(ggml_backend_supports_op(hrx, candidate.output));
            for (const char * flag : { static_cast<const char *>(nullptr), "0", "1x" }) {
                ScopedHrxEnvironment disabled("HRX_ENABLE_QSA_GLUE", flag);
                REQUIRE(!ggml::hrx::supports_qsa_add_dispatch(candidate.output));
            }
            auto bad = *candidate.output;
            bad.view_src = candidate.a;
            REQUIRE(!ggml::hrx::supports_qsa_add_dispatch(&bad));
            bad = *candidate.output;
            bad.ne[1] = 65537;
            REQUIRE(!ggml::hrx::supports_qsa_add_dispatch(&bad));
            ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
            ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
            REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
            for (int replay = 0; replay < 3; ++replay) {
                auto a = make_pattern_f32(128 * tokens, 113 + replay);
                auto b = make_pattern_f32(ggml_nelements(candidate.b), 127 + replay);
                const size_t mask_start = strided ? 128 : 0;
                b[mask_start + replay] = -std::numeric_limits<float>::infinity();
                b[mask_start + replay + 4] = std::numeric_limits<float>::infinity();
                b[mask_start + replay + 8] = std::numeric_limits<float>::quiet_NaN();
                set_tensor_pair_bytes(cpu, reference.a, hrx, candidate.a, a.data(), a.size() * sizeof(float));
                set_tensor_pair_bytes(cpu, reference.b, hrx, candidate.b, b.data(), b.size() * sizeof(float));
                REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
                REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_SUCCESS);
                const auto expected = get_f32_tensor(cpu, reference.output);
                const auto actual = get_f32_tensor(hrx, candidate.output);
                REQUIRE(actual.size() == expected.size());
                for (size_t i = 0; i < expected.size(); ++i) {
                    REQUIRE(std::isnan(expected[i]) ? std::isnan(actual[i]) : actual[i] == expected[i]);
                }
            }
            ggml_backend_buffer_free(cpu_buffer);
            ggml_backend_buffer_free(hrx_buffer);
            ggml_free(cpu_ctx);
            ggml_free(hrx_ctx);
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::fprintf(stderr, "QSA ADD checks passed (T1..8, pooled rows, broadcast/strided, masks, exact replay)\n");
}
