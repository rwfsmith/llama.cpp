static void run_recurrent_concat_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_RECURRENT_CONCAT", "1");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(cpu != nullptr && hrx != nullptr);
    for (int64_t history_count : { 3, 9 }) {
        for (int64_t tokens : { 1, 2, 4, 8 }) {
            for (bool transposed : { false, true }) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                ggml_context * cpu_ctx = ggml_init(params);
                ggml_context * hrx_ctx = ggml_init(params);
                REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
                struct ConcatGraph {
                    ggml_tensor * history;
                    ggml_tensor * input;
                    ggml_tensor * concat;
                    ggml_tensor * consumer;
                    ggml_cgraph * graph;
                };
                const auto build = [=](ggml_context * ctx) {
                    ConcatGraph result = {};
                    result.history = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, history_count, 10240);
                    result.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                        transposed ? 10240 : tokens, transposed ? tokens : 10240);
                    result.concat = ggml_concat(ctx, result.history,
                        transposed ? ggml_transpose(ctx, result.input) : result.input, 0);
                    ggml_set_output(result.concat);
                    result.consumer = ggml_scale(ctx, result.concat, 2.0f);
                    result.graph = ggml_new_graph(ctx);
                    result.graph->uid = next_hrx_test_graph_uid();
                    ggml_build_forward_expand(result.graph, result.consumer);
                    return result;
                };
                const auto reference = build(cpu_ctx);
                const auto candidate = build(hrx_ctx);
                REQUIRE(ggml_backend_supports_op(hrx, candidate.concat));
                for (const char * flag : { static_cast<const char *>(nullptr), "0", "1x" }) {
                    ScopedHrxEnvironment disabled("HRX_ENABLE_RECURRENT_CONCAT", flag);
                    REQUIRE(!ggml_backend_supports_op(hrx, candidate.concat));
                }
                auto malformed = *candidate.concat;
                malformed.op_params[0] = 1;
                REQUIRE(!ggml::hrx::recurrent_concat_supported(&malformed));
                malformed = *candidate.concat;
                auto malformed_input = *candidate.concat->src[1];
                malformed_input.nb[1] += sizeof(float);
                malformed.src[1] = &malformed_input;
                REQUIRE(!ggml::hrx::recurrent_concat_supported(&malformed));
                require_kernel_subsequence(scheduled_kernel_sequence(candidate.graph),
                    { "hrx_owned:ggml_recurrent_concat_f32" });
                ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
                ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
                REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
                for (int replay = 0; replay < 3; ++replay) {
                    const auto history = make_pattern_f32(history_count * 10240, replay + 13);
                    const auto input = make_pattern_f32(tokens * 10240, replay + 43);
                    set_tensor_pair_bytes(cpu, reference.history, hrx, candidate.history,
                                          history.data(), history.size() * sizeof(float));
                    set_tensor_pair_bytes(cpu, reference.input, hrx, candidate.input,
                                          input.data(), input.size() * sizeof(float));
                    REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
                    REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_SUCCESS);
                    REQUIRE(get_f32_tensor(hrx, candidate.concat) == get_f32_tensor(cpu, reference.concat));
                    REQUIRE(get_f32_tensor(hrx, candidate.consumer) == get_f32_tensor(cpu, reference.consumer));
                }
                ggml_backend_buffer_free(cpu_buffer);
                ggml_backend_buffer_free(hrx_buffer);
                ggml_free(cpu_ctx);
                ggml_free(hrx_ctx);
            }
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::fprintf(stderr, "Recurrent CONCAT checks passed (both layouts, history3/9, T1/2/4/8, exact replay)\n");
}
