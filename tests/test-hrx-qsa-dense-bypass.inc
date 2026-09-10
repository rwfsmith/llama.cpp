// CPU-only proof of the full-selection mask identity. This does not instantiate
// a llama model; production graph-reuse/cache transitions need a model-level run.
static void run_qsa_dense_bypass_checks() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    int cases = 0;
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        for (int64_t streams : { 1, 2 }) {
            for (int64_t queries : { 1, 3 }) {
                for (int64_t ratio : { 1, 4 }) {
                    const int64_t budget = 8 + ratio - 1;
                    // Below, at and above the dense/sparse boundary.
                    for (int64_t cells : { budget - 1, budget, budget + 1 }) {
                        for (int scenario = 0; scenario < 4; ++scenario) {
                            ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
                            REQUIRE(ctx != nullptr);
                            const int64_t width = std::min(cells, budget);
                            auto * mask = ggml_new_tensor_4d(ctx, type, cells, queries, 1, streams);
                            auto * scores = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cells, queries, streams);
                            auto * top_k = ggml_cont(ctx, ggml_top_k(ctx, scores, width));
                            top_k = ggml_reshape_4d(ctx, top_k, width, queries, 1, streams);
                            ggml_set_output(top_k);
                            auto * all = ggml_fill(ctx, mask, -INFINITY);
                            all = ggml_view_4d(ctx, all, 1, cells, queries, streams,
                                             all->nb[0], all->nb[1], all->nb[2], 0);
                            auto * ids = ggml_view_4d(ctx, top_k, width, queries, streams, 1,
                                                     top_k->nb[1], top_k->nb[2], streams * top_k->nb[3], 0);
                            auto * zeros = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, width, queries, streams);
                            zeros = ggml_fill(ctx, zeros, 0.0f);
                            auto * selected = ggml_set_rows(ctx, all, zeros, ids);
                            selected = ggml_view_4d(ctx, selected, cells, queries, 1, streams,
                                                   selected->nb[2], selected->nb[3], selected->nb[3], 0);
                            auto * legacy = ggml_add(ctx, selected, mask);
                            auto * graph = ggml_new_graph(ctx);
                            ggml_build_forward_expand(graph, legacy);
                            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
                            REQUIRE(buffer != nullptr);
                            std::vector<float> mask_values(cells * queries * streams);
                            std::vector<float> score_values(mask_values.size());
                            for (int64_t stream = 0; stream < streams; ++stream) {
                                for (int64_t query = 0; query < queries; ++query) {
                                    for (int64_t cell = 0; cell < cells; ++cell) {
                                        const size_t at = (stream * queries + query) * cells + cell;
                                        const bool masked = scenario == 2 ||
                                            (scenario == 1 && cell > (query + stream) % cells);
                                        mask_values[at] = masked ? -INFINITY :
                                            scenario == 3 ? -0.125f * static_cast<float>(cell + stream) : 0.0f;
                                        // Ties, causal restrictions, entirely masked rows and finite mask biases.
                                        score_values[at] = masked ? -INFINITY :
                                            scenario == 0 ? 1.0f : static_cast<float>((cell * 7 + query + stream) % 5);
                                    }
                                }
                            }
                            if (type == GGML_TYPE_F16) {
                                std::vector<ggml_fp16_t> half(mask_values.size());
                                ggml_fp32_to_fp16_row(mask_values.data(), half.data(), half.size());
                                set_tensor_bytes(cpu, mask, half.data(), half.size() * sizeof(ggml_fp16_t));
                            } else {
                                set_tensor_bytes(cpu, mask, mask_values.data(), mask_values.size() * sizeof(float));
                            }
                            set_tensor_bytes(cpu, scores, score_values.data(), score_values.size() * sizeof(float));
                            REQUIRE(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
                            const auto actual = get_f32_tensor(cpu, legacy);
                            const auto original = get_f32_tensor(cpu, mask);
                            REQUIRE(original == mask_values);
                            std::vector<int32_t> chosen(width * queries * streams);
                            ggml_backend_tensor_get(top_k, chosen.data(), 0, chosen.size() * sizeof(int32_t));
                            for (int64_t row = 0; row < queries * streams; ++row) {
                                std::vector<bool> seen(cells, false);
                                for (int64_t i = 0; i < width; ++i) {
                                    const int32_t id = chosen[row * width + i];
                                    REQUIRE(id >= 0 && id < cells && !seen[id]);
                                    seen[id] = true;
                                }
                                for (int64_t cell = 0; cell < cells; ++cell) {
                                    const size_t at = row * cells + cell;
                                    const float expected = seen[cell] ? original[at] : -INFINITY;
                                    REQUIRE(actual[at] == expected);
                                }
                            }
                            if (width == cells) {
                                REQUIRE(actual == original);
                            } else if (scenario == 0 || scenario == 3) {
                                // Removing selection past the threshold is observably wrong.
                                REQUIRE(actual != original);
                            }
                            ++cases;
                            ggml_backend_buffer_free(buffer);
                            ggml_free(ctx);
                        }
                    }
                }
            }
        }
    }
    ggml_backend_free(cpu);
    std::fprintf(stderr, "QSA dense-bypass CPU mask checks passed (%d cases: F16/F32, full permutations, ties, causal/all-masked, streams, threshold)\n",
                 cases);
}
