// Included by test-hrx-ops.cpp after its IQ/MoE helpers.
static const char * iq_packet_kernel_name(ggml_type type, bool packet) {
    return type == GGML_TYPE_IQ3_XXS ?
        (packet ? "hrx_owned:ggml_mul_mat_id_iq3_xxs_f32_packet4" :
                  "hrx_owned:ggml_mul_mat_id_iq3_xxs_f32") :
        (packet ? "hrx_owned:ggml_mul_mat_id_iq4_xs_f32_packet4" :
                  "hrx_owned:ggml_mul_mat_id_iq4_xs_f32");
}

static void run_iq_packet4_scheduling_checks() {
    ScopedHrxEnvironment iq("HRX_ENABLE_IQ_EXPERTS", "1");
    ScopedHrxEnvironment small("HRX_ENABLE_MOE_SMALL_BATCH", "1");
    for (ggml_type type : { GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS }) {
        for (const char * flag : { static_cast<const char *>(nullptr), "0", "true", "1" }) {
            ScopedHrxEnvironment packet("HRX_ENABLE_IQ_PACKET4", flag);
            for (int64_t tokens = 1; tokens <= 9; ++tokens) {
                for (int64_t stride : { 10, 17, 512 }) {
                    ggml_init_params params = {};
                    params.mem_size = 1024 * 1024;
                    params.no_alloc = true;
                    ggml_context * ctx = ggml_init(params);
                    REQUIRE(ctx != nullptr);
                    auto * weight = ggml_new_tensor_3d(ctx, type, 2560, 640, 512);
                    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 1, tokens);
                    auto * ids = make_moe_ids_tensor(ctx, tokens, stride);
                    auto * output = ggml_mul_mat_id(ctx, weight, input, ids);
                    auto * graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, output);
                    auto imported = ggml::hrx::import_ggml_graph(*graph);
                    REQUIRE(imported.valid());
                    ggml::hrx::DispatchScheduler scheduler;
                    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == (tokens <= 8));
                    if (tokens <= 8) {
                        const auto & plan = scheduler.plan();
                        REQUIRE(plan.valid());
                        REQUIRE(plan.dispatches.size() == 1);
                        const auto & dispatch = plan.dispatches[0];
                        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
                                iq_packet_kernel_name(type, flag != nullptr && std::strcmp(flag, "1") == 0));
                        REQUIRE(dispatch.kernel.integer_parameters.at("input_size") == 2560);
                        REQUIRE(dispatch.kernel.integer_parameters.at("output_size") == 640);
                        REQUIRE(dispatch.kernel.integer_parameters.at("expert_count") == 512);
                        REQUIRE(dispatch.kernel.integer_parameters.at("route_count") == 10);
                        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tokens);
                        REQUIRE(dispatch.kernel.integer_parameters.at("route_id_stride") == stride);
                        REQUIRE(dispatch.bindings.size() == 5);
                        REQUIRE(dispatch.bindings[0].length == ggml_nbytes(weight));
                        REQUIRE(dispatch.bindings[1].length == ggml_nbytes(input));
                        REQUIRE(dispatch.bindings[2].length == ((tokens - 1) * stride + 10) * sizeof(int32_t));
                        REQUIRE(dispatch.bindings[3].length == 1168);
                        REQUIRE(dispatch.bindings[4].length == ggml_nbytes(output));
                        REQUIRE(plan.constant_initializations.size() == 1);
                    }
                    ggml_free(ctx);
                }
            }
        }
    }
    // The packet switch must not bypass the existing IQ and small-batch gates.
    ScopedHrxEnvironment packet("HRX_ENABLE_IQ_PACKET4", "1");
    for (bool disable_iq : { false, true }) {
        ScopedHrxEnvironment iq_gate("HRX_ENABLE_IQ_EXPERTS", disable_iq ? "0" : "1");
        ScopedHrxEnvironment small_gate("HRX_ENABLE_MOE_SMALL_BATCH", "0");
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        const int64_t tokens = disable_iq ? 1 : 2;
        auto * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_XXS, 2560, 640, 512);
        auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 1, tokens);
        auto * ids = make_moe_ids_tensor(ctx, tokens, 17);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ggml_mul_mat_id(ctx, weight, input, ids));
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
        ggml_free(ctx);
    }
}

static void run_iq_packet4_reference_case(ggml_type type, int64_t tokens, int64_t stride) {
    ScopedHrxEnvironment iq("HRX_ENABLE_IQ_EXPERTS", "1");
    ScopedHrxEnvironment small("HRX_ENABLE_MOE_SMALL_BATCH", "1");
    constexpr int64_t width = 2560, rows = 640, experts = 512, routes = 10;
    const int64_t assignments = tokens * routes;
    const size_t block_bytes = ggml_type_size(type);
    const size_t row_bytes = ggml_row_size(type, width);
    std::vector<uint8_t> weight(experts * rows * row_bytes, 0);
    const auto original_ids = make_moe_ids(tokens);
    for (int32_t expert : original_ids) {
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t block = 0; block < width / 256; ++block) {
                const size_t base = (static_cast<size_t>(expert) * rows + row) * row_bytes + block * block_bytes;
                // Finite FP16 extremes, signed zero, both signs, and small normal scales.
                // Extreme rows below use same-sign weights to avoid making the usual
                // projection tolerance an ill-conditioned cancellation test.
                const float scales[] = { 0.0f, -0.0f, 0.00006103515625f, -0.00006103515625f,
                                         0.00146484375f, -0.00146484375f, 65504.0f, -65504.0f };
                const ggml_fp16_t scale = ggml_fp32_to_fp16(scales[row % 8]);
                std::memcpy(weight.data() + base, &scale, sizeof(scale));
                uint32_t state = 0xa341316cu ^ (static_cast<uint32_t>(expert) * 0x9e3779b9u) ^
                                 (static_cast<uint32_t>(row) * 0x85ebca6bu) ^ static_cast<uint32_t>(block);
                for (size_t byte = sizeof(scale); byte < block_bytes; ++byte) {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    weight[base + byte] = static_cast<uint8_t>(state);
                }
                if (type == GGML_TYPE_IQ3_XXS) {
                    // Alternate minimum/maximum local scales; preserve random sign codes.
                    for (size_t group = 0; group < 8; ++group) {
                        auto & top = weight[base + 69 + group * 4];
                        top = static_cast<uint8_t>((top & 15) | ((group % 2) ? 0xf0 : 0));
                    }
                    if (row % 8 >= 6) {
                        for (size_t group = 0; group < 8; ++group) {
                            std::memset(weight.data() + base + 66 + group * 4, 0, 4);
                            weight[base + 69 + group * 4] = 0xf0;
                        }
                    }
                } else {
                    // Group scales alternate -32 and +31, hitting both signed endpoints.
                    weight[base + 2] = 0xcc;
                    weight[base + 3] = 0xcc;
                    for (size_t byte = 4; byte < 8; ++byte) {
                        weight[base + byte] = 0xf0;
                    }
                    if (row % 8 >= 6) {
                        std::memset(weight.data() + base + 2, 0xff, block_bytes - 2);
                    }
                }
            }
        }
    }
    const auto * traits = ggml_get_type_traits(type);
    REQUIRE(traits != nullptr && traits->to_float != nullptr);
    std::vector<std::vector<float>> scalar_results(2);
    std::vector<std::vector<float>> expected(2);
    // Fresh backends/graphs make changing the opt-in independent of program caching.
    for (bool packet : { false, true }) {
        ScopedHrxEnvironment option("HRX_ENABLE_IQ_PACKET4", packet ? "1" : "0");
        ggml_backend_t backend = ggml_backend_hrx_init(0);
        REQUIRE(backend != nullptr);
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        auto * w = ggml_new_tensor_3d(ctx, type, width, rows, experts);
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, tokens);
        auto * ids_tensor = make_moe_ids_tensor(ctx, tokens, stride);
        auto * output = ggml_mul_mat_id(ctx, w, x, ids_tensor);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        require_kernel_subsequence(scheduled_kernel_sequence(graph), { iq_packet_kernel_name(type, packet) });
        graph->uid = next_hrx_test_graph_uid();
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer != nullptr);
        set_tensor_bytes(backend, w, weight.data(), weight.size());
        auto ids = original_ids;
        for (int replay = 0; replay < 2; ++replay) {
            if (replay != 0) {
                std::rotate(ids.begin(), ids.begin() + 3, ids.end());
            }
            auto input = make_pattern_f32(width * tokens, 23 + replay * 8, 0.0013f);
            for (float & value : input) {
                value = std::fabs(value) + 0.0001f;
            }
            if (!packet) {
                expected[replay].resize(rows * assignments);
                std::vector<float> decoded(width);
                for (int64_t assignment = 0; assignment < assignments; ++assignment) {
                    for (int64_t row = 0; row < rows; ++row) {
                        const size_t base = (static_cast<size_t>(ids[assignment]) * rows + row) * row_bytes;
                        traits->to_float(weight.data() + base, decoded.data(), width);
                        double sum = 0;
                        for (int64_t col = 0; col < width; ++col) {
                            sum += static_cast<double>(decoded[col]) * input[assignment / routes * width + col];
                        }
                        expected[replay][assignment * rows + row] = static_cast<float>(sum);
                    }
                }
            }
            set_tensor_bytes(backend, x, input.data(), input.size() * sizeof(float));
            set_moe_ids(backend, ids_tensor, ids);
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            const auto actual = get_f32_tensor(backend, output);
            for (float value : actual) {
                REQUIRE(std::isfinite(value));
            }
            require_close(actual, expected[replay], 3.0e-4f, 1.0e-5f);
            if (packet) {
                require_close(actual, scalar_results[replay], 3.0e-4f, 1.0e-5f);
            } else {
                scalar_results[replay] = actual;
            }
            auto invalid_ids = ids;
            invalid_ids[1] = -1;
            invalid_ids[assignments - 2] = experts;
            set_moe_ids(backend, ids_tensor, invalid_ids);
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            const auto invalid_output = get_f32_tensor(backend, output);
            std::vector<float> valid_actual, valid_expected;
            for (int64_t assignment = 0; assignment < assignments; ++assignment) {
                for (int64_t row = 0; row < rows; ++row) {
                    const size_t index = assignment * rows + row;
                    if (invalid_ids[assignment] < 0 || invalid_ids[assignment] >= experts) {
                        REQUIRE(std::isnan(invalid_output[index]));
                    } else {
                        REQUIRE(std::isfinite(invalid_output[index]));
                        valid_actual.push_back(invalid_output[index]);
                        valid_expected.push_back(expected[replay][index]);
                    }
                }
            }
            require_close(valid_actual, valid_expected, 3.0e-4f, 1.0e-5f);
        }
        ggml_backend_hrx_cache_stats stats = {};
        REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &stats));
        REQUIRE(stats.graph_program_hits >= 3 && stats.prepared_program_hits >= 3);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}

static void run_iq_packet4_reference_checks() {
    for (ggml_type type : { GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS }) {
        for (int64_t tokens = 1; tokens <= 8; ++tokens) {
            run_iq_packet4_reference_case(type, tokens, tokens % 2 == 0 ? 512 : 17);
        }
    }
}
