// Reuses the existing raw-F32 GEMVs; the old F32-accum WMMA route has F16 operands.
static const char * dense_f32_gemv_kernel_name(ggml_type type) {
    return type == GGML_TYPE_Q8_0 ? "hrx_owned:ggml_dense_q8_0_gemv_f32" :
                                  "qwen3_moe:ggml_linear_q6k_f32_wave64";
}

static void run_dense_f32_gemv_scheduling_checks() {
    ScopedHrxEnvironment rounded("HRX_ENABLE_DENSE_ROUNDED_GEMV", "0");
    ScopedHrxEnvironment q8("HRX_ENABLE_Q8_GEMV", "0");
    for (const char * accum_flag : { "0", "1" }) {
        ScopedHrxEnvironment accum("HRX_ENABLE_DENSE_F32_ACCUM", accum_flag);
        for (const char * flag : { static_cast<const char *>(nullptr), "0", "true", "1" }) {
            ScopedHrxEnvironment gemv("HRX_ENABLE_DENSE_F32_GEMV", flag);
            for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 }) {
                for (int64_t tokens : { 1, 2, 8, 32 }) {
                    ggml_init_params params = {};
                    params.mem_size = 1024 * 1024;
                    params.no_alloc = true;
                    auto * ctx = ggml_init(params);
                    REQUIRE(ctx != nullptr);
                    auto * w = ggml_new_tensor_2d(ctx, type, 2560, 65);
                    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, tokens);
                    auto * y = ggml_mul_mat(ctx, w, x);
                    auto * graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, y);
                    auto imported = ggml::hrx::import_ggml_graph(*graph);
                    REQUIRE(imported.valid());
                    ggml::hrx::DispatchScheduler scheduler;
                    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
                    const auto & plan = scheduler.plan();
                    REQUIRE(plan.valid() && plan.dispatches.size() == 1);
                    const auto & dispatch = plan.dispatches[0];
                    const bool selected = flag != nullptr && std::strcmp(flag, "1") == 0 &&
                                          tokens == 1 && type != GGML_TYPE_Q4_K;
                    const char * legacy = type == GGML_TYPE_Q4_K ?
                        "qwen3_moe:qwen3_moe_dense_linear_q4k_f16_wmma" : type == GGML_TYPE_Q6_K ?
                        "qwen3_moe:qwen3_moe_dense_linear_q6k_f16_wmma" :
                        "qwen3_moe:qwen3_moe_dense_linear_q8_0_f16_wmma";
                    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
                            (selected ? dense_f32_gemv_kernel_name(type) :
                             std::strcmp(accum_flag, "1") == 0 ? dense_f32_kernel_name(type) : legacy));
                    REQUIRE(dispatch.bindings.size() == 3);
                    REQUIRE(dispatch.bindings[0].length == ggml_nbytes(x));
                    REQUIRE(dispatch.bindings[1].length == ggml_nbytes(w));
                    REQUIRE(dispatch.bindings[2].length == ggml_nbytes(y));
                    if (selected) {
                        REQUIRE(dispatch.kernel.integer_parameters.at("input_size") == 2560);
                        REQUIRE(dispatch.kernel.integer_parameters.at("output_size") == 65);
                        if (type == GGML_TYPE_Q6_K) {
                            REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == 1);
                            REQUIRE(dispatch.kernel.compile_parameters.at("ggml.linear_q6k_f32.token_capacity") == "1");
                            REQUIRE(dispatch.kernel.compile_parameters.at("ggml.linear_q6k_f32.output_capacity") == "65");
                        } else {
                            REQUIRE(dispatch.kernel.compile_parameters.empty());
                        }
                    }
                    ggml_free(ctx);
                }
            }
        }
    }
}

static std::vector<uint8_t> dense_f32_gemv_weights(ggml_type type, int64_t width, int64_t rows, int replay) {
    const size_t block_bytes = ggml_type_size(type);
    const int64_t block_width = ggml_blck_size(type);
    std::vector<uint8_t> weight(ggml_row_size(type, width) * rows);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t block = 0; block < width / block_width; ++block) {
            auto * dst = weight.data() + row * ggml_row_size(type, width) + block * block_bytes;
            uint32_t state = 0xa341316cu ^ static_cast<uint32_t>(row * 7919 + block * 104729 + replay * 31337);
            for (size_t byte = 0; byte < block_bytes; ++byte) {
                state = state * 1664525u + 1013904223u;
                dst[byte] = static_cast<uint8_t>(state >> 24);
            }
            float scale = (row % 2 ? -1.0f : 1.0f) / (replay == 0 ? 1024.0f : 512.0f);
            if (type == GGML_TYPE_Q6_K) {
                for (int i = 0; i < 16; ++i) {
                    const int8_t scales[] = { -4, -1, 1, 4 };
                    std::memcpy(dst + 192 + i, &scales[(i + replay) % 4], 1);
                }
            }
            if (replay == 1 && row == 0) {
                // Constant exactly representable weights isolate lost F32 input bits.
                scale = 0.0625f;
                if (type == GGML_TYPE_Q8_0) {
                    std::memset(dst + 2, 1, 32);
                } else {
                    std::memset(dst, 0x11, 128);
                    std::memset(dst + 128, 0xaa, 64);
                    std::memset(dst + 192, 1, 16);
                }
            }
            if (replay == 2) {
                const float scales[] = { 65504.0f, -65504.0f, 0.00006103515625f,
                                         -0.00006103515625f, 0.0f };
                scale = scales[row % 5];
                if (type == GGML_TYPE_Q8_0) {
                    std::memset(dst + 2, row % 2 ? 0x80 : 0x7f, 32);
                } else {
                    // Signed scale/code endpoints, constant within each row.
                    std::memset(dst, row % 2 ? 0 : 0xff, 192);
                    std::memset(dst + 192, row % 2 ? 0x80 : 0x7f, 16);
                }
            }
            const ggml_fp16_t half = ggml_fp32_to_fp16(scale);
            std::memcpy(dst + (type == GGML_TYPE_Q6_K ? 208 : 0), &half, sizeof(half));
        }
    }
    return weight;
}

static std::vector<float> dense_f32_gemv_reference(ggml_type type, const std::vector<uint8_t> & weight,
                                                  const std::vector<float> & input, int64_t width, int64_t rows) {
    const auto * traits = ggml_get_type_traits(type);
    REQUIRE(traits != nullptr && traits->to_float != nullptr);
    std::vector<float> result(rows), decoded(width);
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(weight.data() + row * ggml_row_size(type, width), decoded.data(), width);
        double sum = 0;
        for (int64_t k = 0; k < width; ++k) {
            sum += static_cast<double>(decoded[k]) * input[k];
        }
        result[row] = static_cast<float>(sum);
        REQUIRE(std::isfinite(result[row]));
    }
    return result;
}

static void run_dense_f32_gemv_reference_case(ggml_type type, int64_t width, int64_t rows, bool run_gpu = true) {
    ScopedHrxEnvironment rounded("HRX_ENABLE_DENSE_ROUNDED_GEMV", "0");
    ScopedHrxEnvironment accum("HRX_ENABLE_DENSE_F32_ACCUM", "1");
    ScopedHrxEnvironment q8("HRX_ENABLE_Q8_GEMV", "0");
    std::vector<float> original;
    for (bool gemv : { false, true }) {
        ScopedHrxEnvironment option("HRX_ENABLE_DENSE_F32_GEMV", gemv ? "1" : "0");
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        auto * w = ggml_new_tensor_2d(ctx, type, width, rows);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
        auto * y = ggml_mul_mat(ctx, w, x);
        auto * graph = ggml_new_graph(ctx);
        graph->uid = next_hrx_test_graph_uid();
        REQUIRE(graph->uid != 0);
        ggml_build_forward_expand(graph, y);
        const auto sequence = scheduled_kernel_sequence(graph);
        REQUIRE(sequence.size() == 1);
        REQUIRE(sequence[0] == (gemv ? dense_f32_gemv_kernel_name(type) : dense_f32_kernel_name(type)));
        ggml_backend_t backend = run_gpu ? ggml_backend_hrx_init(0) : nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        if (run_gpu) {
            REQUIRE(backend != nullptr && ggml_backend_supports_op(backend, y));
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            REQUIRE(buffer != nullptr);
        }
        for (int replay = 0; replay < (gemv ? 3 : 2); ++replay) {
            const auto weight = dense_f32_gemv_weights(type, width, rows, replay);
            auto input = make_pattern_f32(width, 23 + replay * 8, 0.001953125f);
            if (replay == 1) {
                for (int64_t k = 0; k < width; ++k) {
                    input[k] = 1.0003f + static_cast<float>(k % 7) * 0.00001f;
                }
            } else if (replay == 2) {
                std::fill(input.begin(), input.end(), 0.00001f);
            }
            const auto full = dense_f32_gemv_reference(type, weight, input, width, rows);
            const auto expected = gemv ? full : dense_f16_operand_reference(type, weight, input, width, rows, 1);
            if (replay == 0) {
                const auto rounded = dense_f16_operand_reference(type, weight, input, width, rows, 1);
                require_close(full, rounded, kDenseF32AbsTolerance, kDenseF32RelTolerance);
            }
            if (replay == 1) {
                const auto rounded = dense_f16_operand_reference(type, weight, input, width, rows, 1);
                const float allowed = kDenseF32AbsTolerance + kDenseF32RelTolerance * std::fabs(full[0]);
                REQUIRE(std::fabs(full[0] - rounded[0]) > allowed);
            }
            if (run_gpu) {
                // Repeated same-UID execution must observe both changed weights and inputs.
                set_tensor_bytes(backend, w, weight.data(), weight.size());
                set_tensor_bytes(backend, x, input.data(), input.size() * sizeof(float));
                const std::vector<float> dirty(rows, std::numeric_limits<float>::quiet_NaN());
                set_tensor_bytes(backend, y, dirty.data(), dirty.size() * sizeof(float));
                REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                const auto actual = get_f32_tensor(backend, y);
                for (float value : actual) {
                    REQUIRE(std::isfinite(value));
                }
                require_close(actual, expected, kDenseF32AbsTolerance, kDenseF32RelTolerance);
                if (replay == 0) {
                    if (gemv) {
                        require_close(actual, original, kDenseF32AbsTolerance, kDenseF32RelTolerance);
                    } else {
                        original = actual;
                    }
                }
                REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                const auto repeated = get_f32_tensor(backend, y);
                for (float value : repeated) {
                    REQUIRE(std::isfinite(value));
                }
                require_close(repeated, actual, 0.0f, 0.0f);
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

static void run_dense_f32_gemv_reference_checks(bool run_gpu = true) {
    for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q6_K }) {
        for (int64_t width : { 256, 2560, 5120 }) {
            run_dense_f32_gemv_reference_case(type, width, 65, run_gpu);
        }
    }
}

static const char * dense_rounded_gemv_kernel_name(ggml_type type) {
    return type == GGML_TYPE_Q8_0 ? "hrx_owned:ggml_dense_q8_0_rounded_gemv" : "hrx_owned:ggml_dense_q6k_rounded_gemv";
}

static void run_dense_rounded_gemv_scheduling_checks() {
    ScopedHrxEnvironment accum("HRX_ENABLE_DENSE_F32_ACCUM", "1");
    ScopedHrxEnvironment q8("HRX_ENABLE_Q8_GEMV", "0");
    for (const char * raw_flag : { "0", "1" }) {
        ScopedHrxEnvironment raw("HRX_ENABLE_DENSE_F32_GEMV", raw_flag);
        for (const char * flag : { static_cast<const char *>(nullptr), "0", "true", "1" }) {
            ScopedHrxEnvironment rounded("HRX_ENABLE_DENSE_ROUNDED_GEMV", flag);
            for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K }) {
                for (int64_t tokens : { 1, 2, 8, 32 }) {
                    ggml_init_params params = {};
                    params.mem_size         = 1024 * 1024;
                    params.no_alloc         = true;
                    auto * ctx              = ggml_init(params);
                    REQUIRE(ctx != nullptr);
                    auto * w     = ggml_new_tensor_2d(ctx, type, 2560, 65);
                    auto * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, tokens);
                    auto * y     = ggml_mul_mat(ctx, w, x);
                    auto * graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, y);
                    auto imported = ggml::hrx::import_ggml_graph(*graph);
                    REQUIRE(imported.valid());
                    ggml::hrx::DispatchScheduler scheduler;
                    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
                    const auto & plan = scheduler.plan();
                    REQUIRE(plan.valid() && plan.dispatches.size() == 1);
                    const auto & dispatch = plan.dispatches[0];
                    const bool   eligible = tokens == 1 && type != GGML_TYPE_Q4_K;
                    const bool   selected = eligible && flag != nullptr && std::strcmp(flag, "1") == 0;
                    const char * expected = selected ? dense_rounded_gemv_kernel_name(type) :
                                            eligible && std::strcmp(raw_flag, "1") == 0 ?
                                                       dense_f32_gemv_kernel_name(type) :
                                                       dense_f32_kernel_name(type);
                    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected);
                    if (selected) {
                        REQUIRE(dispatch.kernel.integer_parameters.size() == 1);
                        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == 1);
                        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.workload.token_capacity") == "1");
                        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.input_size") ==
                                "2560");
                        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_size") == "65");
                        REQUIRE(dispatch.bindings.size() == 3);
                        REQUIRE(dispatch.bindings[0].length == ggml_nbytes(x));
                        REQUIRE(dispatch.bindings[1].length == ggml_nbytes(w));
                        REQUIRE(dispatch.bindings[2].length == ggml_nbytes(y));
                    }
                    ggml_free(ctx);
                }
            }
        }
    }
}

static void run_dense_rounded_gemv_reference_case(ggml_type type, int64_t width, int64_t rows, bool run_gpu = true) {
    ScopedHrxEnvironment            accum("HRX_ENABLE_DENSE_F32_ACCUM", "1");
    ScopedHrxEnvironment            raw("HRX_ENABLE_DENSE_F32_GEMV", "0");
    ScopedHrxEnvironment            q8("HRX_ENABLE_Q8_GEMV", "0");
    std::vector<std::vector<float>> original(5);
    for (bool rounded : { false, true }) {
        ScopedHrxEnvironment option("HRX_ENABLE_DENSE_ROUNDED_GEMV", rounded ? "1" : "0");
        ggml_init_params     params = {};
        params.mem_size             = 1024 * 1024;
        params.no_alloc             = true;
        auto * ctx                  = ggml_init(params);
        REQUIRE(ctx != nullptr);
        auto * w     = ggml_new_tensor_2d(ctx, type, width, rows);
        auto * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
        auto * y     = ggml_mul_mat(ctx, w, x);
        auto * graph = ggml_new_graph(ctx);
        graph->uid   = next_hrx_test_graph_uid();
        REQUIRE(graph->uid != 0);
        ggml_build_forward_expand(graph, y);
        const auto sequence = scheduled_kernel_sequence(graph);
        REQUIRE(sequence.size() == 1);
        REQUIRE(sequence[0] == (rounded ? dense_rounded_gemv_kernel_name(type) : dense_f32_kernel_name(type)));
        ggml_backend_t        backend = run_gpu ? ggml_backend_hrx_init(0) : nullptr;
        ggml_backend_buffer_t buffer  = nullptr;
        if (run_gpu) {
            REQUIRE(backend != nullptr && ggml_backend_supports_op(backend, y));
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            REQUIRE(buffer != nullptr);
        }
        for (int replay = 0; replay < 5; ++replay) {
            auto         weight      = dense_f32_gemv_weights(type, width, rows, replay == 0 ? 0 : 1);
            auto         input       = make_pattern_f32(width, 23, 0.001953125f);
            const size_t block_bytes = ggml_type_size(type);
            if (replay == 1) {
                for (int64_t k = 0; k < width; ++k) {
                    input[k] = 1.0003f + static_cast<float>(k % 7) * 0.00001f;
                }
                // Non-dyadic block scales exercise the existing coefficient-rounding boundary.
                for (size_t base = ggml_row_size(type, width); base < weight.size(); base += block_bytes) {
                    const ggml_fp16_t d = ggml_fp32_to_fp16((base % 2 ? -1.0f : 1.0f) * 0.003141f);
                    std::memcpy(weight.data() + base + (type == GGML_TYPE_Q6_K ? 208 : 0), &d, sizeof(d));
                }
            } else if (replay >= 2) {
                for (int64_t row = 0; row < rows; ++row) {
                    const float       extremes[] = { 65504.0f, -65504.0f, 0.00006103515625f, -0.00006103515625f, 0.0f };
                    const ggml_fp16_t d          = ggml_fp32_to_fp16(replay == 4 ? extremes[row % 5] : 0.125f);
                    for (int64_t block = 0; block < width / ggml_blck_size(type); ++block) {
                        auto * dst = weight.data() + row * ggml_row_size(type, width) + block * block_bytes;
                        if (type == GGML_TYPE_Q8_0) {
                            std::memset(dst + 2, 1, 32);
                        } else {
                            std::memset(dst, 0x11, 128);
                            std::memset(dst + 128, 0xaa, 64);
                            std::memset(dst + 192, 1, 16);
                        }
                        std::memcpy(dst + (type == GGML_TYPE_Q6_K ? 208 : 0), &d, sizeof(d));
                    }
                }
                std::fill(input.begin(), input.end(), replay == 4 ? 0.001003f : 4096.0f);
                if (replay == 3) {
                    for (int64_t k = width / 2; k < width; ++k) {
                        input[k] = -4096.0f + (k >= width - 16 ? 16.0f : 0.0f);
                    }
                }
            }
            const auto expected = dense_f16_operand_reference(type, weight, input, width, rows, 1);
            if (replay == 1) {
                const auto  full    = dense_f32_gemv_reference(type, weight, input, width, rows);
                const float allowed = kDenseF32AbsTolerance + kDenseF32RelTolerance * std::fabs(expected[0]);
                REQUIRE(std::fabs(full[0] - expected[0]) > allowed);
            } else if (replay == 2) {
                REQUIRE(expected[0] > 65504.0f);
            } else if (replay == 3) {
                REQUIRE(expected[0] == 32.0f);
            }
            if (run_gpu) {
                set_tensor_bytes(backend, w, weight.data(), weight.size());
                set_tensor_bytes(backend, x, input.data(), input.size() * sizeof(float));
                const std::vector<float> dirty(rows, std::numeric_limits<float>::quiet_NaN());
                set_tensor_bytes(backend, y, dirty.data(), dirty.size() * sizeof(float));
                REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                const auto actual = get_f32_tensor(backend, y);
                for (float value : actual) {
                    REQUIRE(std::isfinite(value));
                }
                require_close(actual, expected, kDenseF32AbsTolerance, kDenseF32RelTolerance);
                if (rounded) {
                    require_close(actual, original[replay], kDenseF32AbsTolerance, kDenseF32RelTolerance);
                } else {
                    original[replay] = actual;
                }
                REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                const auto repeated = get_f32_tensor(backend, y);
                for (float value : repeated) {
                    REQUIRE(std::isfinite(value));
                }
                require_close(repeated, actual, 0.0f, 0.0f);
            }
        }
        if (run_gpu) {
            ggml_backend_hrx_cache_stats stats = {};
            REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &stats));
            REQUIRE(stats.graph_program_hits >= 9 && stats.prepared_program_hits >= 9);
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
        }
        ggml_free(ctx);
    }
}

static void run_dense_rounded_gemv_reference_checks(bool run_gpu = true) {
    for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q6_K }) {
        for (int64_t width : { 256, 2560, 5120 }) {
            run_dense_rounded_gemv_reference_case(type, width, 65, run_gpu);
        }
    }
}
