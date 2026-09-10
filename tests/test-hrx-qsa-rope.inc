static void run_qsa_rope_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_QSA_ROPE", "1");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(cpu != nullptr && hrx != nullptr);
    {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        auto * gdn = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 48, 1);
        REQUIRE(!ggml::hrx::qwen4exp_qsa_norm_supported(ggml_rms_norm(ctx, gdn, 1.0e-6f)));
        ggml_free(ctx);
    }
    struct QsaCase {
        int64_t width;
        int64_t heads;
        int64_t tokens;
        bool strided;
        bool norm;
        bool pooled_norm = false;
    };
    const QsaCase cases[] = {
        { 256, 24, 1, true, true }, { 256, 24, 2, true, true }, { 256, 24, 8, true, true },
        { 256, 2, 1, false, true }, { 256, 2, 2, false, true }, { 256, 2, 8, false, true },
        { 128, 4, 1, false, true }, { 128, 4, 2, false, true }, { 128, 4, 8, false, true },
        { 128, 1, 128, false, false }, { 128, 1, 129, false, false },
        { 256, 24, 8, false, false }, { 256, 2, 8, false, false }, { 128, 4, 8, false, false },
        { 128, 1, 8, false, true, true }, { 128, 1, 32, false, true, true },
        { 128, 1, 128, false, true, true }, { 128, 1, 129, false, true, true },
        { 128, 1, 512, false, true, true },
    };
    for (const auto & test : cases) {
        for (bool yarn : { false, true }) {
            ggml_init_params params = {};
            params.mem_size = 2 * 1024 * 1024;
            params.no_alloc = true;
            ggml_context * cpu_ctx = ggml_init(params);
            ggml_context * hrx_ctx = ggml_init(params);
            REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
            struct QsaGraph {
                ggml_tensor * storage;
                ggml_tensor * positions;
                ggml_tensor * norm;
                ggml_tensor * rope;
                ggml_cgraph * graph;
            };
            const auto build = [&](ggml_context * ctx) {
                QsaGraph result = {};
                const int64_t stride = test.width * (test.strided ? 2 : 1);
                result.storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, stride, test.heads, test.tokens);
                ggml_tensor * input = result.storage;
                if (test.strided) {
                    input = ggml_view_3d(ctx, input, test.width, test.heads, test.tokens,
                                        stride * sizeof(float), stride * test.heads * sizeof(float), 0);
                }
                if (test.norm) {
                    if (test.pooled_norm) {
                        input = ggml_reshape_3d(ctx, ggml_scale(ctx, input, 0.25f),
                                                test.width, test.tokens, 1);
                    }
                    result.norm = ggml_rms_norm(ctx, input, yarn ? 1.0e-5f : 1.0e-6f);
                    ggml_set_output(result.norm);
                    input = test.pooled_norm ?
                        ggml_reshape_3d(ctx, result.norm, test.width, 1, test.tokens) : result.norm;
                }
                result.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * test.tokens);
                int sections[] = { 11, 11, 10, 0 };
                result.rope = ggml_rope_multi(ctx, input, result.positions, nullptr, 64, sections,
                    GGML_ROPE_TYPE_IMROPE, 262144, 10000000.0f, yarn ? 0.25f : 1.0f,
                    yarn ? 1.0f : 0.0f, yarn ? 1.125f : 1.0f, 32.0f, 1.0f);
                result.graph = ggml_new_graph(ctx);
                result.graph->uid = next_hrx_test_graph_uid();
                ggml_build_forward_expand(result.graph, result.rope);
                return result;
            };
            const auto reference = build(cpu_ctx);
            const auto candidate = build(hrx_ctx);
            REQUIRE(ggml::hrx::qwen4exp_qsa_rope_supported(candidate.rope));
            REQUIRE(ggml_backend_supports_op(hrx, candidate.rope));
            if (candidate.norm) {
                REQUIRE(ggml::hrx::qwen4exp_qsa_norm_supported(candidate.norm));
                REQUIRE(ggml_backend_supports_op(hrx, candidate.norm));
                require_kernel_subsequence(scheduled_kernel_sequence(candidate.graph),
                    { "hrx_owned:ggml_qsa_norm_f32", "hrx_owned:ggml_qsa_rope_f32" });
            } else {
                require_kernel_subsequence(scheduled_kernel_sequence(candidate.graph),
                    { "hrx_owned:ggml_qsa_rope_f32" });
            }
            {
                ScopedHrxEnvironment disabled("HRX_ENABLE_QSA_ROPE", "0");
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(candidate.rope));
                if (candidate.norm) {
                    REQUIRE(!ggml::hrx::qwen4exp_qsa_norm_supported(candidate.norm));
                }
            }
            const auto reject_integer = [&](int slot, int32_t value) {
                auto bad = *candidate.rope;
                std::memcpy(reinterpret_cast<uint8_t *>(bad.op_params) + slot * sizeof(int32_t), &value, sizeof(value));
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
            };
            reject_integer(1, 32);
            reject_integer(2, GGML_ROPE_TYPE_NEOX);
            reject_integer(2, GGML_ROPE_TYPE_MROPE);
            reject_integer(2, GGML_ROPE_TYPE_VISION);
            reject_integer(11, 10);
            reject_integer(14, 1);
            reject_integer(15, 2);
            {
                auto bad = *candidate.rope;
                auto positions = *candidate.positions;
                positions.ne[0]--;
                bad.src[1] = &positions;
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
                positions = *candidate.positions;
                positions.type = GGML_TYPE_F32;
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
                positions = *candidate.positions;
                positions.nb[0] *= 2;
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
                bad = *candidate.rope;
                bad.src[2] = candidate.storage;
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
                bad = *candidate.rope;
                auto input = *bad.src[0];
                input.type = GGML_TYPE_F16;
                bad.src[0] = &input;
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
                bad = *candidate.rope;
                bad.nb[0] *= 2;
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
                bad = *candidate.rope;
                bad.ne[3] = 2;
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
                bad = *candidate.rope;
                bad.view_src = bad.src[0];
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
                bad = *candidate.rope;
                const float zero = 0.0f;
                std::memcpy(reinterpret_cast<uint8_t *>(bad.op_params) + 6 * sizeof(int32_t), &zero, sizeof(zero));
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
                bad = *candidate.rope;
                const uint32_t nan_bits = 0x7fc00000u;
                std::memcpy(reinterpret_cast<uint8_t *>(bad.op_params) + 8 * sizeof(int32_t),
                            &nan_bits, sizeof(nan_bits));
                REQUIRE(!ggml::hrx::qwen4exp_qsa_rope_supported(&bad));
            }
            if (candidate.norm) {
                auto bad = *candidate.norm;
                auto input = *bad.src[0];
                input.nb[1] += sizeof(float);
                bad.src[0] = &input;
                REQUIRE(!ggml::hrx::qwen4exp_qsa_norm_supported(&bad));
                bad = *candidate.norm;
                bad.ne[2] = 9;
                REQUIRE(!ggml::hrx::qwen4exp_qsa_norm_supported(&bad));
                bad = *candidate.norm;
                const float negative = -1.0f;
                std::memcpy(bad.op_params, &negative, sizeof(negative));
                REQUIRE(!ggml::hrx::qwen4exp_qsa_norm_supported(&bad));
            }
            ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
            ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
            REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
            for (int replay = 0; replay < 4; ++replay) {
                const auto input = make_pattern_f32(ggml_nelements(candidate.storage), 101 + replay);
                std::vector<int32_t> positions(static_cast<size_t>(4 * test.tokens));
                for (int plane = 0; plane < 4; ++plane) {
                    for (int64_t token = 0; token < test.tokens; ++token) {
                        // Unequal planes exercise IMRoPE selection, not just its text/NEOX special case.
                        positions[plane * test.tokens + token] =
                            replay == 0 ? static_cast<int32_t>(token + plane * 7) :
                            replay == 1 ? static_cast<int32_t>(262143 - token * 3 - plane * 11) :
                            replay == 2 ? static_cast<int32_t>(-31 - token - plane * 13) :
                                          static_cast<int32_t>((plane % 2 ? -1 : 1) *
                                              (2147483000LL - token * 3 - plane * 11));
                    }
                }
                set_tensor_pair_bytes(cpu, reference.storage, hrx, candidate.storage,
                                      input.data(), input.size() * sizeof(float));
                set_tensor_pair_bytes(cpu, reference.positions, hrx, candidate.positions,
                                      positions.data(), positions.size() * sizeof(int32_t));
                REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
                REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_SUCCESS);
                if (candidate.norm) {
                    require_close(get_f32_tensor(hrx, candidate.norm), get_f32_tensor(cpu, reference.norm),
                                  1.0e-5f, 1.0e-5f);
                }
                const auto actual = get_f32_tensor(hrx, candidate.rope);
                std::fprintf(stderr, "QSA parity width=%lld heads=%lld tokens=%lld yarn=%d replay=%d\n",
                             static_cast<long long>(test.width), static_cast<long long>(test.heads),
                             static_cast<long long>(test.tokens), yarn ? 1 : 0, replay);
                require_close(actual, get_f32_tensor(cpu, reference.rope), 1.0e-4f, 1.0e-5f);
                const auto before_rope = candidate.norm ? get_f32_tensor(hrx, candidate.norm) : input;
                for (int64_t row = 0; row < test.heads * test.tokens; ++row) {
                    for (int64_t col = 64; col < test.width; ++col) {
                        REQUIRE(actual[row * test.width + col] == before_rope[row * test.width + col]);
                    }
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
    std::fprintf(stderr, "QSA norm/IMRoPE checks passed (GGML CPU, strided Q, T1/2/8, pooled keys, YaRN, replay)\n");
}
