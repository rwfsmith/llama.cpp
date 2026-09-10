static void run_qsa_f16_gather_scheduling_checks() {
    for (const char * flag : { static_cast<const char *>(nullptr), "", "0", "01", "1x", "true", "1" }) {
        ScopedHrxEnvironment enabled("HRX_ENABLE_QSA_F16_GATHER", flag);
        const bool expected = flag != nullptr && std::strcmp(flag, "1") == 0;
        for (bool trusted : { false, true }) {
            ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
            REQUIRE(ctx != nullptr);
            ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, 512);
            ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
            if (trusted) {
                ggml_set_input(ids);
            }
            ggml_tensor * output = ggml_get_rows(ctx, source, ids);
            REQUIRE(ggml::hrx::supports_qsa_f16_gather_dispatch(output) == expected);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);
            auto imported = ggml::hrx::import_ggml_graph(*graph);
            REQUIRE(imported.valid());
            ggml::hrx::DispatchScheduler scheduler;
            REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == expected);
            if (expected) {
                REQUIRE(scheduler.plan().dispatches.size() == 1);
                const auto & dispatch = scheduler.plan().dispatches[0];
                REQUIRE(ggml::hrx::is_qsa_f16_gather_kernel(dispatch.kernel.kernel_id));
                REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "hrx_owned:ggml_qsa_get_rows_f16");
                REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == 128);
                REQUIRE(dispatch.kernel.integer_parameters.at("vocabulary_count") == 512);
                REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == 512);
                REQUIRE(dispatch.bindings[0].length == 128 * 512 * sizeof(ggml_fp16_t));
                REQUIRE(dispatch.bindings[1].length == 512 * sizeof(int32_t));
                REQUIRE(dispatch.bindings[1].trusted == trusted);
                REQUIRE(dispatch.bindings[2].length == 128 * 512 * sizeof(float));
                for (bool initialization : { false, true }) {
                    ggml::hrx::PreparedCommandProgram prepared;
                    ggml::hrx::PreparedCommand command;
                    command.kind = ggml::hrx::CommandKind::Kernel;
                    command.kernel.specialization = dispatch.kernel;
                    command.kernel.bindings.resize(3);
                    command.kernel.bindings[1].binding.trusted = trusted;
                    (initialization ? prepared.initialization_commands : prepared.commands).push_back(command);
                    ggml::hrx::RecordedCommandGraph recorded;
                    const auto replay = ggml::hrx::bind_and_launch_recorded_command_graph(
                        {}, {}, {}, prepared, recorded);
                    REQUIRE(!replay.success);
                    if (trusted) {
                        REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::BuildFailed);
                    } else {
                        REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::Ineligible);
                        REQUIRE(replay.ineligible_reason == "qsa_f16_get_rows_requires_index_validation");
                    }
                }
            }
            ggml_free(ctx);
        }
    }
    ScopedHrxEnvironment enabled("HRX_ENABLE_QSA_F16_GATHER", "1");
    for (int malformed = 0; malformed < 13; ++malformed) {
        ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
        REQUIRE(ctx != nullptr);
        auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, 514);
        auto * source = ggml_view_3d(ctx, storage, 128, 512, 1, 256, 514 * 256, 256);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
        auto * output = ggml_get_rows(ctx, source, ids);
        REQUIRE(ggml::hrx::supports_qsa_f16_gather_dispatch(output));
        if (malformed == 0) { source->type = GGML_TYPE_F32; }
        if (malformed == 1) { ids->type = GGML_TYPE_I64; }
        if (malformed == 2) { output->type = GGML_TYPE_F16; }
        if (malformed == 3) { source->nb[1] += 2; }
        if (malformed == 4) { ids->nb[0] *= 2; }
        if (malformed == 5) { output->nb[1] += 4; }
        if (malformed == 6) { source->ne[0] = 64; }
        if (malformed == 7) { source->ne[2] = 2; }
        if (malformed == 8) { source->ne[1] = 4097; }
        if (malformed == 9) { ids->ne[0] = 4097; }
        if (malformed == 10) { source->view_offs += 1; }
        if (malformed == 11) { source->view_offs = ggml_nbytes(storage) - sizeof(ggml_fp16_t); }
        if (malformed == 12) { output->view_src = source; }
        REQUIRE(!ggml::hrx::supports_qsa_f16_gather_dispatch(output));
        ggml_free(ctx);
    }
    // A truncated imported binding must not get a kernel even if its tensor metadata is valid.
    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    REQUIRE(ctx != nullptr);
    auto * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, 512);
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
    auto * output = ggml_get_rows(ctx, source, ids);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    const_cast<ggml::hrx::Value *>(imported.graph.values().find_tensor(source))->byte_count -= sizeof(ggml_fp16_t);
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
    ggml_free(ctx);
    std::fprintf(stderr, "QSA F16 gather scheduling checks passed (geometry, half spans, trust, replay eligibility)\n");
}

static void run_qsa_f16_gather_checks() {
    run_qsa_f16_gather_scheduling_checks();
    ScopedHrxEnvironment enabled("HRX_ENABLE_QSA_F16_GATHER", "1");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(cpu != nullptr && hrx != nullptr);
    for (int64_t capacity : { 128, 512, 4096 }) {
        for (bool trusted : { false, true }) {
            const int64_t count = capacity;
            ggml_context * host_ctx = ggml_init({ 1024 * 1024, nullptr, true });
            ggml_context * cpu_ctx = ggml_init({ 1024 * 1024, nullptr, true });
            ggml_context * hrx_ctx = ggml_init({ 1024 * 1024, nullptr, true });
            REQUIRE(host_ctx != nullptr && cpu_ctx != nullptr && hrx_ctx != nullptr);
            // Actual blk_cells is a mutable CPU input, not a resident GPU constant.
            ggml_tensor * ids = ggml_new_tensor_1d(host_ctx, GGML_TYPE_I32, count);
            if (trusted) {
                ggml_set_input(ids);
            }
            struct GatherGraph {
                ggml_tensor * storage;
                ggml_tensor * source;
                ggml_tensor * output;
                ggml_tensor * consumer;
                ggml_cgraph * graph;
            };
            const auto build = [&](ggml_context * ctx) {
                GatherGraph result = {};
                result.storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, capacity + 2);
                // The unused sequence stride can include more cache slots than n_kv.
                result.source = ggml_view_3d(ctx, result.storage, 128, capacity, 1,
                                            128 * sizeof(ggml_fp16_t),
                                            (capacity + 2) * 128 * sizeof(ggml_fp16_t),
                                            128 * sizeof(ggml_fp16_t));
                result.output = ggml_get_rows(ctx, result.source, ids);
                ggml_set_output(result.output);
                result.consumer = ggml_cont(ctx, result.output);
                result.graph = ggml_new_graph(ctx);
                result.graph->uid = next_hrx_test_graph_uid();
                ggml_build_forward_expand(result.graph, result.consumer);
                return result;
            };
            const auto reference = build(cpu_ctx);
            const auto candidate = build(hrx_ctx);
            REQUIRE(ggml_backend_supports_op(hrx, candidate.output));
            require_kernel_subsequence(scheduled_kernel_sequence(candidate.graph), { "hrx_owned:ggml_qsa_get_rows_f16" });
            ggml_backend_buffer_t host_buffer = ggml_backend_alloc_ctx_tensors(host_ctx, cpu);
            ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
            ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
            REQUIRE(host_buffer != nullptr && cpu_buffer != nullptr && hrx_buffer != nullptr);
            ggml_backend_hrx_cache_stats before = {};
            REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &before));
            std::vector<int32_t> indices(count);
            std::vector<ggml_fp16_t> halves(128 * (capacity + 2));
            const std::vector<float> dirty(128 * count, -12345.0f);
            const auto same_conversion = [](const std::vector<float> & actual, const std::vector<float> & expected) {
                REQUIRE(actual.size() == expected.size());
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (std::isnan(expected[i])) {
                        REQUIRE(std::isnan(actual[i]));
                    } else {
                        // Widening is exact, including infinities, signed zero and half subnormals.
                        REQUIRE(std::memcmp(&actual[i], &expected[i], sizeof(float)) == 0);
                    }
                }
            };
            for (int replay = 0; replay < 4; ++replay) {
                for (size_t i = 0; i < halves.size(); ++i) {
                    // The 512-row case visits all 65536 FP16 bit patterns.
                    halves[i] = static_cast<ggml_fp16_t>((i + replay * 1711) & 0xffff);
                }
                for (int64_t row = 0; row < count; ++row) {
                    indices[row] = static_cast<int32_t>((row * 17 + replay * 31) % capacity);
                }
                if (replay == 3) {
                    // Empty block members select valid row zero; duplicate rows are legal.
                    indices[0] = indices[1] = 0;
                    indices.back() = static_cast<int32_t>(capacity - 1);
                }
                ggml_backend_tensor_set(ids, indices.data(), 0, indices.size() * sizeof(int32_t));
                set_tensor_pair_bytes(cpu, reference.storage, hrx, candidate.storage,
                                      halves.data(), halves.size() * sizeof(ggml_fp16_t));
                set_tensor_bytes(hrx, candidate.output, dirty.data(), dirty.size() * sizeof(float));
                set_tensor_bytes(hrx, candidate.consumer, dirty.data(), dirty.size() * sizeof(float));
                REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
                REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_SUCCESS);
                same_conversion(get_f32_tensor(hrx, candidate.output), get_f32_tensor(cpu, reference.output));
                same_conversion(get_f32_tensor(hrx, candidate.consumer), get_f32_tensor(cpu, reference.consumer));
                std::vector<ggml_fp16_t> after(halves.size());
                ggml_backend_tensor_get(candidate.storage, after.data(), 0, after.size() * sizeof(ggml_fp16_t));
                REQUIRE(after == halves);
                std::vector<int32_t> ids_after(indices.size());
                ggml_backend_tensor_get(ids, ids_after.data(), 0, ids_after.size() * sizeof(int32_t));
                REQUIRE(ids_after == indices);
                if (!trusted) {
                    for (int32_t bad : { -1, static_cast<int32_t>(capacity),
                                         std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max() }) {
                        auto invalid = indices;
                        invalid[count / 2] = bad;
                        ggml_backend_tensor_set(ids, invalid.data(), 0, invalid.size() * sizeof(int32_t));
                        set_tensor_bytes(hrx, candidate.output, dirty.data(), dirty.size() * sizeof(float));
                        set_tensor_bytes(hrx, candidate.consumer, dirty.data(), dirty.size() * sizeof(float));
                        REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_FAILED);
                        REQUIRE(get_f32_tensor(hrx, candidate.output) == dirty);
                        REQUIRE(get_f32_tensor(hrx, candidate.consumer) == dirty);
                        ggml_backend_tensor_get(candidate.storage, after.data(), 0, after.size() * sizeof(ggml_fp16_t));
                        REQUIRE(after == halves);
                    }
                    ggml_backend_tensor_set(ids, indices.data(), 0, indices.size() * sizeof(int32_t));
                    REQUIRE(ggml_backend_graph_compute(hrx, candidate.graph) == GGML_STATUS_SUCCESS);
                    same_conversion(get_f32_tensor(hrx, candidate.output), get_f32_tensor(cpu, reference.output));
                }
            }
            ggml_backend_hrx_cache_stats after = {};
            REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &after));
            REQUIRE(after.graph_program_hits >= before.graph_program_hits + 3);
            REQUIRE(after.prepared_program_hits >= before.prepared_program_hits + 3);
            ggml_backend_buffer_free(hrx_buffer);
            ggml_backend_buffer_free(cpu_buffer);
            ggml_backend_buffer_free(host_buffer);
            ggml_free(hrx_ctx);
            ggml_free(cpu_ctx);
            ggml_free(host_ctx);
        }
    }
    ggml_backend_free(hrx);
    ggml_backend_free(cpu);
    std::fprintf(stderr, "QSA F16 gather GPU checks passed (CPU conversion, host replay, invalid IDs, source immutability)\n");
}
