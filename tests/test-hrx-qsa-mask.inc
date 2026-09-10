// Include after next_hrx_test_graph_uid() and the existing test helpers.
#include "dispatch_registration/dispatch-copy.h"
#include "dispatch_registration/dispatch-qsa-mask.h"
#include "graph/graph-diagnostics.h"

enum class QsaMaskCase { Fill32, Fill16, Zeros, CastDown, CastUp, CopyDown, CopyUp, Add, BroadcastAdd, Chain };

struct QsaMaskFixture {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * scatter = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
};

static QsaMaskFixture build_qsa_mask_fixture(ggml_context * ctx, QsaMaskCase kind,
                                           int64_t width, int64_t rows, float fill) {
    QsaMaskFixture f;
    const ggml_type type = kind == QsaMaskCase::Fill32 || kind == QsaMaskCase::Zeros ||
                          kind == QsaMaskCase::CastDown || kind == QsaMaskCase::CopyDown ?
                          GGML_TYPE_F32 : GGML_TYPE_F16;
    f.a = kind == QsaMaskCase::Zeros ? ggml_new_tensor_4d(ctx, type, 1, width, rows, 1) :
                                      ggml_new_tensor_4d(ctx, type, width, rows, 1, 1);
    if (kind == QsaMaskCase::Fill32 || kind == QsaMaskCase::Fill16 || kind == QsaMaskCase::Zeros) {
        f.output = ggml_fill(ctx, f.a, fill);
    } else if (kind == QsaMaskCase::CastDown || kind == QsaMaskCase::CastUp) {
        f.output = ggml_cast(ctx, f.a, type == GGML_TYPE_F32 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    } else if (kind == QsaMaskCase::CopyDown || kind == QsaMaskCase::CopyUp) {
        const auto dst_type = type == GGML_TYPE_F32 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        auto * destination = ggml_new_tensor_4d(ctx, dst_type, width, rows, 1, 1);
        f.output = ggml_cpy(ctx, f.a, destination);
    } else if (kind == QsaMaskCase::Chain) {
        // Match build_attn_qsa, leaving selectors external and untrusted just as
        // materialized TOP_K output is. The existing SET_ROWS validator must run.
        auto * filled = ggml_fill(ctx, f.a, -std::numeric_limits<float>::infinity());
        auto * destination = ggml_view_4d(ctx, filled, 1, width, rows, 1,
            filled->nb[0], filled->nb[1], filled->nb[2], 0);
        f.b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 64, rows, 1);
        auto * zeros = ggml_fill(ctx, f.b, 0.0f);
        f.ids = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 64, rows, 1);
        f.scatter = ggml_set_rows(ctx, destination, zeros, f.ids);
        auto * sparse = ggml_view_4d(ctx, f.scatter, width, rows, 1, 1,
            f.scatter->nb[2], f.scatter->nb[3], f.scatter->nb[3], 0);
        f.output = ggml_add(ctx, sparse, f.a);
    } else {
        f.b = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, width,
                                kind == QsaMaskCase::BroadcastAdd ? 1 : rows, 1, 1);
        f.output = ggml_add(ctx, f.a, f.b);
    }
    ggml_set_output(f.output);
    f.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(f.graph, f.output);
    f.graph->uid = next_hrx_test_graph_uid();
    return f;
}

static std::vector<uint8_t> qsa_mask_bytes(ggml_backend_t backend, ggml_tensor * tensor) {
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    ggml_backend_synchronize(backend);
    return bytes;
}

static std::vector<uint8_t> qsa_mask_data(const ggml_tensor * tensor, int generation) {
    const uint32_t f32[] = { 0, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc01234u, 0xffc01234u,
        0x00000001u, 0x80000001u, 0x33800000u, 0x387fc000u, 0x38800000u, 0x3f800000u,
        0xbf800000u, 0x3f801000u, 0x3f803000u, 0x477fe000u, 0x7f7fffffu };
    const uint16_t f16[] = { 0, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0xfe01, 0x0001, 0x8001,
        0x03ff, 0x0400, 0x3c00, 0xbc00, 0x3c01, 0x7bff };
    const size_t size = ggml_type_size(tensor->type);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    for (size_t i = 0; i < bytes.size() / size; ++i) {
        const size_t index = i + static_cast<size_t>(generation * 5);
        if (tensor->type == GGML_TYPE_F32) {
            const uint32_t value = f32[index % (sizeof(f32) / sizeof(f32[0]))];
            std::memcpy(bytes.data() + i * size, &value, size);
        } else {
            const uint16_t value = f16[index % (sizeof(f16) / sizeof(f16[0]))];
            std::memcpy(bytes.data() + i * size, &value, size);
        }
    }
    return bytes;
}

static void require_qsa_mask_exact(const std::vector<uint8_t> & actual,
                                   const std::vector<uint8_t> & expected, ggml_type type) {
    REQUIRE(actual.size() == expected.size());
    const size_t size = ggml_type_size(type);
    for (size_t i = 0; i < actual.size(); i += size) {
        uint32_t a = 0, e = 0;
        std::memcpy(&a, actual.data() + i, size);
        std::memcpy(&e, expected.data() + i, size);
        const uint32_t exponent = type == GGML_TYPE_F16 ? 0x7c00u : 0x7f800000u;
        const uint32_t mantissa = type == GGML_TYPE_F16 ? 0x03ffu : 0x007fffffu;
        if ((e & exponent) == exponent && (e & mantissa) != 0) {
            // NaN payload/sign selection is ISA-dependent, not a mask semantic.
            REQUIRE((a & exponent) == exponent && (a & mantissa) != 0);
        } else {
            REQUIRE(a == e); // Includes signs of zeros and infinities; no tolerance.
        }
    }
}

static void run_qsa_mask_scale_snapshot_checks() {
    const ggml::hrx::ScaleParams cases[] = { { 1.25f, -0.75f }, { 0.0f, 0.0f }, { -2.0f, 0.5f } };
    for (const auto & expected : cases) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
        auto * output = ggml_scale_bias(ctx, input, expected.scale, expected.bias);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        graph->uid = next_hrx_test_graph_uid();
        const auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        const auto json = ggml::hrx::serialize_graph_snapshot_json(imported.graph, "gfx1151", graph->uid);
        const auto loaded = ggml::hrx::load_graph_snapshot_json(json);
        REQUIRE(loaded.valid());
        REQUIRE(loaded.graph.nodes().size() == 1);
        const auto & node = loaded.graph.nodes().back();
        REQUIRE(node.op == GGML_OP_SCALE);
        const auto * actual = ggml::hrx::op_params_as<ggml::hrx::ScaleParams>(node.params);
        REQUIRE(actual != nullptr);
        REQUIRE(actual->scale == expected.scale && actual->bias == expected.bias);
        REQUIRE(ggml::hrx::op_params_equivalent(GGML_OP_SCALE, node.params, *output));
        ggml_free(ctx);
    }
}

static void run_qsa_mask_scheduling_checks() {
    run_qsa_mask_scale_snapshot_checks();
    ScopedHrxEnvironment enabled("HRX_ENABLE_QSA_MASK", "1");
    ScopedHrxEnvironment set_rows("HRX_ENABLE_SET_ROWS", "1");
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t rows : { 1, 8, 32 }) {
        for (QsaMaskCase kind : { QsaMaskCase::Fill32, QsaMaskCase::Fill16, QsaMaskCase::Zeros,
                QsaMaskCase::CastDown, QsaMaskCase::CastUp, QsaMaskCase::CopyDown, QsaMaskCase::CopyUp,
                QsaMaskCase::Add, QsaMaskCase::BroadcastAdd,
                QsaMaskCase::Chain }) {
            if (kind == QsaMaskCase::Chain && rows == 32) {
                continue;
            }
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            auto * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            const auto f = build_qsa_mask_fixture(ctx, kind, 256, rows,
                kind == QsaMaskCase::Zeros ? 0.0f : neg_inf);
            if ((kind == QsaMaskCase::Fill32 || kind == QsaMaskCase::Fill16) && rows != 1) {
                REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(f.output));
                ggml_free(ctx);
                continue;
            }
            REQUIRE(ggml::hrx::supports_qsa_mask_dispatch(f.output));
            auto imported = ggml::hrx::import_ggml_graph(*f.graph);
            REQUIRE(imported.valid());
            ggml::hrx::DispatchScheduler scheduler;
            if (kind == QsaMaskCase::Chain && rows == 8) {
                // Prefill SET_ROWS stays on CPU: accepting the surrounding mask
                // operations must not claim the entire chain as an HRX graph.
                REQUIRE(!ggml::hrx::supports_set_rows_dispatch(f.scatter));
                REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(f.scatter));
                int fills = 0;
                for (int i = 0; i < f.graph->n_nodes; ++i) {
                    if (f.graph->nodes[i]->op == GGML_OP_FILL) {
                        REQUIRE(ggml::hrx::supports_qsa_mask_dispatch(f.graph->nodes[i]) ==
                                (f.graph->nodes[i]->ne[0] == 1));
                        ++fills;
                    }
                }
                REQUIRE(fills == 2);
                REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
                ggml_free(ctx);
                continue;
            }
            REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
            REQUIRE(scheduler.plan().dispatches.size() == (kind == QsaMaskCase::Chain ? 4 : 1));
            if (kind == QsaMaskCase::Chain) {
                bool found = false;
                for (const auto & dispatch : scheduler.plan().dispatches) {
                    if (ggml::hrx::is_set_rows_kernel(dispatch.kernel.kernel_id)) {
                        found = true;
                        REQUIRE(!dispatch.bindings[1].trusted);
                        ggml::hrx::PreparedCommandProgram prepared;
                        ggml::hrx::PreparedCommand command;
                        command.kind = ggml::hrx::CommandKind::Kernel;
                        command.kernel.specialization = dispatch.kernel;
                        command.kernel.bindings.resize(dispatch.bindings.size());
                        prepared.commands.push_back(command);
                        ggml::hrx::RecordedCommandGraph recorded;
                        const auto replay = ggml::hrx::bind_and_launch_recorded_command_graph(
                            {}, {}, {}, prepared, recorded);
                        REQUIRE(!replay.success && replay.event == ggml::hrx::HrxGraphReplayEvent::Ineligible);
                        REQUIRE(replay.ineligible_reason == "set_rows_requires_index_validation");
                    }
                }
                REQUIRE(found);
            }
            for (const char * flag : { static_cast<const char *>(nullptr), "0", "true", "1x" }) {
                ScopedHrxEnvironment disabled("HRX_ENABLE_QSA_MASK", flag);
                REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(f.output));
            }
            auto bad = *f.output;
            bad.ne[3] = 2;
            REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(&bad));
            bad = *f.output;
            bad.nb[1] += ggml_type_size(bad.type);
            if (bad.ne[1] != 1) {
                REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(&bad));
            }
            bad = *f.output;
            bad.view_offs += 1;
            REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(&bad));
            if (kind == QsaMaskCase::Add && rows == 8) {
                bad = *f.output;
                auto rhs = *bad.src[1];
                rhs.ne[1] = 2;
                bad.src[1] = &rhs;
                REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(&bad));
            }
            ggml_free(ctx);
        }
    }
    // FILL params must survive snapshots and full cache metadata validation bit-for-bit.
    ScopedHrxEnvironment validate_params("GGML_HRX_DISABLE_GRAPH_UID_FAST_PATH", "1");
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    auto * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    const auto fill = build_qsa_mask_fixture(ctx, QsaMaskCase::Fill16, 256, 1, 0.0f);
    ggml::hrx::GraphProgramCache cache;
    for (uint32_t bits : { 0u, 0x80000000u, 0xff800000u }) {
        std::memcpy(fill.output->op_params, &bits, sizeof(bits));
        const auto p = ggml::hrx::import_op_params(*fill.output);
        REQUIRE(ggml::hrx::op_params_as<ggml::hrx::FillParams>(p)->value_bits == bits);
        const auto lookup = cache.get_or_build(*fill.graph, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(lookup.valid());
        const auto imported = ggml::hrx::import_ggml_graph(*fill.graph);
        const auto json = ggml::hrx::serialize_graph_snapshot_json(imported.graph, "gfx1151", fill.graph->uid);
        const auto loaded = ggml::hrx::load_graph_snapshot_json(json);
        REQUIRE(loaded.valid());
        REQUIRE(ggml::hrx::op_params_equivalent(GGML_OP_FILL, loaded.graph.nodes().back().params, p));
    }
    REQUIRE(cache.stats().builds == 3 && cache.stats().hits == 0);
    for (uint32_t bits : { 0x7f800000u, 0x7fc00000u, 0x3f800000u }) {
        std::memcpy(fill.output->op_params, &bits, sizeof(bits));
        REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(fill.output));
    }
    for (int64_t rows : { 0, 33 }) {
        const auto invalid = build_qsa_mask_fixture(ctx, QsaMaskCase::Fill16, 256, rows, neg_inf);
        REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(invalid.output));
    }
    const auto invalid = build_qsa_mask_fixture(ctx, QsaMaskCase::Add, 257, 1, 0.0f);
    REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(invalid.output));
    const auto limit = build_qsa_mask_fixture(ctx, QsaMaskCase::CastUp, 131072, 32, 0.0f);
    REQUIRE(ggml::hrx::supports_qsa_mask_dispatch(limit.output));
    REQUIRE(scheduled_kernel_sequence(limit.graph).size() == 1);
    const auto oversized = build_qsa_mask_fixture(ctx, QsaMaskCase::CastUp, 131328, 32, 0.0f);
    REQUIRE(!ggml::hrx::supports_qsa_mask_dispatch(oversized.output));
    ggml_free(ctx);
    std::fprintf(stderr, "QSA mask scheduling/T8 SET_ROWS fallback/bitwise FILL params/mandatory selector validation passed\n");
}

static void run_qsa_mask_mixed_checks(ggml_backend_t cpu, ggml_backend_t hrx) {
    auto * cpu_ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * mixed_ctx = ggml_init({1024 * 1024, nullptr, true});
    REQUIRE(cpu_ctx != nullptr && mixed_ctx != nullptr);
    const auto c = build_qsa_mask_fixture(cpu_ctx, QsaMaskCase::Chain, 512, 8, 0.0f);
    const auto h = build_qsa_mask_fixture(mixed_ctx, QsaMaskCase::Chain, 512, 8, 0.0f);
    auto cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
    REQUIRE(cpu_buffer != nullptr);
    ggml_backend_t backends[] = { hrx, cpu };
    auto * scheduler = ggml_backend_sched_new(backends, nullptr, 2, 64, false, true);
    REQUIRE(scheduler != nullptr);
    for (auto * input : { h.a, h.b, h.ids }) {
        ggml_set_input(input);
        ggml_set_output(input);
        ggml_backend_sched_set_tensor_backend(scheduler, input, cpu);
    }
    REQUIRE(ggml_backend_sched_alloc_graph(scheduler, h.graph));
    REQUIRE(ggml_backend_sched_get_tensor_backend(scheduler, h.scatter) == cpu);
    // An unsupported in-place CPU scatter must never receive a device-only destination.
    REQUIRE(ggml_backend_buffer_is_host(h.scatter->buffer));
    for (int replay = 0; replay < 3; ++replay) {
        const auto a = qsa_mask_data(c.a, replay);
        const auto b = qsa_mask_data(c.b, replay + 1);
        std::vector<int32_t> ids(64 * 8);
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = static_cast<int32_t>((i * 7 + replay * 31) % 512);
        }
        for (const auto & f : { c, h }) {
            ggml_backend_tensor_set(f.a, a.data(), 0, a.size());
            ggml_backend_tensor_set(f.b, b.data(), 0, b.size());
            ggml_backend_tensor_set(f.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        }
        REQUIRE(ggml_backend_graph_compute(cpu, c.graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_sched_graph_compute(scheduler, h.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(scheduler);
        require_qsa_mask_exact(qsa_mask_bytes(hrx, h.output), qsa_mask_bytes(cpu, c.output), c.output->type);
        REQUIRE(qsa_mask_bytes(cpu, h.a) == a);
    }
    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(cpu_buffer);
    ggml_free(cpu_ctx);
    ggml_free(mixed_ctx);
}

static void run_qsa_mask_reference_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_QSA_MASK", "1");
    ScopedHrxEnvironment set_rows("HRX_ENABLE_SET_ROWS", "1");
    auto * cpu = init_cpu_backend();
    auto * hrx = ggml_backend_hrx_init(0);
    REQUIRE(hrx != nullptr);
    run_qsa_mask_mixed_checks(cpu, hrx);
    for (int64_t rows : { 1, 8, 32 }) {
        for (QsaMaskCase kind : { QsaMaskCase::Fill32, QsaMaskCase::Fill16, QsaMaskCase::Zeros,
                QsaMaskCase::CastDown, QsaMaskCase::CastUp, QsaMaskCase::CopyDown, QsaMaskCase::CopyUp,
                QsaMaskCase::Add, QsaMaskCase::BroadcastAdd,
                QsaMaskCase::Chain }) {
            if ((kind == QsaMaskCase::Chain || kind == QsaMaskCase::Fill32 ||
                 kind == QsaMaskCase::Fill16) && rows != 1) {
                continue;
            }
            for (float fill_value : { 0.0f, -0.0f, -std::numeric_limits<float>::infinity() }) {
                if (kind != QsaMaskCase::Fill32 && kind != QsaMaskCase::Fill16 && fill_value != 0.0f) {
                    continue;
                }
                if (kind != QsaMaskCase::Fill32 && kind != QsaMaskCase::Fill16 && std::signbit(fill_value)) {
                    continue;
                }
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                auto * cpu_ctx = ggml_init(params);
                auto * hrx_ctx = ggml_init(params);
                REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
                const auto c = build_qsa_mask_fixture(cpu_ctx, kind, 512, rows, fill_value);
                const auto h = build_qsa_mask_fixture(hrx_ctx, kind, 512, rows, fill_value);
                for (int i = 0; i < h.graph->n_nodes; ++i) {
                    REQUIRE(ggml_backend_supports_op(hrx, h.graph->nodes[i]));
                }
                auto cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
                auto hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
                REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
                ggml_backend_hrx_cache_stats before = {}, first = {};
                REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &before));
                for (int replay = 0; replay < 4; ++replay) {
                    const auto a = qsa_mask_data(c.a, replay);
                    set_tensor_pair_bytes(cpu, c.a, hrx, h.a, a.data(), a.size());
                    std::vector<uint8_t> b;
                    std::vector<int32_t> ids;
                    if (c.b != nullptr) {
                        b = qsa_mask_data(c.b, replay + 1);
                        set_tensor_pair_bytes(cpu, c.b, hrx, h.b, b.data(), b.size());
                    }
                    if (c.ids != nullptr) {
                        ids.resize(64);
                        for (int i = 0; i < 64; ++i) {
                            ids[i] = (i * 7 + replay * 31) % 512;
                        }
                        set_tensor_pair_bytes(cpu, c.ids, hrx, h.ids, ids.data(), ids.size() * sizeof(int32_t));
                    }
                    REQUIRE(ggml_backend_graph_compute(cpu, c.graph) == GGML_STATUS_SUCCESS);
                    REQUIRE(ggml_backend_graph_compute(hrx, h.graph) == GGML_STATUS_SUCCESS);
                    require_qsa_mask_exact(qsa_mask_bytes(hrx, h.output), qsa_mask_bytes(cpu, c.output), c.output->type);
                    REQUIRE(qsa_mask_bytes(hrx, h.a) == a);
                    if (h.b != nullptr) {
                        REQUIRE(qsa_mask_bytes(hrx, h.b) == b);
                    }
                    if (h.ids != nullptr) {
                        const auto unchanged = qsa_mask_bytes(hrx, h.ids);
                        REQUIRE(std::memcmp(unchanged.data(), ids.data(), unchanged.size()) == 0);
                    }
                    ggml_backend_hrx_cache_stats stats = {};
                    REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &stats));
                    if (replay == 0) {
                        REQUIRE(stats.graph_program_builds == before.graph_program_builds + 1);
                        REQUIRE(stats.prepared_program_builds == before.prepared_program_builds + 1);
                        first = stats;
                    } else {
                        REQUIRE(stats.graph_program_builds == first.graph_program_builds);
                        REQUIRE(stats.prepared_program_builds == first.prepared_program_builds);
                        REQUIRE(stats.graph_program_hits == first.graph_program_hits + replay);
                        REQUIRE(stats.prepared_program_hits == first.prepared_program_hits + replay);
                    }
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
    std::fprintf(stderr, "QSA mask CPU differential: exact finite/inf/zero bits, NaN propagation, real cache hits, original chain passed\n");
}
