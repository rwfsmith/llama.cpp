static void run_device_binding_refresh_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_QSA_GLUE", "1");
    for (int variant = 0; variant < 3; ++variant) {
        ggml_backend_t backend = ggml_backend_hrx_init(0);
        ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
        REQUIRE(backend != nullptr && ctx != nullptr);
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 2);
        auto * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 2);
        auto * replacement = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 2);
        auto * spare = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 2);
        auto * output = ggml_add(ctx, input, bias);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        graph->uid = next_hrx_test_graph_uid();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer != nullptr);
        REQUIRE(!ggml_backend_buffer_is_host(buffer));
        const std::vector<float> ones(256, 1.0f), twos(256, 2.0f), tens(256, 10.0f);
        const std::vector<float> sentinel(256, -999.0f);
        set_tensor_bytes(backend, input, ones.data(), ones.size() * sizeof(float));
        set_tensor_bytes(backend, bias, twos.data(), twos.size() * sizeof(float));
        set_tensor_bytes(backend, replacement, tens.data(), tens.size() * sizeof(float));
        set_tensor_bytes(backend, spare, sentinel.data(), sentinel.size() * sizeof(float));
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        require_close(get_f32_tensor(backend, output), std::vector<float>(256, 3.0f), 0.0f, 0.0f);
        ggml_backend_hrx_cache_stats before = {};
        REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &before));

        // The graph/node array stays fixed while the scheduler rebinds device storage.
        if (variant == 0) {
            std::swap(input->data, replacement->data);
        } else if (variant == 1) {
            std::swap(output->data, spare->data);
        } else {
            output->src[0] = replacement;
        }
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        const auto actual = get_f32_tensor(backend, output);
        std::fprintf(stderr, "Device binding refresh variant=%d first=%g\n", variant, actual.front());
        require_close(actual, std::vector<float>(256, variant == 1 ? 3.0f : 12.0f), 0.0f, 0.0f);
        ggml_backend_hrx_cache_stats after = {};
        REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &after));
        REQUIRE(after.graph_program_hits > before.graph_program_hits);
        REQUIRE(after.prepared_program_hits > before.prepared_program_hits);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
    std::fprintf(stderr, "Device binding refresh checks passed (input/output relocation, source replacement)\n");
}
