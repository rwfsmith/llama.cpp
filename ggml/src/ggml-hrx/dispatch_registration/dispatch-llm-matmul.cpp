#include "dispatch-llm-matmul.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kLlmDenseLinearQ4KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_f16_wmma");
static constexpr KernelCatalogRef kLlmDenseLinearQ6KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_f16_wmma");
static constexpr KernelCatalogRef kLlmDenseLinearQ8_0F16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q8_0_f16_wmma");
static constexpr KernelCatalogRef kLlmDenseQ4KF32AccumKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_dense_q4k_f32_accum");
static constexpr KernelCatalogRef kLlmDenseQ6KF32AccumKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_dense_q6k_f32_accum");
static constexpr KernelCatalogRef kLlmDenseQ8_0F32AccumKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_dense_q8_0_f32_accum");
static constexpr KernelCatalogRef kLlmDenseQ8_0GemvKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_dense_q8_0_gemv_f32");
static constexpr KernelCatalogRef kLlmDenseQ6KF32GemvKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_linear_q6k_f32_wave64");
static constexpr KernelCatalogRef kLlmDenseQ8RoundedGemvKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_dense_q8_0_rounded_gemv");
static constexpr KernelCatalogRef kLlmDenseQ6RoundedGemvKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_dense_q6k_rounded_gemv");
static constexpr KernelCatalogRef kLlmQ8NarrowPackKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_q8_narrow_pack");
static constexpr KernelCatalogRef kLlmQ8NarrowDotKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_q8_narrow_dot");
static constexpr KernelCatalogRef kLlmQ8NarrowPackBatchedKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_q8_narrow_pack_batched");
static constexpr KernelCatalogRef kLlmQ8NarrowDotBatchedKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_q8_narrow_dot_batched");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_dense_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 256 == 0;
}

static bool is_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

static bool dense_f32_accum_enabled() {
    const char * flag = std::getenv("HRX_ENABLE_DENSE_F32_ACCUM");
    return flag != nullptr && std::strcmp(flag, "1") == 0;
}

static bool dense_f32_gemv_enabled() {
    const char * flag = std::getenv("HRX_ENABLE_DENSE_F32_GEMV");
    return flag != nullptr && std::strcmp(flag, "1") == 0;
}

static bool dense_rounded_gemv_enabled() {
    const char * flag = std::getenv("HRX_ENABLE_DENSE_ROUNDED_GEMV");
    return flag != nullptr && std::strcmp(flag, "1") == 0;
}

static KernelCatalogRef dense_accumulation_kernel(KernelCatalogRef kernel) {
    if (dense_f32_accum_enabled()) {
        if (kernel.id == kLlmDenseLinearQ4KF16WmmaKernel.id) {
            return kLlmDenseQ4KF32AccumKernel;
        }
        if (kernel.id == kLlmDenseLinearQ6KF16WmmaKernel.id) {
            return kLlmDenseQ6KF32AccumKernel;
        }
        if (kernel.id == kLlmDenseLinearQ8_0F16WmmaKernel.id) {
            return kLlmDenseQ8_0F32AccumKernel;
        }
    }
    return kernel;
}

enum class LlmDenseMatmulRoute {
    Q4K,
    Q6K,
    Q8_0,
};

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

struct LlmDenseMatmulMatch {
    const Value *    input       = nullptr;
    const Value *    weight      = nullptr;
    const Value *    output      = nullptr;
    KernelCatalogRef kernel      = {};
    int64_t          input_size  = 0;
    int64_t          output_size = 0;
    int64_t          token_count = 0;

    bool matched() const {
        return input != nullptr && weight != nullptr && output != nullptr && kernel.id != kUncatalogedKernelId;
    }
};

static LlmDenseMatmulMatch match_llm_dense_matmul(const Graph &       graph,
                                                  const GraphNode *   node,
                                                  LlmDenseMatmulRoute route) {
    LlmDenseMatmulMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(graph, node->inputs[0]);
    const Value * input  = graph_value(graph, node->inputs[1]);
    const Value * output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr || !is_2d(*weight) || !is_2d(*input) ||
        !is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count ||
        !is_llm_supported_query_length(kActiveLlmMoeDispatchProfile, token_count) ||
        !is_supported_dense_input_size(input_size) || !is_supported_dense_output_size(output_size)) {
        return {};
    }

    if (route == LlmDenseMatmulRoute::Q4K && weight->type == GGML_TYPE_Q4_K) {
        match.kernel = kLlmDenseLinearQ4KF16WmmaKernel;
    } else if (route == LlmDenseMatmulRoute::Q6K && weight->type == GGML_TYPE_Q6_K) {
        match.kernel = kLlmDenseLinearQ6KF16WmmaKernel;
    } else if (route == LlmDenseMatmulRoute::Q8_0 && weight->type == GGML_TYPE_Q8_0) {
        const char * enable_gemv = std::getenv("HRX_ENABLE_Q8_GEMV");
        // Explicit F32 WMMA selection takes precedence over the independent raw-F32 GEMV experiment.
        match.kernel = !dense_f32_accum_enabled() && token_count == 1 && weight->byte_count <= UINT32_MAX &&
                               enable_gemv != nullptr && std::strcmp(enable_gemv, "1") == 0 ?
                           kLlmDenseQ8_0GemvKernel : kLlmDenseLinearQ8_0F16WmmaKernel;
    } else {
        return {};
    }

    // The accumulation-only route still stages F16 operands in 32-token WMMA
    // tiles. Reuse the existing raw-F32 packet GEMVs at T1 instead: Q8 uses
    // four independent wave32 rows; Q6 uses a wave64 two-row block cohort.
    if (dense_f32_gemv_enabled() && token_count == 1 && weight->byte_count <= UINT32_MAX) {
        if (weight->type == GGML_TYPE_Q8_0) {
            match.kernel = kLlmDenseQ8_0GemvKernel;
        } else if (weight->type == GGML_TYPE_Q6_K) {
            match.kernel = kLlmDenseQ6KF32GemvKernel;
        }
    }
    // Explicit baseline-operand selection wins over the raw-F32 experiment.
    if (dense_rounded_gemv_enabled() && token_count == 1 && weight->byte_count <= UINT32_MAX) {
        if (weight->type == GGML_TYPE_Q8_0) {
            match.kernel = kLlmDenseQ8RoundedGemvKernel;
        } else if (weight->type == GGML_TYPE_Q6_K) {
            match.kernel = kLlmDenseQ6RoundedGemvKernel;
        }
    }

    match.input       = input;
    match.weight      = weight;
    match.output      = output;
    match.input_size  = input_size;
    match.output_size = output_size;
    match.token_count = token_count;
    return match;
}

}  // namespace

static void build_llm_dense_matmul_dispatch(const LlmDenseMatmulMatch & match,
                                            DispatchMatch &             dispatch_match,
                                            size_t                      root_index) {
    Dispatch dispatch;
    const KernelCatalogRef kernel = dense_accumulation_kernel(match.kernel);
    dispatch.kernel = make_kernel_specialization(kernel);
    if (match.kernel.id == kLlmDenseQ8_0GemvKernel.id) {
        dispatch.kernel.integer_parameters.emplace("input_size", match.input_size);
        dispatch.kernel.integer_parameters.emplace("output_size", match.output_size);
    } else if (match.kernel.id == kLlmDenseQ6KF32GemvKernel.id) {
        dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
        dispatch.kernel.integer_parameters.emplace("input_size", match.input_size);
        dispatch.kernel.integer_parameters.emplace("output_size", match.output_size);
        dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_f32.token_capacity", "1");
        dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_f32.output_capacity",
                                                   to_config_value(match.output_size));
    } else {
        dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                                   to_config_value(match.input_size));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                                   to_config_value(match.output_size));
        if (kernel.id == match.kernel.id && kernel.id != kLlmDenseQ8RoundedGemvKernel.id &&
            kernel.id != kLlmDenseQ6RoundedGemvKernel.id) {
            dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
        }
    }
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static bool match_llm_dense_q4k_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LlmDenseMatmulMatch match =
        match_llm_dense_matmul(context.graph, context.root_node, LlmDenseMatmulRoute::Q4K);
    if (!match.matched()) {
        return false;
    }
    build_llm_dense_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_llm_dense_q6k_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LlmDenseMatmulMatch match =
        match_llm_dense_matmul(context.graph, context.root_node, LlmDenseMatmulRoute::Q6K);
    if (!match.matched()) {
        return false;
    }
    build_llm_dense_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_llm_dense_q8_0_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LlmDenseMatmulMatch match =
        match_llm_dense_matmul(context.graph, context.root_node, LlmDenseMatmulRoute::Q8_0);
    if (!match.matched()) {
        return false;
    }
    build_llm_dense_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_llm_mtp_hc_projection_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return false;
    }
    const Value * weight = graph_value(context.graph, node->inputs[0]);
    const Value * input = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr ||
        !llm_mtp_hc_projection_supported(*weight, *input, *output) ||
        !weight->contiguous || !input->contiguous || !output->contiguous ||
        weight->byte_count != weight->nb[3] || input->byte_count != input->nb[3] ||
        output->byte_count != output->nb[3]) {
        return false;
    }
    // No tensor rewrite or repack: [K,4,T] has the same bytes as [K,4*T].
    const LlmDenseMatmulMatch dense = {
        input, weight, output, kLlmDenseLinearQ4KF16WmmaKernel, weight->ne[0], weight->ne[1],
        input->ne[1] * input->ne[2],
    };
    build_llm_dense_matmul_dispatch(dense, match, context.root_index);
    return true;
}

static bool match_llm_q8_narrow_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return false;
    }
    const Value * weight = graph_value(context.graph, node->inputs[0]);
    const Value * input = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr ||
        !llm_q8_narrow_supported(*weight, *input, *output) ||
        !weight->contiguous || !input->contiguous || !output->contiguous ||
        weight->byte_count != weight->nb[2] || input->byte_count != input->nb[2] ||
        output->byte_count != output->nb[2]) {
        return false;
    }

    const ValueId packed = context.next_plan_value;
    const int64_t tokens = input->ne[1];
    const size_t packed_bytes = ggml_row_size(GGML_TYPE_Q8_0, input->ne[0]) * static_cast<size_t>(tokens);
    Dispatch pack;
    pack.kernel = make_kernel_specialization(tokens == 1 ? kLlmQ8NarrowPackKernel : kLlmQ8NarrowPackBatchedKernel);
    pack.kernel.integer_parameters.emplace("input_size", input->ne[0]);
    pack.bindings = { { input->id, 0, input->byte_count }, { packed, 0, packed_bytes } };
    Dispatch dot;
    dot.kernel = make_kernel_specialization(tokens == 1 ? kLlmQ8NarrowDotKernel : kLlmQ8NarrowDotBatchedKernel);
    dot.kernel.integer_parameters.emplace("input_size", input->ne[0]);
    dot.kernel.integer_parameters.emplace("output_size", output->ne[0]);
    if (tokens > 1) {
        pack.kernel.integer_parameters.emplace("token_count", tokens);
        dot.kernel.integer_parameters.emplace("token_count", tokens);
    }
    dot.bindings = { { packed, 0, packed_bytes }, { weight->id, 0, weight->byte_count },
                     { output->id, 0, output->byte_count } };
    match.transients.push_back({ packed, "llm.q8_narrow.activation", packed_bytes, 256 });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(pack));
    match.dispatches.push_back(std::move(dot));
    return true;
}

void register_llm_matmul_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "llm.matmul.mtp_hc_projection",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        110,
        DispatchSource::Llm,
        match_llm_mtp_hc_projection_dispatch,
    });
    registry.add({
        "llm.matmul.q8_narrow",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        110,
        DispatchSource::Llm,
        match_llm_q8_narrow_dispatch,
    });
    registry.add({
        "llm.matmul.dense_q4k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_llm_dense_q4k_dispatch,
    });
    registry.add({
        "llm.matmul.dense_q6k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_llm_dense_q6k_dispatch,
    });
    registry.add({
        "llm.matmul.dense_q8_0_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_llm_dense_q8_0_dispatch,
    });
}

}  // namespace ggml::hrx
