#include "dispatch-routed-ffn.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#undef GGML_COMMON_IMPL_CPP

namespace ggml::hrx {
namespace {

static bool iq_experts_enabled() {
    const char * value = std::getenv("HRX_ENABLE_IQ_EXPERTS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static bool moe_token_count_supported(int64_t tokens) {
    const char * value = std::getenv("HRX_ENABLE_MOE_SMALL_BATCH");
    return tokens == 1 || (tokens >= 2 && tokens <= 8 && iq_experts_enabled() &&
                          value != nullptr && std::strcmp(value, "1") == 0);
}

// When HRX_TRACE_MOE_DOWN is set, log the first gate in match_decode_routed_ffn_down_qwen4exp() that
// rejects a node. The down matcher has nine independent preconditions and a failure surfaces only as
// a generic "unsupported HRX node ...: MUL_MAT_ID", which says nothing about which one missed.
static bool trace_moe_down_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("HRX_TRACE_MOE_DOWN");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

static void trace_moe_down_reject(const char * reason) {
    if (trace_moe_down_enabled()) {
        std::fprintf(stderr, "hrx: qwen4exp routed down matcher rejected node: %s\n", reason);
        std::fflush(stderr);
    }
}

static void trace_moe_down_reject_counts(const char * reason, size_t views, size_t reductions, bool residual) {
    if (trace_moe_down_enabled()) {
        std::fprintf(stderr, "hrx: qwen4exp routed down matcher rejected node: %s (views=%zu reductions=%zu residual=%d)\n",
                     reason, views, reductions, residual ? 1 : 0);
        std::fflush(stderr);
    }
}

static constexpr KernelCatalogRef kQwenRoutedGateUpSwiGLUQ4KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma");
static constexpr KernelCatalogRef kQwenRoutedGateUpSwiGLUQ4KQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_gate_up_swiglu_q4k_q8");
static constexpr KernelCatalogRef kQwenRoutedGateUpSwiGLUQ4KQ8NextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8");
static constexpr KernelCatalogRef kQwenRoutedDownQ4KF16WmmaGroupedKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q4k_f16_wmma_grouped");
static constexpr KernelCatalogRef kQwenRoutedDownQ6KF16WmmaGroupedKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q6k_f16_wmma_grouped");
static constexpr KernelCatalogRef kQwenRoutedDownQ4KQ8NextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q4k_q8_1_x4_next_q8");
static constexpr KernelCatalogRef kQwenRoutedDownQ6KF32Wave64NextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q6k_f32_wave64_next_q8");
static constexpr KernelCatalogRef kQwenRoutedDownWeightedReduceF16F32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_weighted_reduce_f16_f32");
static constexpr KernelCatalogRef kQwenRoutedDownWeightedReduceNextRmsNormF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32");

// qwen4exp routed-down kernels: these consume 32-block quants (Q5_1/Q8_0/IQ4_NL) for the down
// projection expert weights, since qwen4exp's expert_hidden_size=640 is not a multiple of 256 and
// therefore cannot use the Q4_K/Q6_K 256-superblock kernels above. See dispatch-llm-profiles.h.
static constexpr KernelCatalogRef kQwenRoutedDownQ8_0Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q8_0_q8_1_x4");
static constexpr KernelCatalogRef kQwenRoutedDownQ5_1Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q5_1_q8_1_x4");
static constexpr KernelCatalogRef kQwenRoutedDownIQ4NLKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_iq4_nl_q8_1_x4");

// Standalone f32 -> Q8_1-x4 packer. The qwen30b decode path gets its quantized hidden state for free
// from "llm.rmsnorm.decode_f32_quantize_q8_1_x4", but that matcher is pinned to qwen30b's hidden size
// and roots at an RMS_NORM; qwen4exp feeds its routed FFN from a hyper-connection mix instead, so
// nothing publishes a Q8_1 alternate for it and the qwen4exp gate/up dispatch packs its own input.
static constexpr KernelCatalogRef kGgmlQuantizeQ8_1X4F32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_quantize_q8_1_x4_f32");
static constexpr KernelCatalogRef kZeroF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_zero_f32");
static constexpr KernelCatalogRef kCopyF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_copy_f32");
static constexpr KernelCatalogRef kIQ3XXSExpertKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_mul_mat_id_iq3_xxs_f32");
static constexpr KernelCatalogRef kIQ4XSExpertKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_mul_mat_id_iq4_xs_f32");
static constexpr KernelCatalogRef kSmallDownQ8Kernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_moe_small_down_q8_0");
static constexpr KernelCatalogRef kSmallDownIQ4Kernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_moe_small_down_iq4_nl");
static constexpr KernelCatalogRef kSiluF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_silu_f32");
static constexpr KernelCatalogRef kMulF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_mul_f32");

static constexpr const LlmMoeDispatchProfile & kRoutedFfnProfile                 = kActiveLlmMoeDispatchProfile;
static constexpr int64_t                       kRoutedFfnInputSize               = kRoutedFfnProfile.hidden_size;
static constexpr int64_t                       kRoutedFfnExpertHiddenSize        = kRoutedFfnProfile.expert_hidden_size;
static constexpr int64_t                       kRoutedFfnExpertCount             = kRoutedFfnProfile.expert_count;
static constexpr int64_t                       kRoutedFfnRouteCount              = kRoutedFfnProfile.route_count;
static constexpr size_t                        kRoutedFfnPlanTransientAlignment  = 256;
static constexpr const char *                  kRoutedFfnF16GateUpOutputName     = "qwen.moe.gate_up_swiglu_f16";
static constexpr const char *                  kRoutedFfnF16RoutedDownOutputName = "qwen.moe.routed_down_f16";
static constexpr const char *                  kRoutedFfnQ8GateUpOutputName      = "qwen.decode.moe.gate_up_swiglu_q8";
static constexpr const char *                  kRoutedFfnQ8HiddenOutputName      = "qwen.decode.moe.hidden_q8";
static constexpr const char * kRoutedFfnStagedDownOutputName = "qwen.moe.routed_down_staged_output";

// ggml-alloc recycles a tensor's buffer as soon as its last *node-level* consumer has run, so a
// fused dispatch that keeps one of its inputs live across several ggml nodes can find its own
// destination sitting on top of that input. That is exactly what happens once the MoE router runs on
// HRX: ffn_moe_argsort/ffn_moe_weights_norm die at the routed-down node, so the allocator lands
// ffn_moe_out on their bytes, and the fused kernel then reads the route ids and weights out of the
// same range it accumulates into (the zero-fill below wipes them outright). Detecting that needs the
// ggml addresses rather than HRX storage roots: aliased tensors keep distinct roots in the ValueMap
// but share one tensor->data.
static bool ggml_storage_overlaps(const Value * lhs, size_t lhs_bytes, const Value * rhs, size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs->tensor == nullptr || rhs->tensor == nullptr) {
        return false;
    }
    if (lhs->tensor->buffer != rhs->tensor->buffer || lhs->tensor->data == nullptr || rhs->tensor->data == nullptr) {
        return false;
    }
    const auto * lhs_begin = static_cast<const uint8_t *>(lhs->tensor->data);
    const auto * rhs_begin = static_cast<const uint8_t *>(rhs->tensor->data);
    return lhs_begin < rhs_begin + rhs_bytes && rhs_begin < lhs_begin + lhs_bytes;
}

// qwen4exp-scoped shape constants, deliberately kept distinct from the kRoutedFfn* aliases above
// (which are derived from kActiveLlmMoeDispatchProfile == qwen30b) so the two model geometries can
// never be confused. kActiveLlmMoeDispatchProfile must stay pinned to qwen30b -- see
// dispatch-llm-profiles.h for why.
static constexpr int64_t kQwen4ExpRoutedFfnInputSize        = kQwen4ExpMoeDispatchProfile.hidden_size;
static constexpr int64_t kQwen4ExpRoutedFfnExpertHiddenSize = kQwen4ExpMoeDispatchProfile.expert_hidden_size;
static constexpr int64_t kQwen4ExpRoutedFfnExpertCount      = kQwen4ExpMoeDispatchProfile.expert_count;
static constexpr int64_t kQwen4ExpRoutedFfnRouteCount       = kQwen4ExpMoeDispatchProfile.route_count;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_profile_rms_norm_epsilon(float eps) {
    const float expected = kRoutedFfnProfile.rms_norm_epsilon;
    return eps >= expected * 0.9f && eps <= expected * 1.1f;
}

static bool is_routed_ffn_gate_up_weight(const Value & value) {
    return value.type == GGML_TYPE_Q4_K && value.contiguous &&
           is_shape(value, kRoutedFfnInputSize, kRoutedFfnExpertHiddenSize, kRoutedFfnExpertCount, 1);
}

static bool is_routed_ffn_down_weight(const Value & value) {
    return (value.type == GGML_TYPE_Q4_K || value.type == GGML_TYPE_Q6_K) && value.contiguous &&
           is_shape(value, kRoutedFfnExpertHiddenSize, kRoutedFfnInputSize, kRoutedFfnExpertCount, 1);
}

static bool is_routed_ffn_projection_output(const Value & value, int64_t token_count) {
    return value.type == GGML_TYPE_F32 && value.contiguous &&
           is_shape(value, kRoutedFfnExpertHiddenSize, kRoutedFfnRouteCount, token_count, 1);
}

static bool is_routed_ffn_down_output(const Value & value, int64_t token_count) {
    return value.type == GGML_TYPE_F32 && value.contiguous &&
           is_shape(value, kRoutedFfnInputSize, kRoutedFfnRouteCount, token_count, 1);
}

// qwen4exp siblings of the shape checks above: pure additions, scoped to qwen4exp's geometry and
// its 32-block quant weight types, so the qwen30b matchers/constants above are never touched.
static bool is_routed_ffn_down_weight_qwen4exp(const Value & value) {
    return (value.type == GGML_TYPE_Q5_1 || value.type == GGML_TYPE_Q8_0 || value.type == GGML_TYPE_IQ4_NL) &&
           value.contiguous &&
           is_shape(value, kQwen4ExpRoutedFfnExpertHiddenSize, kQwen4ExpRoutedFfnInputSize, kQwen4ExpRoutedFfnExpertCount, 1);
}

static bool is_routed_ffn_projection_output_qwen4exp(const Value & value, int64_t token_count) {
    return value.type == GGML_TYPE_F32 && value.contiguous &&
           is_shape(value, kQwen4ExpRoutedFfnExpertHiddenSize, kQwen4ExpRoutedFfnRouteCount, token_count, 1);
}

static bool is_routed_ffn_down_output_qwen4exp(const Value & value, int64_t token_count) {
    return value.type == GGML_TYPE_F32 && value.contiguous &&
           is_shape(value, kQwen4ExpRoutedFfnInputSize, kQwen4ExpRoutedFfnRouteCount, token_count, 1);
}

// qwen4exp's routed gate/up expert weights are Q4_K at [hidden=2560, expert_hidden=640, experts=512],
// the transpose of the down geometry checked above, so the two predicates never collide.
static bool is_routed_ffn_gate_up_weight_qwen4exp(const Value & value) {
    return value.type == GGML_TYPE_Q4_K && value.contiguous &&
           is_shape(value, kQwen4ExpRoutedFfnInputSize, kQwen4ExpRoutedFfnExpertHiddenSize,
                    kQwen4ExpRoutedFfnExpertCount, 1);
}

static bool match_decode_iq_expert_qwen4exp_dispatch(const DispatchMatchContext & context,
                                                   DispatchMatch & dispatch_match) {
    const GraphNode * root = context.root_node;
    if (!iq_experts_enabled() || root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3) {
        return false;
    }
    const Value * weight = graph_value(context.graph, root->inputs[0]);
    const Value * input = graph_value(context.graph, root->inputs[1]);
    const Value * ids = graph_value(context.graph, root->inputs[2]);
    const Value * output = graph_value(context.graph, root->output);
    const int64_t tokens = input == nullptr ? 0 : input->ne[2];
    if (weight == nullptr || input == nullptr || ids == nullptr || output == nullptr ||
        (weight->type != GGML_TYPE_IQ3_XXS && weight->type != GGML_TYPE_IQ4_XS) || !weight->contiguous ||
        !is_shape(*weight, kQwen4ExpRoutedFfnInputSize, kQwen4ExpRoutedFfnExpertHiddenSize,
                  kQwen4ExpRoutedFfnExpertCount, 1) ||
        input->type != GGML_TYPE_F32 || !input->contiguous ||
        !moe_token_count_supported(tokens) ||
        !is_shape(*input, kQwen4ExpRoutedFfnInputSize, 1, tokens, 1) ||
        ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
        ids->nb[1] % sizeof(int32_t) != 0 || ids->nb[1] < kQwen4ExpRoutedFfnRouteCount * sizeof(int32_t) ||
        ids->nb[1] > 512 * sizeof(int32_t) ||
        !is_shape(*ids, kQwen4ExpRoutedFfnRouteCount, tokens, 1, 1) ||
        !is_routed_ffn_projection_output_qwen4exp(*output, tokens)) {
        return false;
    }

    // Copy the authoritative GGML tables into a read-only program constant, never the weights.
    std::vector<uint8_t> tables(sizeof(iq3xxs_grid) + sizeof(ksigns_iq2xs) + sizeof(kvalues_iq4nl));
    std::memcpy(tables.data(), iq3xxs_grid, sizeof(iq3xxs_grid));
    std::memcpy(tables.data() + sizeof(iq3xxs_grid), ksigns_iq2xs, sizeof(ksigns_iq2xs));
    std::memcpy(tables.data() + sizeof(iq3xxs_grid) + sizeof(ksigns_iq2xs), kvalues_iq4nl, sizeof(kvalues_iq4nl));
    const ValueId table_value = context.next_plan_value;
    const size_t table_bytes = tables.size();
    const char * table_name = "qwen4exp.iq_expert.tables";
    dispatch_match.transients.push_back({ table_value, table_name, table_bytes, kRoutedFfnPlanTransientAlignment });
    dispatch_match.constant_initializations.push_back({ table_value, table_name, 0, std::move(tables) });

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(weight->type == GGML_TYPE_IQ3_XXS ?
                                                 kIQ3XXSExpertKernel : kIQ4XSExpertKernel);
    dispatch.kernel.integer_parameters.emplace("input_size", kQwen4ExpRoutedFfnInputSize);
    dispatch.kernel.integer_parameters.emplace("output_size", kQwen4ExpRoutedFfnExpertHiddenSize);
    dispatch.kernel.integer_parameters.emplace("expert_count", kQwen4ExpRoutedFfnExpertCount);
    dispatch.kernel.integer_parameters.emplace("route_count", kQwen4ExpRoutedFfnRouteCount);
    dispatch.kernel.integer_parameters.emplace("token_count", tokens);
    dispatch.kernel.integer_parameters.emplace("route_id_stride", ids->nb[1] / sizeof(int32_t));
    dispatch.bindings.push_back({ weight->id, 0, weight->byte_count });
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ ids->id, 0, static_cast<size_t>(tokens - 1) * ids->nb[1] +
                                             kQwen4ExpRoutedFfnRouteCount * sizeof(int32_t) });
    dispatch.bindings.push_back({ table_value, 0, table_bytes });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_decode_iq_swiglu_qwen4exp_dispatch(const DispatchMatchContext & context,
                                                   DispatchMatch & dispatch_match) {
    const GraphNode * root = context.root_node;
    if (!iq_experts_enabled() || root == nullptr || root->op != GGML_OP_GLU || root->inputs.size() != 2) {
        return false;
    }
    const GluParams * params = op_params_as<GluParams>(root->params);
    const Value * gate = graph_value(context.graph, root->inputs[0]);
    const Value * up = graph_value(context.graph, root->inputs[1]);
    const Value * output = graph_value(context.graph, root->output);
    const int64_t tokens = output == nullptr ? 0 : output->ne[2];
    if (params == nullptr || params->op != GGML_GLU_OP_SWIGLU || gate == nullptr || up == nullptr ||
        output == nullptr || !moe_token_count_supported(tokens) ||
        !is_routed_ffn_projection_output_qwen4exp(*gate, tokens) ||
        !is_routed_ffn_projection_output_qwen4exp(*up, tokens) ||
        !is_routed_ffn_projection_output_qwen4exp(*output, tokens)) {
        return false;
    }
    const ValueId silu_output = context.next_plan_value;
    dispatch_match.transients.push_back(
        { silu_output, "qwen4exp.decode.moe.silu", output->byte_count, kRoutedFfnPlanTransientAlignment });
    Dispatch silu;
    silu.kernel = make_kernel_specialization(kSiluF32Kernel);
    silu.kernel.integer_parameters.emplace("element_count", output->element_count);
    silu.bindings.push_back({ gate->id, 0, gate->byte_count });
    silu.bindings.push_back({ silu_output, 0, output->byte_count });
    Dispatch mul;
    mul.kernel = make_kernel_specialization(kMulF32Kernel);
    mul.kernel.integer_parameters.emplace("period", output->element_count);
    mul.kernel.integer_parameters.emplace("outer_count", 1);
    mul.bindings.push_back({ silu_output, 0, output->byte_count });
    mul.bindings.push_back({ up->id, 0, up->byte_count });
    mul.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(silu));
    dispatch_match.dispatches.push_back(std::move(mul));
    return true;
}

static const GraphNode * find_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer != nullptr && consumer->op == op) {
            return consumer;
        }
    }
    return nullptr;
}

static std::vector<const GraphNode *> find_consumers_with_op(const Graph & graph, ValueId value, ggml_op op) {
    std::vector<const GraphNode *> matches;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer != nullptr && consumer->op == op) {
            matches.push_back(consumer);
        }
    }
    return matches;
}

static const GraphNode * find_single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> consumers = find_consumers_with_op(graph, value, op);
    return consumers.size() == 1 ? consumers.front() : nullptr;
}

static const GraphNode * producer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const GraphNode * producer = graph.index().producer(value);
    return producer != nullptr && producer->op == op ? producer : nullptr;
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static size_t expert_table_size(int64_t token_count) {
    return static_cast<size_t>(kRoutedFfnExpertCount + kRoutedFfnExpertCount * token_count) * sizeof(int32_t);
}

static size_t partition_table_size(int64_t token_count) {
    const int64_t assignment_count           = token_count * kRoutedFfnRouteCount;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return static_cast<size_t>(1 + assignment_partition_count + kRoutedFfnExpertCount) * sizeof(int32_t);
}

static size_t f16_gate_up_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kRoutedFfnRouteCount * kRoutedFfnExpertHiddenSize) * sizeof(ggml_fp16_t);
}

static size_t f16_routed_down_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kRoutedFfnRouteCount * kRoutedFfnInputSize) * sizeof(ggml_fp16_t);
}

static size_t q8_1_x4_byte_count(int64_t row_count, int64_t input_size) {
    if (row_count <= 0 || input_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(row_count) * ggml_row_size(GGML_TYPE_Q8_1, input_size);
}

static uint32_t gate_up_completion_counter_count(int64_t token_count,
                                                  int64_t route_count = kRoutedFfnRouteCount,
                                                  int64_t output_size = kRoutedFfnExpertHiddenSize) {
    const int64_t physical_group_count = (output_size + 127) / 128;
    return static_cast<uint32_t>(token_count * route_count * physical_group_count);
}

static void add_routed_down_compile_parameters(Dispatch & dispatch, int64_t token_count) {
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.input_size",
                                               to_config_value(kRoutedFfnExpertHiddenSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.route_count",
                                               to_config_value(kRoutedFfnRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.expert_count",
                                               to_config_value(kRoutedFfnExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.output_size",
                                               to_config_value(kRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(token_count));
}

// qwen4exp sibling of add_routed_down_compile_parameters above, using qwen4exp's shape constants
// instead of the qwen30b-derived kRoutedFfn* ones. Unlike the qwen30b kernels, which share one
// "qwen3_moe.routed_down.*" config namespace, each of the three standalone qwen4exp down kernels
// declares its own quant-suffixed namespace ("routed_down_q5_1", "routed_down_q8_0",
// "routed_down_iq4_nl", plus a matching "workload_<quant>"), so the suffix has to be threaded through
// here or Loom rejects the specialization with "unresolved config ... remains for final compilation".
static void add_qwen4exp_routed_down_compile_parameters(Dispatch &        dispatch,
                                                        int64_t           token_count,
                                                        const std::string & quant_suffix) {
    const std::string down_prefix = "qwen3_moe.routed_down_" + quant_suffix + ".";
    dispatch.kernel.compile_parameters.emplace(down_prefix + "input_size",
                                               to_config_value(kQwen4ExpRoutedFfnExpertHiddenSize));
    dispatch.kernel.compile_parameters.emplace(down_prefix + "route_count",
                                               to_config_value(kQwen4ExpRoutedFfnRouteCount));
    dispatch.kernel.compile_parameters.emplace(down_prefix + "expert_count",
                                               to_config_value(kQwen4ExpRoutedFfnExpertCount));
    dispatch.kernel.compile_parameters.emplace(down_prefix + "output_size",
                                               to_config_value(kQwen4ExpRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload_" + quant_suffix + ".token_capacity",
                                               to_config_value(token_count));
}

struct RoutedGateUpMatch {
    const Value *                       gate_weight    = nullptr;
    const Value *                       up_weight      = nullptr;
    const Value *                       input          = nullptr;
    const Value *                       route_ids      = nullptr;
    const Value *                       gate_output    = nullptr;
    const Value *                       up_output      = nullptr;
    const Value *                       glu_output     = nullptr;
    const CommandPlanMoeRoutingBundle * routing_bundle = nullptr;
    const GraphNode *                   gate_node      = nullptr;
    const GraphNode *                   up_node        = nullptr;
    const GraphNode *                   glu_node       = nullptr;
    int64_t                             token_count    = 0;

    bool matched() const {
        return gate_weight != nullptr && up_weight != nullptr && input != nullptr && route_ids != nullptr &&
               gate_output != nullptr && up_output != nullptr && glu_output != nullptr && routing_bundle != nullptr &&
               gate_node != nullptr && up_node != nullptr && glu_node != nullptr && token_count > 0;
    }
};

struct DecodeRoutedGateUpMatch {
    const Value *                     gate_weight     = nullptr;
    const Value *                     up_weight       = nullptr;
    const Value *                     input           = nullptr;
    const Value *                     route_ids       = nullptr;
    const Value *                     glu_output      = nullptr;
    const CommandPlanAlternateValue * input_alternate = nullptr;
    const GraphNode *                 down_node       = nullptr;
    const Value *                     down_weight     = nullptr;
    const GraphNode *                 gate_node       = nullptr;
    const GraphNode *                 up_node         = nullptr;
    const GraphNode *                 glu_node        = nullptr;
    int64_t                           token_count     = 0;
    int64_t                           route_stride    = 0;
    int64_t                           route_count     = 0;
    bool                              publish_q8      = false;
    // Set when the dispatch packs the f32 input into Q8_1-x4 itself instead of binding an alternate
    // published upstream. qwen4exp has no upstream publisher; see kGgmlQuantizeQ8_1X4F32Kernel.
    bool                              quantize_input  = false;

    bool matched() const {
        return gate_weight != nullptr && up_weight != nullptr && input != nullptr && route_ids != nullptr &&
               glu_output != nullptr && (input_alternate != nullptr || quantize_input) && down_node != nullptr &&
               down_weight != nullptr && gate_node != nullptr && up_node != nullptr && glu_node != nullptr &&
               token_count == 1 && route_count > 0 && route_stride >= route_count;
    }
};

struct RoutedDownMatch {
    const Value *                       input_graph_value = nullptr;
    const CommandPlanAlternateValue *   input_alternate   = nullptr;
    const Value *                       weight            = nullptr;
    const Value *                       output            = nullptr;
    const Value *                       route_ids         = nullptr;
    const CommandPlanMoeRoutingBundle * routing_bundle    = nullptr;
    KernelCatalogRef                    kernel            = {};
    int64_t                             token_count       = 0;

    bool matched() const {
        return input_graph_value != nullptr && input_alternate != nullptr && weight != nullptr && output != nullptr &&
               route_ids != nullptr && routing_bundle != nullptr && kernel.id != kUncatalogedKernelId &&
               token_count > 0;
    }
};

struct WeightedReduceNextRmsNormMatch {
    const GraphNode * rms_node    = nullptr;
    const GraphNode * mul_node    = nullptr;
    const Value *     norm_weight = nullptr;
    const Value *     output      = nullptr;

    bool matched() const {
        return rms_node != nullptr && mul_node != nullptr && norm_weight != nullptr && output != nullptr;
    }
};

struct WeightedReduceMatch {
    const Value *                     route_weights    = nullptr;
    const Value *                     routed_output    = nullptr;
    const CommandPlanAlternateValue * routed_alternate = nullptr;
    const Value *                     residual_input   = nullptr;
    const Value *                     output           = nullptr;
    const GraphNode *                 weighted_node    = nullptr;
    std::vector<const GraphNode *>    views;
    std::vector<const GraphNode *>    reductions;
    const GraphNode *                 residual = nullptr;
    // Set when the routed reduce has no residual ADD to fuse. qwen4exp's routed output is consumed by
    // `ggml_add(moe_out, ffn_shexp)`, and the shared-expert branch feeding it runs on the CPU, so that
    // ADD lands in a different scheduler split and is invisible to this matcher. The dispatch then
    // zero-fills `output` (the ADD-tree terminal) before the down kernel accumulates into it, instead
    // of aliasing a residual buffer into place. Only the qwen4exp matcher ever sets this.
    bool                              residual_missing = false;
    WeightedReduceNextRmsNormMatch    next_rmsnorm;
    int64_t                           token_count = 0;

    bool topology_matched() const {
        return route_weights != nullptr && routed_output != nullptr && output != nullptr &&
               weighted_node != nullptr && !views.empty() && token_count > 0 &&
               (residual_missing || (residual_input != nullptr && residual != nullptr));
    }

    bool matched() const { return topology_matched() && routed_alternate != nullptr; }
};

struct DecodeRoutedDownMatch {
    const Value *                     input_graph_value = nullptr;
    const CommandPlanAlternateValue * input_alternate   = nullptr;
    const Value *                     weight            = nullptr;
    const Value *                     output            = nullptr;
    const Value *                     route_ids         = nullptr;
    WeightedReduceMatch               reduce;
    KernelCatalogRef                  kernel       = {};
    int64_t                           token_count  = 0;
    int64_t                           route_stride = 0;
    bool                              input_is_q8  = false;

    bool matched() const {
        return input_graph_value != nullptr && weight != nullptr && output != nullptr && route_ids != nullptr &&
               reduce.topology_matched() && reduce.next_rmsnorm.matched() && kernel.id != kUncatalogedKernelId &&
               token_count == 1 && route_stride >= kRoutedFfnRouteCount && (!input_is_q8 || input_alternate != nullptr);
    }
};

// qwen4exp sibling of DecodeRoutedDownMatch: the 3 new qwen4exp routed-down kernels (Q5_1/Q8_0/
// IQ4_NL) do not fuse RMSNorm the way the qwen30b "_next_q8" kernel does, so matched() only requires
// reduce.topology_matched() -- NOT reduce.next_rmsnorm.matched(). RMSNorm falls through uncovered to
// a separate (CPU) dispatch, which is the intended hybrid execution model for qwen4exp on HRX.
struct DecodeRoutedDownPlainMatch {
    const Value *                     input_graph_value = nullptr;
    const CommandPlanAlternateValue * input_alternate   = nullptr;
    const Value *                     weight            = nullptr;
    const Value *                     output            = nullptr;
    const Value *                     route_ids         = nullptr;
    WeightedReduceMatch               reduce;
    KernelCatalogRef                  kernel       = {};
    // Each qwen4exp down kernel declares its own quant-suffixed Loom config namespace; see
    // add_qwen4exp_routed_down_compile_parameters().
    std::string                       quant_suffix;
    int64_t                           token_count  = 0;
    int64_t                           route_stride = 0;
    bool                              input_is_q8  = false;
    bool                              pack_input   = false;

    bool matched() const {
        return input_graph_value != nullptr && weight != nullptr && output != nullptr && route_ids != nullptr &&
               reduce.topology_matched() && kernel.id != kUncatalogedKernelId && !quant_suffix.empty() &&
               moe_token_count_supported(token_count) && route_stride >= kQwen4ExpRoutedFfnRouteCount &&
               (!input_is_q8 || input_alternate != nullptr || pack_input);
    }
};

static bool bundle_matches_moe_routing(const CommandPlanMoeRoutingBundle & bundle,
                                       ValueId                             route_ids,
                                       int64_t                             token_count) {
    return bundle.route_ids == route_ids && bundle.route_weights.value >= 0 && bundle.expert_table.value >= 0 &&
           bundle.partition_table.value >= 0 && bundle.expert_table_byte_count == expert_table_size(token_count) &&
           bundle.partition_table_byte_count == partition_table_size(token_count) &&
           bundle.token_count == token_count && bundle.route_count == kRoutedFfnRouteCount &&
           bundle.expert_count == kRoutedFfnExpertCount && bundle.route_stride >= kRoutedFfnRouteCount;
}

static bool match_same_route_projection(const Graph &     graph,
                                        const GraphNode & node,
                                        ValueId           expected_input,
                                        ValueId           expected_route_ids,
                                        int64_t           token_count,
                                        const Value *&    weight,
                                        const Value *&    output) {
    if (node.op != GGML_OP_MUL_MAT_ID || node.inputs.size() != 3 || node.inputs[1] != expected_input ||
        node.inputs[2] != expected_route_ids) {
        return false;
    }
    weight = graph_value(graph, node.inputs[0]);
    output = graph_value(graph, node.output);
    return weight != nullptr && output != nullptr && is_routed_ffn_gate_up_weight(*weight) &&
           is_routed_ffn_projection_output(*output, token_count);
}

// qwen4exp sibling of match_same_route_projection(): identical topology test, but against qwen4exp's
// gate/up weight and projection-output geometry.
static bool match_same_route_projection_qwen4exp(const Graph &     graph,
                                                 const GraphNode & node,
                                                 ValueId           expected_input,
                                                 ValueId           expected_route_ids,
                                                 int64_t           token_count,
                                                 const Value *&    weight,
                                                 const Value *&    output) {
    if (node.op != GGML_OP_MUL_MAT_ID || node.inputs.size() != 3 || node.inputs[1] != expected_input ||
        node.inputs[2] != expected_route_ids) {
        return false;
    }
    weight = graph_value(graph, node.inputs[0]);
    output = graph_value(graph, node.output);
    return weight != nullptr && output != nullptr && is_routed_ffn_gate_up_weight_qwen4exp(*weight) &&
           is_routed_ffn_projection_output_qwen4exp(*output, token_count);
}

static RoutedDownMatch match_routed_ffn_down_grouped(const DispatchMatchContext & context) {
    RoutedDownMatch   match;
    const GraphNode * root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * weight      = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_routed_ffn_down_weight(*weight) || !is_routed_ffn_projection_output(*input, input->ne[2]) ||
        route_ids->type != GGML_TYPE_I32 || !is_shape(*route_ids, kRoutedFfnRouteCount, input->ne[2], 1, 1)) {
        return {};
    }

    const int64_t token_count = input->ne[2];
    if (!is_llm_supported_query_length(kRoutedFfnProfile, token_count) ||
        !is_routed_ffn_down_output(*root_output, token_count)) {
        return {};
    }

    const CommandPlanMoeRoutingBundle * routing_bundle = context.plan.metadata.find_moe_routing_bundle(route_ids->id);
    if (routing_bundle == nullptr || !bundle_matches_moe_routing(*routing_bundle, route_ids->id, token_count)) {
        return {};
    }

    const CommandPlanAlternateValue * input_alternate =
        find_alternate_value(context.plan, input->id, GGML_TYPE_F16, f16_gate_up_output_size(token_count));
    if (input_alternate == nullptr) {
        return {};
    }

    match.input_graph_value = input;
    match.input_alternate   = input_alternate;
    match.weight            = weight;
    match.output            = root_output;
    match.route_ids         = route_ids;
    match.routing_bundle    = routing_bundle;
    match.kernel            = weight->type == GGML_TYPE_Q4_K ? kQwenRoutedDownQ4KF16WmmaGroupedKernel :
                                                               kQwenRoutedDownQ6KF16WmmaGroupedKernel;
    match.token_count       = token_count;
    return match;
}

static RoutedGateUpMatch match_routed_ffn_gate_up_swiglu(const DispatchMatchContext & context) {
    RoutedGateUpMatch match;
    const GraphNode * root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * root_weight = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (root_weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_routed_ffn_gate_up_weight(*root_weight) || input->type != GGML_TYPE_F32 || !input->contiguous ||
        !is_shape(*input, kRoutedFfnInputSize, 1, input->ne[2], 1) || route_ids->type != GGML_TYPE_I32 ||
        !is_shape(*route_ids, kRoutedFfnRouteCount, input->ne[2], 1, 1)) {
        return {};
    }

    const int64_t token_count = input->ne[2];
    if (!is_llm_supported_query_length(kRoutedFfnProfile, token_count) ||
        !is_routed_ffn_projection_output(*root_output, token_count)) {
        return {};
    }

    const CommandPlanMoeRoutingBundle * routing_bundle = context.plan.metadata.find_moe_routing_bundle(route_ids->id);
    if (routing_bundle == nullptr || !bundle_matches_moe_routing(*routing_bundle, route_ids->id, token_count)) {
        return {};
    }

    const GraphNode * glu_node = find_consumer_with_op(context.graph, root->output, GGML_OP_GLU);
    if (glu_node == nullptr || glu_node->inputs.size() != 2) {
        return {};
    }
    const GluParams * glu_params = op_params_as<GluParams>(glu_node->params);
    if (glu_params == nullptr || glu_params->op != GGML_GLU_OP_SWIGLU) {
        return {};
    }

    const GraphNode * gate_node = producer_with_op(context.graph, glu_node->inputs[0], GGML_OP_MUL_MAT_ID);
    const GraphNode * up_node   = producer_with_op(context.graph, glu_node->inputs[1], GGML_OP_MUL_MAT_ID);
    if (gate_node == nullptr || up_node == nullptr || gate_node == up_node || (gate_node != root && up_node != root)) {
        return {};
    }

    const Value * gate_weight = nullptr;
    const Value * gate_output = nullptr;
    const Value * up_weight   = nullptr;
    const Value * up_output   = nullptr;
    if (!match_same_route_projection(context.graph, *gate_node, input->id, route_ids->id, token_count, gate_weight,
                                     gate_output) ||
        !match_same_route_projection(context.graph, *up_node, input->id, route_ids->id, token_count, up_weight,
                                     up_output)) {
        return {};
    }
    if (!same_shape(*gate_output, *up_output)) {
        return {};
    }

    const Value * glu_output = graph_value(context.graph, glu_node->output);
    if (glu_output == nullptr || glu_output->kind != ValueKind::Transient ||
        !is_routed_ffn_projection_output(*glu_output, token_count)) {
        return {};
    }

    match.gate_weight    = gate_weight;
    match.up_weight      = up_weight;
    match.input          = input;
    match.route_ids      = route_ids;
    match.gate_output    = gate_output;
    match.up_output      = up_output;
    match.glu_output     = glu_output;
    match.routing_bundle = routing_bundle;
    match.gate_node      = gate_node;
    match.up_node        = up_node;
    match.glu_node       = glu_node;
    match.token_count    = token_count;
    return match;
}

static DecodeRoutedGateUpMatch match_decode_routed_ffn_gate_up_swiglu(const DispatchMatchContext & context) {
    DecodeRoutedGateUpMatch match;
    const GraphNode *       root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * root_weight = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (root_weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_routed_ffn_gate_up_weight(*root_weight) || input->type != GGML_TYPE_F32 || !input->contiguous ||
        !is_shape(*input, kRoutedFfnInputSize, 1, 1, 1) || route_ids->type != GGML_TYPE_I32 ||
        !is_shape(*route_ids, kRoutedFfnRouteCount, 1, 1, 1) || !is_routed_ffn_projection_output(*root_output, 1)) {
        return {};
    }

    const int64_t                     route_stride    = static_cast<int64_t>(route_ids->nb[1] / sizeof(int32_t));
    const CommandPlanAlternateValue * input_alternate = find_alternate_value(
        context.graph, context.plan, input->id, GGML_TYPE_Q8_1, q8_1_x4_byte_count(1, kRoutedFfnInputSize));
    if (input_alternate == nullptr) {
        return {};
    }

    const GraphNode * glu_node = find_consumer_with_op(context.graph, root->output, GGML_OP_GLU);
    if (glu_node == nullptr || glu_node->inputs.size() != 2) {
        return {};
    }
    const GluParams * glu_params = op_params_as<GluParams>(glu_node->params);
    if (glu_params == nullptr || glu_params->op != GGML_GLU_OP_SWIGLU) {
        return {};
    }

    const GraphNode * gate_node = producer_with_op(context.graph, glu_node->inputs[0], GGML_OP_MUL_MAT_ID);
    const GraphNode * up_node   = producer_with_op(context.graph, glu_node->inputs[1], GGML_OP_MUL_MAT_ID);
    if (gate_node == nullptr || up_node == nullptr || gate_node == up_node || (gate_node != root && up_node != root)) {
        return {};
    }

    const Value * gate_weight = nullptr;
    const Value * gate_output = nullptr;
    const Value * up_weight   = nullptr;
    const Value * up_output   = nullptr;
    if (!match_same_route_projection(context.graph, *gate_node, input->id, route_ids->id, 1, gate_weight,
                                     gate_output) ||
        !match_same_route_projection(context.graph, *up_node, input->id, route_ids->id, 1, up_weight, up_output) ||
        !same_shape(*gate_output, *up_output)) {
        return {};
    }

    const Value * glu_output = graph_value(context.graph, glu_node->output);
    if (glu_output == nullptr || glu_output->kind != ValueKind::Transient ||
        !is_routed_ffn_projection_output(*glu_output, 1)) {
        return {};
    }

    const GraphNode * down_node   = find_single_consumer_with_op(context.graph, glu_output->id, GGML_OP_MUL_MAT_ID);
    const Value *     down_weight = down_node == nullptr || down_node->inputs.size() != 3 ?
                                        nullptr :
                                        graph_value(context.graph, down_node->inputs[0]);
    if (down_node == nullptr || down_weight == nullptr || !is_routed_ffn_down_weight(*down_weight)) {
        return {};
    }

    match.gate_weight     = gate_weight;
    match.up_weight       = up_weight;
    match.input           = input;
    match.route_ids       = route_ids;
    match.glu_output      = glu_output;
    match.input_alternate = input_alternate;
    match.down_node       = down_node;
    match.down_weight     = down_weight;
    match.gate_node       = gate_node;
    match.up_node         = up_node;
    match.glu_node        = glu_node;
    match.token_count     = 1;
    match.route_stride    = route_stride;
    match.route_count     = kRoutedFfnRouteCount;
    match.publish_q8      = down_weight->type == GGML_TYPE_Q4_K;
    return match;
}

// qwen4exp routed gate/up + SwiGLU, decode (single token). Mirrors
// match_decode_routed_ffn_gate_up_swiglu() above but against qwen4exp's geometry
// (hidden 2560, expert_hidden 640, experts 512, route 10) via the kQwen4ExpRoutedFfn* constants.
//
// Registered *alongside* the qwen30b matcher rather than switching kActiveLlmMoeDispatchProfile:
// the two disagree on every shape, so each declines what the other claims and qwen30b is untouched.
//
// publish_q8 is unconditionally true here. All three qwen4exp down kernels (Q5_1/Q8_0/IQ4_NL)
// consume a Q8_1-x4 quantized activation, and match_decode_routed_ffn_down_qwen4exp() looks up
// exactly that alternate -- so without publishing it the down projection can never match, which is
// why the down path was gated off until now.
static DecodeRoutedGateUpMatch match_decode_routed_ffn_gate_up_swiglu_qwen4exp(
    const DispatchMatchContext & context) {
    DecodeRoutedGateUpMatch match;
    const GraphNode *       root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * root_weight = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (root_weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_routed_ffn_gate_up_weight_qwen4exp(*root_weight) || input->type != GGML_TYPE_F32 || !input->contiguous ||
        !is_shape(*input, kQwen4ExpRoutedFfnInputSize, 1, 1, 1) || route_ids->type != GGML_TYPE_I32 ||
        route_ids->nb[0] != sizeof(int32_t) || route_ids->nb[1] % sizeof(int32_t) != 0 ||
        !is_shape(*route_ids, kQwen4ExpRoutedFfnRouteCount, 1, 1, 1) ||
        !is_routed_ffn_projection_output_qwen4exp(*root_output, 1)) {
        return {};
    }

    const int64_t route_stride = static_cast<int64_t>(route_ids->nb[1] / sizeof(int32_t));
    if (input->byte_count < static_cast<size_t>(kQwen4ExpRoutedFfnInputSize) * sizeof(float)) {
        return {};
    }

    const GraphNode * glu_node = find_consumer_with_op(context.graph, root->output, GGML_OP_GLU);
    if (glu_node == nullptr || glu_node->inputs.size() != 2) {
        return {};
    }
    const GluParams * glu_params = op_params_as<GluParams>(glu_node->params);
    if (glu_params == nullptr || glu_params->op != GGML_GLU_OP_SWIGLU) {
        return {};
    }

    const GraphNode * gate_node = producer_with_op(context.graph, glu_node->inputs[0], GGML_OP_MUL_MAT_ID);
    const GraphNode * up_node   = producer_with_op(context.graph, glu_node->inputs[1], GGML_OP_MUL_MAT_ID);
    if (gate_node == nullptr || up_node == nullptr || gate_node == up_node || (gate_node != root && up_node != root)) {
        return {};
    }

    const Value * gate_weight = nullptr;
    const Value * gate_output = nullptr;
    const Value * up_weight   = nullptr;
    const Value * up_output   = nullptr;
    if (!match_same_route_projection_qwen4exp(context.graph, *gate_node, input->id, route_ids->id, 1, gate_weight,
                                              gate_output) ||
        !match_same_route_projection_qwen4exp(context.graph, *up_node, input->id, route_ids->id, 1, up_weight,
                                              up_output) ||
        !same_shape(*gate_output, *up_output)) {
        return {};
    }

    const Value * glu_output = graph_value(context.graph, glu_node->output);
    // Deliberately no ValueKind::Transient requirement, unlike the qwen30b matcher above. Transient
    // is reserved for values a dispatch plan synthesizes; import_ggml_graph() classifies every tensor
    // that comes out of a ggml graph as External (see graph.cpp -- treating a scheduler-owned tensor
    // as transient scratch bound kernels to uninitialized arena memory, so that rule was removed), and
    // the SwiGLU result is an ordinary ggml node. The invariant that check was reaching for -- that
    // nothing outside this fusion observes the SwiGLU result -- is already established below by
    // find_single_consumer_with_op(), which requires the down projection to be its only consumer.
    if (glu_output == nullptr || !is_routed_ffn_projection_output_qwen4exp(*glu_output, 1)) {
        return {};
    }

    const GraphNode * down_node   = find_single_consumer_with_op(context.graph, glu_output->id, GGML_OP_MUL_MAT_ID);
    const Value *     down_weight = down_node == nullptr || down_node->inputs.size() != 3 ?
                                        nullptr :
                                        graph_value(context.graph, down_node->inputs[0]);
    if (down_node == nullptr || down_weight == nullptr || !is_routed_ffn_down_weight_qwen4exp(*down_weight)) {
        return {};
    }

    match.gate_weight     = gate_weight;
    match.up_weight       = up_weight;
    match.input           = input;
    match.route_ids       = route_ids;
    match.glu_output      = glu_output;
    match.input_alternate = nullptr;
    match.down_node       = down_node;
    match.down_weight     = down_weight;
    match.gate_node       = gate_node;
    match.up_node         = up_node;
    match.glu_node        = glu_node;
    match.token_count     = 1;
    match.route_stride    = route_stride;
    match.route_count     = kQwen4ExpRoutedFfnRouteCount;
    match.publish_q8      = true;
    match.quantize_input  = true;
    return match;
}

static WeightedReduceNextRmsNormMatch match_qwen_weighted_reduce_next_rmsnorm(const DispatchMatchContext & context,
                                                                              const Value &                residual) {
    WeightedReduceNextRmsNormMatch match;
    const GraphNode * rms_node = find_single_consumer_with_op(context.graph, residual.id, GGML_OP_RMS_NORM);
    if (rms_node == nullptr || rms_node->inputs.size() != 1) {
        return match;
    }
    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(rms_node->params);
    if (rms_params == nullptr || !is_profile_rms_norm_epsilon(rms_params->eps)) {
        return {};
    }

    const Value * rms = graph_value(context.graph, rms_node->output);
    if (rms == nullptr || rms->type != GGML_TYPE_F32 || !same_shape(*rms, residual)) {
        return {};
    }

    const GraphNode * mul_node = find_single_consumer_with_op(context.graph, rms_node->output, GGML_OP_MUL);
    if (mul_node == nullptr || mul_node->inputs.size() != 2) {
        return {};
    }

    const Value * norm_weight = nullptr;
    for (ValueId input : mul_node->inputs) {
        if (input != rms_node->output) {
            norm_weight = graph_value(context.graph, input);
        }
    }
    const Value * output = graph_value(context.graph, mul_node->output);
    if (norm_weight == nullptr || output == nullptr || norm_weight->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !norm_weight->contiguous || !output->contiguous ||
        !is_shape(*norm_weight, kRoutedFfnInputSize, 1, 1, 1) || !same_shape(*output, residual)) {
        return {};
    }

    match.rms_node    = rms_node;
    match.mul_node    = mul_node;
    match.norm_weight = norm_weight;
    match.output      = output;
    return match;
}

static bool append_node_if_uncovered(const DispatchMatchContext &     context,
                                     const GraphNode *                node,
                                     std::vector<const GraphNode *> & nodes) {
    size_t index = 0;
    if (node == nullptr || !context.graph.index().node_index(node, index) || index >= context.covered_nodes.size() ||
        context.covered_nodes[index]) {
        return false;
    }
    for (const GraphNode * existing : nodes) {
        if (existing == node) {
            return true;
        }
    }
    nodes.push_back(node);
    return true;
}

static bool node_is_covered(const DispatchMatchContext & context, const GraphNode * node) {
    size_t index = 0;
    return node != nullptr && context.graph.index().node_index(node, index) && index < context.covered_nodes.size() &&
           context.covered_nodes[index];
}

static bool input_ready_at_root(const DispatchMatchContext & context, const Value * source) {
    while (source != nullptr) {
        const GraphNode * producer = context.graph.index().producer(source->id);
        if (producer == nullptr || node_is_covered(context, producer)) {
            return true;
        }
        if (producer->op == GGML_OP_VIEW || producer->op == GGML_OP_RESHAPE ||
            producer->op == GGML_OP_PERMUTE || producer->op == GGML_OP_TRANSPOSE) {
            source = producer->inputs.empty() ? nullptr : graph_value(context.graph, producer->inputs[0]);
            continue;
        }
        size_t index = 0;
        return context.graph.index().node_index(producer, index) && index < context.root_index;
    }
    return false;
}

static bool residual_input_is_safe_for_in_place(const DispatchMatchContext & context,
                                                const WeightedReduceMatch &  match) {
    if (match.residual_input == nullptr || match.residual == nullptr) {
        return false;
    }
    for (const GraphNode * consumer : context.graph.index().consumers(match.residual_input->id)) {
        if (consumer == match.residual || node_is_covered(context, consumer)) {
            continue;
        }
        return false;
    }
    return true;
}

static const Value * find_qwen_route_weights_for_route_ids(const Graph & graph,
                                                           ValueId       route_ids,
                                                           int64_t       token_count,
                                                           int64_t       route_count) {
    const GraphNode * get_rows = find_single_consumer_with_op(graph, route_ids, GGML_OP_GET_ROWS);
    if (get_rows == nullptr || get_rows->inputs.size() != 2) {
        return nullptr;
    }
    const Value * selected = graph_value(graph, get_rows->output);
    if (selected == nullptr || selected->type != GGML_TYPE_F32 ||
        !is_shape(*selected, 1, route_count, token_count, 1)) {
        return nullptr;
    }
    const GraphNode * reshape      = find_single_consumer_with_op(graph, get_rows->output, GGML_OP_RESHAPE);
    const Value *     flat_weights = reshape == nullptr ? nullptr : graph_value(graph, reshape->output);
    if (flat_weights == nullptr || flat_weights->type != GGML_TYPE_F32 ||
        !is_shape(*flat_weights, route_count, token_count, 1, 1)) {
        return nullptr;
    }
    const GraphNode * sum_rows = find_single_consumer_with_op(graph, reshape->output, GGML_OP_SUM_ROWS);
    const GraphNode * clamp =
        sum_rows == nullptr ? nullptr : find_single_consumer_with_op(graph, sum_rows->output, GGML_OP_CLAMP);
    const GraphNode * div =
        clamp == nullptr ? nullptr : find_single_consumer_with_op(graph, clamp->output, GGML_OP_DIV);
    const GraphNode * output_reshape =
        div == nullptr ? nullptr : find_single_consumer_with_op(graph, div->output, GGML_OP_RESHAPE);
    const Value * weights = output_reshape == nullptr ? nullptr : graph_value(graph, output_reshape->output);
    if (weights == nullptr || weights->type != GGML_TYPE_F32 ||
        !is_shape(*weights, 1, route_count, token_count, 1)) {
        return nullptr;
    }
    return weights;
}

// qwen4exp reaches its normalized route weights differently from qwen30b. The qwen30b lookup above
// walks the router chain (GET_ROWS -> RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE), which only
// works because qwen30b's MoE router dispatch pulls that whole chain into the same HRX split. qwen4exp
// has no router dispatch -- its 10-of-512 top-k topology matches nothing in dispatch-moe-router.cpp --
// so the normalization trio deliberately stays on the CPU (see hrx_moe_route_weight_norm_qwen4exp()),
// and the HRX split's Graph does not contain those nodes at all. Walking the chain there always fails.
//
// What the split does contain is the weighted-reduce MUL, and the normalized weights are simply one of
// its two inputs, materialized by ggml-backend as a cross-backend copy. Pick them out by shape: the
// route weights are [1, route_count, token_count] while the routed output is [hidden, route_count,
// token_count], so the two can never be confused.
static const Value * find_route_weights_from_weighted_mul(const Graph &     graph,
                                                          const GraphNode * weighted,
                                                          const Value *     routed_output,
                                                          int64_t           token_count,
                                                          int64_t           route_count) {
    if (weighted == nullptr || weighted->op != GGML_OP_MUL || weighted->inputs.size() != 2 ||
        routed_output == nullptr) {
        return nullptr;
    }
    for (ValueId input : weighted->inputs) {
        const Value * value = graph_value(graph, input);
        if (value == nullptr || value->id.value == routed_output->id.value) {
            continue;
        }
        if (value->type == GGML_TYPE_F32 && is_shape(*value, 1, route_count, token_count, 1)) {
            return value;
        }
    }
    return nullptr;
}

static WeightedReduceMatch match_routed_ffn_down_weighted_reduce_topology(const DispatchMatchContext & context,
                                                                          const GraphNode *            weighted,
                                                                          const Value *                routed_output,
                                                                          const Value *                route_weights) {
    WeightedReduceMatch match;
    if (weighted == nullptr || weighted->op != GGML_OP_MUL || weighted->inputs.size() != 2 ||
        routed_output == nullptr || route_weights == nullptr || !context.graph.has_index()) {
        return match;
    }
    if (!node_has_input_or_alias(context.graph, *weighted, routed_output->id) ||
        !node_has_input_or_alias(context.graph, *weighted, route_weights->id)) {
        return {};
    }
    const Value * weighted_output = graph_value(context.graph, weighted->output);
    if (!is_routed_ffn_down_output(*routed_output, routed_output->ne[2]) || route_weights->type != GGML_TYPE_F32 ||
        !route_weights->contiguous || !is_shape(*route_weights, 1, kRoutedFfnRouteCount, routed_output->ne[2], 1) ||
        weighted_output == nullptr || !same_shape(*weighted_output, *routed_output)) {
        return {};
    }

    const int64_t token_count = routed_output->ne[2];
    if (!is_llm_supported_query_length(kRoutedFfnProfile, token_count)) {
        return {};
    }

    std::vector<const GraphNode *> views =
        layout_alias_consumers_with_op(context.graph, weighted->output, GGML_OP_VIEW);
    if (views.size() != kRoutedFfnRouteCount) {
        return {};
    }

    std::set<int32_t>              routed_values;
    std::vector<const GraphNode *> owned_views;
    for (const GraphNode * view : views) {
        const Value * value = view == nullptr ? nullptr : graph_value(context.graph, view->output);
        if (value == nullptr || value->type != GGML_TYPE_F32 ||
            !is_shape(*value, kRoutedFfnInputSize, token_count, 1, 1) ||
            !append_node_if_uncovered(context, view, owned_views)) {
            return {};
        }
        routed_values.insert(view->output.value);
    }

    std::vector<const GraphNode *> reductions;
    bool                           changed = true;
    while (changed) {
        changed                        = false;
        const std::set<int32_t> values = routed_values;
        for (int32_t value : values) {
            for (const GraphNode * add : find_consumers_with_op(context.graph, ValueId(value), GGML_OP_ADD)) {
                if (add == nullptr || add->inputs.size() != 2) {
                    continue;
                }
                bool already_owned = false;
                for (const GraphNode * reduction : reductions) {
                    if (reduction == add) {
                        already_owned = true;
                        break;
                    }
                }
                if (already_owned) {
                    continue;
                }
                bool all_routed = true;
                for (ValueId input : add->inputs) {
                    all_routed = all_routed && routed_values.count(input.value) != 0;
                }
                if (!all_routed) {
                    continue;
                }
                const Value * output = graph_value(context.graph, add->output);
                if (output == nullptr || output->type != GGML_TYPE_F32 ||
                    !is_shape(*output, kRoutedFfnInputSize, token_count, 1, 1) ||
                    !append_node_if_uncovered(context, add, reductions)) {
                    return {};
                }
                routed_values.insert(add->output.value);
                changed = true;
            }
        }
    }

    const GraphNode * residual       = nullptr;
    const Value *     residual_input = nullptr;
    for (int32_t value : routed_values) {
        for (const GraphNode * add : find_consumers_with_op(context.graph, ValueId(value), GGML_OP_ADD)) {
            if (add == nullptr || add->inputs.size() != 2) {
                continue;
            }
            bool is_reduction = false;
            for (const GraphNode * reduction : reductions) {
                if (reduction == add) {
                    is_reduction = true;
                    break;
                }
            }
            if (is_reduction) {
                continue;
            }
            int           routed_input_count = 0;
            const Value * non_routed_input   = nullptr;
            for (ValueId input : add->inputs) {
                if (routed_values.count(input.value) != 0) {
                    ++routed_input_count;
                } else {
                    non_routed_input = graph_value(context.graph, input);
                }
            }
            if (routed_input_count != 1 || non_routed_input == nullptr || residual != nullptr) {
                return {};
            }
            residual       = add;
            residual_input = non_routed_input;
        }
    }
    if (residual == nullptr || reductions.size() + 1 != views.size()) {
        return {};
    }
    const Value * output = graph_value(context.graph, residual->output);
    if (output == nullptr || residual_input == nullptr || output->type != GGML_TYPE_F32 ||
        residual_input->type != GGML_TYPE_F32 || !is_shape(*output, kRoutedFfnInputSize, token_count, 1, 1) ||
        !same_shape(*output, *residual_input) || output->byte_count != residual_input->byte_count ||
        !output->contiguous || !residual_input->contiguous) {
        return {};
    }

    match.route_weights  = route_weights;
    match.routed_output  = routed_output;
    match.residual_input = residual_input;
    match.output         = output;
    match.weighted_node  = weighted;
    match.views          = std::move(owned_views);
    match.reductions     = std::move(reductions);
    match.residual       = residual;
    match.next_rmsnorm   = match_qwen_weighted_reduce_next_rmsnorm(context, *output);
    match.token_count    = token_count;
    return match;
}

// qwen4exp sibling of match_routed_ffn_down_weighted_reduce_topology above: a full parallel copy
// substituting qwen4exp's route_count/input_size/shape-check throughout, rather than threading new
// parameters through the qwen30b-scoped original (chosen to keep the existing qwen30b matcher's
// behavior provably unchanged). match_qwen_weighted_reduce_next_rmsnorm() is still called at the end
// (harmless -- it uses qwen30b's kRoutedFfnInputSize/kRoutedFfnProfile internally and will simply
// fail to match qwen4exp's larger norm_weight shape / different residual, which is fine here since
// callers of this qwen4exp sibling only require topology_matched(), not next_rmsnorm.matched()).
static WeightedReduceMatch match_routed_ffn_down_weighted_reduce_topology_qwen4exp(
    const DispatchMatchContext & context,
    const GraphNode *            weighted,
    const Value *                routed_output,
    const Value *                route_weights,
    bool                         fuse_residual = true) {
    WeightedReduceMatch match;
    if (weighted == nullptr || weighted->op != GGML_OP_MUL || weighted->inputs.size() != 2 ||
        routed_output == nullptr || route_weights == nullptr || !context.graph.has_index()) {
        return match;
    }
    if (!node_has_input_or_alias(context.graph, *weighted, routed_output->id) ||
        !node_has_input_or_alias(context.graph, *weighted, route_weights->id)) {
        trace_moe_down_reject("reduce: weighted MUL inputs");
        return {};
    }
    const Value * weighted_output = graph_value(context.graph, weighted->output);
    if (!is_routed_ffn_down_output_qwen4exp(*routed_output, routed_output->ne[2]) ||
        route_weights->type != GGML_TYPE_F32 || !route_weights->contiguous ||
        !is_shape(*route_weights, 1, kQwen4ExpRoutedFfnRouteCount, routed_output->ne[2], 1) ||
        weighted_output == nullptr || !same_shape(*weighted_output, *routed_output)) {
        trace_moe_down_reject("reduce: weighted MUL shapes");
        return {};
    }

    const int64_t token_count = routed_output->ne[2];
    if (!moe_token_count_supported(token_count)) {
        trace_moe_down_reject("reduce: query length");
        return {};
    }
    if (token_count > 1 && (weighted_output->type != GGML_TYPE_F32 || !weighted_output->contiguous)) {
        return {};
    }

    std::vector<const GraphNode *> views =
        layout_alias_consumers_with_op(context.graph, weighted->output, GGML_OP_VIEW);
    if (views.size() != kQwen4ExpRoutedFfnRouteCount) {
        trace_moe_down_reject("reduce: per-expert VIEW count");
        return {};
    }

    std::set<int32_t>              routed_values;
    std::vector<const GraphNode *> owned_views;
    std::map<int32_t, uint32_t> route_masks;
    uint32_t seen_routes = 0;
    for (const GraphNode * view : views) {
        const Value * value = view == nullptr ? nullptr : graph_value(context.graph, view->output);
        if (value == nullptr || value->type != GGML_TYPE_F32 ||
            !is_shape(*value, kQwen4ExpRoutedFfnInputSize, token_count, 1, 1) ||
            !append_node_if_uncovered(context, view, owned_views)) {
            trace_moe_down_reject("reduce: per-expert VIEW shape/coverage");
            return {};
        }
        if (token_count > 1) {
            if (value->storage_root != weighted_output->storage_root ||
                value->storage_offset < weighted_output->storage_offset ||
                value->nb[0] != sizeof(float) || value->nb[1] != weighted_output->nb[2]) {
                return {};
            }
            const size_t offset = value->storage_offset - weighted_output->storage_offset;
            const size_t route = offset / weighted_output->nb[1];
            if (offset % weighted_output->nb[1] != 0 || route >= 10 ||
                (seen_routes & (1u << route)) != 0) {
                return {};
            }
            route_masks[view->output.value] = 1u << route;
            seen_routes |= 1u << route;
        }
        routed_values.insert(view->output.value);
    }

    std::vector<const GraphNode *> reductions;
    bool                           changed = true;
    while (changed) {
        changed                        = false;
        const std::set<int32_t> values = routed_values;
        for (int32_t value : values) {
            for (const GraphNode * add : find_consumers_with_op(context.graph, ValueId(value), GGML_OP_ADD)) {
                if (add == nullptr || add->inputs.size() != 2) {
                    continue;
                }
                bool already_owned = false;
                for (const GraphNode * reduction : reductions) {
                    if (reduction == add) {
                        already_owned = true;
                        break;
                    }
                }
                if (already_owned) {
                    continue;
                }
                bool all_routed = true;
                for (ValueId input : add->inputs) {
                    all_routed = all_routed && routed_values.count(input.value) != 0;
                }
                if (!all_routed) {
                    continue;
                }
                const Value * output = graph_value(context.graph, add->output);
                if (output == nullptr || output->type != GGML_TYPE_F32 ||
                    !is_shape(*output, kQwen4ExpRoutedFfnInputSize, token_count, 1, 1) ||
                    !append_node_if_uncovered(context, add, reductions)) {
                    trace_moe_down_reject("reduce: ADD-tree node shape/coverage");
                    return {};
                }
                if (token_count > 1) {
                    const uint32_t lhs = route_masks[add->inputs[0].value];
                    const uint32_t rhs = route_masks[add->inputs[1].value];
                    if ((lhs & rhs) != 0 || lhs == 0 || rhs == 0) {
                        return {};
                    }
                    route_masks[add->output.value] = lhs | rhs;
                }
                routed_values.insert(add->output.value);
                changed = true;
            }
        }
    }

    const GraphNode * residual       = nullptr;
    const Value *     residual_input = nullptr;
    for (int32_t value : routed_values) {
        for (const GraphNode * add : find_consumers_with_op(context.graph, ValueId(value), GGML_OP_ADD)) {
            if (add == nullptr || add->inputs.size() != 2) {
                continue;
            }
            bool is_reduction = false;
            for (const GraphNode * reduction : reductions) {
                if (reduction == add) {
                    is_reduction = true;
                    break;
                }
            }
            if (is_reduction) {
                continue;
            }
            int           routed_input_count = 0;
            const Value * non_routed_input   = nullptr;
            for (ValueId input : add->inputs) {
                if (routed_values.count(input.value) != 0) {
                    ++routed_input_count;
                } else {
                    non_routed_input = graph_value(context.graph, input);
                }
            }
            if (routed_input_count != 1 || non_routed_input == nullptr || residual != nullptr) {
                trace_moe_down_reject("reduce: ambiguous residual ADD");
                return {};
            }
            residual       = add;
            residual_input = non_routed_input;
        }
    }
    if (reductions.size() + 1 != views.size()) {
        trace_moe_down_reject_counts("reduce: incomplete ADD tree", views.size(), reductions.size(),
                                     residual != nullptr);
        return {};
    }

    const Value * output = nullptr;
    if (residual == nullptr || !fuse_residual) {
        // No eligible residual ADD. Write the ADD tree's terminal value directly;
        // the dispatch zero-fills it first so the kernel's unconditional `output += routed_sum`
        // accumulation still produces the plain reduce. The terminal is the one reduction output that
        // no other reduction consumes.
        std::set<int32_t> consumed;
        for (const GraphNode * add : reductions) {
            for (ValueId input : add->inputs) {
                consumed.insert(input.value);
            }
        }
        for (const GraphNode * add : reductions) {
            if (consumed.count(add->output.value) != 0) {
                continue;
            }
            if (output != nullptr) {
                trace_moe_down_reject("reduce: ADD tree has multiple terminals");
                return {};
            }
            output = graph_value(context.graph, add->output);
        }
        if (output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous ||
            !is_shape(*output, kQwen4ExpRoutedFfnInputSize, token_count, 1, 1)) {
            trace_moe_down_reject("reduce: ADD tree terminal shape/layout");
            return {};
        }
        match.residual_missing = true;
        residual               = nullptr;
        residual_input         = nullptr;
    } else {
        output = graph_value(context.graph, residual->output);
        if (output == nullptr || residual_input == nullptr || output->type != GGML_TYPE_F32 ||
            residual_input->type != GGML_TYPE_F32 ||
            !is_shape(*output, kQwen4ExpRoutedFfnInputSize, token_count, 1, 1) ||
            !same_shape(*output, *residual_input) || output->byte_count != residual_input->byte_count ||
            !output->contiguous || !residual_input->contiguous) {
            trace_moe_down_reject("reduce: residual output shape/layout");
            return {};
        }
    }

    match.route_weights  = route_weights;
    match.routed_output  = routed_output;
    match.residual_input = residual_input;
    match.output         = output;
    match.weighted_node  = weighted;
    match.views          = std::move(owned_views);
    match.reductions     = std::move(reductions);
    match.residual       = residual;
    match.next_rmsnorm   = match_qwen_weighted_reduce_next_rmsnorm(context, *output);
    match.token_count    = token_count;
    return match;
}

static WeightedReduceMatch match_routed_ffn_down_weighted_reduce(const DispatchMatchContext & context) {
    WeightedReduceMatch match;
    const GraphNode *   weighted = context.root_node;
    if (weighted == nullptr || weighted->op != GGML_OP_MUL || weighted->inputs.size() != 2 ||
        !context.graph.has_index()) {
        return match;
    }

    const Value * routed_output = nullptr;
    const Value * route_weights = nullptr;
    for (ValueId input : weighted->inputs) {
        const Value * value = graph_value(context.graph, input);
        if (value == nullptr) {
            return {};
        }
        if (is_routed_ffn_down_output(*value, value->ne[2])) {
            routed_output = value;
        } else if (value->type == GGML_TYPE_F32 && value->contiguous &&
                   is_shape(*value, 1, kRoutedFfnRouteCount, value->ne[2], 1)) {
            route_weights = value;
        }
    }
    match = match_routed_ffn_down_weighted_reduce_topology(context, weighted, routed_output, route_weights);
    if (!match.topology_matched()) {
        return {};
    }
    const int64_t                     token_count = match.token_count;
    const CommandPlanAlternateValue * routed_alternate =
        find_alternate_value(context.plan, routed_output->id, GGML_TYPE_F16, f16_routed_down_output_size(token_count));
    if (routed_alternate == nullptr) {
        return {};
    }

    bool known_route_weights = false;
    for (const CommandPlanMoeRoutingBundle & bundle : context.plan.metadata.moe_routing_bundles()) {
        if (bundle.route_weights == route_weights->id &&
            bundle_matches_moe_routing(bundle, bundle.route_ids, token_count)) {
            known_route_weights = true;
            break;
        }
    }
    if (!known_route_weights) {
        return {};
    }

    match.routed_alternate = routed_alternate;
    return match;
}

static bool match_routed_ffn_gate_up_swiglu_q4k_f16_wmma_dispatch(const DispatchMatchContext & context,
                                                                  DispatchMatch &              dispatch_match) {
    const RoutedGateUpMatch match = match_routed_ffn_gate_up_swiglu(context);
    if (!match.matched()) {
        return false;
    }

    const ValueId f16_output(context.next_plan_value.value);
    const size_t  f16_output_bytes = f16_gate_up_output_size(match.token_count);
    dispatch_match.transients.push_back(
        { f16_output, kRoutedFfnF16GateUpOutputName, f16_output_bytes, kRoutedFfnPlanTransientAlignment });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.glu_output->id, f16_output, GGML_TYPE_F16, f16_output_bytes, kRoutedFfnF16GateUpOutputName },
            metadata_status)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRoutedGateUpSwiGLUQ4KF16WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.input_size",
                                               to_config_value(kRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.expert_count",
                                               to_config_value(kRoutedFfnExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.route_count",
                                               to_config_value(kRoutedFfnRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.output_size",
                                               to_config_value(kRoutedFfnExpertHiddenSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));

    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back(
        { match.routing_bundle->expert_table, 0, match.routing_bundle->expert_table_byte_count });
    dispatch.bindings.push_back(
        { match.routing_bundle->partition_table, 0, match.routing_bundle->partition_table_byte_count });
    dispatch.bindings.push_back({ match.gate_weight->id, 0, match.gate_weight->byte_count });
    dispatch.bindings.push_back({ match.up_weight->id, 0, match.up_weight->byte_count });
    dispatch.bindings.push_back({ f16_output, 0, f16_output_bytes });

    if (!append_covered_node(context, match.gate_node, dispatch_match) ||
        !append_covered_node(context, match.up_node, dispatch_match) ||
        !append_covered_node(context, match.glu_node, dispatch_match)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_decode_routed_ffn_gate_up_swiglu_q4k_q8_dispatch(const DispatchMatchContext & context,
                                                                   DispatchMatch &              dispatch_match) {
    const DecodeRoutedGateUpMatch match = match_decode_routed_ffn_gate_up_swiglu(context);
    if (!match.matched()) {
        return false;
    }

    const size_t q8_output_bytes =
        q8_1_x4_byte_count(match.token_count * kRoutedFfnRouteCount, kRoutedFfnExpertHiddenSize);
    const ValueId q8_output           = match.publish_q8 ? ValueId(context.next_plan_value.value) : ValueId();
    const ValueId completion_counters = match.publish_q8 ? ValueId(context.next_plan_value.value + 1) : ValueId();

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.publish_q8 ? kQwenRoutedGateUpSwiGLUQ4KQ8NextQ8Kernel :
                                                                    kQwenRoutedGateUpSwiGLUQ4KQ8Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("route_count", kRoutedFfnRouteCount);
    dispatch.kernel.integer_parameters.emplace("route_stride", match.route_stride);
    dispatch.kernel.integer_parameters.emplace("expert_count", kRoutedFfnExpertCount);
    dispatch.kernel.integer_parameters.emplace("output_size", kRoutedFfnExpertHiddenSize);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.input_size",
                                               to_config_value(kRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.expert_count",
                                               to_config_value(kRoutedFfnExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.route_count",
                                               to_config_value(kRoutedFfnRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.output_size",
                                               to_config_value(kRoutedFfnExpertHiddenSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));

    const size_t route_id_length = static_cast<size_t>(match.token_count * match.route_stride) * sizeof(int32_t);
    dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    dispatch.bindings.push_back({ match.route_ids->id, 0, route_id_length });
    dispatch.bindings.push_back({ match.gate_weight->id, 0, match.gate_weight->byte_count });
    dispatch.bindings.push_back({ match.up_weight->id, 0, match.up_weight->byte_count });
    dispatch.bindings.push_back({ match.glu_output->id, 0, match.glu_output->byte_count });
    if (match.publish_q8) {
        dispatch.bindings.push_back(
            { completion_counters, 0, gate_up_completion_counter_count(match.token_count) * sizeof(int32_t) });
        dispatch.bindings.push_back({ q8_output, 0, q8_output_bytes });
        dispatch_match.completion_counter_requests.push_back({
            completion_counters,
            "qwen.decode.moe.gate_up_completion_counters",
            gate_up_completion_counter_count(match.token_count),
        });
        dispatch_match.transients.push_back(
            { q8_output, kRoutedFfnQ8GateUpOutputName, q8_output_bytes, kRoutedFfnPlanTransientAlignment });
        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.glu_output->id, q8_output, GGML_TYPE_Q8_1, q8_output_bytes, kRoutedFfnQ8GateUpOutputName },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
    }

    if (!append_covered_node(context, match.gate_node, dispatch_match) ||
        !append_covered_node(context, match.up_node, dispatch_match) ||
        !append_covered_node(context, match.glu_node, dispatch_match)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

// qwen4exp sibling of the dispatch above. Always emits the "next_q8" kernel variant so the SwiGLU
// result is republished as a Q8_1-x4 alternate for match_decode_routed_ffn_down_qwen4exp() to bind.
static bool match_decode_routed_ffn_gate_up_swiglu_q4k_q8_qwen4exp_dispatch(const DispatchMatchContext & context,
                                                                            DispatchMatch & dispatch_match) {
    const DecodeRoutedGateUpMatch match = match_decode_routed_ffn_gate_up_swiglu_qwen4exp(context);
    if (!match.matched()) {
        return false;
    }

    const size_t q8_output_bytes =
        q8_1_x4_byte_count(match.token_count * kQwen4ExpRoutedFfnRouteCount, kQwen4ExpRoutedFfnExpertHiddenSize);
    const size_t  q8_input_bytes      = q8_1_x4_byte_count(match.token_count, kQwen4ExpRoutedFfnInputSize);
    const ValueId q8_input            = ValueId(context.next_plan_value.value);
    const ValueId q8_output           = ValueId(context.next_plan_value.value + 1);
    const ValueId completion_counters = ValueId(context.next_plan_value.value + 2);
    const uint32_t counter_count = gate_up_completion_counter_count(
        match.token_count, kQwen4ExpRoutedFfnRouteCount, kQwen4ExpRoutedFfnExpertHiddenSize);
    if (q8_input_bytes == 0 || q8_output_bytes == 0) {
        return false;
    }

    // Pass 1: pack the f32 hidden state into Q8_1-x4. The gate/up kernel only reads that layout, and
    // for qwen4exp nothing upstream publishes it (see kGgmlQuantizeQ8_1X4F32Kernel).
    const int64_t q8_group_count = match.token_count * ((kQwen4ExpRoutedFfnInputSize + 127) / 128);
    Dispatch      quantize;
    quantize.kernel = make_kernel_specialization(kGgmlQuantizeQ8_1X4F32Kernel);
    quantize.kernel.integer_parameters.emplace("token_count", match.token_count);
    quantize.kernel.integer_parameters.emplace("input_size", kQwen4ExpRoutedFfnInputSize);
    quantize.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                               to_config_value(q8_group_count));
    quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    quantize.bindings.push_back({ q8_input, 0, q8_input_bytes });

    // Pass 2: the fused gate + up + SwiGLU itself.
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRoutedGateUpSwiGLUQ4KQ8NextQ8Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("route_count", kQwen4ExpRoutedFfnRouteCount);
    dispatch.kernel.integer_parameters.emplace("route_stride", match.route_stride);
    dispatch.kernel.integer_parameters.emplace("expert_count", kQwen4ExpRoutedFfnExpertCount);
    dispatch.kernel.integer_parameters.emplace("output_size", kQwen4ExpRoutedFfnExpertHiddenSize);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.input_size",
                                               to_config_value(kQwen4ExpRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.expert_count",
                                               to_config_value(kQwen4ExpRoutedFfnExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.route_count",
                                               to_config_value(kQwen4ExpRoutedFfnRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.output_size",
                                               to_config_value(kQwen4ExpRoutedFfnExpertHiddenSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));

    // The route ids are a [route_count, 1] view into the 512-wide top-k selection, so nb[1] (and hence
    // route_stride) is 512 while the tensor itself is only route_count elements long. The kernel reads
    // route_ids[token * route_stride + r], so the extent it actually touches ends at the last token's
    // route_count-th id -- binding token_count * route_stride would run past the tensor.
    const size_t route_id_length =
        static_cast<size_t>((match.token_count - 1) * match.route_stride + kQwen4ExpRoutedFfnRouteCount) *
        sizeof(int32_t);
    dispatch.bindings.push_back({ q8_input, 0, q8_input_bytes });
    dispatch.bindings.push_back({ match.route_ids->id, 0, route_id_length });
    dispatch.bindings.push_back({ match.gate_weight->id, 0, match.gate_weight->byte_count });
    dispatch.bindings.push_back({ match.up_weight->id, 0, match.up_weight->byte_count });
    dispatch.bindings.push_back({ match.glu_output->id, 0, match.glu_output->byte_count });
    dispatch.bindings.push_back(
        { completion_counters, 0, counter_count * sizeof(int32_t) });
    dispatch.bindings.push_back({ q8_output, 0, q8_output_bytes });

    dispatch_match.completion_counter_requests.push_back({
        completion_counters,
        "qwen.decode.moe.gate_up_completion_counters",
        counter_count,
    });
    dispatch_match.transients.push_back(
        { q8_input, "qwen4exp.decode.moe.hidden_q8", q8_input_bytes, kRoutedFfnPlanTransientAlignment });
    dispatch_match.transients.push_back(
        { q8_output, kRoutedFfnQ8GateUpOutputName, q8_output_bytes, kRoutedFfnPlanTransientAlignment });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.glu_output->id, q8_output, GGML_TYPE_Q8_1, q8_output_bytes, kRoutedFfnQ8GateUpOutputName },
            metadata_status)) {
        dispatch_match.status.append(metadata_status);
        return false;
    }

    if (!append_covered_node(context, match.gate_node, dispatch_match) ||
        !append_covered_node(context, match.up_node, dispatch_match) ||
        !append_covered_node(context, match.glu_node, dispatch_match)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(quantize));
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static DecodeRoutedDownMatch match_decode_routed_ffn_down_next_q8(const DispatchMatchContext & context) {
    DecodeRoutedDownMatch match;
    const GraphNode *     root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * weight      = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_routed_ffn_down_weight(*weight) || !is_routed_ffn_projection_output(*input, 1) ||
        !is_routed_ffn_down_output(*root_output, 1) || route_ids->type != GGML_TYPE_I32 ||
        route_ids->nb[0] != sizeof(int32_t) || route_ids->nb[1] % sizeof(int32_t) != 0 ||
        !is_shape(*route_ids, kRoutedFfnRouteCount, 1, 1, 1)) {
        return {};
    }

    const int64_t                     route_stride = static_cast<int64_t>(route_ids->nb[1] / sizeof(int32_t));
    const GraphNode *                 glu_node     = producer_with_op(context.graph, input->id, GGML_OP_GLU);
    const GraphNode *                 gate_node    = glu_node == nullptr || glu_node->inputs.size() != 2 ?
                                                         nullptr :
                                                         producer_with_op(context.graph, glu_node->inputs[0], GGML_OP_MUL_MAT_ID);
    const Value *                     gate_input   = gate_node == nullptr || gate_node->inputs.size() != 3 ?
                                                         nullptr :
                                                         graph_value(context.graph, gate_node->inputs[1]);
    const CommandPlanAlternateValue * gate_input_q8 =
        gate_input == nullptr ? nullptr :
                                find_alternate_value(context.graph, context.plan, gate_input->id, GGML_TYPE_Q8_1,
                                                     q8_1_x4_byte_count(1, kRoutedFfnInputSize));
    if (gate_input_q8 == nullptr) {
        return {};
    }

    const Value *     route_weights =
        find_qwen_route_weights_for_route_ids(context.graph, route_ids->id, 1, kRoutedFfnRouteCount);
    const GraphNode * weighted =
        find_single_consumer_with_op_through_layout_aliases(context.graph, root_output->id, GGML_OP_MUL);
    WeightedReduceMatch reduce =
        match_routed_ffn_down_weighted_reduce_topology(context, weighted, root_output, route_weights);
    if (!reduce.topology_matched() || !reduce.next_rmsnorm.matched() ||
        !residual_input_is_safe_for_in_place(context, reduce)) {
        return {};
    }

    const bool                        input_is_q8 = weight->type == GGML_TYPE_Q4_K;
    const CommandPlanAlternateValue * input_alternate =
        input_is_q8 ? find_alternate_value(context.graph, context.plan, input->id, GGML_TYPE_Q8_1,
                                           q8_1_x4_byte_count(kRoutedFfnRouteCount, kRoutedFfnExpertHiddenSize)) :
                      nullptr;
    if (input_is_q8 && input_alternate == nullptr) {
        return {};
    }

    match.input_graph_value = input;
    match.input_alternate   = input_alternate;
    match.weight            = weight;
    match.output            = reduce.output;
    match.route_ids         = route_ids;
    match.reduce            = std::move(reduce);
    match.kernel =
        weight->type == GGML_TYPE_Q4_K ? kQwenRoutedDownQ4KQ8NextQ8Kernel : kQwenRoutedDownQ6KF32Wave64NextQ8Kernel;
    match.token_count  = 1;
    match.route_stride = route_stride;
    match.input_is_q8  = input_is_q8;
    return match;
}

static bool match_decode_routed_ffn_down_next_q8_dispatch(const DispatchMatchContext & context,
                                                          DispatchMatch &              dispatch_match) {
    const DecodeRoutedDownMatch match = match_decode_routed_ffn_down_next_q8(context);
    if (!match.matched()) {
        return false;
    }

    const ValueId completion_counter_value(context.next_plan_value.value);
    const ValueId q8_output(context.next_plan_value.value + 1);
    const size_t  q8_output_bytes = q8_1_x4_byte_count(match.token_count, kRoutedFfnInputSize);

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("input_size", kRoutedFfnExpertHiddenSize);
    dispatch.kernel.integer_parameters.emplace("route_count", kRoutedFfnRouteCount);
    dispatch.kernel.integer_parameters.emplace("route_id_stride", match.route_stride);
    dispatch.kernel.integer_parameters.emplace("expert_count", kRoutedFfnExpertCount);
    dispatch.kernel.integer_parameters.emplace("output_size", kRoutedFfnInputSize);
    add_routed_down_compile_parameters(dispatch, match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(kRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");

    const size_t route_id_length = static_cast<size_t>(match.token_count * match.route_stride) * sizeof(int32_t);
    if (match.input_is_q8) {
        dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    } else {
        dispatch.bindings.push_back({ match.input_graph_value->id, 0, match.input_graph_value->byte_count });
    }
    dispatch.bindings.push_back({ match.route_ids->id, 0, route_id_length });
    dispatch.bindings.push_back({ match.reduce.route_weights->id, 0, match.reduce.route_weights->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    dispatch.bindings.push_back(
        { match.reduce.next_rmsnorm.norm_weight->id, 0, match.reduce.next_rmsnorm.norm_weight->byte_count });
    dispatch.bindings.push_back({ completion_counter_value, 0, sizeof(int32_t) });
    dispatch.bindings.push_back({ q8_output, 0, q8_output_bytes });

    dispatch_match.value_aliases.push_back({ match.reduce.residual_input->id, match.output->id });
    dispatch_match.completion_counter_requests.push_back({
        completion_counter_value,
        "qwen.decode.moe.routed_down_completion_counter",
        1,
    });
    dispatch_match.transients.push_back(
        { q8_output, kRoutedFfnQ8HiddenOutputName, q8_output_bytes, kRoutedFfnPlanTransientAlignment });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.reduce.next_rmsnorm.output->id, q8_output, GGML_TYPE_Q8_1, q8_output_bytes,
              kRoutedFfnQ8HiddenOutputName },
            metadata_status)) {
        dispatch_match.status.append(metadata_status);
        return false;
    }

    if (!append_covered_node(context, context.root_node, dispatch_match) ||
        !append_covered_node(context, match.reduce.weighted_node, dispatch_match)) {
        return false;
    }
    for (const GraphNode * view : match.reduce.views) {
        if (!append_covered_node(context, view, dispatch_match)) {
            return false;
        }
    }
    for (const GraphNode * reduction : match.reduce.reductions) {
        if (!append_covered_node(context, reduction, dispatch_match)) {
            return false;
        }
    }
    if (!append_covered_node(context, match.reduce.residual, dispatch_match) ||
        !append_covered_node(context, match.reduce.next_rmsnorm.rms_node, dispatch_match) ||
        !append_covered_node(context, match.reduce.next_rmsnorm.mul_node, dispatch_match)) {
        return false;
    }

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

// qwen4exp decode-time routed-down matcher: mirrors match_decode_routed_ffn_down_next_q8 above, but
// targets the 3 new qwen4exp kernels (Q5_1/Q8_0/IQ4_NL) which consume a plain Q8_1-quantized
// activation (no fused rmsnorm/republish) and end at the plain residual-add. Deliberately does NOT
// replicate the gate_input_q8 precondition check (see report) -- dropped as a non-fundamental
// sanity check rather than threaded through for qwen4exp's narrower dispatch set.
static DecodeRoutedDownPlainMatch match_decode_routed_ffn_down_qwen4exp(const DispatchMatchContext & context) {
    DecodeRoutedDownPlainMatch match;
    const GraphNode *          root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * weight      = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr) {
        return {};
    }
    if (!is_routed_ffn_down_weight_qwen4exp(*weight)) {
        trace_moe_down_reject("down weight quant/shape");
        return {};
    }
    const int64_t tokens = input->ne[2];
    if (!moe_token_count_supported(tokens) || (tokens > 1 && weight->type == GGML_TYPE_Q5_1) ||
        !is_routed_ffn_projection_output_qwen4exp(*input, tokens)) {
        trace_moe_down_reject("projection input shape");
        return {};
    }
    if (!is_routed_ffn_down_output_qwen4exp(*root_output, tokens)) {
        trace_moe_down_reject("down output shape");
        return {};
    }
    if (route_ids->type != GGML_TYPE_I32 || route_ids->nb[0] != sizeof(int32_t) ||
        route_ids->nb[1] % sizeof(int32_t) != 0 ||
        route_ids->nb[1] < kQwen4ExpRoutedFfnRouteCount * sizeof(int32_t) ||
        route_ids->nb[1] > 512 * sizeof(int32_t) ||
        !is_shape(*route_ids, kQwen4ExpRoutedFfnRouteCount, tokens, 1, 1)) {
        trace_moe_down_reject("route id layout");
        return {};
    }

    const int64_t route_stride = static_cast<int64_t>(route_ids->nb[1] / sizeof(int32_t));

    const GraphNode * weighted =
        find_single_consumer_with_op_through_layout_aliases(context.graph, root_output->id, GGML_OP_MUL);
    if (weighted == nullptr) {
        trace_moe_down_reject("no single MUL consumer (weighted reduce)");
        return {};
    }
    const Value * route_weights = find_qwen_route_weights_for_route_ids(context.graph, route_ids->id, tokens,
                                                                        kQwen4ExpRoutedFfnRouteCount);
    if (route_weights == nullptr) {
        route_weights = find_route_weights_from_weighted_mul(context.graph, weighted, root_output, tokens,
                                                             kQwen4ExpRoutedFfnRouteCount);
    }
    if (route_weights == nullptr) {
        trace_moe_down_reject("route weights not found");
        return {};
    }
    // This fusion executes at MUL_MAT_ID, not at its later weighted MUL. Layout aliases
    // must be traced to their producer so a future router result is never read early.
    if (!input_ready_at_root(context, route_weights)) {
        trace_moe_down_reject("route weights producer is after down root");
        return {};
    }
    WeightedReduceMatch reduce =
        match_routed_ffn_down_weighted_reduce_topology_qwen4exp(context, weighted, root_output, route_weights);
    if (!reduce.topology_matched()) {
        trace_moe_down_reject("weighted reduce topology");
        return {};
    }
    if (!reduce.residual_missing && (!residual_input_is_safe_for_in_place(context, reduce) ||
                                     !input_ready_at_root(context, reduce.residual_input))) {
        // Shared SwiGLU can merge a later shared-FFN producer into this split.
        // Keep the routed reduce on GPU, but leave the final ADD at its original position.
        reduce = match_routed_ffn_down_weighted_reduce_topology_qwen4exp(
            context, weighted, root_output, route_weights, false);
        if (!reduce.topology_matched()) {
            trace_moe_down_reject("weighted reduce without residual");
            return {};
        }
    }

    // Unlike the qwen30b next_q8 kernel (which conditionally reads a raw or Q8_1-republished
    // activation depending on weight->type), all 3 new qwen4exp kernels always consume a Q8_1-x4
    // quantized activation (via ggml_q8_1_x4_block), so this is unconditionally true.
    const bool                        input_is_q8 = true;
    const CommandPlanAlternateValue * input_alternate =
        find_alternate_value(context.graph, context.plan, input->id, GGML_TYPE_Q8_1,
                             q8_1_x4_byte_count(tokens * kQwen4ExpRoutedFfnRouteCount,
                                               kQwen4ExpRoutedFfnExpertHiddenSize));
    if (input_alternate == nullptr && !iq_experts_enabled()) {
        trace_moe_down_reject("no Q8_1-x4 alternate for the GLU activation");
        return {};
    }

    match.input_graph_value = input;
    match.input_alternate   = input_alternate;
    match.weight            = weight;
    match.output            = reduce.output;
    match.route_ids         = route_ids;
    match.reduce            = std::move(reduce);
    match.kernel            = weight->type == GGML_TYPE_Q8_0 ? kQwenRoutedDownQ8_0Kernel :
                              weight->type == GGML_TYPE_Q5_1 ? kQwenRoutedDownQ5_1Kernel :
                                                               kQwenRoutedDownIQ4NLKernel;
    if (tokens > 1) {
        match.kernel = weight->type == GGML_TYPE_Q8_0 ? kSmallDownQ8Kernel : kSmallDownIQ4Kernel;
    }
    match.quant_suffix      = weight->type == GGML_TYPE_Q8_0 ? "q8_0" :
                              weight->type == GGML_TYPE_Q5_1 ? "q5_1" :
                                                               "iq4_nl";
    match.token_count  = tokens;
    match.route_stride = route_stride;
    match.input_is_q8  = input_is_q8;
    match.pack_input   = input_alternate == nullptr;
    return match;
}

static bool match_decode_routed_ffn_down_qwen4exp_dispatch(const DispatchMatchContext & context,
                                                            DispatchMatch &              dispatch_match) {
    const DecodeRoutedDownPlainMatch match = match_decode_routed_ffn_down_qwen4exp(context);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("input_size", kQwen4ExpRoutedFfnExpertHiddenSize);
    dispatch.kernel.integer_parameters.emplace("route_count", kQwen4ExpRoutedFfnRouteCount);
    dispatch.kernel.integer_parameters.emplace("route_id_stride", match.route_stride);
    dispatch.kernel.integer_parameters.emplace("expert_count", kQwen4ExpRoutedFfnExpertCount);
    dispatch.kernel.integer_parameters.emplace("output_size", kQwen4ExpRoutedFfnInputSize);
    if (match.token_count == 1) {
        add_qwen4exp_routed_down_compile_parameters(dispatch, match.token_count, match.quant_suffix);
    }

    // Same [route_count, 1]-view-into-512 stride/extent split as the gate/up dispatch above.
    const size_t route_id_length =
        static_cast<size_t>((match.token_count - 1) * match.route_stride + kQwen4ExpRoutedFfnRouteCount) *
        sizeof(int32_t);

    // Stage routing overlap and residual accumulation. A graph output may be External,
    // so aliasing it to the residual is illegal; even a legal alias can overwrite an
    // externally observable residual. Copy-in/out preserves both GGML storage identities.
    const bool routing_overlaps_output =
        ggml_storage_overlaps(match.output, match.output->byte_count, match.route_ids, route_id_length) ||
        ggml_storage_overlaps(match.output, match.output->byte_count, match.reduce.route_weights,
                              match.reduce.route_weights->byte_count);
    if (routing_overlaps_output && (match.output->type != GGML_TYPE_F32 || !match.output->contiguous)) {
        trace_moe_down_reject("routing arrays alias an output that cannot be staged");
        return false;
    }

    const bool stage_output = routing_overlaps_output || !match.reduce.residual_missing;
    const ValueId staged_output = context.next_plan_value;
    const ValueId output_binding = stage_output ? staged_output : match.output->id;
    if (stage_output) {
        dispatch_match.transients.push_back(
            { staged_output, kRoutedFfnStagedDownOutputName, match.output->byte_count,
              kRoutedFfnPlanTransientAlignment });
    }

    if (match.pack_input) {
        const ValueId q8_input(context.next_plan_value.value + 1);
        const size_t q8_bytes =
            q8_1_x4_byte_count(match.token_count * kQwen4ExpRoutedFfnRouteCount, kQwen4ExpRoutedFfnExpertHiddenSize);
        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kGgmlQuantizeQ8_1X4F32Kernel);
        quantize.kernel.integer_parameters.emplace("token_count", match.token_count * kQwen4ExpRoutedFfnRouteCount);
        quantize.kernel.integer_parameters.emplace("input_size", kQwen4ExpRoutedFfnExpertHiddenSize);
        quantize.kernel.compile_parameters.emplace(
            "ggml.quantize_q8_1_x4.group_capacity",
            to_config_value(match.token_count * kQwen4ExpRoutedFfnRouteCount *
                            ((kQwen4ExpRoutedFfnExpertHiddenSize + 127) / 128)));
        quantize.bindings.push_back({ match.input_graph_value->id, 0, match.input_graph_value->byte_count });
        quantize.bindings.push_back({ q8_input, 0, q8_bytes });
        dispatch_match.transients.push_back(
            { q8_input, "qwen4exp.decode.moe.down_input_q8", q8_bytes, kRoutedFfnPlanTransientAlignment });
        dispatch_match.dispatches.push_back(std::move(quantize));
        dispatch.bindings.push_back({ q8_input, 0, q8_bytes });
    } else {
        dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    }
    dispatch.bindings.push_back({ match.route_ids->id, 0, route_id_length });
    dispatch.bindings.push_back({ match.reduce.route_weights->id, 0, match.reduce.route_weights->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ output_binding, 0, match.output->byte_count });

    if (!append_covered_node(context, context.root_node, dispatch_match) ||
        !append_covered_node(context, match.reduce.weighted_node, dispatch_match)) {
        return false;
    }
    for (const GraphNode * view : match.reduce.views) {
        if (!append_covered_node(context, view, dispatch_match)) {
            return false;
        }
    }
    for (const GraphNode * reduction : match.reduce.reductions) {
        if (!append_covered_node(context, reduction, dispatch_match)) {
            return false;
        }
    }
    if (!match.reduce.residual_missing && !append_covered_node(context, match.reduce.residual, dispatch_match)) {
        return false;
    }

    // Without a residual to accumulate onto, the destination has to start at zero: the routed-down
    // kernels all end in an unconditional `output[c] = output[c] + routed_sum`, and `output` here is an
    // ordinary graph tensor whose buffer the ggml allocator may have recycled from an earlier node.
    if (match.reduce.residual_missing) {
        Dispatch zero;
        zero.kernel = make_kernel_specialization(kZeroF32Kernel);
        zero.kernel.integer_parameters.emplace("element_count", match.output->element_count);
        zero.bindings.push_back({ output_binding, 0, match.output->byte_count });
        dispatch_match.dispatches.push_back(std::move(zero));
    } else {
        Dispatch copy_in;
        copy_in.kernel = make_kernel_specialization(kCopyF32Kernel);
        copy_in.kernel.integer_parameters.emplace("element_count", match.output->element_count);
        copy_in.bindings.push_back({ match.reduce.residual_input->id, 0, match.output->byte_count });
        copy_in.bindings.push_back({ output_binding, 0, match.output->byte_count });
        dispatch_match.dispatches.push_back(std::move(copy_in));
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    if (stage_output) {
        Dispatch copy_out;
        copy_out.kernel = make_kernel_specialization(kCopyF32Kernel);
        copy_out.kernel.integer_parameters.emplace("element_count", match.output->element_count);
        copy_out.bindings.push_back({ staged_output, 0, match.output->byte_count });
        copy_out.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        dispatch_match.dispatches.push_back(std::move(copy_out));
    }
    return true;
}

static bool build_routed_ffn_down_grouped_dispatch(const DispatchMatchContext & context,
                                                   DispatchMatch &              dispatch_match,
                                                   KernelCatalogRef             expected_kernel) {
    const RoutedDownMatch match = match_routed_ffn_down_grouped(context);
    if (!match.matched() || match.kernel.id != expected_kernel.id) {
        return false;
    }

    const ValueId f16_output(context.next_plan_value.value);
    const size_t  f16_output_bytes = f16_routed_down_output_size(match.token_count);
    dispatch_match.transients.push_back(
        { f16_output, kRoutedFfnF16RoutedDownOutputName, f16_output_bytes, kRoutedFfnPlanTransientAlignment });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.output->id, f16_output, GGML_TYPE_F16, f16_output_bytes, kRoutedFfnF16RoutedDownOutputName },
            metadata_status)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    add_routed_down_compile_parameters(dispatch, match.token_count);
    dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    dispatch.bindings.push_back(
        { match.routing_bundle->expert_table, 0, match.routing_bundle->expert_table_byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ f16_output, 0, f16_output_bytes });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_routed_ffn_down_weighted_reduce_dispatch(const DispatchMatchContext & context,
                                                           DispatchMatch &              dispatch_match) {
    const WeightedReduceMatch match = match_routed_ffn_down_weighted_reduce(context);
    if (!match.matched()) {
        return false;
    }
    const bool use_next_rmsnorm = match.next_rmsnorm.matched() && match.output->kind == ValueKind::Transient &&
                                  residual_input_is_safe_for_in_place(context, match);

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(use_next_rmsnorm ? kQwenRoutedDownWeightedReduceNextRmsNormF32Kernel :
                                                                    kQwenRoutedDownWeightedReduceF16F32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    add_routed_down_compile_parameters(dispatch, match.token_count);
    if (use_next_rmsnorm) {
        dispatch_match.value_aliases.push_back({ match.residual_input->id, match.output->id });
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(kRoutedFfnInputSize));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
        dispatch.bindings.push_back({ match.route_weights->id, 0, match.route_weights->byte_count });
        dispatch.bindings.push_back({ match.routed_alternate->alternate_value, 0, match.routed_alternate->byte_count });
        dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        dispatch.bindings.push_back(
            { match.next_rmsnorm.norm_weight->id, 0, match.next_rmsnorm.norm_weight->byte_count });
        dispatch.bindings.push_back({ match.next_rmsnorm.output->id, 0, match.next_rmsnorm.output->byte_count });
    } else {
        dispatch.bindings.push_back({ match.route_weights->id, 0, match.route_weights->byte_count });
        dispatch.bindings.push_back({ match.routed_alternate->alternate_value, 0, match.routed_alternate->byte_count });
        dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    }

    if (!append_covered_node(context, match.weighted_node, dispatch_match)) {
        return false;
    }
    for (const GraphNode * view : match.views) {
        if (!append_covered_node(context, view, dispatch_match)) {
            return false;
        }
    }
    for (const GraphNode * reduction : match.reductions) {
        if (!append_covered_node(context, reduction, dispatch_match)) {
            return false;
        }
    }
    if (!append_covered_node(context, match.residual, dispatch_match)) {
        return false;
    }
    if (use_next_rmsnorm && (!append_covered_node(context, match.next_rmsnorm.rms_node, dispatch_match) ||
                             !append_covered_node(context, match.next_rmsnorm.mul_node, dispatch_match))) {
        return false;
    }

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_routed_ffn_down_q4k_f16_wmma_grouped_dispatch(const DispatchMatchContext & context,
                                                                DispatchMatch &              dispatch_match) {
    return build_routed_ffn_down_grouped_dispatch(context, dispatch_match, kQwenRoutedDownQ4KF16WmmaGroupedKernel);
}

static bool match_routed_ffn_down_q6k_f16_wmma_grouped_dispatch(const DispatchMatchContext & context,
                                                                DispatchMatch &              dispatch_match) {
    return build_routed_ffn_down_grouped_dispatch(context, dispatch_match, kQwenRoutedDownQ6KF16WmmaGroupedKernel);
}

}  // namespace

bool is_q4_routed_gate_up_kernel(uint64_t kernel_id) {
    return kernel_id == kQwenRoutedGateUpSwiGLUQ4KQ8Kernel.id ||
           kernel_id == kQwenRoutedGateUpSwiGLUQ4KQ8NextQ8Kernel.id;
}

size_t q4_routed_gate_up_id_byte_count(int64_t tokens, int64_t routes, int64_t stride, int64_t experts) {
    if (tokens < 1 || tokens > 512 || routes < 1 || routes > 16 ||
        stride < routes || stride > 512 || experts < 1 || experts > 512) {
        return 0;
    }
    return static_cast<size_t>((tokens - 1) * stride + routes) * sizeof(int32_t);
}

Q4RoutedIdValidation validate_q4_routed_gate_up_id_bytes(
        const void * data, size_t size, int64_t tokens, int64_t routes, int64_t stride, int64_t experts) {
    Q4RoutedIdValidation result;
    const size_t required = q4_routed_gate_up_id_byte_count(tokens, routes, stride, experts);
    if (data == nullptr || required == 0 || size < required) {
        return result;
    }
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t route = 0; route < routes; ++route) {
            int32_t id;
            std::memcpy(&id, bytes + (token * stride + route) * sizeof(id), sizeof(id));
            if (id < 0 || id >= experts) {
                result.invalid_id = id;
                result.token = token;
                result.route = route;
                return result;
            }
        }
    }
    result.valid = true; // Repeated experts are valid; padding is deliberately not examined.
    return result;
}

void register_routed_ffn_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "llm.routed_ffn.decode_iq_swiglu_qwen4exp",
        GGML_OP_GLU,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_decode_iq_swiglu_qwen4exp_dispatch,
    });
    registry.add({
        "llm.routed_ffn.decode_iq_expert_qwen4exp",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_decode_iq_expert_qwen4exp_dispatch,
    });
    registry.add({
        "llm.routed_ffn.decode_gate_up_swiglu_q4k_q8",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        1200,
        DispatchSource::Llm,
        match_decode_routed_ffn_gate_up_swiglu_q4k_q8_dispatch,
    });
    registry.add({
        "llm.routed_ffn.decode_gate_up_swiglu_q4k_q8_qwen4exp",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        1190,
        DispatchSource::Llm,
        match_decode_routed_ffn_gate_up_swiglu_q4k_q8_qwen4exp_dispatch,
    });
    registry.add({
        "llm.routed_ffn.decode_down_next_q8",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        1150,
        DispatchSource::Llm,
        match_decode_routed_ffn_down_next_q8_dispatch,
    });
    registry.add({
        "llm.routed_ffn.decode_down_qwen4exp",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        1140,
        DispatchSource::Llm,
        match_decode_routed_ffn_down_qwen4exp_dispatch,
    });
    registry.add({
        "llm.routed_ffn.gate_up_swiglu_q4k_f16_wmma",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Llm,
        match_routed_ffn_gate_up_swiglu_q4k_f16_wmma_dispatch,
    });
    registry.add({
        "llm.routed_ffn.down_q4k_f16_wmma_grouped",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        900,
        DispatchSource::Llm,
        match_routed_ffn_down_q4k_f16_wmma_grouped_dispatch,
    });
    registry.add({
        "llm.routed_ffn.down_q6k_f16_wmma_grouped",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        900,
        DispatchSource::Llm,
        match_routed_ffn_down_q6k_f16_wmma_grouped_dispatch,
    });
    registry.add({
        "llm.routed_ffn.down_weighted_reduce",
        GGML_OP_MUL,
        DispatchMatchKind::Fused,
        800,
        DispatchSource::Llm,
        match_routed_ffn_down_weighted_reduce_dispatch,
    });
}

}  // namespace ggml::hrx
