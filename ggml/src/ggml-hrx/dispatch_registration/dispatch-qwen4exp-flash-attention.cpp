#include "dispatch-qwen4exp-flash-attention.h"

#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

// qwen4exp (Qwen3.8-Flash-Next, HF architecture "qwen4exp") QSA decode-attention kernels. These
// live in the shared kernels/qwen_moe/manifest.json catalog file but are registered under their
// own "qwen4exp" family tag (see each export's "family" field there) -- do not change this to
// "qwen_moe", it will fail kernel catalog lookup.
static constexpr KernelCatalogRef kQwen4ExpAttentionDecodeKernel =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_attention_decode");
static constexpr KernelCatalogRef kQwen4ExpAttentionDecodeSplitPartialKernel =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_attention_decode_split_partial");
static constexpr KernelCatalogRef kQwen4ExpAttentionDecodeSplitWmmaPartialKernel =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_attention_decode_split_wmma_partial");
static constexpr KernelCatalogRef kQwen4ExpAttentionDecodeSplitReduceKernel =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_attention_decode_split_reduce");

// qwen4exp's QSA attention shape. Deliberately different from the qwen3_moe precedent's
// kQwenAttentionHeadSize=128 / kQwenQueryHeadCount=32 / kQwenKeyValueHeadCount=4 -- qwen4exp's GGUF
// metadata reports attention.key_length/attention.value_length=256 with 24 query heads and 2 KV
// heads (12 query heads per KV group). This was the shape bug that had to be fixed in the .loom
// kernels this session; do not reintroduce the qwen3_moe constants here.
static constexpr int64_t kQwen4ExpAttentionHeadSize = 256;
static constexpr int64_t kQwen4ExpQueryHeadCount    = 24;
static constexpr int64_t kQwen4ExpKeyValueHeadCount = 2;
static constexpr int64_t kQwen4ExpHiddenSize        = kQwen4ExpQueryHeadCount * kQwen4ExpAttentionHeadSize;
static constexpr int64_t kQwen4ExpDecodeRowCapacity  = 16;

// qwen38_attention_decode_split_wmma_partial's token_count launch parameter is hardware-range
// limited to [1,4] (see attention_decode_split.loom's `index.assume ... [range(...,1,4)]`).
static constexpr int64_t kQwen4ExpSplitWmmaMaxTokenCount = 4;

// qwen38.attention.cache_capacity's declared config.decl range is [1,262144]; reuse that as the
// upper bound for any key/value context length this dispatch will consider.
static constexpr int64_t kQwen4ExpMaxKeyValueTokenCount = 262144;

// Threshold between the single-workgroup-per-head qwen38_attention_decode kernel (always
// kQwen4ExpQueryHeadCount workgroups, regardless of context length) and the split+reduce pipeline
// (kQwen4ExpKeyValueHeadCount * decode_split_count parallel partial workgroups). There is no
// qwen4exp-specific profiling behind this exact cutover; 2048 mirrors
// is_supported_decode_key_value_token_count's cap in the qwen3_moe precedent
// (dispatch-qwen-flash-attention.cpp), which is at least an already-used round number in this
// codebase. A human should re-tune this once real gfx1151 decode-latency numbers exist.
static constexpr int64_t kQwen4ExpSimpleDecodeMaxKeyValueTokenCount = 2048;

// decode_split_count supplied via the qwen38.attention.decode_split_count config key whenever the
// split pipeline is dispatched. 16 splits x 2 KV heads = 32 parallel partial workgroups, a
// reasonable middle ground for gfx1151's CU count. Likewise not backed by hardware profiling --
// tunable, should be reviewed alongside the threshold above.
static constexpr int64_t kQwen4ExpDecodeSplitCount = 16;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-6f;
}

static bool is_supported_qwen4exp_key_value_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= kQwen4ExpMaxKeyValueTokenCount;
}

static bool is_qwen4exp_flash_decode_query_length(int64_t query_length) {
    return query_length >= 1 && query_length < kQwen4ExpDecodeRowCapacity;
}

static bool has_query_layout(const Value & value, int64_t query_head_count) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(query_head_count * kQwen4ExpAttentionHeadSize) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(kQwen4ExpAttentionHeadSize) * element_size);
}

static bool has_key_value_layout(const Value & value, int64_t key_value_head_count) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(key_value_head_count * kQwen4ExpAttentionHeadSize) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(kQwen4ExpAttentionHeadSize) * element_size);
}

static bool has_mask_layout(const Value & value, int64_t key_value_token_count) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(key_value_token_count) * element_size;
}

// Layout of build_attn_mha's raw (pre-reshape, pre-gate) flash-attention output:
// [head_size, head_count, token_count, 1].
static bool has_output_layout(const Value & value, int64_t query_head_count) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(kQwen4ExpAttentionHeadSize) * element_size &&
           value.nb[2] == static_cast<size_t>(query_head_count * kQwen4ExpAttentionHeadSize) * element_size;
}

// Layout shared by qwen4exp.cpp's sigmoided output gate (gate_sigmoid = ggml_sigmoid(cont_2d(view
// of Qcur_full))) and the gated MUL output that replaces build_attn_mha's raw flash-attention
// result once qwen4exp.cpp applies its per-layer output gate. Both are plain contiguous 2-D
// [hidden_size, token_count] f32 tensors -- unlike has_output_layout's pre-gate/pre-reshape
// [head_size, head_count, token_count] layout.
static bool has_flat_hidden_layout(const Value & value, int64_t hidden_size) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(hidden_size) * element_size;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static bool has_qwen4exp_flash_attention_params(const GraphNode & node) {
    const FlashAttnExtParams * params = op_params_as<FlashAttnExtParams>(node.params);
    if (params == nullptr) {
        return false;
    }
    const float expected_scale = 1.0f / std::sqrt(static_cast<float>(kQwen4ExpAttentionHeadSize));
    return nearly_equal(params->scale, expected_scale) && nearly_equal(params->max_bias, 0.0f) &&
           nearly_equal(params->logit_softcap, 0.0f) &&
           (params->prec == GGML_PREC_DEFAULT || params->prec == GGML_PREC_F32);
}

static ValueId match_value(const DispatchMatchContext & context, const DispatchMatch & dispatch_match,
                           int32_t offset) {
    return ValueId(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size()) +
                   static_cast<int32_t>(dispatch_match.completion_counter_requests.size()) + offset);
}

// Shared topology match for both qwen4exp decode dispatch destinations (qwen38_attention_decode
// and the split+reduce pipeline). Unlike the qwen3_moe precedent, qwen4exp.cpp's build_layer_attn
// wraps build_attn_mha's raw FLASH_ATTN_EXT result in a per-layer output gate:
//
//   cur (build_attn_mha's ggml_reshape_2d of the raw flash-attn result, a layout alias)
//   gate = ggml_sigmoid(ctx0, ggml_cont_2d(view of the wq projection's interleaved gate half))
//   cur  = ggml_mul(ctx0, cur, gate)
//
// qwen38_attention_decode / qwen38_attention_decode_split_reduce already expect their own `gate`
// binding to be this *sigmoided* value (see attention_decode.loom's `%gated = scalar.mulf
// %attention_value, %gate_value` -- no logistic call there; only qwen38_attention_query_prepare
// _decode applies scalar.logisticf, for a different, upstream purpose). So this match must cover
// FLASH_ATTN_EXT + the reshape + the MUL, and bind the MUL's *other* operand directly as `gate`
// without re-deriving or undoing the sigmoid -- whatever produced it (a CONT + SIGMOID chain, not
// a layout alias) is dispatched/covered separately.
struct Qwen4ExpFlashAttentionGateMatch {
    const Value *     query                 = nullptr;
    const Value *     key                   = nullptr;
    const Value *     value                 = nullptr;
    const Value *     mask                  = nullptr;
    const Value *     gate                  = nullptr;
    const Value *     output                = nullptr;
    const GraphNode * output_layout         = nullptr;
    const GraphNode * mul_node              = nullptr;
    int64_t           query_token_count     = 0;
    int64_t           key_value_token_count = 0;
    int64_t           query_head_count      = 0;
    int64_t           key_value_head_count  = 0;

    bool matched() const {
        return query != nullptr && key != nullptr && value != nullptr && mask != nullptr && gate != nullptr &&
               output != nullptr && output_layout != nullptr && mul_node != nullptr;
    }
};

static bool trace_qsa_attention_enabled() {
    const char * value = std::getenv("HRX_TRACE_QSA_ATTN");
    return value != nullptr && value[0] != '0';
}

// Reports the first predicate that rejected a FLASH_ATTN_EXT, once per distinct stage. The abort
// message for an unmatched node only names the node, which is not enough to tell which of the dozen
// shape/layout/topology conditions below actually failed.
static Qwen4ExpFlashAttentionGateMatch qsa_attention_decline(const char * stage) {
    if (trace_qsa_attention_enabled()) {
        static std::set<std::string> seen;
        if (seen.insert(stage).second) {
            std::fprintf(stderr, "HRX qsa_attn decline: %s\n", stage);
        }
    }
    return {};
}

static Qwen4ExpFlashAttentionGateMatch match_qwen4exp_flash_attention_gate(const Graph &     graph,
                                                                           const GraphNode * node) {
    Qwen4ExpFlashAttentionGateMatch match;
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4) {
        return match;
    }
    if (!has_qwen4exp_flash_attention_params(*node)) {
        return qsa_attention_decline("flash_attention_params");
    }

    const Value * query        = graph_value(graph, node->inputs[0]);
    const Value * key          = graph_value(graph, node->inputs[1]);
    const Value * value        = graph_value(graph, node->inputs[2]);
    const Value * mask         = graph_value(graph, node->inputs[3]);
    const Value * flash_output = graph_value(graph, node->output);
    if (query == nullptr || key == nullptr || value == nullptr || mask == nullptr || flash_output == nullptr) {
        return qsa_attention_decline("missing_operand");
    }
    if (query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F16 || value->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || flash_output->type != GGML_TYPE_F32) {
        return qsa_attention_decline("operand_types");
    }
    if (query->ne[0] != kQwen4ExpAttentionHeadSize || key->ne[0] != kQwen4ExpAttentionHeadSize ||
        value->ne[0] != kQwen4ExpAttentionHeadSize || flash_output->ne[0] != kQwen4ExpAttentionHeadSize) {
        return qsa_attention_decline("head_size");
    }
    if (query->ne[3] != 1 || key->ne[3] != 1 || value->ne[3] != 1 || flash_output->ne[3] != 1 || mask->ne[2] != 1 ||
        mask->ne[3] != 1) {
        return qsa_attention_decline("trailing_dims");
    }

    const int64_t query_token_count     = query->ne[1];
    const int64_t query_head_count      = query->ne[2];
    const int64_t key_value_capacity    = key->ne[1];
    const int64_t key_value_head_count  = key->ne[2];
    const int64_t key_value_token_count = mask->ne[0];
    if (!is_qwen4exp_flash_decode_query_length(query_token_count) ||
        !is_supported_qwen4exp_key_value_token_count(key_value_token_count) ||
        query_head_count != kQwen4ExpQueryHeadCount || key_value_head_count != kQwen4ExpKeyValueHeadCount ||
        key_value_capacity < key_value_token_count || value->ne[1] != key_value_capacity ||
        value->ne[2] != key_value_head_count || mask->ne[1] != query_token_count ||
        flash_output->ne[1] != query_head_count || flash_output->ne[2] != query_token_count) {
        return qsa_attention_decline("counts");
    }
    if (!has_query_layout(*query, query_head_count) || !has_key_value_layout(*key, key_value_head_count) ||
        !has_key_value_layout(*value, key_value_head_count) || !has_mask_layout(*mask, key_value_token_count) ||
        !has_output_layout(*flash_output, query_head_count)) {
        return qsa_attention_decline("operand_layout");
    }

    // qwen4exp.cpp's build_layer_attn is the only consumer of build_attn_mha's raw output (it is
    // immediately reshaped, then multiplied by the sigmoided gate). Require exactly one real
    // consumer so this fused dispatch can never silently drop some other reader of the raw,
    // pre-gate attention value.
    if (graph.index().consumers(flash_output->id).size() != 1) {
        return qsa_attention_decline("flash_output_consumer_count");
    }
    const GraphNode * output_layout = find_single_layout_alias_consumer(graph, flash_output->id);
    const GraphNode * mul_node =
        find_single_consumer_with_op_through_layout_aliases(graph, flash_output->id, GGML_OP_MUL);
    if (output_layout == nullptr || mul_node == nullptr || mul_node->inputs.size() != 2) {
        return qsa_attention_decline(output_layout == nullptr ? "no_output_layout_alias" : "no_gate_mul");
    }

    const ValueId reshaped_output = output_layout->output;
    ValueId       gate_id;
    if (mul_node->inputs[0] == reshaped_output) {
        gate_id = mul_node->inputs[1];
    } else if (mul_node->inputs[1] == reshaped_output) {
        gate_id = mul_node->inputs[0];
    } else {
        return qsa_attention_decline("gate_operand_position");
    }

    const Value * gate         = graph_value(graph, gate_id);
    const Value * gated_output = graph_value(graph, mul_node->output);
    if (gate == nullptr || gated_output == nullptr || gate->type != GGML_TYPE_F32 ||
        gated_output->type != GGML_TYPE_F32 || gate->ne[0] != kQwen4ExpHiddenSize ||
        gate->ne[1] != query_token_count || gate->ne[2] != 1 || gate->ne[3] != 1 ||
        gated_output->ne[0] != kQwen4ExpHiddenSize || gated_output->ne[1] != query_token_count ||
        gated_output->ne[2] != 1 || gated_output->ne[3] != 1 ||
        !has_flat_hidden_layout(*gate, kQwen4ExpHiddenSize) ||
        !has_flat_hidden_layout(*gated_output, kQwen4ExpHiddenSize)) {
        return qsa_attention_decline("gate_or_output_layout");
    }

    // Defensive check confirming the gate operand really is a sigmoid (qwen4exp.cpp's
    // gate_sigmoid = ggml_sigmoid(ctx0, gate)). Not load-bearing for the byte-level binding below
    // (whatever produced `gate` is dispatched/covered separately regardless), but cheap insurance
    // against silently binding an unrelated second MUL operand.
    const GraphNode *   gate_producer     = graph.index().producer(gate_id);
    const UnaryParams * gate_unary_params = gate_producer != nullptr ? op_params_as<UnaryParams>(gate_producer->params)
                                                                     : nullptr;
    if (gate_producer == nullptr || gate_producer->op != GGML_OP_UNARY || gate_unary_params == nullptr ||
        gate_unary_params->op != GGML_UNARY_OP_SIGMOID) {
        return qsa_attention_decline("gate_not_sigmoid");
    }

    match.query                 = query;
    match.key                   = key;
    match.value                 = value;
    match.mask                  = mask;
    match.gate                  = gate;
    match.output                = gated_output;
    match.output_layout         = output_layout;
    match.mul_node              = mul_node;
    match.query_token_count     = query_token_count;
    match.key_value_token_count = key_value_token_count;
    match.query_head_count      = query_head_count;
    match.key_value_head_count  = key_value_head_count;
    return match;
}

static bool cover_qwen4exp_gate_match(const DispatchMatchContext &            context,
                                      const Qwen4ExpFlashAttentionGateMatch & match, DispatchMatch & dispatch_match) {
    dispatch_match.covered_nodes.push_back(context.root_index);
    if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.output_layout,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.mul_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    return true;
}

// Synthesizes a small read-only i32 buffer used as the `control` binding (the QSA decode kernels'
// runtime "position" scalar(s)), writing the same `position` value into each of `element_count`
// slots.
static void push_position_control(DispatchMatch & dispatch_match, ValueId control_value, const char * name,
                                  int32_t position, size_t element_count) {
    std::vector<uint8_t> data(element_count * sizeof(int32_t));
    for (size_t i = 0; i < element_count; ++i) {
        std::memcpy(data.data() + i * sizeof(int32_t), &position, sizeof(int32_t));
    }
    dispatch_match.transients.push_back({ control_value, name, data.size(), 256 });
    dispatch_match.constant_initializations.push_back({ control_value, name, 0, std::move(data) });
}

}  // namespace

// Smaller kv-cache/context decode path: qwen38_attention_decode uses a fixed
// kQwen4ExpQueryHeadCount workgroups regardless of context length, one dispatch per query-token
// row (mirroring the qwen3_moe precedent's per-row decode-split loop). `control` (position) is
// conservatively shared across all rows: qwen38_attention_decode has no token_ordinal to add to
// it, and every row's own bound `mask` slice independently encodes that row's true causal
// visibility via -inf regardless of how generous this shared value makes the kernel's internal
// context_count loop bound.
static bool match_qwen4exp_attention_decode_dispatch(const DispatchMatchContext & context,
                                                     DispatchMatch &              dispatch_match) {
    const Qwen4ExpFlashAttentionGateMatch match =
        match_qwen4exp_flash_attention_gate(context.graph, context.root_node);
    if (!match.matched() || match.key_value_token_count > kQwen4ExpSimpleDecodeMaxKeyValueTokenCount) {
        return false;
    }

    const size_t hidden_row_bytes = static_cast<size_t>(kQwen4ExpHiddenSize) * sizeof(float);
    const size_t mask_row_bytes   = static_cast<size_t>(match.key_value_token_count) * sizeof(ggml_fp16_t);

    const ValueId control_value = match_value(context, dispatch_match, 0);
    push_position_control(dispatch_match, control_value, "qwen4exp.decode.flash_attention.control",
                          static_cast<int32_t>(match.key_value_token_count - 1), 1);

    for (int64_t row = 0; row < match.query_token_count; ++row) {
        Dispatch dispatch;
        dispatch.kernel = make_kernel_specialization(kQwen4ExpAttentionDecodeKernel);
        dispatch.kernel.compile_parameters.emplace("qwen38.attention.cache_capacity",
                                                   to_config_value(match.key_value_token_count));
        dispatch.bindings.push_back({ control_value, 0, sizeof(int32_t) });
        dispatch.bindings.push_back(
            { match.query->id, static_cast<size_t>(row) * match.query->nb[1], hidden_row_bytes });
        dispatch.bindings.push_back(
            { match.gate->id, static_cast<size_t>(row) * match.gate->nb[1], hidden_row_bytes });
        dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
        dispatch.bindings.push_back({ match.value->id, 0, match.value->byte_count });
        dispatch.bindings.push_back({ match.mask->id, static_cast<size_t>(row) * match.mask->nb[1], mask_row_bytes });
        dispatch.bindings.push_back(
            { match.output->id, static_cast<size_t>(row) * match.output->nb[1], hidden_row_bytes });
        dispatch_match.dispatches.push_back(std::move(dispatch));
    }

    return cover_qwen4exp_gate_match(context, match, dispatch_match);
}

// Larger kv-cache/context decode path: the 3-kernel split+reduce pipeline, mirroring the
// qwen3_moe precedent's decode-split-next-q8 partial+reduce orchestration. query_token_count == 1
// uses the scalar qwen38_attention_decode_split_partial kernel (avoids WMMA's fixed tiling
// overhead when there is only one token to amortize it over); 2-4 tokens use the WMMA-tiled
// qwen38_attention_decode_split_wmma_partial kernel, looping one dispatch per token_ordinal (each
// launch's own workgroup count is a fixed kQwen4ExpKeyValueHeadCount * decode_split_count,
// independent of token_count -- see attention_decode_split.loom's
// `%workgroup_count = index.mul %key_head_count, %split_count`, so batching multiple tokens does
// *not* collapse into a single dispatch). query_token_count in [5, kQwen4ExpDecodeRowCapacity)
// is an accepted gap: not matched here, falls back to the generic backend.
static bool match_qwen4exp_attention_decode_split_dispatch(const DispatchMatchContext & context,
                                                           DispatchMatch &              dispatch_match) {
    const Qwen4ExpFlashAttentionGateMatch match =
        match_qwen4exp_flash_attention_gate(context.graph, context.root_node);
    if (!match.matched() || match.key_value_token_count <= kQwen4ExpSimpleDecodeMaxKeyValueTokenCount ||
        match.query_token_count > kQwen4ExpSplitWmmaMaxTokenCount) {
        return false;
    }

    const size_t mask_row_bytes = static_cast<size_t>(match.key_value_token_count) * sizeof(ggml_fp16_t);
    // partials layout: [token_count]x2(key_value_head_count)x[split_count]x12x258 f32 scalars (see
    // attention_decode_split.loom's %partial_view declaration) -- one shared buffer per match,
    // written by the partial kernel(s) and consumed by the reduce kernel(s) below.
    const size_t partial_count =
        static_cast<size_t>(match.query_token_count) * 2 * static_cast<size_t>(kQwen4ExpDecodeSplitCount) * 12 * 258;
    const size_t partial_bytes = partial_count * sizeof(float);

    const ValueId control_value  = match_value(context, dispatch_match, 0);
    const ValueId partials_value = match_value(context, dispatch_match, 1);
    dispatch_match.transients.push_back(
        { partials_value, "qwen4exp.decode.flash_attention.partials", partial_bytes, 256 });

    const std::string cache_capacity_config     = to_config_value(match.key_value_token_count);
    const std::string decode_split_count_config = to_config_value(kQwen4ExpDecodeSplitCount);

    if (match.query_token_count == 1) {
        push_position_control(dispatch_match, control_value, "qwen4exp.decode.flash_attention.control",
                              static_cast<int32_t>(match.key_value_token_count - 1), 1);

        Dispatch partial_dispatch;
        partial_dispatch.kernel = make_kernel_specialization(kQwen4ExpAttentionDecodeSplitPartialKernel);
        partial_dispatch.kernel.compile_parameters.emplace("qwen38.attention.cache_capacity", cache_capacity_config);
        partial_dispatch.kernel.compile_parameters.emplace("qwen38.attention.decode_split_count",
                                                           decode_split_count_config);
        partial_dispatch.bindings.push_back({ control_value, 0, sizeof(int32_t) });
        partial_dispatch.bindings.push_back({ match.query->id, 0, match.query->byte_count });
        partial_dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
        partial_dispatch.bindings.push_back({ match.value->id, 0, match.value->byte_count });
        partial_dispatch.bindings.push_back({ match.mask->id, 0, mask_row_bytes });
        partial_dispatch.bindings.push_back({ partials_value, 0, partial_bytes });
        dispatch_match.dispatches.push_back(std::move(partial_dispatch));

        Dispatch reduce_dispatch;
        reduce_dispatch.kernel = make_kernel_specialization(kQwen4ExpAttentionDecodeSplitReduceKernel);
        reduce_dispatch.kernel.integer_parameters.emplace("token_count", 1);
        reduce_dispatch.kernel.integer_parameters.emplace("token_ordinal", 0);
        reduce_dispatch.kernel.compile_parameters.emplace("qwen38.attention.decode_split_count",
                                                          decode_split_count_config);
        reduce_dispatch.bindings.push_back({ match.gate->id, 0, match.gate->byte_count });
        reduce_dispatch.bindings.push_back({ partials_value, 0, partial_bytes });
        reduce_dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        dispatch_match.dispatches.push_back(std::move(reduce_dispatch));
    } else {
        // qwen38_attention_decode_split_wmma_partial computes `position = control[
        // control_position_index] + token_ordinal` (attention_decode_split.loom), so -- unlike the
        // single-token kernels above -- a shared conservative constant (key_value_token_count - 1)
        // would push position out of its declared `lt(position, cache_capacity)` bound for every
        // row past the first. Row `t`'s true position in a chronologically-ordered decode/MTP-draft
        // batch is key_value_token_count - query_token_count + t, so position_base is exactly that
        // expression evaluated at t=0; control_position_index is always 0 here.
        const int32_t position_base = static_cast<int32_t>(match.key_value_token_count - match.query_token_count);
        push_position_control(dispatch_match, control_value, "qwen4exp.decode.flash_attention.control", position_base,
                              2);

        for (int64_t row = 0; row < match.query_token_count; ++row) {
            Dispatch partial_dispatch;
            partial_dispatch.kernel = make_kernel_specialization(kQwen4ExpAttentionDecodeSplitWmmaPartialKernel);
            partial_dispatch.kernel.integer_parameters.emplace("token_count", match.query_token_count);
            partial_dispatch.kernel.integer_parameters.emplace("token_ordinal", row);
            partial_dispatch.kernel.integer_parameters.emplace("control_position_index", 0);
            partial_dispatch.kernel.compile_parameters.emplace("qwen38.attention.cache_capacity",
                                                               cache_capacity_config);
            partial_dispatch.kernel.compile_parameters.emplace("qwen38.attention.decode_split_count",
                                                               decode_split_count_config);
            partial_dispatch.bindings.push_back({ control_value, 0, 2 * sizeof(int32_t) });
            partial_dispatch.bindings.push_back({ match.query->id, 0, match.query->byte_count });
            partial_dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
            partial_dispatch.bindings.push_back({ match.value->id, 0, match.value->byte_count });
            partial_dispatch.bindings.push_back(
                { match.mask->id, static_cast<size_t>(row) * match.mask->nb[1], mask_row_bytes });
            partial_dispatch.bindings.push_back({ partials_value, 0, partial_bytes });
            dispatch_match.dispatches.push_back(std::move(partial_dispatch));

            Dispatch reduce_dispatch;
            reduce_dispatch.kernel = make_kernel_specialization(kQwen4ExpAttentionDecodeSplitReduceKernel);
            reduce_dispatch.kernel.integer_parameters.emplace("token_count", match.query_token_count);
            reduce_dispatch.kernel.integer_parameters.emplace("token_ordinal", row);
            reduce_dispatch.kernel.compile_parameters.emplace("qwen38.attention.decode_split_count",
                                                              decode_split_count_config);
            reduce_dispatch.bindings.push_back({ match.gate->id, 0, match.gate->byte_count });
            reduce_dispatch.bindings.push_back({ partials_value, 0, partial_bytes });
            reduce_dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
            dispatch_match.dispatches.push_back(std::move(reduce_dispatch));
        }
    }

    return cover_qwen4exp_gate_match(context, match, dispatch_match);
}

void register_qwen4exp_flash_attention_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen4exp.flash_attention_decode_split",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        200,
        DispatchSource::Qwen,
        match_qwen4exp_attention_decode_split_dispatch,
    });
    registry.add({
        "qwen4exp.flash_attention_decode",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen4exp_attention_decode_dispatch,
    });
}

}  // namespace ggml::hrx
