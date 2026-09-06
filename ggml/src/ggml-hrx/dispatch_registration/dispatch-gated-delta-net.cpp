#include "dispatch-gated-delta-net.h"

#include "dispatch-llm-profiles.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

// Kernel catalog references (family "qwen4exp", see kernel-corpus/kernels/qwen_moe/manifest.json).
static constexpr KernelCatalogRef kQwenGdnConvPrepareDecodeKernel =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_gdn_conv_prepare_decode");
static constexpr KernelCatalogRef kQwenGdnRecurrentDecodeKernel =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_gdn_recurrent_decode");
static constexpr KernelCatalogRef kQwenGdnNormGateDecodeKernel =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_gdn_norm_gate_decode");

// qwen4exp GDN decode-only (K=1, single token, single sequence) shape profile. Field names mirror
// qwen4exp.cpp's build_layer_attn_linear()/build_conv_state_at(), where head_k_dim = head_v_dim =
// hparams.ssm_d_state, num_k_heads = hparams.ssm_n_group, num_v_heads = hparams.ssm_dt_rank. Chunked
// prefill (K>1, n_seq_tokens>1) is intentionally left CPU-fallback -- see per-matcher comments below.
static constexpr int64_t kGdnHeadDim        = 128;
static constexpr int64_t kGdnKeyHeadCount   = 16;
static constexpr int64_t kGdnValueHeadCount = 48;
// conv_channels = head_k_dim*num_k_heads*2 (q,k) + head_v_dim*num_v_heads (v) = 4096 + 6144 = 10240.
static constexpr int64_t kGdnConvChannels    = kGdnHeadDim * kGdnKeyHeadCount * 2 + kGdnHeadDim * kGdnValueHeadCount;
static constexpr int64_t kGdnConvKernelSize  = 4;  // hparams.ssm_d_conv
static constexpr int64_t kGdnConvHistoryLen  = kGdnConvKernelSize - 1;
static constexpr size_t  kGdnQkInverseNormByteCount = static_cast<size_t>(2 * kGdnKeyHeadCount) * sizeof(float);
static constexpr size_t  kGdnRawOutputByteCount =
    static_cast<size_t>(kGdnValueHeadCount * kGdnHeadDim) * sizeof(float);

static constexpr float kQwen4ExpRmsNormEpsilon = kQwen4ExpMoeDispatchProfile.rms_norm_epsilon;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool is_qwen4exp_rms_norm_epsilon(float eps) {
    return std::fabs(eps - kQwen4ExpRmsNormEpsilon) <= 1.0e-12f;
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

// ---------------------------------------------------------------------------------------------
// qwen4exp.gdn_conv_prepare_decode
//
// Matches: CONCAT(conv_history, TRANSPOSE(qkv_mixed)) -> SSM_CONV -> UNARY(SILU)
//          -> { VIEW(q), VIEW(k), VIEW(v) } -> { L2_NORM(q), L2_NORM(k) }
//
// This is qwen4exp.cpp's build_conv_state_at() (the CONCAT) feeding build_layer_attn_linear()'s
// conv+silu+per-head L2-norm prelude. The root is CONCAT rather than SSM_CONV (as the task brief's
// op list suggested) because the HRX dispatch scheduler only ever covers a match's root node plus
// nodes *downstream* of it in one topological pass -- rooting at SSM_CONV would leave CONCAT
// permanently unmatched (SSM_CONV's own producer), since nothing else claims it (CONCAT is not a
// layout-alias op eligible for the generic auto-elide fallback). TRANSPOSE, by contrast, *is* a
// pure layout-alias op (graph.cpp is_layout_alias_op()) and is left uncovered here deliberately --
// it auto-elides on its own turn, so this match binds straight through its input instead of its
// output (see qkv_input below); n_seq_tokens==1 makes that transpose a byte-identical no-op anyway.
//
// The kernel never materializes L2_NORM(q)/L2_NORM(k)'s literal "normalized q/k" output: normalization
// is deferred and folded into a small `qk_inverse_norm` reciprocal-scale buffer applied at point of use
// inside qwen38_gdn_recurrent_decode. So L2_NORM(q)/L2_NORM(k) must be swept into *this* match (they
// never get a written value of their own) and `qkv_silu` is bound to the shared, un-sliced SILU output
// -- the kernel does its own q/k/v slicing internally via fixed channel-offset constants.
struct ConvPrepareMatch {
    const GraphNode * concat_node   = nullptr;
    const GraphNode * ssm_conv_node = nullptr;
    const GraphNode * silu_node     = nullptr;
    const GraphNode * view_q_node   = nullptr;
    const GraphNode * view_k_node   = nullptr;
    const GraphNode * view_v_node   = nullptr;
    const GraphNode * l2norm_q_node = nullptr;
    const GraphNode * l2norm_k_node = nullptr;
    const Value *      qkv_input    = nullptr;  // pre-transpose conv input (qkv_mixed), [10240,1,1,1]
    const Value *      conv_weight  = nullptr;  // ssm_conv1d weight, [4,10240,1,1]
    const Value *      conv_history = nullptr;  // persistent conv state slice, [3,10240,1,1]
    const Value *      qkv_silu     = nullptr;  // shared SILU output, [10240,1,1,1]

    bool matched() const {
        return concat_node != nullptr && ssm_conv_node != nullptr && silu_node != nullptr &&
               view_q_node != nullptr && view_k_node != nullptr && view_v_node != nullptr &&
               l2norm_q_node != nullptr && l2norm_k_node != nullptr && qkv_input != nullptr &&
               conv_weight != nullptr && conv_history != nullptr && qkv_silu != nullptr;
    }
};

static ConvPrepareMatch match_qwen4exp_gdn_conv_prepare(const Graph & graph, const GraphNode * node) {
    ConvPrepareMatch match;
    if (node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const Value * conv_history = graph_value(graph, node->inputs[0]);
    if (conv_history == nullptr || !is_shape(*conv_history, kGdnConvHistoryLen, kGdnConvChannels, 1, 1)) {
        return {};
    }

    // node->inputs[1] is ggml_transpose(qkv_mixed); bind through to the transpose's own input instead
    // of its output (see comment above) -- valid because n_seq_tokens==1 makes the transpose a byte-
    // identical metadata no-op. qkv_input's own shape (ne[1]==1) is this match's decode-only gate.
    const Value *      transpose_output = graph_value(graph, node->inputs[1]);
    const GraphNode * transpose_node   = transpose_output == nullptr ? nullptr :
                                                                       graph.index().producer(transpose_output->id);
    if (transpose_node == nullptr || transpose_node->op != GGML_OP_TRANSPOSE || transpose_node->inputs.size() != 1) {
        return {};
    }
    const Value * qkv_input = graph_value(graph, transpose_node->inputs[0]);
    if (qkv_input == nullptr || !is_shape(*qkv_input, kGdnConvChannels, 1, 1, 1)) {
        return {};  // n_seq_tokens > 1 (prefill) shows up here as ne[1] > 1; decode-only gate.
    }

    const Value * concat_output = graph_value(graph, node->output);
    if (concat_output == nullptr || !is_shape(*concat_output, kGdnConvKernelSize, kGdnConvChannels, 1, 1)) {
        return {};
    }

    const std::vector<const GraphNode *> & concat_consumers = graph.index().consumers(node->output);
    if (concat_consumers.size() != 1 || concat_consumers.front() == nullptr) {
        return {};
    }
    const GraphNode * ssm_conv_node = concat_consumers.front();
    if (ssm_conv_node->op != GGML_OP_SSM_CONV || ssm_conv_node->inputs.size() != 2 ||
        ssm_conv_node->inputs[0] != node->output) {
        return {};
    }
    const Value * conv_weight     = graph_value(graph, ssm_conv_node->inputs[1]);
    const Value * ssm_conv_output = graph_value(graph, ssm_conv_node->output);
    if (conv_weight == nullptr || ssm_conv_output == nullptr ||
        !is_shape(*conv_weight, kGdnConvKernelSize, kGdnConvChannels, 1, 1) ||
        !is_shape(*ssm_conv_output, kGdnConvChannels, 1, 1, 1)) {
        return {};
    }

    const std::vector<const GraphNode *> & ssm_conv_consumers = graph.index().consumers(ssm_conv_node->output);
    if (ssm_conv_consumers.size() != 1 || ssm_conv_consumers.front() == nullptr) {
        return {};
    }
    const GraphNode * silu_node = ssm_conv_consumers.front();
    if (silu_node->op != GGML_OP_UNARY || silu_node->inputs.size() != 1) {
        return {};
    }
    const UnaryParams * silu_params = op_params_as<UnaryParams>(silu_node->params);
    if (silu_params == nullptr || silu_params->op != GGML_UNARY_OP_SILU) {
        return {};
    }
    const Value * qkv_silu = graph_value(graph, silu_node->output);
    if (qkv_silu == nullptr || !is_shape(*qkv_silu, kGdnConvChannels, 1, 1, 1)) {
        return {};
    }

    // SILU's output feeds q/k/v as three VIEW slices; q and k are further L2-normalized (per-head,
    // head_k_dim=128 over num_k_heads=16), v is not (it stays raw post-SiLU, per the kernel's own
    // internal slicing). Identify the pattern structurally (shape + which views feed an L2_NORM)
    // rather than by hardcoding byte offsets, since the kernel's own binding is to the whole,
    // unsliced qkv_silu value regardless of exactly how ggml chose to lay the three views out.
    const std::vector<const GraphNode *> views =
        layout_alias_consumers_with_op(graph, silu_node->output, GGML_OP_VIEW);
    if (views.size() != 3) {
        return {};
    }
    const std::vector<const GraphNode *> l2norms =
        consumers_with_op_through_layout_aliases(graph, silu_node->output, GGML_OP_L2_NORM);
    if (l2norms.size() != 2) {
        return {};
    }

    const GraphNode * l2norm_q = l2norms[0];
    const GraphNode * l2norm_k = l2norms[1];
    const GraphNode * view_q   = nullptr;
    const GraphNode * view_k   = nullptr;
    const GraphNode * view_v   = nullptr;
    for (const GraphNode * l2norm : { l2norm_q, l2norm_k }) {
        if (l2norm->op != GGML_OP_L2_NORM || l2norm->inputs.size() != 1) {
            return {};
        }
        const L2NormParams * params = op_params_as<L2NormParams>(l2norm->params);
        if (params == nullptr || !is_qwen4exp_rms_norm_epsilon(params->eps)) {
            return {};
        }
        const Value * l2norm_output = graph_value(graph, l2norm->output);
        if (l2norm_output == nullptr || !is_shape(*l2norm_output, kGdnHeadDim, kGdnKeyHeadCount, 1, 1)) {
            return {};
        }
        const GraphNode * source_view = graph.index().producer(l2norm->inputs[0]);
        if (source_view == nullptr || source_view->op != GGML_OP_VIEW) {
            return {};
        }
        if (l2norm == l2norm_q) {
            view_q = source_view;
        } else {
            view_k = source_view;
        }
    }
    for (const GraphNode * view : views) {
        if (view != view_q && view != view_k) {
            if (view_v != nullptr) {
                return {};  // more than one leftover view -- not the expected 2-normalized + 1-raw pattern
            }
            view_v = view;
        }
    }
    if (view_q == nullptr || view_k == nullptr || view_v == nullptr) {
        return {};
    }
    const Value * view_v_output = graph_value(graph, view_v->output);
    if (view_v_output == nullptr || !is_shape(*view_v_output, kGdnHeadDim, kGdnValueHeadCount, 1, 1)) {
        return {};
    }

    match.concat_node   = node;
    match.ssm_conv_node = ssm_conv_node;
    match.silu_node     = silu_node;
    match.view_q_node   = view_q;
    match.view_k_node   = view_k;
    match.view_v_node   = view_v;
    match.l2norm_q_node = l2norm_q;
    match.l2norm_k_node = l2norm_k;
    match.qkv_input     = qkv_input;
    match.conv_weight   = conv_weight;
    match.conv_history  = conv_history;
    match.qkv_silu      = qkv_silu;
    return match;
}

// ---------------------------------------------------------------------------------------------
// qwen4exp.gdn_norm_gate_decode
//
// Matches: RMS_NORM -> MUL(norm_weight) -> SIGMOID(z) -> MUL(gated)
//
// This is qwen4exp.cpp's build_norm_gated(): build_norm(..., LLM_NORM_RMS, ...) emits RMS_NORM+MUL
// internally, then a separate `gated = ggml_sigmoid(ctx0, gate)` branch (SIGMOID, not SILU -- per the
// comment at that call site), then `ggml_mul(normalized, gated)`. Mirrors dispatch-rmsnorm.cpp's
// RmsNormMatch pattern, extended two more nodes for the gate.
struct NormGateMatch {
    const GraphNode * rms_node      = nullptr;
    const GraphNode * mul_node      = nullptr;
    const GraphNode * sigmoid_node  = nullptr;
    const GraphNode * gated_mul_node = nullptr;
    const Value *      raw_output   = nullptr;  // RMS_NORM's input
    const Value *      norm_weight  = nullptr;
    const Value *      z            = nullptr;  // SIGMOID's input
    const Value *      output       = nullptr;  // gated MUL's output

    bool matched() const {
        return rms_node != nullptr && mul_node != nullptr && sigmoid_node != nullptr &&
               gated_mul_node != nullptr && raw_output != nullptr && norm_weight != nullptr && z != nullptr &&
               output != nullptr;
    }
};

static NormGateMatch match_qwen4exp_gdn_norm_gate(const Graph & graph, const GraphNode * node) {
    NormGateMatch match;
    if (node == nullptr || node->op != GGML_OP_RMS_NORM || node->inputs.size() != 1 || !graph.has_index()) {
        return match;
    }

    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(node->params);
    if (rms_params == nullptr || !is_qwen4exp_rms_norm_epsilon(rms_params->eps)) {
        return {};
    }
    const Value * raw_output = graph_value(graph, node->inputs[0]);
    const Value * rms_output = graph_value(graph, node->output);
    if (raw_output == nullptr || rms_output == nullptr ||
        !is_shape(*raw_output, kGdnHeadDim, kGdnValueHeadCount, 1, 1) ||
        !is_shape(*rms_output, kGdnHeadDim, kGdnValueHeadCount, 1, 1)) {
        return {};
    }

    const std::vector<const GraphNode *> & rms_consumers = graph.index().consumers(node->output);
    if (rms_consumers.size() != 1 || rms_consumers.front() == nullptr) {
        return {};
    }
    const GraphNode * mul_node = rms_consumers.front();
    if (mul_node->op != GGML_OP_MUL || mul_node->inputs.size() != 2) {
        return {};
    }
    const Value * norm_weight = nullptr;
    for (ValueId input : mul_node->inputs) {
        if (input != node->output) {
            norm_weight = graph_value(graph, input);
        }
    }
    const Value * mul_output = graph_value(graph, mul_node->output);
    if (norm_weight == nullptr || mul_output == nullptr || !is_shape(*norm_weight, kGdnHeadDim, 1, 1, 1) ||
        !is_shape(*mul_output, kGdnHeadDim, kGdnValueHeadCount, 1, 1)) {
        return {};
    }

    const std::vector<const GraphNode *> & mul_consumers = graph.index().consumers(mul_node->output);
    if (mul_consumers.size() != 1 || mul_consumers.front() == nullptr) {
        return {};
    }
    const GraphNode * gated_mul_node = mul_consumers.front();
    if (gated_mul_node->op != GGML_OP_MUL || gated_mul_node->inputs.size() != 2) {
        return {};
    }

    const Value * sigmoid_output = nullptr;
    for (ValueId input : gated_mul_node->inputs) {
        if (input != mul_node->output) {
            sigmoid_output = graph_value(graph, input);
        }
    }
    const GraphNode * sigmoid_node = sigmoid_output == nullptr ? nullptr :
                                                                 graph.index().producer(sigmoid_output->id);
    if (sigmoid_node == nullptr || sigmoid_node->op != GGML_OP_UNARY || sigmoid_node->inputs.size() != 1) {
        return {};
    }
    const UnaryParams * sigmoid_params = op_params_as<UnaryParams>(sigmoid_node->params);
    if (sigmoid_params == nullptr || sigmoid_params->op != GGML_UNARY_OP_SIGMOID) {
        return {};
    }
    const Value * z      = graph_value(graph, sigmoid_node->inputs[0]);
    const Value * output = graph_value(graph, gated_mul_node->output);
    if (z == nullptr || output == nullptr || sigmoid_output == nullptr ||
        !is_shape(*z, kGdnHeadDim, kGdnValueHeadCount, 1, 1) ||
        !is_shape(*sigmoid_output, kGdnHeadDim, kGdnValueHeadCount, 1, 1) ||
        !is_shape(*output, kGdnHeadDim, kGdnValueHeadCount, 1, 1)) {
        return {};
    }

    match.rms_node       = node;
    match.mul_node       = mul_node;
    match.sigmoid_node   = sigmoid_node;
    match.gated_mul_node = gated_mul_node;
    match.raw_output     = raw_output;
    match.norm_weight    = norm_weight;
    match.z              = z;
    match.output         = output;
    return match;
}

}  // namespace

static bool match_qwen4exp_gdn_conv_prepare_decode_dispatch(const DispatchMatchContext & context,
                                                             DispatchMatch &              match) {
    const ConvPrepareMatch conv_match = match_qwen4exp_gdn_conv_prepare(context.graph, context.root_node);
    if (!conv_match.matched()) {
        return false;
    }

    const ValueId qk_inverse_norm_value = context.next_plan_value;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenGdnConvPrepareDecodeKernel);
    dispatch.kernel.compile_parameters.emplace("qwen4exp.model.rms_epsilon", "0.000001");
    dispatch.bindings.push_back({ conv_match.qkv_input->id, 0, conv_match.qkv_input->byte_count });
    dispatch.bindings.push_back({ conv_match.conv_weight->id, 0, conv_match.conv_weight->byte_count });
    dispatch.bindings.push_back({ conv_match.conv_history->id, 0, conv_match.conv_history->byte_count });
    dispatch.bindings.push_back({ conv_match.qkv_silu->id, 0, conv_match.qkv_silu->byte_count });
    dispatch.bindings.push_back({ qk_inverse_norm_value, 0, kGdnQkInverseNormByteCount });

    Status metadata_status;
    if (!match.metadata.append_alternate_value(
            { conv_match.qkv_silu->id, qk_inverse_norm_value, GGML_TYPE_F32, kGdnQkInverseNormByteCount,
              "qwen4exp.gdn.qk_inverse_norm" },
            metadata_status)) {
        match.status.append(metadata_status);
        return false;
    }

    if (!append_covered_node(context, conv_match.concat_node, match) ||
        !append_covered_node(context, conv_match.ssm_conv_node, match) ||
        !append_covered_node(context, conv_match.silu_node, match) ||
        !append_covered_node(context, conv_match.view_q_node, match) ||
        !append_covered_node(context, conv_match.view_k_node, match) ||
        !append_covered_node(context, conv_match.view_v_node, match) ||
        !append_covered_node(context, conv_match.l2norm_q_node, match) ||
        !append_covered_node(context, conv_match.l2norm_k_node, match)) {
        return false;
    }

    match.transients.push_back(
        { qk_inverse_norm_value, "qwen4exp.gdn.qk_inverse_norm", kGdnQkInverseNormByteCount, 256 });
    match.dispatches.push_back(std::move(dispatch));
    return match.status.success();
}

// qwen4exp.gdn_recurrent_decode: single-node match on GGML_OP_GATED_DELTA_NET (K=1 decode only). q/k/v
// bind to the shared qkv_silu value produced by qwen4exp.gdn_conv_prepare_decode (backward-traced
// through each of their L2_NORM/VIEW producer chains, since the literal L2_NORM outputs are never
// materialized -- see that matcher's comment); g/beta bind directly to op->src[3]/src[4] since they are
// already the fully-activated (post-sigmoid) tensors ggml_gated_delta_net() itself consumes -- this
// dispatch does not need to know or care whether their own upstream SIGMOID producers ran on HRX or
// CPU, since ggml_backend_sched transparently stages any value that crosses a backend split boundary.
//
// *** dst-tail state contract (previously a KNOWN CONTRACT MISMATCH; now fixed, see below) ***
// ggml's own graph (delta-net-base.cpp build_delta_net_fused()/build_recurrent_attn(), the decode/
// !keep branch used here) unconditionally extracts "new_state" via a VIEW into this op's *own* `dst`
// tail (byte offset == the attention-output byte count, kGdnRawOutputByteCount) and ggml_cpy's it into
// the persistent ssm_states_all recurrent-state cache -- i.e. the graph's contract is "write output and
// updated state as one packed dst buffer, state in the tail". The CPU reference kernel
// (ggml-cpu/ops.cpp:ggml_compute_forward_gated_delta_net_one_chunk, K==1 path) honors this exactly
// (its `s_out` pointer is set directly into dst's tail). qwen38_gdn_recurrent_decode now honors it too:
// `state` (src[5]) is bound READ-ONLY (the previous recurrent state) and a *second* binding aliases the
// same `output`/dst Value at byte offset kGdnRawOutputByteCount, giving the kernel a write-only
// "new_state" buffer that is exactly dst's tail -- the same bytes the graph's ggml_cpy reads from. An
// earlier version of this kernel instead mutated `state` in place and left dst's tail unwritten, so the
// unconditional ggml_cpy silently overwrote every decode step's correct update with dst's
// never-initialized tail; that correctness bug is what this two-binding split fixes.
static bool match_qwen4exp_gdn_recurrent_decode_dispatch(const DispatchMatchContext & context,
                                                          DispatchMatch &              match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_GATED_DELTA_NET || node->inputs.size() != 6 ||
        !context.graph.has_index()) {
        return false;
    }

    const Value * q      = graph_value(context.graph, node->inputs[0]);
    const Value * k      = graph_value(context.graph, node->inputs[1]);
    const Value * v      = graph_value(context.graph, node->inputs[2]);
    const Value * gate   = graph_value(context.graph, node->inputs[3]);
    const Value * beta   = graph_value(context.graph, node->inputs[4]);
    const Value * state  = graph_value(context.graph, node->inputs[5]);
    const Value * output = graph_value(context.graph, node->output);
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr || beta == nullptr || state == nullptr ||
        output == nullptr) {
        return false;
    }

    // Decode-only (K=1) shape gate; this also excludes any other delta-net-family model's own GDN
    // decode (e.g. qwen3next/kimi-linear both build on the same delta-net-base.cpp), since those use
    // different head-dim/head-count profiles than qwen4exp's 128/16/48.
    if (!is_shape(*q, kGdnHeadDim, kGdnKeyHeadCount, 1, 1) || !is_shape(*k, kGdnHeadDim, kGdnKeyHeadCount, 1, 1) ||
        !is_shape(*v, kGdnHeadDim, kGdnValueHeadCount, 1, 1) || !is_shape(*gate, 1, kGdnValueHeadCount, 1, 1) ||
        !is_shape(*beta, 1, kGdnValueHeadCount, 1, 1) ||
        !is_shape(*state, kGdnHeadDim, kGdnHeadDim, kGdnValueHeadCount, 1) ||
        !is_shape(*output, kGdnValueHeadCount * kGdnHeadDim, 1 + kGdnHeadDim, 1, 1)) {
        return false;
    }
    const GatedDeltaNetParams * gdn_params = op_params_as<GatedDeltaNetParams>(node->params);
    if (gdn_params == nullptr || gdn_params->k != 1) {
        return false;  // K>1 is chunked/prefill; intentionally left CPU-fallback.
    }

    // Backward-trace q/k/v through their L2_NORM/VIEW/SILU producer chains to the shared, already-
    // covered qkv_silu value (see qwen4exp.gdn_conv_prepare_decode's comment for why L2_NORM's literal
    // output is never materialized). All three must resolve to the *same* SILU value, and their
    // immediate producers must already be covered -- confirming the expected upstream fusion actually
    // ran, rather than some other, coincidentally-shaped graph.
    const GraphNode * l2norm_q = context.graph.index().producer(q->id);
    const GraphNode * l2norm_k = context.graph.index().producer(k->id);
    const GraphNode * view_v   = context.graph.index().producer(v->id);
    size_t            l2norm_q_index = 0;
    size_t            l2norm_k_index = 0;
    size_t            view_v_index   = 0;
    if (l2norm_q == nullptr || l2norm_q->op != GGML_OP_L2_NORM || l2norm_q->inputs.size() != 1 ||
        l2norm_k == nullptr || l2norm_k->op != GGML_OP_L2_NORM || l2norm_k->inputs.size() != 1 ||
        view_v == nullptr || view_v->op != GGML_OP_VIEW || view_v->inputs.size() != 1 ||
        !context.graph.index().node_index(l2norm_q, l2norm_q_index) ||
        !context.graph.index().node_index(l2norm_k, l2norm_k_index) ||
        !context.graph.index().node_index(view_v, view_v_index) ||
        l2norm_q_index >= context.covered_nodes.size() || !context.covered_nodes[l2norm_q_index] ||
        l2norm_k_index >= context.covered_nodes.size() || !context.covered_nodes[l2norm_k_index] ||
        view_v_index >= context.covered_nodes.size() || !context.covered_nodes[view_v_index]) {
        return false;
    }

    const Value * view_q_value = graph_value(context.graph, l2norm_q->inputs[0]);
    const Value * view_k_value = graph_value(context.graph, l2norm_k->inputs[0]);
    const GraphNode * view_q = view_q_value == nullptr ? nullptr : context.graph.index().producer(view_q_value->id);
    const GraphNode * view_k = view_k_value == nullptr ? nullptr : context.graph.index().producer(view_k_value->id);
    if (view_q == nullptr || view_q->op != GGML_OP_VIEW || view_q->inputs.size() != 1 || view_k == nullptr ||
        view_k->op != GGML_OP_VIEW || view_k->inputs.size() != 1) {
        return false;
    }
    const Value * silu_from_q = graph_value(context.graph, view_q->inputs[0]);
    const Value * silu_from_k = graph_value(context.graph, view_k->inputs[0]);
    const Value * silu_from_v = graph_value(context.graph, view_v->inputs[0]);
    if (silu_from_q == nullptr || silu_from_k == nullptr || silu_from_v == nullptr ||
        silu_from_q->id != silu_from_k->id || silu_from_q->id != silu_from_v->id) {
        return false;  // q/k/v must all come from the same shared SILU output
    }

    const CommandPlanAlternateValue * qk_inverse_norm =
        find_alternate_value(context.graph, context.plan, silu_from_q->id, GGML_TYPE_F32, kGdnQkInverseNormByteCount);
    if (qk_inverse_norm == nullptr) {
        return false;  // no qwen4exp.gdn_conv_prepare_decode match produced the scale-factor transient
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenGdnRecurrentDecodeKernel);
    dispatch.bindings.push_back({ silu_from_q->id, 0, silu_from_q->byte_count });
    dispatch.bindings.push_back({ qk_inverse_norm->alternate_value, 0, qk_inverse_norm->byte_count });
    dispatch.bindings.push_back({ gate->id, 0, gate->byte_count });
    dispatch.bindings.push_back({ beta->id, 0, beta->byte_count });
    dispatch.bindings.push_back({ state->id, 0, state->byte_count });
    // new_state aliases dst's tail (the same bytes the graph's own VIEW+ggml_cpy reads new_state from,
    // see the launch-site comment above this function): offset kGdnRawOutputByteCount, length
    // state->byte_count (dst's total byte_count minus the attention-output prefix; ggml_gated_delta_net's
    // ne[] construction in ggml.c guarantees these are equal for this K=1/S_v=128/H=48/n_seqs=1 shape).
    dispatch.bindings.push_back({ output->id, kGdnRawOutputByteCount, state->byte_count });
    // raw_output is dst's prefix (the small attention-output region); see new_state above for the tail.
    dispatch.bindings.push_back({ output->id, 0, kGdnRawOutputByteCount });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_qwen4exp_gdn_norm_gate_decode_dispatch(const DispatchMatchContext & context,
                                                          DispatchMatch &              match) {
    const NormGateMatch norm_match = match_qwen4exp_gdn_norm_gate(context.graph, context.root_node);
    if (!norm_match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenGdnNormGateDecodeKernel);
    dispatch.kernel.compile_parameters.emplace("qwen4exp.model.rms_epsilon", "0.000001");
    dispatch.bindings.push_back({ norm_match.raw_output->id, 0, norm_match.raw_output->byte_count });
    dispatch.bindings.push_back({ norm_match.norm_weight->id, 0, norm_match.norm_weight->byte_count });
    dispatch.bindings.push_back({ norm_match.z->id, 0, norm_match.z->byte_count });
    dispatch.bindings.push_back({ norm_match.output->id, 0, norm_match.output->byte_count });

    if (!append_covered_node(context, norm_match.rms_node, match) ||
        !append_covered_node(context, norm_match.mul_node, match) ||
        !append_covered_node(context, norm_match.sigmoid_node, match) ||
        !append_covered_node(context, norm_match.gated_mul_node, match)) {
        return false;
    }
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_gdn_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen4exp.gdn_conv_prepare_decode",
        GGML_OP_CONCAT,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Qwen,
        match_qwen4exp_gdn_conv_prepare_decode_dispatch,
    });
    registry.add({
        "qwen4exp.gdn_recurrent_decode",
        GGML_OP_GATED_DELTA_NET,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen4exp_gdn_recurrent_decode_dispatch,
    });
    registry.add({
        // Priority 1300 (above dispatch-rmsnorm.cpp's highest, 1200) is deliberate: this shares
        // GGML_OP_RMS_NORM as a root op with the pre-existing qwen3_moe rmsnorm matchers, and the
        // lowest-priority one of those ("qwen.rmsnorm_f32.mul_weight", priority 1000) has no check on
        // what consumes its MUL(weight) output -- it would otherwise structurally match this pattern's
        // own RMS_NORM+MUL prefix too (hidden_size=128 satisfies its is_supported_hidden_size(), and
        // the ssm_norm weight's [128] shape satisfies its is_weight_shape()). If that matcher won this
        // node first, it would claim just the 2-node prefix and strand the SIGMOID+MUL(gated) nodes
        // with no matcher and no layout-alias eligibility -- a hard scheduling failure. This dispatch
        // must therefore be tried first.
        "qwen4exp.gdn_norm_gate_decode",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        1300,
        DispatchSource::Qwen,
        match_qwen4exp_gdn_norm_gate_decode_dispatch,
    });
}

}  // namespace ggml::hrx
