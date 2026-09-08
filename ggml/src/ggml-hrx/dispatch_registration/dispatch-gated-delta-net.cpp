#include "dispatch-gated-delta-net.h"

#include "dispatch-llm-profiles.h"
#include "ggml-impl.h"
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
static constexpr KernelCatalogRef kQwenGdnRecurrentDecodeSplitKernel =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_gdn_recurrent_decode_split");
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
// Matches: SSM_CONV -> UNARY(SILU)
//
// This is qwen4exp.cpp's build_layer_attn_linear()'s conv+silu+per-head L2-norm prelude, fed by
// build_conv_state_at()'s CONCAT. The root is SSM_CONV, *not* CONCAT, even though CONCAT is
// SSM_CONV's own producer and topologically comes first -- ggml's own cross-backend scheduler
// (ggml-backend.cpp's ggml_backend_sched_backend_id_from_cur(), *not* this backend's
// device_supports_op) always assigns this specific CONCAT to CPU, independently of anything this
// file declares support for: the CONCAT's source chain roots through build_rs()'s ggml_get_rows
// gather of the persistent recurrent conv-state buffer, and ggml prefers to run an op on the same
// backend as its inputs. That assignment happens during ggml's graph-splitting pass, *before* HRX's
// own DispatchScheduler ever sees a Graph -- CONCAT is carved into a separate CPU split and never
// appears as a node in the Graph this dispatcher's traversal visits at all (confirmed by tracing
// graph_compute()'s per-split input: the split containing this SSM_CONV has no CONCAT node in it).
// A matcher rooted at CONCAT therefore never fires; SSM_CONV is the only reachable root.
//
// Because CONCAT is not part of this match, `conv_input` below binds directly to SSM_CONV's own
// src[0] -- the CONCAT's already-materialized [4,conv_channels,1,1] output value -- without tracing
// into how it was produced (Graph/GraphIndex only record producer info for nodes actually present
// in the current split; graph_value() still resolves the *value* itself correctly for any
// cross-split/external tensor, the same way conv_weight below has always been resolved). The
// four-tap window's own persistent-state write-back is a *separate* graph node
// (build_conv_state_at()'s per-rollback-slot ggml_cpy, reading this same CONCAT output's tail) that
// this dispatch does not need to cover or otherwise participate in -- it runs independently,
// wherever ggml schedules it, same as CONCAT itself.
//
// TRANSPOSE is a pure layout-alias op (graph.cpp is_layout_alias_op()) that used to be traced
// through explicitly here (back when this matcher rooted at CONCAT); now that SSM_CONV's src[0] is
// consumed as one already-assembled external value, there is nothing left to trace -- TRANSPOSE
// auto-elides on its own turn same as any other alias op, entirely outside this match.
//
// This match covers exactly two nodes -- SSM_CONV and its SILU -- and writes exactly one value, the
// shared `qkv_silu` output. The q/k/v VIEWs and the per-head L2_NORM(q)/L2_NORM(k) are NOT part of it.
// They used to be: the kernel folds q/k normalization into a small `qk_inverse_norm` reciprocal-scale
// transient applied at point of use inside qwen38_gdn_recurrent_decode, so the literal L2_NORM outputs
// were never materialized and the nodes were swept in here to hide that. That is only sound while
// qwen38_gdn_recurrent_decode runs in the SAME split, and GGML_SCHED_DEBUG=2 shows it never does --
// ggml always orders the CPU-resident beta/gate chain between the two, forcing a split boundary. The
// swept-in L2_NORMs therefore left q_conv_predelta/k_conv_predelta unwritten for their real consumer,
// which is precisely the "conv-prepare computes garbage" symptom. Leaving them out makes this dispatch
// self-contained and correct. The kernel still writes its `qk_inverse_norm` binding; nothing reads it
// while the recurrent step is CPU-side (see hrx_gdn_group_disabled("gdncore") in ggml-hrx.cpp).
struct ConvPrepareMatch {
    const GraphNode * ssm_conv_node = nullptr;
    const GraphNode * silu_node     = nullptr;
    const Value *      conv_input   = nullptr;  // CONCAT's own output (external to this split), [4,10240,1,1]
    const Value *      conv_weight  = nullptr;  // ssm_conv1d weight, [4,10240,1,1]
    const Value *      qkv_silu     = nullptr;  // shared SILU output, [10240,1,1,1]

    bool matched() const {
        return ssm_conv_node != nullptr && silu_node != nullptr && conv_input != nullptr &&
               conv_weight != nullptr && qkv_silu != nullptr;
    }
};

static ConvPrepareMatch match_qwen4exp_gdn_conv_prepare(const Graph & graph, const GraphNode * node) {
    ConvPrepareMatch match;
    if (node == nullptr || node->op != GGML_OP_SSM_CONV || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const Value * conv_input = graph_value(graph, node->inputs[0]);
    if (conv_input == nullptr || !is_shape(*conv_input, kGdnConvKernelSize, kGdnConvChannels, 1, 1)) {
        return {};  // n_seq_tokens > 1 (prefill) shows up here as a differently-shaped conv window.
    }
    const Value * conv_weight     = graph_value(graph, node->inputs[1]);
    const Value * ssm_conv_output = graph_value(graph, node->output);
    if (conv_weight == nullptr || ssm_conv_output == nullptr ||
        !is_shape(*conv_weight, kGdnConvKernelSize, kGdnConvChannels, 1, 1) ||
        !is_shape(*ssm_conv_output, kGdnConvChannels, 1, 1, 1)) {
        return {};
    }

    const std::vector<const GraphNode *> & ssm_conv_consumers = graph.index().consumers(node->output);
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

    // The q/k/v VIEW slices and the two L2_NORMs that this matcher used to identify are deliberately
    // no longer part of the match. They are not covered (see the dispatch function below for why),
    // and now that device_supports_op() declines GGML_OP_L2_NORM they are not even present in this
    // split's Graph -- so requiring them here would make the match fail outright, leaving SSM_CONV
    // itself unclaimed and stranding the split. The shape checks above are specific enough on their
    // own: a [4,10240] conv window against a [4,10240] kernel feeding a [10240] SILU is qwen4exp's
    // decode-time GDN prelude and nothing else.

    match.ssm_conv_node = node;
    match.silu_node     = silu_node;
    match.conv_input    = conv_input;
    match.conv_weight   = conv_weight;
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
    dispatch.bindings.push_back({ conv_match.conv_input->id, 0, conv_match.conv_input->byte_count });
    dispatch.bindings.push_back({ conv_match.conv_weight->id, 0, conv_match.conv_weight->byte_count });
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

    // Cover ONLY the two nodes this kernel actually writes: SSM_CONV (whose conv_output_raw is a
    // pure intermediate, use=1, consumed solely by the SILU below) and the SILU itself, whose
    // output IS the `qkv_silu` binding above.
    //
    // The q/k/v VIEWs and the two L2_NORMs are deliberately NOT covered any more, even though
    // match_qwen4exp_gdn_conv_prepare() still identifies them as a structural guard. Covering them
    // was only sound while qwen4exp.gdn_recurrent_decode ran in the SAME split, because this kernel
    // never materializes the L2_NORM outputs -- it folds normalization into the split-local
    // `qk_inverse_norm` transient instead. GGML_SCHED_DEBUG=2 shows that precondition can never hold
    // for this model: ggml topologically orders the CPU-resident beta/gate chain (MUL_MAT(ssm_alpha)
    // -> ADD(ssm_dt) -> SOFTPLUS -> MUL(ssm_a) -> MUL_MAT(ssm_beta) -> SIGMOID) between the conv
    // prelude and GATED_DELTA_NET, forcing a split boundary between them. Claiming the L2_NORMs
    // therefore left q_conv_predelta/k_conv_predelta permanently unwritten for their real (CPU)
    // consumer -- the actual cause of the long-standing "conv-prepare computes garbage" note.
    // Leaving them uncovered lets L2_NORM run normally and keeps this dispatch numerically correct
    // on its own; see hrx_gdn_group_disabled() in ggml-hrx.cpp for the matching capability change.
    if (!append_covered_node(context, conv_match.ssm_conv_node, match) ||
        !append_covered_node(context, conv_match.silu_node, match)) {
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

// qwen4exp.gdn_recurrent_decode_split: the conv-prelude-independent form of the matcher above, and the
// one that actually fires for this model. Same root op and same shape gate, but q/k/v bind directly to
// ggml_gated_delta_net's own src[0]/src[1]/src[2] -- the materialized L2_NORM(q), L2_NORM(k) and the v
// VIEW -- instead of being backward-traced to conv-prepare's packed `qkv_silu` value and its
// `qk_inverse_norm` transient.
//
// The packed variant additionally requires those L2_NORM/VIEW producer nodes to be *covered in the same
// split*, because qwen38_gdn_conv_prepare_decode never materializes normalized q/k. GGML_SCHED_DEBUG=2
// shows ggml always orders the CPU-resident beta/gate chain (MUL_MAT(ssm_alpha) -> ADD(ssm_dt) ->
// SOFTPLUS -> MUL(ssm_a) -> MUL_MAT(ssm_beta) -> SIGMOID) between the conv prelude and
// GATED_DELTA_NET, so that precondition can never hold and GATED_DELTA_NET was left on the CPU
// reference implementation for every layer. Because this variant reads only real graph values, it has
// no covered-node or alternate-value preconditions at all and ggml_backend_sched transparently stages
// whichever of its inputs happen to live on the other backend.
//
// The dst-tail state contract is identical to the packed variant -- see its comment above: `state`
// (src[5]) binds READ-ONLY and a second binding aliases dst at byte offset kGdnRawOutputByteCount to
// give the kernel the write-only `new_state` region that the graph's own VIEW + ggml_cpy reads back
// into ssm_states_all.
static bool match_qwen4exp_gdn_recurrent_decode_split_dispatch(const DispatchMatchContext & context,
                                                                DispatchMatch &              match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_GATED_DELTA_NET || node->inputs.size() != 6) {
        return false;
    }

    const Value * q      = graph_value(context.graph, node->inputs[0]);
    const Value * k      = graph_value(context.graph, node->inputs[1]);
    const Value * v      = graph_value(context.graph, node->inputs[2]);
    const Value * gate   = graph_value(context.graph, node->inputs[3]);
    const Value * beta   = graph_value(context.graph, node->inputs[4]);
    const Value * state  = graph_value(context.graph, node->inputs[5]);
    const Value * output = graph_value(context.graph, node->output);
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr || beta == nullptr ||
        state == nullptr || output == nullptr) {
        return false;
    }

    // Decode-only (K=1) shape gate, matching hrx_gdn_gated_delta_net_decode_supported() in ggml-hrx.cpp.
    // This also excludes the other delta-net-family models built on delta-net-base.cpp (qwen3next,
    // kimi-linear, ...), which use different head-dim/head-count profiles than qwen4exp's 128/16/48.
    if (!is_shape(*q, kGdnHeadDim, kGdnKeyHeadCount, 1, 1) ||
        !is_shape(*k, kGdnHeadDim, kGdnKeyHeadCount, 1, 1) ||
        !is_shape(*v, kGdnHeadDim, kGdnValueHeadCount, 1, 1) ||
        !is_shape(*gate, 1, kGdnValueHeadCount, 1, 1) || !is_shape(*beta, 1, kGdnValueHeadCount, 1, 1) ||
        !is_shape(*state, kGdnHeadDim, kGdnHeadDim, kGdnValueHeadCount, 1) ||
        !is_shape(*output, kGdnValueHeadCount * kGdnHeadDim, 1 + kGdnHeadDim, 1, 1)) {
        return false;
    }
    const GatedDeltaNetParams * gdn_params = op_params_as<GatedDeltaNetParams>(node->params);
    if (gdn_params == nullptr || gdn_params->k != 1) {
        return false;  // K>1 is chunked/prefill; intentionally left CPU-fallback.
    }

    // The kernel indexes q/k/v as flat, tightly packed f32 rows, so a permuted or gappy view would be
    // read incorrectly rather than declined. q and k are L2_NORM outputs (always freshly materialized
    // and contiguous); v is a VIEW into the conv output whose rows are tight for n_seq_tokens == 1.
    if (!q->contiguous || !k->contiguous || !v->contiguous || !gate->contiguous || !beta->contiguous ||
        !state->contiguous) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenGdnRecurrentDecodeSplitKernel);
    dispatch.bindings.push_back({ q->id, 0, q->byte_count });
    dispatch.bindings.push_back({ k->id, 0, k->byte_count });
    dispatch.bindings.push_back({ v->id, 0, v->byte_count });
    dispatch.bindings.push_back({ gate->id, 0, gate->byte_count });
    dispatch.bindings.push_back({ beta->id, 0, beta->byte_count });
    dispatch.bindings.push_back({ state->id, 0, state->byte_count });
    dispatch.bindings.push_back({ output->id, kGdnRawOutputByteCount, state->byte_count });
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
        GGML_OP_SSM_CONV,
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
        // Lower priority than the packed variant above, so that if conv-prepare ever does land in the
        // same split (which would let the fused form skip re-reading q/k and reuse its transient), that
        // one still wins. In practice ggml always separates them and this is the variant that fires.
        "qwen4exp.gdn_recurrent_decode_split",
        GGML_OP_GATED_DELTA_NET,
        DispatchMatchKind::SingleOp,
        50,
        DispatchSource::Qwen,
        match_qwen4exp_gdn_recurrent_decode_split_dispatch,
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
