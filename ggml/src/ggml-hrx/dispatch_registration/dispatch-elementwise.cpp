// Elementwise activations (GGML_OP_UNARY) and row-wise L2 normalization (GGML_OP_L2_NORM).
//
// These are arithmetically trivial, and that is exactly why they matter. A node the HRX backend
// cannot run is a backend boundary: ggml_backend_sched has to close the current HRX split, stage its
// tensors back to host memory, run one op on the CPU, and re-enter the GPU. A GGML_SCHED_DEBUG=2
// census of qwen4exp decode showed 1110 splits over 5081 nodes -- a mean of 4.6 nodes per split, with
// 550 splits holding two nodes or fewer -- and measured 2.36 s/token against a ~1.0 s/token CPU-only
// floor. Split count, not GPU node count, is what that workload is paying for.
//
// The single largest contributor was one repeating signature: 36 identical CPU splits (one per GDN
// layer) of SSM_CONV, SILU, L2_NORM, L2_NORM, MUL_MAT, ADD, SOFTPLUS, MUL, MUL_MAT, SIGMOID -- the
// GDN front end. Every MUL_MAT/ADD/MUL in it already had HRX coverage; the chain was held on the CPU
// by the activations and the two L2_NORMs alone, which then also stranded the GATED_DELTA_NET that
// consumes them into a split of its own.
//
// Coverage here is deliberately generic (any contiguous same-shaped f32 node) rather than scoped to
// qwen4exp's shapes, because these ops carry no shape-dependent risk: the kernels are flat maps over
// element_count, and L2_NORM's only structure is its ne[0] row length. That also picks up the 326
// SIGMOID / 134 SILU nodes in the MoE router and shared-expert paths for free.

#include "dispatch-elementwise.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

namespace ggml::hrx {

// Bisect switches. These mirror identically-named helpers in ggml-hrx.cpp and must stay in lockstep
// with them: if a capability predicate claims a node the matcher here then declines, the whole split
// aborts with "unsupported HRX node" rather than falling back for that one node.
static bool mul_outer_broadcast_enabled() {
    const char * enabled = std::getenv("HRX_MUL_OUTER");
    return enabled == nullptr || enabled[0] != '0';
}

static bool repeat_dispatch_enabled() {
    const char * enabled = std::getenv("HRX_ENABLE_REPEAT");
    return enabled != nullptr && enabled[0] != '0';
}

static constexpr KernelCatalogRef kSigmoidF32Kernel  = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_sigmoid_f32");
static constexpr KernelCatalogRef kSiluF32Kernel     = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_silu_f32");
static constexpr KernelCatalogRef kReluF32Kernel     = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_relu_f32");
static constexpr KernelCatalogRef kSoftplusF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_softplus_f32");
static constexpr KernelCatalogRef kL2NormF32Kernel   = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_l2_norm_f32");
static constexpr KernelCatalogRef kDenseMatmulF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_dense_matmul_f32");

// Mirrors hrx_dense_matmul_f32_supported() in ggml-hrx.cpp. Keep the two in sync: a MUL_MAT that
// device_supports_op() claims but no matcher roots at aborts its whole split rather than falling
// back to the CPU for that one node.
static constexpr int64_t kDenseMatmulF32MaxInputSize   = 131072;
static constexpr int64_t kDenseMatmulF32MaxTokenCount  = 2048;
// Deliberately far below what the kernel can address. This route exists for small per-layer F32
// gates, and the one other F32 MUL_MAT in a qwen4exp decode graph is the MoE router logits
// projection ([2560]x[n_expert] = 512 wide). That node must stay CPU-side: dispatch-moe-router.cpp
// roots its multi-op match at the SOFT_MAX above it and walks down to the ARGSORT/TOP_K/GET_ROWS
// that select route weights. Claiming the logits MUL_MAT drags the SOFT_MAX into the HRX split
// while GET_ROWS stays behind, and the router matcher then hard-aborts the graph with
// "missing GET_ROWS from reshaped probabilities and top-k ids". 256 clears qwen4exp's 48-wide
// ssm_alpha/ssm_beta gates with room to spare while excluding any plausible expert count.
static constexpr int64_t kDenseMatmulF32MaxOutputSize  = 256;
static constexpr KernelCatalogRef kMulF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_mul_f32");
static constexpr KernelCatalogRef kMulOuterF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_mul_outer_f32");
static constexpr KernelCatalogRef kRepeatF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_repeat_f32");
// Matches the index.assume ranges in hrx_owned/repeat_f32.loom.
static constexpr int64_t kRepeatF32MaxPeriod = 1048576;
static constexpr int64_t kRepeatF32MaxTile   = 65536;
// Matches the index.assume ranges in hrx_owned/mul_f32.loom.
static constexpr int64_t kMulF32MaxExtent = 1048576;

// ggml lets src1 be repeated into src0's shape. hrx_owned/mul_f32.loom handles the contiguous-prefix
// form of that: src1 agrees with src0 on a leading run of dimensions and is 1 above it, so
// dst[i] = src0[i] * src1[i % period] where period is the product of that leading run. Returns the
// period, or 0 if src1 does not have that form (a stride-broadcast such as [1,ne1] into [ne0,ne1]
// needs a different addressing scheme and is left on the CPU).
static int64_t mul_broadcast_period(const Value & lhs, const Value & rhs) {
    int64_t period      = 1;
    bool    seen_ones   = false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (rhs.ne[i] == lhs.ne[i] && !seen_ones) {
            period *= lhs.ne[i];
            continue;
        }
        if (rhs.ne[i] == 1) {
            seen_ones = true;
            continue;
        }
        return 0;  // Partial repeat along this axis: not a contiguous prefix.
    }
    return period;
}

// The complementary direction: src1 is 1 on a leading run of dimensions and agrees with src0 above
// it, so dst[i] = src0[i] * src1[i / period]. qwen4exp emits this as f32[2560,4] * f32[1,4] in
// build_hc_combine() and build_ple(), and f32[640,10] * f32[1,10] for ffn_moe_weighted; those are the
// 185 lone-MUL CPU splits a decode census shows, each costing two split boundaries. Returns the
// period (the product of the broadcast prefix), or 0 if src1 does not have this form.
static int64_t mul_outer_broadcast_period(const Value & lhs, const Value & rhs) {
    int64_t period   = 1;
    bool    matching = false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (!matching && rhs.ne[i] == 1) {
            // A dimension src0 is also 1 on carries no elements either way, so it neither ends the
            // broadcast prefix nor contributes to the period.
            if (lhs.ne[i] != 1) {
                period *= lhs.ne[i];
            }
            continue;
        }
        if (rhs.ne[i] != lhs.ne[i]) {
            return 0;  // Partial repeat along this axis: not a contiguous broadcast prefix.
        }
        matching = true;
    }
    return period;
}

static const Value * elementwise_graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool elementwise_same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

// Shared precondition for every kernel in this file: one f32 input mapped elementwise onto one
// f32 output of identical shape, both tightly packed, with an element count the kernels' index
// range assumptions can represent.
static bool elementwise_f32_pair(const Graph & graph,
                                 const GraphNode * node,
                                 const Value **    input_out,
                                 const Value **    output_out) {
    if (node == nullptr || node->inputs.size() != 1) {
        return false;
    }
    const Value * input  = elementwise_graph_value(graph, node->inputs[0]);
    const Value * output = elementwise_graph_value(graph, node->output);
    if (input == nullptr || output == nullptr) {
        return false;
    }
    if (input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return false;
    }
    if (!input->contiguous || !output->contiguous || !elementwise_same_shape(*input, *output)) {
        return false;
    }
    if (output->element_count <= 0 ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *input_out  = input;
    *output_out = output;
    return true;
}

// Kept in sync with hrx_unary_f32_supported() in ggml-hrx.cpp. Any ggml_unary_op not listed here has
// no kernel, so device_supports_op() must decline it -- claiming it would strand its split.
static bool unary_kernel_for(ggml_unary_op op, KernelCatalogRef & kernel_out) {
    switch (op) {
        case GGML_UNARY_OP_SIGMOID:
            kernel_out = kSigmoidF32Kernel;
            return true;
        case GGML_UNARY_OP_SILU:
            kernel_out = kSiluF32Kernel;
            return true;
        case GGML_UNARY_OP_RELU:
            kernel_out = kReluF32Kernel;
            return true;
        case GGML_UNARY_OP_SOFTPLUS:
            kernel_out = kSoftplusF32Kernel;
            return true;
        default:
            return false;
    }
}

static bool match_unary_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_UNARY) {
        return false;
    }
    const UnaryParams * params = op_params_as<UnaryParams>(node->params);
    if (params == nullptr) {
        return false;
    }
    KernelCatalogRef kernel = kSigmoidF32Kernel;
    if (!unary_kernel_for(params->op, kernel)) {
        return false;
    }
    const Value * input  = nullptr;
    const Value * output = nullptr;
    if (!elementwise_f32_pair(context.graph, node, &input, &output)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_l2_norm_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_L2_NORM) {
        return false;
    }
    const Value * input  = nullptr;
    const Value * output = nullptr;
    if (!elementwise_f32_pair(context.graph, node, &input, &output)) {
        return false;
    }
    // ggml normalizes over ne[0] only; ne[1..3] are independent rows. One workgroup owns one row, so
    // the kernel needs the row length and the flattened row count.
    const int64_t row_length = output->ne[0];
    const int64_t row_count  = output->ne[1] * output->ne[2] * output->ne[3];
    if (row_length <= 0 || row_count <= 0) {
        return false;
    }
    const L2NormParams * params = op_params_as<L2NormParams>(node->params);
    if (params == nullptr || !(params->eps >= 0.0f)) {
        return false;  // Rejects NaN as well as negatives; ggml asserts eps >= 0.
    }

    // The epsilon reaches the kernel as a compile-time config because DispatchBinding and
    // KernelSpecialization carry no f32 channel, matching how every RMSNorm dispatch in this backend
    // supplies its own eps. Round-tripping through the widest exact decimal form keeps the compiled
    // constant bit-identical to the value ggml would have used on the CPU.
    char epsilon_text[32];
    std::snprintf(epsilon_text, sizeof(epsilon_text), "%.9g", static_cast<double>(params->eps));

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kL2NormF32Kernel);
    dispatch.kernel.integer_parameters.emplace("row_length", row_length);
    dispatch.kernel.integer_parameters.emplace("row_count", row_count);
    dispatch.kernel.compile_parameters.emplace("ggml.l2_norm_f32.epsilon", epsilon_text);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

// GGML_OP_MUL. See the header comment on hrx_owned/mul_f32.loom: device_supports_op() has always
// claimed every MUL through its `default: return true`, but the only MUL kernel was the one fused
// into the routed-FFN dispatch, so an unfused MUL that got pulled into an HRX split aborted the
// graph. This is the generic fallback that makes that claim honest.
static bool match_mul_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL || node->inputs.size() != 2) {
        return false;
    }
    const Graph & graph  = context.graph;
    const Value * lhs    = elementwise_graph_value(graph, node->inputs[0]);
    const Value * rhs    = elementwise_graph_value(graph, node->inputs[1]);
    const Value * output = elementwise_graph_value(graph, node->output);
    if (lhs == nullptr || rhs == nullptr || output == nullptr) {
        return false;
    }
    if (lhs->type != GGML_TYPE_F32 || rhs->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return false;
    }
    if (!lhs->contiguous || !rhs->contiguous || !output->contiguous || !elementwise_same_shape(*lhs, *output)) {
        return false;
    }
    // Two broadcast directions, two exports. The inner (prefix) form is tried first so a same-shape
    // MUL -- which satisfies both -- keeps the geometry it already used.
    bool    outer_broadcast = false;
    int64_t period          = mul_broadcast_period(*lhs, *rhs);
    int64_t outer_count     = 0;
    if (period > 0 && period == rhs->element_count && output->element_count > 0 &&
        output->element_count % period == 0) {
        outer_count = output->element_count / period;
    } else {
        if (!mul_outer_broadcast_enabled()) {
            return false;
        }
        period = mul_outer_broadcast_period(*lhs, *rhs);
        if (period <= 0 || output->element_count <= 0 || output->element_count % period != 0) {
            return false;
        }
        outer_count = output->element_count / period;
        if (outer_count != rhs->element_count) {
            return false;
        }
        outer_broadcast = true;
    }
    if (period > kMulF32MaxExtent || outer_count > kMulF32MaxExtent) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(outer_broadcast ? kMulOuterF32Kernel : kMulF32Kernel);
    dispatch.kernel.integer_parameters.emplace("period", period);
    dispatch.kernel.integer_parameters.emplace("outer_count", outer_count);
    dispatch.bindings.push_back({ lhs->id, 0, lhs->byte_count });
    dispatch.bindings.push_back({ rhs->id, 0, rhs->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

// Decomposes a GGML_OP_REPEAT into the one form hrx_owned/repeat_f32.loom implements: dst is src
// tiled along exactly one axis, so dst[outer][mid][i] = src[outer][i]. Returns false for anything
// else (a repeat along two axes, or a partial tile), which leaves the node on the CPU.
static bool repeat_f32_geometry(const Value & src,
                                const Value & dst,
                                int64_t *     period_out,
                                int64_t *     mid_out,
                                int64_t *     outer_out) {
    int repeat_axis = -1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src.ne[i] == dst.ne[i]) {
            continue;
        }
        if (src.ne[i] != 1 || dst.ne[i] <= 1 || repeat_axis >= 0) {
            return false;
        }
        repeat_axis = i;
    }

    // No repeat axis at all is a plain copy; treating it as a single tile keeps the geometry valid.
    const int     axis   = repeat_axis < 0 ? GGML_MAX_DIMS : repeat_axis;
    const int64_t mid    = repeat_axis < 0 ? 1 : dst.ne[repeat_axis];
    int64_t       period = 1;
    int64_t       outer  = 1;
    for (int i = 0; i < axis; ++i) {
        period *= dst.ne[i];
    }
    for (int i = axis + 1; i < GGML_MAX_DIMS; ++i) {
        outer *= dst.ne[i];
    }

    if (period <= 0 || mid <= 0 || outer <= 0) {
        return false;
    }
    if (period * mid * outer != dst.element_count || period * outer != src.element_count) {
        return false;
    }
    if (period > kRepeatF32MaxPeriod || mid > kRepeatF32MaxTile || outer > kRepeatF32MaxTile) {
        return false;
    }

    *period_out = period;
    *mid_out    = mid;
    *outer_out  = outer;
    return true;
}

// GGML_OP_REPEAT. qwen4exp materialises [n_embd, 1, T] -> [n_embd, hc, T] three times per
// hyper-connected block; a decode census leaves all 98 of them on the CPU, 96 as lone single-node
// splits between two HRX runs, which is two split boundaries spent to run one broadcast copy.
static bool match_repeat_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_REPEAT || node->inputs.size() != 1) {
        return false;
    }
    if (!repeat_dispatch_enabled()) {
        return false;
    }
    const Graph & graph  = context.graph;
    const Value * input  = elementwise_graph_value(graph, node->inputs[0]);
    const Value * output = elementwise_graph_value(graph, node->output);
    if (input == nullptr || output == nullptr) {
        return false;
    }
    if (input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return false;
    }
    if (!input->contiguous || !output->contiguous || input->element_count <= 0 ||
        output->element_count <= 0) {
        return false;
    }

    int64_t period = 0;
    int64_t mid    = 0;
    int64_t outer  = 0;
    if (!repeat_f32_geometry(*input, *output, &period, &mid, &outer)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRepeatF32Kernel);
    dispatch.kernel.integer_parameters.emplace("period", period);
    dispatch.kernel.integer_parameters.emplace("mid_count", mid);
    dispatch.kernel.integer_parameters.emplace("outer_count", outer);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static constexpr KernelCatalogRef kScaleF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_scale_f32");

// GGML_OP_SCALE: dst = src*scale + bias, both floats in op_params. See the header comment on
// hrx_owned/scale_f32.loom -- SCALE is the most common first-op of a CPU split in a qwen4exp decode
// graph, often as a lone node inside an otherwise contiguous HRX run.
static bool match_scale_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_SCALE || node->inputs.size() != 1) {
        return false;
    }
    const Value * input  = nullptr;
    const Value * output = nullptr;
    if (!elementwise_f32_pair(context.graph, node, &input, &output)) {
        return false;
    }
    const ScaleParams * params = op_params_as<ScaleParams>(node->params);
    if (params == nullptr) {
        return false;
    }
    // The zero-fill (0, 0) form stays with common.zero_f32, which writes the destination without
    // reading the source and is what llama.cpp's recurrent-state clear relies on.
    if (params->scale == 0.0f && params->bias == 0.0f) {
        return false;
    }

    char scale_text[32];
    char bias_text[32];
    std::snprintf(scale_text, sizeof(scale_text), "%.9g", static_cast<double>(params->scale));
    std::snprintf(bias_text, sizeof(bias_text), "%.9g", static_cast<double>(params->bias));

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kScaleF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.scale_f32.scale", scale_text);
    dispatch.kernel.compile_parameters.emplace("ggml.scale_f32.bias", bias_text);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

// Unquantized f32 weights. The quantized dense routes live in dispatch-llm-matmul.cpp; this covers
// the small F32 projections those routes cannot, which for qwen4exp means the per-layer
// ssm_alpha/ssm_beta [2560]x[48] GDN gates. See the header comment on hrx_owned/dense_f32.loom for
// why two ~123K-MAC nodes are worth a kernel: they sit mid-chain in the GDN prelude, and
// ggml_backend_sched will not fragment a contiguous run, so leaving them on the CPU pins every
// downstream node there too.
static bool match_dense_matmul_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return false;
    }
    const Graph & graph  = context.graph;
    const Value * weight = elementwise_graph_value(graph, node->inputs[0]);
    const Value * input  = elementwise_graph_value(graph, node->inputs[1]);
    const Value * output = elementwise_graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr) {
        return false;
    }
    const auto is_2d = [](const Value & v) { return v.ne[2] == 1 && v.ne[3] == 1; };
    if (weight->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return false;
    }
    if (!is_2d(*weight) || !is_2d(*input) || !is_2d(*output) || !weight->contiguous || !input->contiguous ||
        !output->contiguous) {
        return false;
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count) {
        return false;
    }
    if (input_size < 1 || input_size > kDenseMatmulF32MaxInputSize || output_size < 1 ||
        output_size > kDenseMatmulF32MaxOutputSize || token_count < 1 ||
        token_count > kDenseMatmulF32MaxTokenCount) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kDenseMatmulF32Kernel);
    dispatch.kernel.integer_parameters.emplace("input_size", input_size);
    dispatch.kernel.integer_parameters.emplace("output_size", output_size);
    dispatch.kernel.integer_parameters.emplace("token_count", token_count);
    dispatch.bindings.push_back({ weight->id, 0, weight->byte_count });
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_elementwise_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        // Priority 0: these are the fallback single-op forms. Any fused matcher that wants to absorb
        // an activation into a larger dispatch (the routed-FFN SwiGLU kernels, qwen4exp's GDN
        // conv-prepare SILU) registers above this and still wins.
        "common.unary_f32",
        GGML_OP_UNARY,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_unary_f32_dispatch,
    });
    registry.add({
        "common.l2_norm_f32",
        GGML_OP_L2_NORM,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_l2_norm_f32_dispatch,
    });
    registry.add({
        // Priority 0 so common.zero_f32 keeps the zero-fill form it is registered for; this matcher
        // declines that case explicitly as well.
        "common.scale_f32",
        GGML_OP_SCALE,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_scale_f32_dispatch,
    });
    registry.add({
        // Priority 0: the routed-FFN dispatch fuses its own MULs and registers above this, so this
        // only ever picks up the nodes nothing else claimed.
        "common.mul_f32",
        GGML_OP_MUL,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_mul_f32_dispatch,
    });
    registry.add({
        "common.repeat_f32",
        GGML_OP_REPEAT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_repeat_f32_dispatch,
    });
    registry.add({
        // Priority 0 for the same reason as the activations: the quantized dense routes and the fused
        // qwen.* projection matchers register above this and keep winning on the weights they cover.
        "common.dense_matmul_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_dense_matmul_f32_dispatch,
    });
}

}  // namespace ggml::hrx
