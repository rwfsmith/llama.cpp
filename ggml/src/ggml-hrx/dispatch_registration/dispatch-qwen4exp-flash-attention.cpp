#include "dispatch-qwen4exp-flash-attention.h"

#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstring>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kDecode =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_attention_decode");
static constexpr KernelCatalogRef kPartial =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_attention_decode_split_partial");
static constexpr KernelCatalogRef kReduce =
    GGML_HRX_KERNEL_REF("qwen4exp", "qwen38_attention_decode_split_reduce");
static constexpr int64_t kHead = 256;
static constexpr int64_t kQueryHeads = 24;
static constexpr int64_t kKvHeads = 2;
static constexpr int64_t kHidden = kHead * kQueryHeads;
static constexpr int64_t kSplits = 16;
static constexpr size_t kRowBytes = kHidden * sizeof(float);

template<class T>
static bool attention_shapes(const T * q, const T * k, const T * v, const T * mask, const T * out) {
    if (!q || !k || !v || !mask || !out ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || out->type != GGML_TYPE_F32) {
        return false;
    }
    if (q->ne[0] != kHead || q->ne[1] < 1 || q->ne[1] >= 16 || q->ne[2] != kQueryHeads ||
        k->ne[0] != kHead || v->ne[0] != kHead || k->ne[2] != kKvHeads || v->ne[2] != kKvHeads ||
        k->ne[1] < 1 || k->ne[1] > 262144 || v->ne[1] != k->ne[1] ||
        mask->ne[0] != k->ne[1] || mask->ne[1] < q->ne[1] ||
        mask->ne[2] != 1 || mask->ne[3] != 1 ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || out->ne[3] != 1 ||
        out->ne[0] != kHead || out->ne[1] != kQueryHeads || out->ne[2] != q->ne[1]) {
        return false;
    }
    // q and the non-transposed F16 cache arrive permuted (0,2,1,3).
    return q->nb[0] == sizeof(float) && q->nb[1] == kRowBytes &&
           q->nb[2] == kHead * sizeof(float) &&
           k->nb[0] == sizeof(ggml_fp16_t) && v->nb[0] == sizeof(ggml_fp16_t) &&
           k->nb[1] == kHead * kKvHeads * sizeof(ggml_fp16_t) && v->nb[1] == k->nb[1] &&
           k->nb[2] == kHead * sizeof(ggml_fp16_t) && v->nb[2] == k->nb[2] &&
           mask->nb[0] == sizeof(ggml_fp16_t) &&
           mask->nb[1] == static_cast<size_t>(mask->ne[0]) * sizeof(ggml_fp16_t) &&
           out->nb[0] == sizeof(float) && out->nb[1] == kHead * sizeof(float) &&
           out->nb[2] == kRowBytes;
}

static bool attention_params(float scale, float bias, float softcap, ggml_prec prec) {
    return std::fabs(scale - 0.0625f) <= 1.0e-6f && bias == 0.0f && softcap == 0.0f &&
           (prec == GGML_PREC_DEFAULT || prec == GGML_PREC_F32);
}

static bool has_sparse_kv_limit(const ggml_tensor * op) {
    // n_kv_max has no public getter and is not yet represented in FlashAttnExtParams.
    int32_t limit = 0;
    std::memcpy(&limit, reinterpret_cast<const uint8_t *>(op->op_params) + 4 * sizeof(int32_t), sizeof(limit));
    return limit != 0;
}

static bool input_ready(const DispatchMatchContext & context, ValueId id) {
    // Also inspect aliases: a view's position must not hide an uncomputed producer.
    for (size_t hop = 0; hop <= context.graph.values().size(); ++hop) {
        const GraphNode * producer = context.graph.index().producer(id);
        if (!producer) {
            return true;
        }
        size_t index = 0;
        if (!context.graph.index().node_index(producer, index) || index >= context.root_index) {
            return false;
        }
        if (!is_layout_alias_node(context.graph, *producer) || producer->inputs.empty()) {
            return true;
        }
        id = producer->inputs[0];
    }
    return false;
}

static bool match_attention(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (!node || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4) {
        return false;
    }
    const auto * params = op_params_as<FlashAttnExtParams>(node->params);
    if (!params || !attention_params(params->scale, params->max_bias, params->logit_softcap, params->prec)) {
        return false;
    }
    const Value * q = context.graph.values().find(node->inputs[0]);
    const Value * k = context.graph.values().find(node->inputs[1]);
    const Value * v = context.graph.values().find(node->inputs[2]);
    const Value * mask = context.graph.values().find(node->inputs[3]);
    const Value * out = context.graph.values().find(node->output);
    if (!attention_shapes(q, k, v, mask, out)) {
        return false;
    }
    for (ValueId input : node->inputs) {
        if (!input_ready(context, input)) {
            return false;
        }
    }
    if (out->tensor && has_sparse_kv_limit(out->tensor)) {
        return false;
    }

    const ValueId control(context.next_plan_value.value);
    const ValueId identity_gate(control.value + 1);
    const ValueId partials(control.value + 2);
    const int32_t position = static_cast<int32_t>(k->ne[1] - 1);
    std::vector<uint8_t> control_data(sizeof(position));
    std::memcpy(control_data.data(), &position, sizeof(position));
    match.transients.push_back({ control, "qwen4exp.attention.control", sizeof(position), 256 });
    match.constant_initializations.push_back({ control, "qwen4exp.attention.control", 0, std::move(control_data) });

    // build_layer_attn expands FLASH before CONT(gate_view) and SIGMOID. Binding that
    // future gate violates readiness; writing its future MUL allocation can clobber
    // still-live tensors under gallocr. Reuse the gated kernel with a private identity
    // gate and write ONLY the FLASH result at its own lifetime. CONT/SIGMOID/MUL remain
    // ordinary graph nodes. This is intentionally not an attention+gate fusion.
    std::vector<uint8_t> ones(kRowBytes);
    const float one = 1.0f;
    for (size_t offset = 0; offset < ones.size(); offset += sizeof(one)) {
        std::memcpy(ones.data() + offset, &one, sizeof(one));
    }
    match.transients.push_back({ identity_gate, "qwen4exp.attention.identity_gate", kRowBytes, 256 });
    match.constant_initializations.push_back(
        { identity_gate, "qwen4exp.attention.identity_gate", 0, std::move(ones) });

    const bool split = k->ne[1] > 2048;
    const size_t partial_bytes = kKvHeads * kSplits * 12 * 258 * sizeof(float);
    if (split) {
        match.transients.push_back({ partials, "qwen4exp.attention.partials", partial_bytes, 256 });
    }
    for (int64_t row = 0; row < q->ne[1]; ++row) {
        Dispatch dispatch;
        dispatch.kernel = make_kernel_specialization(split ? kPartial : kDecode);
        dispatch.kernel.compile_parameters.emplace("qwen38.attention.cache_capacity", std::to_string(k->ne[1]));
        if (split) {
            dispatch.kernel.compile_parameters.emplace("qwen38.attention.decode_split_count", std::to_string(kSplits));
        }
        dispatch.bindings.push_back({ control, 0, sizeof(position) });
        dispatch.bindings.push_back({ q->id, static_cast<size_t>(row) * q->nb[1], kRowBytes });
        if (!split) {
            dispatch.bindings.push_back({ identity_gate, 0, kRowBytes });
        }
        dispatch.bindings.push_back({ k->id, 0, k->byte_count });
        dispatch.bindings.push_back({ v->id, 0, v->byte_count });
        dispatch.bindings.push_back({ mask->id, static_cast<size_t>(row) * mask->nb[1], mask->nb[1] });
        dispatch.bindings.push_back(split ? DispatchBinding{ partials, 0, partial_bytes } :
            DispatchBinding{ out->id, static_cast<size_t>(row) * out->nb[2], kRowBytes });
        match.dispatches.push_back(std::move(dispatch));
        if (split) {
            // Scalar partials per row honor the entire explicit mask. The WMMA variant
            // infers a causal position from row ordinal, which a sparse mask cannot prove.
            Dispatch reduce;
            reduce.kernel = make_kernel_specialization(kReduce);
            // Runtime launch indices, NOT kernel.def specialization parameters.
            // The generated catalog corrects the pinned manifest's workload entry.
            reduce.kernel.integer_parameters.emplace("token_count", 1);
            reduce.kernel.integer_parameters.emplace("token_ordinal", 0);
            reduce.kernel.compile_parameters.emplace("qwen38.attention.decode_split_count", std::to_string(kSplits));
            reduce.bindings.push_back({ identity_gate, 0, kRowBytes });
            reduce.bindings.push_back({ partials, 0, partial_bytes });
            reduce.bindings.push_back({ out->id, static_cast<size_t>(row) * out->nb[2], kRowBytes });
            match.dispatches.push_back(std::move(reduce));
        }
    }
    match.covered_nodes.push_back(context.root_index);
    return true;
}

}  // namespace

bool supports_qwen4exp_flash_attention_dispatch(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_FLASH_ATTN_EXT || op->src[4] ||
        has_sparse_kv_limit(op)) {
        return false;
    }
    float params[3];
    std::memcpy(params, op->op_params, sizeof(params));
    return attention_params(params[0], params[1], params[2], ggml_flash_attn_ext_get_prec(op)) &&
           attention_shapes(op->src[0], op->src[1], op->src[2], op->src[3], op);
}

void register_qwen4exp_flash_attention_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen4exp.flash_attention_decode",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_attention,
    });
}

}  // namespace ggml::hrx
