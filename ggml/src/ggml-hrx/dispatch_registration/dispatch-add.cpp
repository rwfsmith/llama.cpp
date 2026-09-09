#include "dispatch-add.h"
#include "dispatch-copy.h"
#include "dispatch-llm-profiles.h"
#include "dispatch-rmsnorm.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kAddF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_add_f32");

template <typename T> static bool same_shape(const T & lhs, const T & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool contiguous(const Value & value) { return value.contiguous; }
static bool contiguous(const ggml_tensor & value) { return ggml_is_contiguous(&value); }
static int64_t elements(const Value & value) { return value.element_count; }
static int64_t elements(const ggml_tensor & value) { return ggml_nelements(&value); }

static bool storage_covers(const Value & value) {
    return value.storage_offset <= value.storage_byte_count &&
           value.byte_count <= value.storage_byte_count - value.storage_offset;
}

static bool storage_covers(const ggml_tensor & value) {
    const ggml_tensor * root = &value;
    size_t offset = 0;
    while (root->view_src != nullptr) {
        if (root->view_offs > std::numeric_limits<size_t>::max() - offset) {
            return false;
        }
        offset += root->view_offs;
        root = root->view_src;
    }
    const size_t bytes = ggml_nbytes(root);
    return offset <= bytes && ggml_nbytes(&value) <= bytes - offset;
}

static bool enabled(const char * name) {
    const char * flag = std::getenv(name);
    return flag != nullptr && std::strcmp(flag, "1") == 0;
}

template <typename T> static bool row_bias_geometry(const T & a, const T & b, const T & output) {
    const char * flag = std::getenv("HRX_ENABLE_SMALL_BATCH_GLUE");
    return flag != nullptr && std::strcmp(flag, "1") == 0 &&
           a.type == GGML_TYPE_F32 && b.type == GGML_TYPE_F32 && output.type == GGML_TYPE_F32 &&
           same_shape(a, output) && a.ne[0] >= 1 && a.ne[0] <= 32768 &&
           a.ne[1] >= 2 && a.ne[1] <= 8 && a.ne[2] == 1 && a.ne[3] == 1 &&
           b.ne[0] == a.ne[0] && b.ne[1] == 1 && b.ne[2] == 1 && b.ne[3] == 1 &&
           contiguous(a) && contiguous(b) && contiguous(output) &&
           a.nb[0] == sizeof(float) && b.nb[0] == sizeof(float) && output.nb[0] == sizeof(float) &&
           a.nb[1] == static_cast<size_t>(a.ne[0]) * sizeof(float) &&
           output.nb[1] == static_cast<size_t>(a.ne[0]) * sizeof(float);
}

template <typename T> static bool hc_row_layout(const T & value) {
    return value.ne[0] == 2560 && value.ne[1] >= 2 && value.ne[1] <= 8 &&
           qwen4exp_hc_grouped_norm_token_count_supported(value.ne[1]) &&
           value.ne[2] == 1 && value.ne[3] == 1 && value.nb[0] == sizeof(float) &&
           (value.nb[1] == 2560 * sizeof(float) || value.nb[1] == 10240 * sizeof(float));
}

template <typename T> static bool add_geometry(const T & a, const T & b, const T & output) {
    if (row_bias_geometry(a, b, output)) {
        return true;
    }
    if (output.type != GGML_TYPE_F32 || a.type != GGML_TYPE_F32 || b.type != GGML_TYPE_F32 ||
        !same_shape(output, a) || !same_shape(output, b) || !contiguous(output) ||
        elements(output) <= 0 || static_cast<uint64_t>(elements(output)) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (contiguous(a) && contiguous(b)) {
        return true;
    }
    CopyF32Geometry ga, gb;
    const bool hc_compatible = hc_row_layout(a) && hc_row_layout(b) && hc_row_layout(output);
    if (!hc_compatible && (!enabled("HRX_ENABLE_SMALL_BATCH_GLUE") || output.ne[0] <= 0 ||
                         output.ne[0] > 32768 || elements(output) / output.ne[0] < 2 ||
                         elements(output) / output.ne[0] > 8)) {
        return false;
    }
    return storage_covers(a) && storage_covers(b) && storage_covers(output) &&
           copy_f32_geometry(a, output, ga) && copy_f32_geometry(b, output, gb) &&
           ga.row_length == output.ne[0] && gb.row_length == output.ne[0] &&
           ga.row_count >= 2 && ga.row_count <= 8 && gb.row_count == ga.row_count;
}

bool supports_add_f32_dispatch(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_ADD && op->src[0] != nullptr && op->src[1] != nullptr &&
           add_geometry(*op->src[0], *op->src[1], *op);
}

bool is_broadcast_add_candidate(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_ADD && op->src[0] != nullptr && op->src[1] != nullptr &&
           (!same_shape(*op, *op->src[0]) || !same_shape(*op, *op->src[1]));
}

bool is_hc_collapse_add_candidate(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD) {
        return false;
    }
    for (int operand = 0; operand < 2; ++operand) {
        const ggml_tensor * value = op->src[operand];
        if (value == nullptr) {
            continue;
        }
        if (value->ne[0] == 2560 && value->nb[1] == 10240 * sizeof(float)) {
            return true;
        }
        // Retain HC provenance when a malformed stride/width no longer matches.
        while (value->view_src != nullptr) {
            value = value->view_src;
            if (value->ne[0] == 10240 || (value->ne[0] == 2560 && value->ne[1] == 4)) {
                return true;
            }
        }
    }
    return false;
}

struct MoeFoldProof {
    const ggml_tensor * weighted = nullptr;
    uint32_t routes = 0;
};

static bool moe_weighted_rows(const ggml_tensor * weighted) {
    if (weighted == nullptr || weighted->op != GGML_OP_MUL || weighted->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(weighted) || weighted->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * down = weighted->src[0];
    const ggml_tensor * route_weights = weighted->src[1];
    while (down != nullptr && down->view_src != nullptr) {
        down = down->view_src;
    }
    if (down == nullptr || down->op != GGML_OP_MUL_MAT_ID) {
        down = weighted->src[1];
        route_weights = weighted->src[0];
        while (down != nullptr && down->view_src != nullptr) {
            down = down->view_src;
        }
    }
    if (down == nullptr || down->op != GGML_OP_MUL_MAT_ID || down->src[0] == nullptr ||
        down->src[1] == nullptr || down->src[2] == nullptr || route_weights == nullptr ||
        !same_shape(*down, *weighted) || down->type != GGML_TYPE_F32 || !ggml_is_contiguous(down) ||
        route_weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(route_weights) ||
        route_weights->ne[0] != 1 || route_weights->ne[1] != weighted->ne[1] ||
        route_weights->ne[2] != weighted->ne[2] || route_weights->ne[3] != 1) {
        return false;
    }
    const auto * weight = down->src[0];
    const auto * input = down->src[1];
    const auto * ids = down->src[2];
    const bool qwen4 = weighted->ne[0] == 2560 && weighted->ne[1] == 10 &&
        weight->ne[0] == 640 && weight->ne[1] == 2560 && weight->ne[2] == 512 &&
        weighted->ne[2] >= 2 && weighted->ne[2] <= 8 &&
        (weight->type == GGML_TYPE_Q8_0 || weight->type == GGML_TYPE_IQ4_NL) &&
        enabled("HRX_ENABLE_IQ_EXPERTS") && enabled("HRX_ENABLE_MOE_SMALL_BATCH");
    const bool qwen30 = weighted->ne[0] == kQwen30BMoeDispatchProfile.hidden_size &&
        weighted->ne[1] == kQwen30BMoeDispatchProfile.route_count &&
        weight->ne[0] == kQwen30BMoeDispatchProfile.expert_hidden_size &&
        weight->ne[1] == weighted->ne[0] && weight->ne[2] == kQwen30BMoeDispatchProfile.expert_count &&
        is_llm_supported_query_length(kQwen30BMoeDispatchProfile, weighted->ne[2]) &&
        (weight->type == GGML_TYPE_Q4_K || weight->type == GGML_TYPE_Q6_K);
    return (qwen4 || qwen30) && weight->ne[3] == 1 && ggml_is_contiguous(weight) &&
        input->type == GGML_TYPE_F32 && ggml_is_contiguous(input) &&
        input->ne[0] == weight->ne[0] && input->ne[1] == weighted->ne[1] &&
        input->ne[2] == weighted->ne[2] && input->ne[3] == 1 &&
        ids->type == GGML_TYPE_I32 && ids->ne[0] == weighted->ne[1] &&
        ids->ne[1] == weighted->ne[2] && ids->ne[2] == 1 && ids->ne[3] == 1 &&
        ids->nb[0] == sizeof(int32_t) && ids->nb[1] % sizeof(int32_t) == 0 &&
        ids->nb[1] >= static_cast<size_t>(ids->ne[0]) * sizeof(int32_t) &&
        ids->nb[1] <= static_cast<size_t>(weight->ne[2]) * sizeof(int32_t);
}

static MoeFoldProof moe_fold_proof(const ggml_tensor * node, int depth) {
    if (node == nullptr || depth > 16 || node->type != GGML_TYPE_F32 ||
        node->ne[2] != 1 || node->ne[3] != 1) {
        return {};
    }
    if (node->op == GGML_OP_ADD) {
        if (node->src[0] == nullptr || node->src[1] == nullptr || !ggml_is_contiguous(node) ||
            !same_shape(*node, *node->src[0]) || !same_shape(*node, *node->src[1])) {
            return {};
        }
        const auto a = moe_fold_proof(node->src[0], depth + 1);
        const auto b = moe_fold_proof(node->src[1], depth + 1);
        if (a.weighted == nullptr || a.weighted != b.weighted || (a.routes & b.routes) != 0) {
            return {};
        }
        return { a.weighted, a.routes | b.routes };
    }
    if (node->op != GGML_OP_VIEW || node->view_src == nullptr) {
        return {};
    }
    const ggml_tensor * root = node;
    size_t offset = 0;
    while (root->view_src != nullptr) {
        if (root->view_offs > std::numeric_limits<size_t>::max() - offset) {
            return {};
        }
        offset += root->view_offs;
        root = root->view_src;
    }
    if (!moe_weighted_rows(root) || node->ne[0] != root->ne[0] || node->ne[1] != root->ne[2] ||
        node->nb[0] != sizeof(float) || node->nb[1] != root->nb[2] || offset % root->nb[1] != 0) {
        return {};
    }
    const size_t route = offset / root->nb[1];
    if (route >= static_cast<size_t>(root->ne[1]) || route >= 32) {
        return {};
    }
    return { root, uint32_t(1) << route };
}

bool supports_fused_moe_add(const ggml_tensor * op) {
    // Capability sees ancestors, not consumers: prove a disjoint subtree from one
    // routed weighted projection. The down matcher still verifies the complete fold.
    return op != nullptr && op->op == GGML_OP_ADD && moe_fold_proof(op, 0).weighted != nullptr;
}

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool supports_add_f32_dispatch(const Graph & graph, const GraphNode * node) {
    if (node == nullptr || node->op != GGML_OP_ADD || node->inputs.size() != 2) {
        return false;
    }
    const Value * output = graph_value(graph, node->output);
    const Value * a      = graph_value(graph, node->inputs[0]);
    const Value * b      = graph_value(graph, node->inputs[1]);
    if (output == nullptr || a == nullptr || b == nullptr) {
        return false;
    }
    return add_geometry(*a, *b, *output);
}

static bool match_add_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    if (!supports_add_f32_dispatch(context.graph, context.root_node)) {
        return false;
    }
    const Value * output = graph_value(context.graph, context.root_node->output);
    const Value * a      = graph_value(context.graph, context.root_node->inputs[0]);
    const Value * b      = graph_value(context.graph, context.root_node->inputs[1]);
    if (output == nullptr || a == nullptr || b == nullptr) {
        return false;
    }

    if (row_bias_geometry(*a, *b, *output)) {
        // Freeze both operands before writing any token: output can alias the
        // activation or bias storage, including offsets crossing token boundaries.
        const Value * operands[] = { a, b };
        const ValueId input_copy(context.next_plan_value.value);
        const ValueId bias_copy(context.next_plan_value.value + 1);
        for (int i = 0; i < 2; ++i) {
            const ValueId packed(context.next_plan_value.value + i);
            CopyF32Geometry geometry;
            if (!copy_f32_geometry(*operands[i], *operands[i], geometry)) {
                return false;
            }
            match.transients.push_back({ packed, i == 0 ? "row_bias.input" : "row_bias.bias",
                                         operands[i]->byte_count, 256 });
            match.dispatches.push_back(make_copy_f32_dispatch(
                geometry, operands[i]->id, operands[i]->byte_count, packed, operands[i]->byte_count));
        }
        const size_t row_bytes = static_cast<size_t>(a->ne[0]) * sizeof(float);
        for (int64_t token = 0; token < a->ne[1]; ++token) {
            Dispatch dispatch;
            dispatch.kernel = make_kernel_specialization(kAddF32Kernel);
            dispatch.kernel.integer_parameters.emplace("element_count", a->ne[0]);
            const size_t offset = static_cast<size_t>(token) * row_bytes;
            dispatch.bindings.push_back({ input_copy, offset, row_bytes });
            dispatch.bindings.push_back({ bias_copy, 0, row_bytes });
            dispatch.bindings.push_back({ output->id, offset, row_bytes });
            match.dispatches.push_back(std::move(dispatch));
        }
        match.covered_nodes.push_back(context.root_index);
        return true;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kAddF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    if (!a->contiguous || !b->contiguous) {
        // Snapshot both operands before any destination write, including a contiguous
        // operand that partially overlaps the output or the other stream view.
        const Value * operands[] = { a, b };
        for (int i = 0; i < 2; ++i) {
            const ValueId packed(context.next_plan_value.value + i);
            CopyF32Geometry geometry;
            if (!copy_f32_geometry(*operands[i], *output, geometry)) {
                return false;
            }
            match.transients.push_back({ packed, i == 0 ? "row_add.lhs" : "row_add.rhs", output->byte_count, 256 });
            match.dispatches.push_back(make_copy_f32_dispatch(
                geometry, operands[i]->id, operands[i]->byte_count, packed, output->byte_count));
            dispatch.bindings.push_back({ packed, 0, output->byte_count });
        }
    } else {
        dispatch.bindings.push_back({ a->id, 0, a->byte_count });
        dispatch.bindings.push_back({ b->id, 0, b->byte_count });
    }
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_add_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.add_f32",
        GGML_OP_ADD,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_add_f32_dispatch,
    });
}

}  // namespace ggml::hrx
