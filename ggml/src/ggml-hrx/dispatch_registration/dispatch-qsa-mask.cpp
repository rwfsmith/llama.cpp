#include "dispatch-qsa-mask.h"

#include "ggml-impl.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdlib>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr auto kZero = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_zero_f32");
static constexpr auto kFill = GGML_HRX_KERNEL_REF("hrx_owned", "ggml_qsa_mask_fill");
static constexpr auto kCast = GGML_HRX_KERNEL_REF("hrx_owned", "ggml_qsa_mask_cast");
static constexpr auto kAdd = GGML_HRX_KERNEL_REF("hrx_owned", "ggml_qsa_mask_add_f16");

static bool dense_aligned(const ggml_tensor * value) {
    return value != nullptr && (value->type == GGML_TYPE_F32 || value->type == GGML_TYPE_F16) &&
           ggml_is_contiguous(value) && value->view_offs % ggml_type_size(value->type) == 0;
}

static bool mask_shape(const ggml_tensor * value) {
    return dense_aligned(value) && value->ne[0] >= 256 && value->ne[0] <= 131072 &&
           value->ne[0] % 256 == 0 && value->ne[1] >= 1 && value->ne[1] <= 32 &&
           value->ne[2] == 1 && value->ne[3] == 1;
}

static const ggml_tensor * storage_root(const ggml_tensor * value) {
    while (value->view_src != nullptr) {
        value = value->view_src;
    }
    return value;
}

static bool fill_shape(const ggml_tensor * value) {
    const char * set_rows = std::getenv("HRX_ENABLE_SET_ROWS");
    // The mask becomes an in-place SET_ROWS destination. Unsupported multi-query
    // scatters run on CPU and cannot write a device-only FILL allocation.
    const bool device_scatter = set_rows != nullptr && set_rows[0] != '\0' && set_rows[0] != '0';
    return (mask_shape(value) && value->ne[1] == 1 && device_scatter) ||
           (dense_aligned(value) && value->type == GGML_TYPE_F32 && value->ne[0] == 1 &&
            value->ne[1] >= 1 && value->ne[1] <= 4096 &&
            value->ne[2] >= 1 && value->ne[2] <= 32 && value->ne[3] == 1);
}

static bool match_qsa_mask(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const Value * output = context.graph.values().find(node->output);
    if (output == nullptr || !supports_qsa_mask_dispatch(output->tensor)) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    if (node->op == GGML_OP_FILL) {
        const auto * params = op_params_as<FillParams>(node->params);
        if (params == nullptr || node->inputs.size() != 1 ||
            params->value_bits != static_cast<uint32_t>(ggml_get_op_params_i32(output->tensor, 0))) {
            return false;
        }
        // Reuse the existing zero kernel; FILL's source supplies shape, not data.
        dispatch.kernel.kernel_id = (output->type == GGML_TYPE_F32 && params->value_bits == 0 ? kZero : kFill).id;
        if (dispatch.kernel.kernel_id == kFill.id) {
            const uint32_t bits = output->type == GGML_TYPE_F32 ? params->value_bits :
                params->value_bits == 0xff800000u ? 0xfc00u : params->value_bits >> 16;
            dispatch.kernel.integer_parameters.emplace("element_bytes", ggml_type_size(output->type));
            dispatch.kernel.integer_parameters.emplace("value_bits", bits);
        }
    } else {
        if (node->inputs.size() != 2) {
            return false;
        }
        const Value * a = context.graph.values().find(node->inputs[0]);
        const Value * b = context.graph.values().find(node->inputs[1]);
        if (a == nullptr || b == nullptr) {
            return false;
        }
        dispatch.bindings.push_back({ a->id, 0, a->byte_count });
        if (node->op == GGML_OP_CPY) {
            dispatch.kernel.kernel_id = kCast.id;
            dispatch.kernel.integer_parameters.emplace("source_f16", a->type == GGML_TYPE_F16 ? 1 : 0);
        } else {
            dispatch.kernel.kernel_id = kAdd.id;
            dispatch.kernel.integer_parameters.emplace("rhs_count", b->element_count);
            dispatch.bindings.push_back({ b->id, 0, b->byte_count });
        }
    }
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

bool supports_qsa_mask_dispatch(const ggml_tensor * op) {
    const char * enabled = std::getenv("HRX_ENABLE_QSA_MASK");
    if (enabled == nullptr || enabled[0] != '1' || enabled[1] != '\0' || op == nullptr) {
        return false;
    }
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (op->op == GGML_OP_FILL) {
        if (!fill_shape(op) || !dense_aligned(a) || op->view_src != nullptr ||
            op->type != a->type || !ggml_are_same_shape(op, a)) {
            return false;
        }
        const uint32_t bits = static_cast<uint32_t>(ggml_get_op_params_i32(op, 0));
        return bits == 0 || bits == 0x80000000u || bits == 0xff800000u;
    }
    if (!mask_shape(op) || !mask_shape(a)) {
        return false;
    }
    if (op->op == GGML_OP_CPY) {
        // ggml_cast uses src[1]==op; ggml_cpy instead aliases its destination.
        return mask_shape(b) && op->type != a->type && op->type == b->type &&
               ggml_are_same_shape(op, a) && ggml_are_same_shape(op, b) &&
               ggml_are_same_stride(op, b) && storage_root(op) == storage_root(b) &&
               op->view_offs == b->view_offs && storage_root(op) != storage_root(a);
    }
    return op->op == GGML_OP_ADD && op->view_src == nullptr && mask_shape(b) &&
           op->type == GGML_TYPE_F16 && a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16 &&
           ggml_are_same_shape(op, a) && b->ne[0] == op->ne[0] &&
           (b->ne[1] == 1 || b->ne[1] == op->ne[1]);
}

void register_qsa_mask_dispatches(DispatchRegistryBuilder & registry) {
    for (ggml_op op : { GGML_OP_FILL, GGML_OP_CPY, GGML_OP_ADD }) {
        registry.add({ "qwen4exp.qsa_mask", op, DispatchMatchKind::SingleOp, 100,
                       DispatchSource::Qwen, match_qsa_mask });
    }
}

}  // namespace ggml::hrx
