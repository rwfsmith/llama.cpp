#include "dispatch-swiglu.h"

#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kSwigluF32Kernel = GGML_HRX_KERNEL_REF("hrx_owned", "ggml_swiglu_split_f32");

static bool swiglu_enabled() {
    const char * flag = std::getenv("HRX_ENABLE_SWIGLU");
    return flag != nullptr && std::strcmp(flag, "1") == 0;
}

static bool storage_covers(const Value & value) {
    return value.storage_offset % sizeof(float) == 0 &&
           value.storage_offset <= value.storage_byte_count &&
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
    return offset % sizeof(float) == 0 && offset <= ggml_nbytes(root) &&
           ggml_nbytes(&value) <= ggml_nbytes(root) - offset;
}

template <typename T> static bool swiglu_layout(const T & value) {
    return value.type == GGML_TYPE_F32 &&
           value.ne[0] >= 1 && value.ne[0] <= 32768 &&
           value.ne[1] >= 1 && value.ne[1] <= 8 && value.ne[2] == 1 && value.ne[3] == 1 &&
           value.nb[0] == sizeof(float) &&
           value.nb[1] == static_cast<size_t>(value.ne[0]) * sizeof(float) &&
           storage_covers(value);
}

template <typename T> static bool swiglu_geometry(const T & gate, const T & up, const T & output) {
    return swiglu_layout(gate) && swiglu_layout(up) && swiglu_layout(output) &&
           gate.ne[0] == output.ne[0] && gate.ne[1] == output.ne[1] &&
           up.ne[0] == output.ne[0] && up.ne[1] == output.ne[1];
}

bool supports_swiglu_f32_dispatch(const ggml_tensor * op) {
    return swiglu_enabled() && op != nullptr && op->op == GGML_OP_GLU &&
           ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU &&
           op->src[0] != nullptr && op->src[1] != nullptr && op->view_src == nullptr &&
           swiglu_geometry(*op->src[0], *op->src[1], *op);
}

static bool match_swiglu_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (!swiglu_enabled() || node == nullptr || node->op != GGML_OP_GLU || node->inputs.size() != 2) {
        return false;
    }
    const auto * params = op_params_as<GluParams>(node->params);
    const auto * gate = context.graph.values().find(node->inputs[0]);
    const auto * up = context.graph.values().find(node->inputs[1]);
    const auto * output = context.graph.values().find(node->output);
    if (params == nullptr || params->op != GGML_GLU_OP_SWIGLU ||
        gate == nullptr || up == nullptr || output == nullptr ||
        !swiglu_geometry(*gate, *up, *output) ||
        output->storage_root != output->id ||
        output->storage == gate->storage || output->storage == up->storage) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSwigluF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    // ggml_glu_split applies SiLU to src[0], including when the two inputs alias.
    dispatch.bindings.push_back({ gate->id, 0, gate->byte_count });
    dispatch.bindings.push_back({ up->id, 0, up->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_swiglu_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({ "common.swiglu_split_f32", GGML_OP_GLU, DispatchMatchKind::SingleOp,
                   0, DispatchSource::Common, match_swiglu_f32 });
}

}  // namespace ggml::hrx
