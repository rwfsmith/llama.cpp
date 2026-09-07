#include "dispatch-scale.h"

#include "ggml.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kZeroF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_zero_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

// HRX only implements the zero-fill form of SCALE (dst = a*0 + 0). That is the one llama.cpp applies
// to a pre-allocated buffer: build_rs() clears a recurrent state slot with ggml_scale_inplace(s, 0),
// and ggml_backend_sched aborts outright rather than falling back when the buffer's backend declines
// an op on a pre-allocated tensor. Every other scale factor is left to the CPU, where it is an
// ordinary schedulable node.
static bool zero_fill_scale(const GraphNode & node) {
    const ScaleParams * params = op_params_as<ScaleParams>(node.params);
    return params != nullptr && params->scale == 0.0f && params->bias == 0.0f;
}

static bool supports_zero_f32_dispatch(const Graph & graph, const GraphNode * node) {
    if (node == nullptr || node->op != GGML_OP_SCALE || node->inputs.size() != 1 || !zero_fill_scale(*node)) {
        return false;
    }
    const Value * output = graph_value(graph, node->output);
    if (output == nullptr) {
        return false;
    }
    return output->type == GGML_TYPE_F32 && output->contiguous && output->element_count > 0 &&
           static_cast<uint64_t>(output->element_count) <= std::numeric_limits<uint32_t>::max();
}

static bool match_zero_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    if (!supports_zero_f32_dispatch(context.graph, context.root_node)) {
        return false;
    }
    const Value * output = graph_value(context.graph, context.root_node->output);
    if (output == nullptr) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kZeroF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_scale_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.zero_f32",
        GGML_OP_SCALE,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_zero_f32_dispatch,
    });
}

}  // namespace ggml::hrx
