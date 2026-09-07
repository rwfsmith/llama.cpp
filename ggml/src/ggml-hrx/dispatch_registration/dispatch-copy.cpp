#include "dispatch-copy.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kCopyF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_copy_f32");
static constexpr KernelCatalogRef kCopyRowsF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_copy_rows_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

// ggml_cpy() builds its node as a view of the destination: src[0] is the source, src[1] is the
// destination, and the node's own output aliases the destination's storage. Writing the output
// therefore writes the destination in place, so only the source needs a separate binding. ggml_cont()
// is the same operation with an implicit, freshly allocated contiguous destination and a single input.
static bool copy_like_node(const GraphNode * node) {
    if (node == nullptr) {
        return false;
    }
    if (node->op == GGML_OP_CPY) {
        return node->inputs.size() == 2;
    }
    return node->op == GGML_OP_CONT && node->inputs.size() == 1;
}

// HRX implements the same-length contiguous f32 case. That is what llama.cpp's build_rs() needs to
// carry recurrent state slots forward, and it must run on HRX because the recurrent state cache is
// pre-allocated there -- ggml_backend_sched aborts rather than falling back when a pre-allocated
// tensor's backend declines its op. Type-converting copies have no kernel and are declined.
static bool supports_copy_f32_dispatch(const Graph & graph, const GraphNode * node) {
    if (!copy_like_node(node)) {
        return false;
    }
    const Value * output = graph_value(graph, node->output);
    const Value * source = graph_value(graph, node->inputs[0]);
    if (output == nullptr || source == nullptr) {
        return false;
    }
    return output->type == GGML_TYPE_F32 && source->type == GGML_TYPE_F32 && output->contiguous &&
           source->contiguous && output->element_count == source->element_count && output->element_count > 0 &&
           static_cast<uint64_t>(output->element_count) <= std::numeric_limits<uint32_t>::max();
}

static bool match_copy_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    if (!supports_copy_f32_dispatch(context.graph, context.root_node)) {
        return false;
    }
    const Value * output = graph_value(context.graph, context.root_node->output);
    const Value * source = graph_value(context.graph, context.root_node->inputs[0]);
    if (output == nullptr || source == nullptr) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kCopyF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.bindings.push_back({ source->id, 0, source->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

// The same copy through a strided view. llama.cpp's GDN conv-prepare writes the rolling convolution
// window back into the conv-state cache with ggml_cpy() from a column slice of the CONCAT output, which
// is row-strided rather than contiguous. Declining it does not merely cost one kernel: the CPY sits
// between the CONCAT and the SSM_CONV in graph order, so a CPU fallback splits the conv-prepare chain
// across backends and qwen4exp.gdn_conv_prepare_decode (rooted at SSM_CONV) can never cover the CONCAT.
// Rows must be element-contiguous (nb[0] == sizeof(float)) and 2D; anything else stays on the CPU.
static bool copy_rows_geometry(const Value & value, int64_t & row_length, int64_t & row_stride) {
    if (value.type != GGML_TYPE_F32 || value.ne[2] != 1 || value.ne[3] != 1) {
        return false;
    }
    if (value.nb[0] != sizeof(float) || value.nb[1] % sizeof(float) != 0) {
        return false;
    }
    row_length = value.ne[0];
    row_stride = static_cast<int64_t>(value.nb[1] / sizeof(float));
    return row_length > 0 && row_stride >= row_length;
}

static bool match_copy_rows_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (!copy_like_node(node)) {
        return false;
    }
    const Value * output = graph_value(context.graph, node->output);
    const Value * source = graph_value(context.graph, node->inputs[0]);
    if (output == nullptr || source == nullptr) {
        return false;
    }
    // The contiguous case has its own, cheaper kernel.
    if (output->contiguous && source->contiguous) {
        return false;
    }
    int64_t source_row_length = 0;
    int64_t source_row_stride = 0;
    int64_t output_row_length = 0;
    int64_t output_row_stride = 0;
    if (!copy_rows_geometry(*source, source_row_length, source_row_stride) ||
        !copy_rows_geometry(*output, output_row_length, output_row_stride)) {
        return false;
    }
    if (source_row_length != output_row_length || source->ne[1] != output->ne[1] ||
        source->element_count != output->element_count || source->element_count <= 0 ||
        static_cast<uint64_t>(source->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kCopyRowsF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", source->element_count);
    dispatch.kernel.integer_parameters.emplace("row_length", source_row_length);
    dispatch.kernel.integer_parameters.emplace("source_row_stride", source_row_stride);
    dispatch.kernel.integer_parameters.emplace("output_row_stride", output_row_stride);
    dispatch.bindings.push_back({ source->id, 0, source->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_copy_dispatch(DispatchRegistryBuilder & registry) {
    for (const ggml_op root_op : { GGML_OP_CPY, GGML_OP_CONT }) {
        registry.add({
            "common.copy_f32",
            root_op,
            DispatchMatchKind::SingleOp,
            0,
            DispatchSource::Common,
            match_copy_f32_dispatch,
        });
        registry.add({
            "common.copy_rows_f32",
            root_op,
            DispatchMatchKind::SingleOp,
            0,
            DispatchSource::Common,
            match_copy_rows_f32_dispatch,
        });
    }
}

}  // namespace ggml::hrx
