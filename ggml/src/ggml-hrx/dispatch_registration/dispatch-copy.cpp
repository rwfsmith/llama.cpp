#include "dispatch-copy.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kCopyF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_copy_f32");
static constexpr KernelCatalogRef kCopyRowsF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_copy_rows_f32");
static constexpr KernelCatalogRef kSetRowsKernel = GGML_HRX_KERNEL_REF("hrx_owned", "ggml_set_rows_f32");

bool is_set_rows_kernel(uint64_t kernel_id) {
    return kernel_id == kSetRowsKernel.id;
}

static const ggml_tensor * storage_root(const ggml_tensor * tensor) {
    while (tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static size_t storage_offset(const ggml_tensor * tensor) {
    size_t offset = 0;
    while (tensor->view_src != nullptr) {
        offset += tensor->view_offs;
        tensor = tensor->view_src;
    }
    return offset;
}

static bool row_write_geometry(const ggml_tensor * tensor, size_t element_size) {
    if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
        tensor->nb[0] != element_size || tensor->nb[1] % element_size != 0) {
        return false;
    }
    const size_t width = static_cast<size_t>(tensor->ne[0]);
    const size_t rows = static_cast<size_t>(tensor->ne[1]);
    const size_t stride = tensor->nb[1] / element_size;
    return width <= 512 && stride >= width && stride <= kCopyF32MaxElements &&
           rows - 1 <= (kCopyF32MaxElements - width) / stride;
}

bool supports_set_rows_dispatch(const ggml_tensor * op) {
    const char * enabled = std::getenv("HRX_ENABLE_SET_ROWS");
    if (enabled == nullptr || enabled[0] == '\0' || enabled[0] == '0' ||
        op == nullptr || op->op != GGML_OP_SET_ROWS ||
        op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr) {
        return false;
    }
    const ggml_tensor * source = op->src[0];
    const ggml_tensor * ids = op->src[1];
    const ggml_tensor * destination = op->src[2];
    if (source->type != GGML_TYPE_F32 || (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
        destination->type != op->type || (ids->type != GGML_TYPE_I32 && ids->type != GGML_TYPE_I64) ||
        !row_write_geometry(source, sizeof(float)) || !row_write_geometry(op, ggml_type_size(op->type))) {
        return false;
    }
    // Only the flattened qwen4exp KV, raw indexer, and scalar cache rows are enabled.
    const int64_t width = source->ne[0];
    if ((width != 1 && width != 128 && width != 256 && width != 512) ||
        width != op->ne[0] || source->ne[1] > 4096 ||
        ids->ne[0] != source->ne[1] || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1) {
        return false;
    }
    const size_t index_size = ggml_type_size(ids->type);
    if (ids->nb[0] < index_size || ids->nb[0] % index_size != 0 ||
        ids->nb[0] / index_size > kCopyF32MaxElements ||
        static_cast<size_t>(ids->ne[0] - 1) > (kCopyF32MaxElements - 1) / (ids->nb[0] / index_size)) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (op->ne[d] != destination->ne[d] || op->nb[d] != destination->nb[d]) {
            return false;
        }
    }
    return op->view_src != nullptr && storage_root(op) == storage_root(destination) &&
           storage_offset(op) == storage_offset(destination) &&
           storage_root(op) != storage_root(source) && storage_root(op) != storage_root(ids);
}

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
template<class T>
static bool copy_element_count(const T & value, int64_t & count) {
    if (value.type != GGML_TYPE_F32) {
        return false;
    }
    count = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (value.ne[d] <= 0 || value.ne[d] > kCopyF32MaxElements / count) {
            return false;
        }
        count *= value.ne[d];
    }
    return true;
}

static bool copy_contiguous(const Value & value) { return value.contiguous; }
static bool copy_contiguous(const ggml_tensor & value) { return ggml_is_contiguous(&value); }
static size_t copy_bytes(const Value & value) { return value.byte_count; }
static size_t copy_bytes(const ggml_tensor & value) { return ggml_nbytes(&value); }
static int64_t copy_elements(const Value & value) { return value.element_count; }
static int64_t copy_elements(const ggml_tensor & value) { return ggml_nelements(&value); }

template<class T>
static bool copy_rows_geometry(const T & value, int64_t count, int64_t & stride) {
    if (value.nb[0] != sizeof(float) || value.nb[1] % sizeof(float) != 0 ||
        value.nb[1] / sizeof(float) < static_cast<uint64_t>(value.ne[0]) ||
        value.nb[1] / sizeof(float) > static_cast<uint64_t>(kCopyF32MaxElements)) {
        return false;
    }
    stride = static_cast<int64_t>(value.nb[1] / sizeof(float));
    size_t expected_stride = value.nb[1];
    for (int d = 2; d < GGML_MAX_DIMS; ++d) {
        if (static_cast<uint64_t>(value.ne[d - 1]) >
            std::numeric_limits<size_t>::max() / expected_stride) {
            return false;
        }
        expected_stride *= static_cast<size_t>(value.ne[d - 1]);
        // Singleton axes are never traversed and may carry arbitrary strides.
        if (value.ne[d] > 1 && value.nb[d] != expected_stride) {
            return false;
        }
    }
    const size_t rows = static_cast<size_t>(count / value.ne[0]);
    const size_t row_bytes = static_cast<size_t>(value.ne[0]) * sizeof(float);
    if (rows - 1 > (std::numeric_limits<size_t>::max() - row_bytes) / value.nb[1]) {
        return false;
    }
    return (rows - 1) * value.nb[1] + row_bytes <= copy_bytes(value);
}

template<class T>
static bool copy_f32_geometry_impl(const T & source, const T & output, CopyF32Geometry & geometry) {
    geometry = {};
    int64_t count = 0, output_count = 0;
    if (!copy_element_count(source, count) || !copy_element_count(output, output_count) ||
        count != output_count || count != copy_elements(source) || count != copy_elements(output)) {
        return false;
    }
    CopyF32Geometry candidate;
    candidate.element_count = count;
    candidate.row_length = source.ne[0];
    candidate.row_count = count / source.ne[0];
    if (copy_contiguous(source) && copy_contiguous(output)) {
        const size_t bytes = static_cast<size_t>(count) * sizeof(float);
        if (bytes > copy_bytes(source) || bytes > copy_bytes(output)) {
            return false;
        }
        candidate.contiguous = true;
        candidate.source_row_stride = candidate.output_row_stride = candidate.row_length;
    } else {
        if (!copy_rows_geometry(source, count, candidate.source_row_stride)) {
            return false;
        }
        if (copy_contiguous(output)) {
            if (static_cast<size_t>(count) * sizeof(float) > copy_bytes(output)) {
                return false;
            }
            candidate.output_row_stride = candidate.row_length;
        } else if (output.ne[0] != source.ne[0] ||
                   !copy_rows_geometry(output, count, candidate.output_row_stride)) {
            return false;
        }
    }
    geometry = candidate;
    return true;
}

bool copy_f32_geometry(const Value & source, const Value & output, CopyF32Geometry & geometry) {
    return copy_f32_geometry_impl(source, output, geometry);
}

bool copy_f32_geometry(const ggml_tensor & source, const ggml_tensor & output, CopyF32Geometry & geometry) {
    return copy_f32_geometry_impl(source, output, geometry);
}

bool supports_copy_f32_dispatch(const ggml_tensor * op) {
    if (!op || !op->src[0] ||
        (op->op != GGML_OP_CONT && (op->op != GGML_OP_CPY || !op->src[1]))) {
        return false;
    }
    CopyF32Geometry geometry;
    return copy_f32_geometry(*op->src[0], *op, geometry);
}

Dispatch make_copy_f32_dispatch(const CopyF32Geometry & geometry, ValueId source, size_t source_bytes,
                               ValueId output, size_t output_bytes) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(geometry.contiguous ? kCopyF32Kernel : kCopyRowsF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", geometry.element_count);
    if (!geometry.contiguous) {
        dispatch.kernel.integer_parameters.emplace("row_length", geometry.row_length);
        dispatch.kernel.integer_parameters.emplace("source_row_stride", geometry.source_row_stride);
        dispatch.kernel.integer_parameters.emplace("output_row_stride", geometry.output_row_stride);
    }
    dispatch.bindings.push_back({ source, 0, source_bytes });
    dispatch.bindings.push_back({ output, 0, output_bytes });
    return dispatch;
}

static bool match_copy_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    if (!copy_like_node(context.root_node)) {
        return false;
    }
    const Value * output = graph_value(context.graph, context.root_node->output);
    const Value * source = graph_value(context.graph, context.root_node->inputs[0]);
    CopyF32Geometry geometry;
    if (output == nullptr || source == nullptr || !copy_f32_geometry(*source, *output, geometry) ||
        !geometry.contiguous) {
        return false;
    }

    Dispatch dispatch = make_copy_f32_dispatch(geometry, source->id, source->byte_count, output->id, output->byte_count);

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

// The same copy through a strided view. llama.cpp's GDN conv-prepare writes the rolling convolution
// window back into the conv-state cache with ggml_cpy() from a column slice of the CONCAT output, which
// is row-strided rather than contiguous. Declining it does not merely cost one kernel: the CPY sits
// between the CONCAT and the SSM_CONV in graph order, so a CPU fallback splits the conv-prepare chain
// across backends and qwen4exp.gdn_conv_prepare_decode (rooted at SSM_CONV) can never cover the CONCAT.
// QSA also copies [256,24,T] query/gate views: uniformly spaced outer rows can
// use exactly the same kernel by flattening ne[1..3] into a single row count.

static bool match_copy_rows_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (!copy_like_node(node)) {
        return false;
    }
    const Value * output = graph_value(context.graph, node->output);
    const Value * source = graph_value(context.graph, node->inputs[0]);
    CopyF32Geometry geometry;
    if (output == nullptr || source == nullptr || !copy_f32_geometry(*source, *output, geometry) ||
        geometry.contiguous) {
        return false;
    }

    Dispatch dispatch = make_copy_f32_dispatch(geometry, source->id, source->byte_count, output->id, output->byte_count);

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_set_rows_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_SET_ROWS || node->inputs.size() != 3) {
        return false;
    }
    const Value * output = graph_value(context.graph, node->output);
    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * ids = graph_value(context.graph, node->inputs[1]);
    const Value * destination = graph_value(context.graph, node->inputs[2]);
    if (output == nullptr || source == nullptr || ids == nullptr || destination == nullptr ||
        !supports_set_rows_dispatch(output->tensor) || output->storage != destination->storage ||
        output->storage_offset != destination->storage_offset) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSetRowsKernel);
    dispatch.kernel.integer_parameters = {
        { "row_length", source->ne[0] },
        { "row_count", source->ne[1] },
        { "cache_rows", output->ne[1] },
        { "source_stride", static_cast<int64_t>(source->nb[1] / sizeof(float)) },
        { "output_stride", static_cast<int64_t>(output->nb[1] / ggml_type_size(output->type)) },
        { "index_stride", static_cast<int64_t>(ids->nb[0] / ggml_type_size(ids->type)) },
        { "output_f16", output->type == GGML_TYPE_F16 ? 1 : 0 },
        { "index_i64", ids->type == GGML_TYPE_I64 ? 1 : 0 },
    };
    dispatch.bindings.push_back({ source->id, 0, source->byte_count });
    dispatch.bindings.push_back({ ids->id, 0, ids->byte_count });
    // SET_ROWS aliases src[2]. ReadWrite preserves all rows and padding not selected by IDs.
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_copy_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.set_rows_f32",
        GGML_OP_SET_ROWS,
        DispatchMatchKind::SingleOp,
        -100,
        DispatchSource::Common,
        match_set_rows_dispatch,
    });
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
