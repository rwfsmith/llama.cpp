#include "dispatch-gather-add.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <cstdlib>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kGatherAddF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_gather_add_f32");
static constexpr KernelCatalogRef kGetRowsF32Kernel = GGML_HRX_KERNEL_REF("hrx_owned", "ggml_get_rows_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_2d_f32(const Value & value) {
    return value.type == GGML_TYPE_F32 && value.contiguous && value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 &&
           value.ne[3] == 1;
}

static bool is_row_id_tensor(const Value & value, int64_t output_token_count) {
    return value.type == GGML_TYPE_I32 && value.contiguous && value.element_count == output_token_count &&
           output_token_count > 0;
}

static bool is_supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

struct GatherAddMatch {
    const GraphNode * first_get_rows        = nullptr;
    const GraphNode * second_get_rows       = nullptr;
    const GraphNode * add_node              = nullptr;
    const Value *     first_source          = nullptr;
    const Value *     second_source         = nullptr;
    const Value *     row_ids               = nullptr;
    const Value *     output                = nullptr;
    size_t            first_get_rows_index  = 0;
    size_t            second_get_rows_index = 0;
    size_t            add_node_index        = 0;
    int64_t           source_token_count    = 0;
    int64_t           output_token_count    = 0;
    int64_t           hidden_size           = 0;

    bool matched() const {
        return first_get_rows != nullptr && second_get_rows != nullptr && add_node != nullptr &&
               first_source != nullptr && second_source != nullptr && row_ids != nullptr && output != nullptr;
    }
};

static const GraphNode * find_single_add_consumer(const Graph & graph, const GraphNode & get_rows) {
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(get_rows.output);
    if (consumers.size() != 1) {
        return nullptr;
    }
    const GraphNode * add = consumers.front();
    return add != nullptr && add->op == GGML_OP_ADD && add->inputs.size() == 2 ? add : nullptr;
}

static const GraphNode * peer_get_rows_input(const Graph &     graph,
                                             const GraphNode & add,
                                             const GraphNode & root_get_rows) {
    ValueId peer_output;
    if (add.inputs[0] == root_get_rows.output) {
        peer_output = add.inputs[1];
    } else if (add.inputs[1] == root_get_rows.output) {
        peer_output = add.inputs[0];
    } else {
        return nullptr;
    }

    const GraphNode * peer = graph.index().producer(peer_output);
    return peer != nullptr && peer->op == GGML_OP_GET_ROWS && peer->inputs.size() == 2 ? peer : nullptr;
}

static GatherAddMatch match_gather_add_f32(const Graph & graph, const GraphNode * node, size_t node_index) {
    GatherAddMatch match;
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const GraphNode * add_node = find_single_add_consumer(graph, *node);
    const GraphNode * peer     = add_node != nullptr ? peer_get_rows_input(graph, *add_node, *node) : nullptr;
    if (add_node == nullptr || peer == nullptr) {
        return {};
    }

    size_t peer_index = 0;
    size_t add_index  = 0;
    if (!graph.index().node_index(peer, peer_index) || !graph.index().node_index(add_node, add_index)) {
        return {};
    }

    const Value * first_source  = graph_value(graph, node->inputs[0]);
    const Value * first_ids     = graph_value(graph, node->inputs[1]);
    const Value * first_output  = graph_value(graph, node->output);
    const Value * second_source = graph_value(graph, peer->inputs[0]);
    const Value * second_ids    = graph_value(graph, peer->inputs[1]);
    const Value * second_output = graph_value(graph, peer->output);
    const Value * output        = graph_value(graph, add_node->output);
    if (first_source == nullptr || first_ids == nullptr || first_output == nullptr || second_source == nullptr ||
        second_ids == nullptr || second_output == nullptr || output == nullptr) {
        return {};
    }
    if (node->inputs[1] != peer->inputs[1]) {
        return {};
    }
    if (!is_2d_f32(*first_source) || !is_2d_f32(*second_source) || !is_2d_f32(*first_output) ||
        !is_2d_f32(*second_output) || !is_2d_f32(*output)) {
        return {};
    }
    if (!same_shape(*first_source, *second_source) || !same_shape(*first_output, *second_output) ||
        !same_shape(*first_output, *output)) {
        return {};
    }

    const int64_t hidden_size        = first_source->ne[0];
    const int64_t source_token_count = first_source->ne[1];
    const int64_t output_token_count = first_output->ne[1];
    if (first_output->ne[0] != hidden_size || !is_row_id_tensor(*first_ids, output_token_count) ||
        !is_row_id_tensor(*second_ids, output_token_count)) {
        return {};
    }
    if (!is_supported_hidden_size(hidden_size) || !is_supported_token_count(source_token_count) ||
        !is_supported_token_count(output_token_count)) {
        return {};
    }

    match.first_get_rows        = node;
    match.second_get_rows       = peer;
    match.add_node              = add_node;
    match.first_source          = first_source;
    match.second_source         = second_source;
    match.row_ids               = first_ids;
    match.output                = output;
    match.first_get_rows_index  = node_index;
    match.second_get_rows_index = peer_index;
    match.add_node_index        = add_index;
    match.source_token_count    = source_token_count;
    match.output_token_count    = output_token_count;
    match.hidden_size           = hidden_size;
    return match;
}

}  // namespace

bool is_f32_get_rows_kernel(uint64_t kernel_id) {
    return kernel_id == kGetRowsF32Kernel.id;
}

bool supports_f32_get_rows_dispatch(const ggml_tensor * op) {
    const char * enabled = std::getenv("HRX_ENABLE_F32_GET_ROWS");
    if (enabled == nullptr || enabled[0] == '\0' || enabled[0] == '0' ||
        op == nullptr || op->op != GGML_OP_GET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * source = op->src[0];
    const ggml_tensor * ids = op->src[1];
    if (source->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32 ||
        source->ne[2] != 1 || source->ne[3] != 1 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        op->ne[2] != 1 || op->ne[3] != 1 || op->view_src != nullptr ||
        !ggml_is_contiguous(source) || !ggml_is_contiguous(ids) || !ggml_is_contiguous(op) ||
        source->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t) || op->nb[0] != sizeof(float) ||
        source->view_offs % sizeof(float) != 0 || ids->view_offs % sizeof(int32_t) != 0) {
        return false;
    }
    const int64_t width = source->ne[0];
    const int64_t capacity = source->ne[1];
    const int64_t count = ids->ne[0];
    return width > 0 && width <= kF32GetRowsMaxWidth &&
           capacity > 0 && capacity <= kF32GetRowsMaxRows && count > 0 && count <= kF32GetRowsMaxRows &&
           width <= kF32GetRowsMaxElements / capacity && width <= kF32GetRowsMaxElements / count &&
           op->ne[0] == width && op->ne[1] == count;
}

static bool match_get_rows_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return false;
    }
    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * ids = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (source == nullptr || ids == nullptr || output == nullptr ||
        !supports_f32_get_rows_dispatch(output->tensor) ||
        source->storage == output->storage || ids->storage == output->storage) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsF32Kernel);
    dispatch.kernel.integer_parameters = {
        { "hidden_size", source->ne[0] },
        { "vocabulary_count", source->ne[1] },
        { "token_count", ids->ne[0] },
    };
    dispatch.bindings.push_back({ source->id, 0, source->byte_count });
    dispatch.bindings.push_back({ ids->id, 0, ids->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_gather_add_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GatherAddMatch gather_add = match_gather_add_f32(context.graph, context.root_node, context.root_index);
    if (!gather_add.matched() || gather_add.first_get_rows_index >= context.covered_nodes.size() ||
        gather_add.second_get_rows_index >= context.covered_nodes.size() ||
        gather_add.add_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[gather_add.first_get_rows_index] ||
        context.covered_nodes[gather_add.second_get_rows_index] || context.covered_nodes[gather_add.add_node_index]) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGatherAddF32Kernel);
    dispatch.kernel.integer_parameters.emplace("source_token_count", gather_add.source_token_count);
    dispatch.kernel.integer_parameters.emplace("output_token_count", gather_add.output_token_count);
    dispatch.kernel.integer_parameters.emplace("hidden_size", gather_add.hidden_size);
    dispatch.bindings.push_back({ gather_add.first_source->id, 0, gather_add.first_source->byte_count });
    dispatch.bindings.push_back({ gather_add.second_source->id, 0, gather_add.second_source->byte_count });
    dispatch.bindings.push_back({ gather_add.row_ids->id, 0, gather_add.row_ids->byte_count });
    dispatch.bindings.push_back({ gather_add.output->id, 0, gather_add.output->byte_count });

    match.covered_nodes.push_back(gather_add.first_get_rows_index);
    match.covered_nodes.push_back(gather_add.second_get_rows_index);
    match.covered_nodes.push_back(gather_add.add_node_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_gather_add_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.get_rows_f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        -100,
        DispatchSource::Common,
        match_get_rows_f32_dispatch,
    });
    registry.add({
        "common.gather_add_f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Common,
        match_gather_add_f32_dispatch,
    });
}

}  // namespace ggml::hrx
