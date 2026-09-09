#include "dispatch-qwen-preamble.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kQ8EmbeddingKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_token_embedding_q8_0_f32");
static constexpr KernelCatalogRef kQ4EmbeddingKernel =
    GGML_HRX_KERNEL_REF("hrx_owned", "ggml_token_embedding_q4_k_f32");

bool is_q8_embedding_kernel(uint64_t kernel_id) {
    return kernel_id == kQ8EmbeddingKernel.id;
}

bool is_q4_embedding_kernel(uint64_t kernel_id) {
    return kernel_id == kQ4EmbeddingKernel.id;
}

static bool supports_raw_embedding_dispatch(const ggml_tensor * op, ggml_type type, const char * flag) {
    const char * enabled = std::getenv(flag);
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0 ||
        op == nullptr || op->op != GGML_OP_GET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * ids = op->src[1];
    if (weight->type != type || ids->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32 ||
        weight->op != GGML_OP_NONE || weight->view_src != nullptr ||
        ids->view_src != nullptr || op->view_src != nullptr || op == weight || op == ids) {
        return false;
    }
    const int64_t width = weight->ne[0];
    const int64_t vocabulary = weight->ne[1];
    const int64_t tokens = ids->ne[0];
    const bool q4 = type == GGML_TYPE_Q4_K;
    if ((q4 ? width != 2560 : (width < 32 || width > 32768 || width % 32 != 0)) ||
        vocabulary < 1 || vocabulary > 262144 || tokens < 1 || tokens > (q4 ? 8 : 2048) ||
        weight->ne[2] != 1 || weight->ne[3] != 1 ||
        ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        op->ne[0] != width || op->ne[1] != tokens || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    return ggml_is_contiguous(weight) && ggml_is_contiguous(ids) && ggml_is_contiguous(op) &&
           weight->nb[0] == ggml_type_size(type) && weight->nb[1] == ggml_row_size(type, width) &&
           ids->nb[0] == sizeof(int32_t) && op->nb[0] == sizeof(float) &&
           op->nb[1] == static_cast<size_t>(width) * sizeof(float) &&
           ggml_nbytes(weight) <= UINT32_MAX && ggml_nbytes(op) <= UINT32_MAX;
}

bool supports_q8_embedding_dispatch(const ggml_tensor * op) {
    return supports_raw_embedding_dispatch(op, GGML_TYPE_Q8_0, "HRX_ENABLE_Q8_EMBEDDING");
}

bool supports_q4_embedding_dispatch(const ggml_tensor * op) {
    return supports_raw_embedding_dispatch(op, GGML_TYPE_Q4_K, "HRX_ENABLE_Q4_EMBEDDING");
}

namespace {

static constexpr KernelCatalogRef kQwenTokenEmbeddingQ4KKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen_token_embedding_q4k");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_1d_or_2d_column(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] == 1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_hidden_size(int64_t hidden_size) {
    return hidden_size == 2048;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_supported_vocabulary_count(int64_t vocabulary_count) {
    return vocabulary_count >= 1 && vocabulary_count <= 262144;
}

struct QwenTokenEmbeddingMatch {
    const Value * token_ids        = nullptr;
    const Value * weight           = nullptr;
    const Value * output           = nullptr;
    int64_t       token_count      = 0;
    int64_t       vocabulary_count = 0;
    int64_t       hidden_size      = 0;

    bool matched() const { return token_ids != nullptr && weight != nullptr && output != nullptr; }
};

static QwenTokenEmbeddingMatch match_qwen_token_embedding(const Graph & graph, const GraphNode * node) {
    QwenTokenEmbeddingMatch match;
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight    = graph_value(graph, node->inputs[0]);
    const Value * token_ids = graph_value(graph, node->inputs[1]);
    const Value * output    = graph_value(graph, node->output);
    if (weight == nullptr || token_ids == nullptr || output == nullptr) {
        return {};
    }
    if (weight->type != GGML_TYPE_Q4_K || token_ids->type != GGML_TYPE_I32 || output->type != GGML_TYPE_F32) {
        return {};
    }
    if (!weight->contiguous || !token_ids->contiguous || !output->contiguous) {
        return {};
    }
    if (!is_2d(*weight) || !is_1d_or_2d_column(*token_ids) || !is_2d(*output)) {
        return {};
    }

    const int64_t hidden_size      = weight->ne[0];
    const int64_t vocabulary_count = weight->ne[1];
    const int64_t token_count      = token_ids->ne[0];
    if (output->ne[0] != hidden_size || output->ne[1] != token_count) {
        return {};
    }
    if (!is_supported_hidden_size(hidden_size) || !is_supported_vocabulary_count(vocabulary_count) ||
        !is_supported_token_count(token_count)) {
        return {};
    }

    match.token_ids        = token_ids;
    match.weight           = weight;
    match.output           = output;
    match.token_count      = token_count;
    match.vocabulary_count = vocabulary_count;
    match.hidden_size      = hidden_size;
    return match;
}

}  // namespace

static bool match_qwen_token_embedding_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenTokenEmbeddingMatch match = match_qwen_token_embedding(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenTokenEmbeddingQ4KKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("vocabulary_count", match.vocabulary_count);
    dispatch.bindings.push_back({ match.token_ids->id, 0, match.token_ids->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_raw_embedding_dispatch(const DispatchMatchContext & context, DispatchMatch & match, bool q4) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return false;
    }
    const Value * weight = graph_value(context.graph, node->inputs[0]);
    const Value * ids = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (weight == nullptr || ids == nullptr || output == nullptr ||
        !(q4 ? supports_q4_embedding_dispatch(output->tensor) : supports_q8_embedding_dispatch(output->tensor)) ||
        weight->tensor != output->tensor->src[0] || ids->tensor != output->tensor->src[1] ||
        output->storage == weight->storage || output->storage == ids->storage) {
        return false;
    }
    for (const Value * value : { weight, ids, output }) {
        if (!value->contiguous || value->byte_count != ggml_nbytes(value->tensor) ||
            value->storage_offset != 0 || value->storage_byte_count < value->byte_count) {
            return false;
        }
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(q4 ? kQ4EmbeddingKernel : kQ8EmbeddingKernel);
    dispatch.kernel.integer_parameters.emplace("hidden_size", weight->ne[0]);
    dispatch.kernel.integer_parameters.emplace("vocabulary_count", weight->ne[1]);
    dispatch.kernel.integer_parameters.emplace("token_count", ids->ne[0]);
    dispatch.bindings.push_back({ weight->id, 0, weight->byte_count });
    dispatch.bindings.push_back({ ids->id, 0, ids->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_q8_embedding_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    return match_raw_embedding_dispatch(context, match, false);
}

static bool match_q4_embedding_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    return match_raw_embedding_dispatch(context, match, true);
}

void register_qwen_preamble_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.preamble.token_embedding_q4_k",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_q4_embedding_dispatch,
    });
    registry.add({
        "qwen.preamble.token_embedding_q8_0",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_q8_embedding_dispatch,
    });
    registry.add({
        "qwen.preamble.token_embedding_q4k",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen_token_embedding_dispatch,
    });
}

}  // namespace ggml::hrx
