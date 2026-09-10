#include "dispatch-qwen4exp-rope.h"

#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

constexpr KernelCatalogRef kNorm = GGML_HRX_KERNEL_REF("hrx_owned", "ggml_qsa_norm_f32");
constexpr KernelCatalogRef kRope = GGML_HRX_KERNEL_REF("hrx_owned", "ggml_qsa_rope_f32");

bool enabled() {
    const char * flag = std::getenv("HRX_ENABLE_QSA_ROPE");
    return flag && std::strcmp(flag, "1") == 0;
}

bool shape_supported(const ggml_tensor & t, bool pooled, bool pooled_norm) {
    if (t.type != GGML_TYPE_F32 || t.ne[3] != 1 || t.ne[2] < 1) {
        return false;
    }
    if (pooled && t.ne[0] == 128 && t.ne[1] == 1) {
        // These are pooled cache blocks, not the current query token count.
        return t.ne[2] <= 65536;
    }
    if (pooled_norm && t.ne[0] == 128 && t.ne[2] == 1) {
        return t.ne[1] >= 1 && t.ne[1] <= 65536;
    }
    return t.ne[2] <= 8 &&
        ((t.ne[0] == 256 && (t.ne[1] == 24 || t.ne[1] == 2)) ||
         (t.ne[0] == 128 && t.ne[1] == 4));
}

bool layout_supported(const ggml_tensor & t, bool strided_query) {
    const size_t row = static_cast<size_t>(t.ne[0]) * sizeof(float);
    const bool dense = t.nb[1] == row;
    const bool query = strided_query && t.ne[0] == 256 && t.ne[1] == 24 && t.nb[1] == 2 * row;
    return t.nb[0] == sizeof(float) && (dense || query) &&
        t.nb[2] == t.nb[1] * static_cast<size_t>(t.ne[1]) &&
        t.nb[3] == t.nb[2] * static_cast<size_t>(t.ne[2]);
}

bool tensors_supported(const ggml_tensor * op, bool rope) {
    // The indexer mean is reshaped to [128, blocks, 1] before normalization.
    // Preserve the existing fused GDN norm/gate route, which has different provenance.
    const bool pooled_norm = !rope && op && op->src[0] && op->src[0]->op == GGML_OP_RESHAPE &&
        op->src[0]->src[0] && op->src[0]->src[0]->op == GGML_OP_SCALE;
    return enabled() && op && op->src[0] && !op->view_src &&
        shape_supported(*op, rope, pooled_norm) && ggml_are_same_shape(op, op->src[0]) &&
        op->src[0]->type == GGML_TYPE_F32 &&
        layout_supported(*op, false) && layout_supported(*op->src[0], !rope);
}

float param_float(const ggml_tensor * op, int slot) {
    float value;
    std::memcpy(&value, reinterpret_cast<const uint8_t *>(op->op_params) + slot * sizeof(int32_t), sizeof(value));
    return value;
}

int32_t param_int(const ggml_tensor * op, int slot) {
    int32_t value;
    std::memcpy(&value, reinterpret_cast<const uint8_t *>(op->op_params) + slot * sizeof(int32_t), sizeof(value));
    return value;
}

std::string float_config(float value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return stream.str();
}

bool value_matches_tensor(const Value * value, const ggml_tensor * tensor) {
    if (!value || !tensor || value->type != tensor->type || value->byte_count != ggml_nbytes(tensor)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value->ne[i] != tensor->ne[i] || value->nb[i] != tensor->nb[i]) {
            return false;
        }
    }
    return true;
}

bool match_qsa(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (!node || (node->op != GGML_OP_RMS_NORM && node->op != GGML_OP_ROPE)) {
        return false;
    }
    const bool rope = node->op == GGML_OP_ROPE;
    if (node->inputs.size() != (rope ? 2u : 1u)) {
        return false;
    }
    const Value * out = context.graph.values().find(node->output);
    const ggml_tensor * op = out ? out->tensor : nullptr;
    // RopeParams does not retain sections/offset. Never guess those for a synthetic graph.
    if (!(rope ? qwen4exp_qsa_rope_supported(op) : qwen4exp_qsa_norm_supported(op)) ||
        !op_params_equivalent(node->op, node->params, *op) || !value_matches_tensor(out, op)) {
        return false;
    }
    const Value * input = context.graph.values().find(node->inputs[0]);
    if (!value_matches_tensor(input, op->src[0]) ||
        context.graph.values().same_storage(input->id, out->id)) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(rope ? kRope : kNorm);
    dispatch.kernel.integer_parameters.emplace("width", out->ne[0]);
    dispatch.kernel.integer_parameters.emplace("heads", out->ne[1]);
    dispatch.kernel.integer_parameters.emplace("tokens", out->ne[2]);
    if (rope) {
        const Value * positions = context.graph.values().find(node->inputs[1]);
        if (!value_matches_tensor(positions, op->src[1]) ||
            context.graph.values().same_storage(positions->id, out->id)) {
            return false;
        }
        float correction[2];
        ggml_rope_yarn_corr_dims(64, param_int(op, 4), param_float(op, 5),
                                param_float(op, 9), param_float(op, 10), correction);
        const float ext = param_float(op, 7);
        float magnitude = param_float(op, 8);
        if (ext != 0.0f) {
            magnitude *= 1.0f + 0.1f * std::log(1.0f / param_float(op, 6));
        }
        const auto config = [&](const char * key, float value) {
            dispatch.kernel.compile_parameters.emplace(std::string("ggml.qsa_rope.") + key, float_config(value));
        };
        config("theta_scale", std::pow(param_float(op, 5), -2.0f / 64.0f));
        config("freq_scale", param_float(op, 6));
        config("ext_factor", ext);
        config("magnitude", magnitude);
        config("corr_low", correction[0]);
        config("corr_width", std::fmax(0.001f, correction[1] - correction[0]));
        dispatch.bindings.push_back({ input->id, 0, input->byte_count });
        dispatch.bindings.push_back({ positions->id, 0, positions->byte_count });
    } else {
        dispatch.kernel.integer_parameters.emplace("source_stride", input->nb[1] / sizeof(float));
        dispatch.kernel.compile_parameters.emplace("ggml.qsa_norm.epsilon", float_config(param_float(op, 0)));
        dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    }
    dispatch.bindings.push_back({ out->id, 0, out->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

bool qwen4exp_qsa_norm_supported(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_RMS_NORM || !tensors_supported(op, false)) {
        return false;
    }
    const float eps = param_float(op, 0);
    return std::isfinite(eps) && eps > 0.0f;
}

bool qwen4exp_qsa_rope_supported(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_ROPE || !tensors_supported(op, true) || !op->src[1] || op->src[2] ||
        param_int(op, 1) != 64 || param_int(op, 2) != GGML_ROPE_TYPE_IMROPE ||
        param_int(op, 11) != 11 || param_int(op, 12) != 11 || param_int(op, 13) != 10 ||
        param_int(op, 14) != 0 || param_int(op, 15) != 0 || param_int(op, 4) <= 0) {
        return false;
    }
    const ggml_tensor * pos = op->src[1];
    if (pos->type != GGML_TYPE_I32 || !ggml_is_contiguous(pos) ||
        pos->ne[0] != 4 * op->ne[2] || pos->ne[1] != 1 || pos->ne[2] != 1 || pos->ne[3] != 1) {
        return false;
    }
    for (int slot = 5; slot <= 10; ++slot) {
        if (!std::isfinite(param_float(op, slot))) {
            return false;
        }
    }
    return param_float(op, 5) > 1.0f && param_float(op, 5) <= 1.0e10f &&
        param_float(op, 6) >= 1.0e-6f && param_float(op, 6) <= 1.0f &&
        param_float(op, 7) >= 0.0f && param_float(op, 7) <= 1.0f &&
        param_float(op, 8) > 0.0f && param_float(op, 8) <= 16.0f &&
        param_float(op, 9) >= param_float(op, 10) && param_float(op, 10) > 0.0f;
}

void register_qwen4exp_qsa_rope_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({ "qwen4exp.qsa_norm", GGML_OP_RMS_NORM, DispatchMatchKind::SingleOp, 100,
                   DispatchSource::Qwen, match_qsa });
    registry.add({ "qwen4exp.qsa_rope", GGML_OP_ROPE, DispatchMatchKind::SingleOp, 100,
                   DispatchSource::Qwen, match_qsa });
}

}  // namespace ggml::hrx
