#pragma once

#include "dispatch-registry.h"

#include <cstdlib>
#include <cstring>

namespace ggml::hrx {

// HC streams are adjacent columns, with tokens outside them. The weight is shared, not batched.
template <typename Tensor>
bool llm_mtp_hc_projection_supported(const Tensor & weight, const Tensor & input, const Tensor & output) {
    const char * flag = std::getenv("HRX_ENABLE_MTP_HC_PROJECTION");
    if (flag == nullptr || std::strcmp(flag, "1") != 0 ||
        weight.type != GGML_TYPE_Q4_K || input.type != GGML_TYPE_F32 || output.type != GGML_TYPE_F32 ||
        weight.ne[0] != 5120 || weight.ne[1] != 2560 || weight.ne[2] != 1 || weight.ne[3] != 1 ||
        input.ne[0] != 5120 || input.ne[1] != 4 || input.ne[2] < 1 || input.ne[2] > 8 || input.ne[3] != 1 ||
        output.ne[0] != 2560 || output.ne[1] != 4 || output.ne[2] != input.ne[2] || output.ne[3] != 1) {
        return false;
    }
    for (const Tensor * tensor : { &weight, &input, &output }) {
        if (tensor->nb[0] != ggml_type_size(tensor->type) ||
            tensor->nb[1] != ggml_row_size(tensor->type, tensor->ne[0]) ||
            tensor->nb[2] != tensor->nb[1] * static_cast<size_t>(tensor->ne[1]) ||
            tensor->nb[3] != tensor->nb[2] * static_cast<size_t>(tensor->ne[2])) {
            return false;
        }
    }
    return true;
}

// Shared by the tensor capability probe and imported-graph matcher. Contiguous T=1..8.
template <typename Tensor>
bool llm_q8_narrow_supported(const Tensor & weight, const Tensor & input, const Tensor & output) {
    const char * flag = std::getenv("HRX_ENABLE_Q8_NARROW");
    if (flag == nullptr || std::strcmp(flag, "1") != 0 ||
        weight.type != GGML_TYPE_Q8_0 || input.type != GGML_TYPE_F32 || output.type != GGML_TYPE_F32 ||
        (weight.ne[0] != 320 && weight.ne[0] != 640) || weight.ne[1] < 1 || weight.ne[1] > 10240 ||
        input.ne[0] != weight.ne[0] || input.ne[1] < 1 || input.ne[1] > 8 ||
        output.ne[0] != weight.ne[1] || output.ne[1] != input.ne[1]) {
        return false;
    }
    for (const Tensor * tensor : { &weight, &input, &output }) {
        if (tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
            tensor->nb[0] != ggml_type_size(tensor->type) ||
            tensor->nb[1] != ggml_row_size(tensor->type, tensor->ne[0]) ||
            tensor->nb[2] != tensor->nb[1] * static_cast<size_t>(tensor->ne[1]) ||
            tensor->nb[3] != tensor->nb[2]) {
            return false;
        }
    }
    return true;
}

void register_llm_matmul_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
