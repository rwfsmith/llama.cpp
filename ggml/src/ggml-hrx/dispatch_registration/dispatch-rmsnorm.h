#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

bool qwen4exp_hc_grouped_norm_token_count_supported(int64_t tokens);

template <typename Tensor> bool qwen4exp_hc_grouped_norm_layout_supported(const Tensor & tensor) {
    return tensor.type == GGML_TYPE_F32 && tensor.ne[0] == 2560 && tensor.ne[1] == 4 &&
           qwen4exp_hc_grouped_norm_token_count_supported(tensor.ne[2]) && tensor.ne[3] == 1 &&
           tensor.nb[0] == sizeof(float) && tensor.nb[1] == 2560 * sizeof(float) &&
           tensor.nb[2] == 10240 * sizeof(float);
}

bool qwen4exp_hc_grouped_norm_rms_supported(const ggml_tensor * op);

void register_qwen_rmsnorm_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
