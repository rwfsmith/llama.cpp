#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

bool qwen4exp_qsa_norm_supported(const ggml_tensor * op);
bool qwen4exp_qsa_rope_supported(const ggml_tensor * op);
void register_qwen4exp_qsa_rope_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
