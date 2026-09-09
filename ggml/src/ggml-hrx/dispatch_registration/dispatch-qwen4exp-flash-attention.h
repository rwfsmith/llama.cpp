#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

// Shape/parameter predicate only; callers retain the default-off HRX_ENABLE_QSA_ATTN policy.
bool supports_qwen4exp_flash_attention_dispatch(const ggml_tensor * op);

void register_qwen4exp_flash_attention_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
