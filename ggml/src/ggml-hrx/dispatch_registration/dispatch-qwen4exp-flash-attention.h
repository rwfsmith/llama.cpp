#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

void register_qwen4exp_flash_attention_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
