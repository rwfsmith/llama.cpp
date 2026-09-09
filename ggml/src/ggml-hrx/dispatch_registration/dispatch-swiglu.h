#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

bool supports_swiglu_f32_dispatch(const ggml_tensor * op);
void register_swiglu_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
