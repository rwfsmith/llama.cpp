#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

bool supports_add_f32_dispatch(const ggml_tensor * op);
bool supports_qsa_add_dispatch(const ggml_tensor * op);
bool is_hc_collapse_add_candidate(const ggml_tensor * op);
bool is_broadcast_add_candidate(const ggml_tensor * op);
bool supports_fused_moe_add(const ggml_tensor * op);

void register_add_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
