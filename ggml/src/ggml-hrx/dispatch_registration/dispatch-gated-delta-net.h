#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

// Opt-in core only: F32, T=2..8, one sequence, scalar gates, 1<=K<=T.
// Q/K/V may have padded head/token strides; snapshot slot 0 is newest.
bool supports_gdn_prefill_dispatch(const ggml_tensor * op);

// HRX_ENABLE_GDN_CONV_PREFILL=1: dense F32 four-tap SSM_CONV, T=2..8,
// 10240 channels, one sequence. CONCAT, SILU and history writes stay separate.
bool supports_gdn_conv_prefill_dispatch(const ggml_tensor * op);

void register_gdn_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
