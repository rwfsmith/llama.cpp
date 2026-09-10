#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

// HRX_ENABLE_QSA_MASK=1. Dense one-stream masks, KV width 256..131072
// (multiple of 256), 1..32 query/padded rows. FILL is limited to +/-0 and -inf.
// Also accepts F32 selector-zero tensors [1,1..4096,1..32,1].
// ADD allows identical masks or a one-row RHS; no arbitrary broadcast/in-place ops.
bool supports_qsa_mask_dispatch(const ggml_tensor * op);
void register_qsa_mask_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
