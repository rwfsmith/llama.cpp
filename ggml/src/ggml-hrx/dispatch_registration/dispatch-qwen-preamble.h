#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

// Raw, contiguous 2D Q8_0 leaf weights, contiguous I32 IDs and F32 rows.
// Exact opt-in; views and spans exceeding the kernel's u32 addressing are declined.
bool supports_q8_embedding_dispatch(const ggml_tensor * op);
bool is_q8_embedding_kernel(uint64_t kernel_id);
// Opt-in MTP Q4_K rows: hidden=2560, tokens=1..8; the legacy 2048 route is unchanged.
bool supports_q4_embedding_dispatch(const ggml_tensor * op);
bool is_q4_embedding_kernel(uint64_t kernel_id);

void register_qwen_preamble_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
