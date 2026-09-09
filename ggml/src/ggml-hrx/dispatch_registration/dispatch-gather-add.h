#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

inline constexpr int64_t kF32GetRowsMaxWidth = 1048576;
inline constexpr int64_t kF32GetRowsMaxRows = 4096;
inline constexpr int64_t kF32GetRowsMaxElements = 134217728;

// HRX_ENABLE_F32_GET_ROWS=1: contiguous 2D F32 state rows and contiguous I32 IDs.
bool supports_f32_get_rows_dispatch(const ggml_tensor * op);
bool is_f32_get_rows_kernel(uint64_t kernel_id);

void register_gather_add_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
