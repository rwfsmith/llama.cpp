#pragma once

#include "dispatch-registry.h"

#include <cstdint>

namespace ggml::hrx {

inline constexpr int64_t kCopyF32MaxElements = 134217728;

struct CopyF32Geometry {
    int64_t element_count = 0;
    int64_t row_length = 0;
    int64_t row_count = 0;
    int64_t source_row_stride = 0;
    int64_t output_row_stride = 0;
    bool contiguous = false;
};

// Uniform row strides across all traversed outer axes; contiguous destinations
// may reshape freely, strided destinations must retain row width and row count.
bool copy_f32_geometry(const Value & source, const Value & output, CopyF32Geometry & geometry);
bool copy_f32_geometry(const ggml_tensor & source, const ggml_tensor & output, CopyF32Geometry & geometry);
bool supports_copy_f32_dispatch(const ggml_tensor * op);
Dispatch make_copy_f32_dispatch(const CopyF32Geometry & geometry, ValueId source, size_t source_bytes,
                               ValueId output, size_t output_bytes);

// HRX_ENABLE_SET_ROWS=1: experimental 2D F32 -> F16/F32 writes, widths 1/128/256/512.
// IDs must be unique and in range; runtime validation fails the graph before writing,
// unless the matcher proved the IDs come from a trusted host-populated producer (see
// DispatchBinding::trusted), in which case the synchronous validation readback and
// replay exclusion are skipped for that binding.
// Shared by the scheduler claim and dispatch matcher.
bool supports_set_rows_dispatch(const ggml_tensor * op);
bool is_set_rows_kernel(uint64_t kernel_id);

// True only for a tensor that is provably populated exclusively by host code before this
// graph runs (a leaf with no producer op, explicitly marked as a graph input) -- e.g. the
// KV-cache k_idxs/v_idxs tensors -- and therefore safe to exempt from GPU-side index
// validation. HRX_ENABLE_TRUSTED_INDEX_VIEWS=1 also accepts contiguous, same-type
// VIEW chains rooted at such an input (recurrent-state row selection).
// Exposed for direct unit testing; see the definition for the full rationale.
bool is_trusted_host_index_producer(const ggml_tensor * tensor);

// HRX_ENABLE_QSA_F16_GATHER=1: dense [128, rows] F16 cache to F32, one sequence.
// Uses the same trusted-input and untrusted-ID validation contract as F32 GET_ROWS.
bool supports_qsa_f16_gather_dispatch(const ggml_tensor * op);
bool is_qsa_f16_gather_kernel(uint64_t kernel_id);

void register_copy_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
