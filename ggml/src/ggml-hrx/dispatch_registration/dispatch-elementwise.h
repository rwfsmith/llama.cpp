#pragma once

#include "dispatch-registry.h"

#include <cstdint>

namespace ggml::hrx {

bool dense_f32_shape_supported(int64_t input_size, int64_t output_size, int64_t token_count);
bool recurrent_concat_supported(const ggml_tensor * op);
bool qsa_bf16_projection_supported(const ggml_tensor * op);
bool gdn_norm_prefill_supported(const ggml_tensor * op);

void register_elementwise_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
