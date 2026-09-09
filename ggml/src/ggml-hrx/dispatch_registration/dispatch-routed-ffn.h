#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

bool is_q4_routed_gate_up_kernel(uint64_t kernel_id);
size_t q4_routed_gate_up_id_byte_count(int64_t tokens, int64_t routes, int64_t stride, int64_t experts);

struct Q4RoutedIdValidation {
    bool valid = false;
    int64_t invalid_id = 0;
    int64_t token = -1;
    int64_t route = -1;
};

Q4RoutedIdValidation validate_q4_routed_gate_up_id_bytes(
    const void * data, size_t size, int64_t tokens, int64_t routes, int64_t stride, int64_t experts);

void register_routed_ffn_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
