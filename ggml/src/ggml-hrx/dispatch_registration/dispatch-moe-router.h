#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

bool is_moe_router_top8_kernel(uint64_t kernel_id);

void register_moe_router_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
