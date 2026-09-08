#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

void register_elementwise_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
