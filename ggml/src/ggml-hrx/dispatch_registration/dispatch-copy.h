#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

void register_copy_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
