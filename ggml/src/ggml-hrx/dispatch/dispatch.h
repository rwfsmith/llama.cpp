#pragma once

#include "graph/value-map.h"
#include "kernel-corpus/kernel-corpus-catalog.h"
#include "kernel-corpus/kernel-types.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ggml::hrx {

struct KernelSpecialization {
    uint64_t                           kernel_id = kUncatalogedKernelId;
    std::map<std::string, int64_t>     integer_parameters;
    std::map<std::string, std::string> compile_parameters;
};

inline KernelSpecialization make_kernel_specialization(KernelCatalogRef ref) {
    KernelSpecialization kernel;
    kernel.kernel_id = ref.id;
    return kernel;
}

struct DispatchBinding {
    ValueId value;
    size_t  offset = 0;
    size_t  length = 0;
    // True only when the matcher proved this value is written exclusively by trusted
    // host code before the command executes (e.g. a `ggml_set_input()` leaf with no
    // GPU-side producer in this graph), so per-launch index validation and replay
    // exclusion are unnecessary for it. Defaults to untrusted for every existing
    // binding; only specific matchers that verify the producer set this to true.
    bool    trusted = false;
};

struct Dispatch {
    KernelSpecialization         kernel;
    std::vector<DispatchBinding> bindings;
};

}  // namespace ggml::hrx
