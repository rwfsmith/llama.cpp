#include "graph-traversal.h"

namespace ggml::hrx {

GraphTraversalOrder GraphTraversalOrder::build(const Graph & graph) {
    GraphTraversalOrder result;
    result.nodes_.reserve(graph.nodes().size());
    // ggml's allocator assigns overlapping buffers according to this order. Logical
    // data dependencies do not describe those storage anti-dependencies: selecting
    // a ready matmul or ADD/MUL followup early can overwrite another live tensor.
    // Preserve the order even before allocation so cached plans agree with replay.
    for (const GraphNode & node : graph.nodes()) {
        result.nodes_.push_back(&node);
    }
    return result;
}

}  // namespace ggml::hrx
