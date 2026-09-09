#pragma once

#include "graph.h"

#include <vector>

namespace ggml::hrx {

class GraphTraversalOrder {
  public:
    GraphTraversalOrder() = default;

    // Graph nodes must already be in execution order, as supplied by ggml.
    // Fusion still has to respect producer readiness and storage lifetimes.
    static GraphTraversalOrder build(const Graph & graph);

    const std::vector<const GraphNode *> & nodes() const { return nodes_; }

  private:
    std::vector<const GraphNode *> nodes_;
};

}  // namespace ggml::hrx
