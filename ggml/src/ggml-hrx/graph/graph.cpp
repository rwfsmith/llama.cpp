#include "graph.h"

#include "ggml-impl.h"

#include <cassert>
#include <utility>

namespace ggml::hrx {

GraphIndex GraphIndex::build(const Graph & graph) {
    GraphIndex                     index;
    const std::vector<GraphNode> & nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const GraphNode & node = nodes[i];
        index.node_indices_.emplace(&node, i);
        index.producers_.emplace(node.output.value, &node);
        for (ValueId input : node.inputs) {
            index.consumers_[input.value].push_back(&node);
        }
    }
    return index;
}

const GraphNode * GraphIndex::producer(ValueId value) const {
    const auto found = producers_.find(value.value);
    return found == producers_.end() ? nullptr : found->second;
}

const std::vector<const GraphNode *> & GraphIndex::consumers(ValueId value) const {
    static const std::vector<const GraphNode *> empty;
    const auto                                  found = consumers_.find(value.value);
    return found == consumers_.end() ? empty : found->second;
}

bool GraphIndex::has_single_consumer(ValueId value) const {
    return consumers(value).size() == 1;
}

bool GraphIndex::node_index(const GraphNode * node, size_t & index) const {
    const auto found = node_indices_.find(node);
    if (found == node_indices_.end()) {
        return false;
    }
    index = found->second;
    return true;
}

Graph::Graph(const Graph & other) : values_(other.values_), nodes_(other.nodes_) {
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
}

Graph & Graph::operator=(const Graph & other) {
    if (this == &other) {
        return *this;
    }
    values_ = other.values_;
    nodes_  = other.nodes_;
    index_.reset();
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
    return *this;
}

Graph::Graph(Graph && other) : values_(std::move(other.values_)), nodes_(std::move(other.nodes_)) {
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
}

Graph & Graph::operator=(Graph && other) {
    if (this == &other) {
        return *this;
    }
    values_ = std::move(other.values_);
    nodes_  = std::move(other.nodes_);
    index_.reset();
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
    return *this;
}

GraphNode & Graph::add_node(ggml_op op, ValueId output, std::vector<ValueId> inputs) {
    index_.reset();
    GraphNode node;
    node.op     = op;
    node.output = output;
    node.inputs = std::move(inputs);
    nodes_.push_back(std::move(node));
    return nodes_.back();
}

Status Graph::build_index() {
    index_ = GraphIndex::build(*this);
    return {};
}

const GraphIndex & Graph::index() const {
    assert(index_.has_value());
    return *index_;
}

GraphImportResult import_ggml_graph(const ggml_cgraph & graph) {
    GraphImportResult result;
    for (int i = 0; i < graph.n_nodes; ++i) {
        if (graph.nodes[i] == nullptr) {
            result.status.log("ggml graph contains a null node");
            return result;
        }
    }

    // Every tensor in a ggml graph owns storage that the scheduler assigned, and ggml expects each node to be
    // materialized there. Treating any of them as transient scratch is unsound: an earlier rule classified a
    // tensor as external only when no node in the split consumed it, which inverted the test for inputs --
    // a split input is consumed by definition, so it was bound to uninitialized arena memory and every kernel
    // read zeros instead of the real activations. Leaves escaped that rule, which is why single-op backend
    // tests passed while real graphs produced garbage. The transient arena stays available for values that
    // fused dispatch plans synthesize, which are not ggml tensors and carry no scheduler-owned storage.
    ValueMap & values = result.graph.values();
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor *  node = graph.nodes[i];
        std::vector<ValueId> inputs;
        for (const ggml_tensor * source : node->src) {
            if (source == nullptr) {
                continue;
            }
            inputs.push_back(values.get_or_add_tensor_value(source, ValueKind::External));
        }

        const ValueId   output     = values.get_or_add_tensor_value(node, ValueKind::External);
        GraphNode &     graph_node = result.graph.add_node(node->op, output, std::move(inputs));
        graph_node.params          = import_op_params(*node);
    }

    result.status.append(result.graph.build_index());
    return result;
}

bool is_layout_alias_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

bool is_layout_alias_node(const Graph & graph, const GraphNode & node) {
    return is_layout_alias_op(node.op) && node.inputs.size() == 1 &&
           graph.values().same_storage(node.output, node.inputs[0]);
}

}  // namespace ggml::hrx
