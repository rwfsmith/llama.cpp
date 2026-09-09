#include "dispatch-scheduler.h"

#include "backend-buffer-binding.h"
#include "ggml.h"
#include "graph/graph-traversal.h"
#include "kernel-corpus/kernel-corpus.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

// When HRX_SURVEY_UNSUPPORTED is set, schedule_graph() keeps walking the graph after it hits a node
// no dispatch matches, logging every one instead of bailing at the first. Scheduling still fails, so
// nothing is executed -- this only exists so that bringing a new model up costs one run per pass
// rather than one run per missing kernel. Nodes past the first are best-effort: a surveyed node is
// marked covered so traversal can continue, which lets later matchers see its output as available.
static bool survey_unsupported_nodes_enabled() {
    const char * value = std::getenv("HRX_SURVEY_UNSUPPORTED");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// When HRX_TRACE_DISPATCH is set, every dispatch registration that wins a match is printed once, the
// first time it fires, with the shape of the node it rooted at. A backend that computes the right
// graph but the wrong numbers can only be bisected by moving whole op classes to the CPU, which says
// "something under GGML_OP_MUL is wrong" and no more; this says *which matcher* under GGML_OP_MUL
// actually ran on this model. Output is deduplicated by name and shape, so a 48-layer decode prints a
// couple of dozen lines rather than tens of thousands.
static bool trace_dispatch_matches_enabled() {
    const char * value = std::getenv("HRX_TRACE_DISPATCH");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static void trace_dispatch_match(const Graph &                    graph,
                                 const GraphNode *                node,
                                 const DispatchMatchDiagnostics & diagnostics) {
    const char * name = nullptr;
    for (const DispatchRegistrationAttempt & attempt : diagnostics.attempts) {
        if (attempt.matched) {
            name = attempt.name.c_str();
            break;
        }
    }
    if (name == nullptr || node == nullptr) {
        return;
    }
    std::ostringstream stream;
    stream << name << " " << ggml_op_name(node->op) << " ";
    const Value * output = graph.values().find(node->output);
    if (output != nullptr) {
        stream << ggml_type_name(output->type) << "[" << output->ne[0] << "," << output->ne[1] << "," << output->ne[2]
               << "," << output->ne[3] << "]";
    }
    static std::set<std::string> seen;
    if (seen.insert(stream.str()).second) {
        fprintf(stderr, "HRX dispatch: %s\n", stream.str().c_str());
    }
}

static bool match_covers_root(const DispatchMatch & match, size_t root_index) {
    return std::find(match.covered_nodes.begin(), match.covered_nodes.end(), root_index) != match.covered_nodes.end();
}

static bool match_overlaps_covered_nodes(const DispatchMatch & match, const std::vector<bool> & covered_nodes) {
    for (const size_t node_index : match.covered_nodes) {
        if (node_index >= covered_nodes.size() || covered_nodes[node_index]) {
            return true;
        }
    }
    return false;
}

// A match's dispatches are emitted below at the traversal position of its *root* node, not at the
// position of the last node it covers. Every value they bind therefore has to be live by the time
// the root runs. That is not automatic for a fused match: ggml may order an independent producer
// between the root and the last covered node, and the fused kernel would then read a buffer that
// nothing has written yet. The graph, the bindings and the kernel all look correct in that case and
// only the numerics are wrong, so the failure is silent and extremely expensive to chase -- it cost
// a full bisect to find in qwen4exp's GDN norm gate, where the z projection is ordered after the
// RMS_NORM the fusion roots at.
//
// A bound value is acceptable when it is produced before the root, produced by a node this match
// covers (the fusion subsumes that node, so it is computing the value itself rather than reading a
// stale one), or has no producer at all -- a weight, a graph input, or a plan transient, all of
// which are materialized outside the traversal.
static bool match_binds_value_produced_after_root(const Graph &         graph,
                                                  const DispatchMatch & match,
                                                  size_t                root_index,
                                                  const Dispatch *&     offending_dispatch,
                                                  ValueId &             offending_value) {
    const auto covers = [&match](size_t index) {
        return std::find(match.covered_nodes.begin(), match.covered_nodes.end(), index) != match.covered_nodes.end();
    };
    const auto scan = [&](const std::vector<Dispatch> & dispatches) {
        for (const Dispatch & dispatch : dispatches) {
            for (const DispatchBinding & binding : dispatch.bindings) {
                const GraphNode * producer = graph.index().producer(binding.value);
                if (producer == nullptr) {
                    continue;
                }
                size_t producer_index = 0;
                if (!graph.index().node_index(producer, producer_index)) {
                    continue;
                }
                if (producer_index < root_index || covers(producer_index)) {
                    continue;
                }
                offending_dispatch = &dispatch;
                offending_value    = binding.value;
                return true;
            }
        }
        return false;
    };
    return scan(match.initialization_dispatches) || scan(match.dispatches);
}

struct FusionStorageSpan {
    hrx_buffer_t buffer = nullptr;
    size_t begin = 0;
    size_t end = 0;
};

static bool fusion_resident_span(const Graph & graph, ValueId id, size_t offset, size_t length,
                                FusionStorageSpan & span) {
    const Value * value = graph.values().find(id);
    ValueBufferBinding binding;
    if (value == nullptr || value->tensor == nullptr || value->tensor->data == nullptr || length == 0 ||
        !ggml_backend_hrx_resolve_value_buffer(value->tensor, binding) ||
        binding.buffer == nullptr || binding.host_data != nullptr ||
        offset > binding.length || length > binding.length - offset ||
        binding.offset > std::numeric_limits<size_t>::max() - offset) {
        return false;
    }
    const size_t begin = binding.offset + offset;
    if (length > std::numeric_limits<size_t>::max() - begin) {
        return false;
    }
    // Same resident-buffer resolution as command-program-resolver: no host staging
    // or logical ValueStorageId is substituted for the actual buffer and offset.
    span = { binding.buffer, begin, begin + length };
    return true;
}

static const char * fusion_tensor_name(const Graph & graph, ValueId value) {
    const Value * found = graph.values().find(value);
    return found != nullptr && found->tensor != nullptr ? found->tensor->name : "<synthetic>";
}

static void trace_fusion_storage(const Graph & graph, const DispatchTarget & target,
                                 const DispatchMatch & match, size_t root_index,
                                 const std::vector<bool> & covered_nodes,
                                 const DispatchMatchDiagnostics & diagnostics) {
    const char * enabled = std::getenv("HRX_TRACE_FUSION_STORAGE");
    static std::atomic<unsigned> reports{0};
    constexpr unsigned limit = 64;
    if (enabled == nullptr || enabled[0] == '\0' || enabled[0] == '0' ||
        reports.load(std::memory_order_relaxed) >= limit) {
        return;
    }
    const char * rule = "<unknown>";
    for (const auto & attempt : diagnostics.attempts) {
        if (attempt.matched) {
            rule = attempt.name.c_str();
            break;
        }
    }
    const auto covered = [&](size_t index) {
        return covered_nodes[index] ||
               std::find(match.covered_nodes.begin(), match.covered_nodes.end(), index) != match.covered_nodes.end();
    };
    const auto scan = [&](const std::vector<Dispatch> & dispatches) {
        for (const Dispatch & dispatch : dispatches) {
            const auto kernel = resolve_kernel_definition(get_qwen_kernel_corpus(), target.architecture,
                                                          dispatch.kernel.kernel_id);
            if (!kernel.found() || kernel.definition->bindings.size() != dispatch.bindings.size()) {
                continue;
            }
            for (size_t slot = 0; slot < dispatch.bindings.size(); ++slot) {
                if (kernel.definition->bindings[slot].access == ResourceAccess::Read) {
                    continue;
                }
                const DispatchBinding & write = dispatch.bindings[slot];
                const GraphNode * producer = graph.index().producer(write.value);
                size_t producer_index = 0;
                FusionStorageSpan written;
                if (producer == nullptr || !graph.index().node_index(producer, producer_index) ||
                    producer_index <= root_index ||
                    std::find(match.covered_nodes.begin(), match.covered_nodes.end(), producer_index) ==
                        match.covered_nodes.end() ||
                    !fusion_resident_span(graph, write.value, write.offset, write.length, written)) {
                    continue;
                }
                for (size_t i = root_index + 1; i < producer_index; ++i) {
                    const GraphNode & node = graph.nodes()[i];
                    if (covered(i) || node.op == GGML_OP_NONE || is_layout_alias_node(graph, node)) {
                        continue;
                    }
                    const auto inspect = [&](ValueId value_id, const char * role) {
                        const Value * value = graph.values().find(value_id);
                        FusionStorageSpan other;
                        if (value == nullptr ||
                            !fusion_resident_span(graph, value_id, 0, value->byte_count, other) ||
                            written.buffer != other.buffer || written.begin >= other.end ||
                            other.begin >= written.end) {
                            return;
                        }
                        const unsigned report = reports.fetch_add(1, std::memory_order_relaxed);
                        if (report >= limit) {
                            return;
                        }
                        std::fprintf(stderr,
                            "HRX fusion-storage candidate=%u rule=%s root=%zu/%s/%s "
                            "write=%d/%s producer=%zu buffer=%p span=[%zu,%zu) "
                            "intervening=%zu/%s/%s role=%s value=%d/%s span=[%zu,%zu)\n",
                            report + 1, rule, root_index, ggml_op_name(graph.nodes()[root_index].op),
                            fusion_tensor_name(graph, graph.nodes()[root_index].output),
                            write.value.value, fusion_tensor_name(graph, write.value), producer_index,
                            static_cast<void *>(written.buffer), written.begin, written.end,
                            i, ggml_op_name(node.op), fusion_tensor_name(graph, node.output), role,
                            value_id.value, fusion_tensor_name(graph, value_id), other.begin, other.end);
                    };
                    for (const ValueId input : node.inputs) {
                        inspect(input, "input");
                    }
                    inspect(node.output, "output");
                    if (reports.load(std::memory_order_relaxed) >= limit) {
                        return;
                    }
                }
            }
        }
    };
    // Diagnostic only: overlaps are candidates, not sufficient grounds to reject
    // a match. A later fusion can subsume one of these still-uncovered nodes.
    scan(match.initialization_dispatches);
    scan(match.dispatches);
}

static bool try_match_registration(const Graph &              graph,
                                   const GraphNode *          node,
                                   size_t                     node_index,
                                   const std::vector<bool> &  covered_nodes,
                                   const CommandPlan &        plan,
                                   const DispatchRegistry &   registry,
                                   ValueId                    next_plan_value,
                                   DispatchMatch &            match,
                                   DispatchMatchDiagnostics * diagnostics) {
    const DispatchMatchContext context = {
        graph, node, node_index, covered_nodes, plan, next_plan_value,
    };
    return registry.match(context, match, diagnostics);
}

static void clear_plan_results(CommandPlan & plan) {
    plan.initialization_dispatches.clear();
    plan.dispatches.clear();
    plan.transients.clear();
    plan.constant_initializations.clear();
    plan.completion_counter_requests.clear();
    plan.metadata.clear();
}

static void append_value_summary(std::ostringstream & stream, const Graph & graph, ValueId value_id) {
    const Value * value = graph.values().find(value_id);
    if (value == nullptr) {
        stream << value_id.value << ":missing";
        return;
    }
    stream << value_id.value << ":" << ggml_type_name(value->type) << "[" << value->ne[0] << "," << value->ne[1] << ","
           << value->ne[2] << "," << value->ne[3] << "]";
    const GraphNode * producer = graph.index().producer(value_id);
    if (producer != nullptr) {
        stream << "<-" << ggml_op_name(producer->op);
    }
}

static void append_node_summary(std::ostringstream & stream, const Graph & graph, const GraphNode * node) {
    if (node == nullptr) {
        stream << "null";
        return;
    }
    size_t node_index = 0;
    if (graph.index().node_index(node, node_index)) {
        stream << node_index << ":";
    }
    stream << ggml_op_name(node->op);
}

static std::string unsupported_node_message(const Graph & graph, size_t index, const GraphNode & node) {
    std::ostringstream stream;
    stream << "unsupported HRX node " << index << ": " << ggml_op_name(node.op) << " output=";
    append_value_summary(stream, graph, node.output);
    stream << " inputs=[";
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        append_value_summary(stream, graph, node.inputs[i]);
    }
    stream << "]";
    stream << " consumers=[";
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(node.output);
    for (size_t i = 0; i < consumers.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        append_node_summary(stream, graph, consumers[i]);
    }
    stream << "]";
    return stream.str();
}

static bool value_is_available(const Graph & graph, ValueId value, const std::vector<bool> & covered_nodes) {
    const GraphNode * producer = graph.index().producer(value);
    if (producer == nullptr) {
        return true;
    }
    size_t producer_index = 0;
    return graph.index().node_index(producer, producer_index) && producer_index < covered_nodes.size() &&
           covered_nodes[producer_index];
}

static bool can_elide_layout_alias_node(const Graph &             graph,
                                        const GraphNode &         node,
                                        const std::vector<bool> & covered_nodes) {
    return is_layout_alias_node(graph, node) && value_is_available(graph, node.inputs[0], covered_nodes);
}

// A node whose output holds no elements computes nothing, so it can be covered without a dispatch.
// llama.cpp emits these deliberately: build_rs() clears a recurrent state slot through a view that is
// zero-sized whenever no slot needs resetting, and worst-case reserve graphs carry zero-token nodes.
// The backend cannot simply decline them, because ggml_backend_sched aborts outright -- rather than
// falling back to the CPU -- when a pre-allocated tensor such as the recurrent state cache lands in a
// buffer whose backend refuses the op. Eliding them here is what makes claiming them safe.
static bool node_output_is_empty(const Graph & graph, const GraphNode & node) {
    const Value * output = graph.values().find(node.output);
    return output != nullptr && output->element_count == 0;
}

static bool apply_value_aliases(Graph & graph, const DispatchMatch & match, Status & status) {
    for (const DispatchValueAliasRequest & alias : match.value_aliases) {
        Status alias_status = graph.values().alias_storage(alias.target_value, alias.source_value);
        if (!alias_status.success()) {
            status.append(alias_status);
            return false;
        }
    }
    return true;
}

}  // namespace

bool DispatchScheduler::schedule_graph(Graph & graph, const DispatchTarget & target) {
    return this->schedule_graph(graph, target, nullptr);
}

bool DispatchScheduler::schedule_graph(Graph &                       graph,
                                       const DispatchTarget &        target,
                                       DispatchScheduleDiagnostics * diagnostics) {
    plan_ = {};
    if (diagnostics != nullptr) {
        *diagnostics = {};
    }
    const DispatchRegistry * registry = find_dispatch_registry(target);
    if (registry == nullptr) {
        plan_.status.log("no HRX dispatch registry for target %s", target.architecture.c_str());
        return false;
    }
    const std::vector<GraphNode> & nodes = graph.nodes();
    if (!graph.has_index()) {
        plan_.status.log("HRX graph is missing graph index");
        return false;
    }
    std::vector<bool>         covered_nodes(nodes.size(), false);
    Status                    pending_diagnostics;
    const bool                survey_unsupported   = survey_unsupported_nodes_enabled();
    const bool                trace_matches        = trace_dispatch_matches_enabled();
    bool                      unsupported_surveyed = false;
    const GraphTraversalOrder traversal = GraphTraversalOrder::build(graph);
    for (const GraphNode * node : traversal.nodes()) {
        size_t i = 0;
        if (node == nullptr || !graph.index().node_index(node, i)) {
            plan_.status.log("HRX traversal references a node outside the graph");
            clear_plan_results(plan_);
            return false;
        }
        if (covered_nodes[i]) {
            continue;
        }
        DispatchMatch            match;
        const ValueId            next_plan_value(static_cast<int32_t>(graph.values().size() + plan_.transients.size() +
                                                                      plan_.completion_counter_requests.size()));
        DispatchMatchDiagnostics match_diagnostics;
        if (!try_match_registration(graph, node, i, covered_nodes, plan_, *registry, next_plan_value, match,
                                    &match_diagnostics)) {
            if (can_elide_layout_alias_node(graph, *node, covered_nodes) || node_output_is_empty(graph, *node)) {
                pending_diagnostics.append(match.status);
                covered_nodes[i] = true;
                continue;
            }
            plan_.status.append(pending_diagnostics);
            plan_.status.append(match.status);
            const std::string message = unsupported_node_message(graph, i, *node);
            plan_.status.log("%s", message.c_str());
            if (diagnostics != nullptr) {
                diagnostics->unsupported_node_index = i;
                diagnostics->unsupported_node       = node;
                diagnostics->unsupported_message    = message;
                diagnostics->match                  = std::move(match_diagnostics);
            }
            if (survey_unsupported) {
                unsupported_surveyed = true;
                covered_nodes[i]     = true;
                pending_diagnostics  = {};
                continue;
            }
            clear_plan_results(plan_);
            return false;
        }
        if (trace_matches) {
            trace_dispatch_match(graph, node, match_diagnostics);
        }
        if (match.covered_nodes.empty() || match.dispatches.empty() || !match_covers_root(match, i) ||
            match_overlaps_covered_nodes(match, covered_nodes)) {
            plan_.status.log("invalid HRX dispatch match for node %zu: %s", i, ggml_op_name(node->op));
            if (diagnostics != nullptr) {
                diagnostics->unsupported_node_index = i;
                diagnostics->unsupported_node       = node;
                diagnostics->unsupported_message    = "invalid HRX dispatch match";
                diagnostics->match                  = std::move(match_diagnostics);
            }
            clear_plan_results(plan_);
            return false;
        }
        const Dispatch * offending_dispatch = nullptr;
        ValueId          offending_value;
        if (match_binds_value_produced_after_root(graph, match, i, offending_dispatch, offending_value)) {
            plan_.status.log(
                "HRX dispatch kernel_id=%llu for node %zu (%s) binds value %d, which is produced after the node "
                "it is emitted at; the matcher must decline this graph ordering",
                offending_dispatch != nullptr ? (unsigned long long) offending_dispatch->kernel.kernel_id : 0ULL, i,
                ggml_op_name(node->op), offending_value.value);
            if (diagnostics != nullptr) {
                diagnostics->unsupported_node_index = i;
                diagnostics->unsupported_node       = node;
                diagnostics->unsupported_message    = "HRX dispatch binds a value produced after its root";
                diagnostics->match                  = std::move(match_diagnostics);
            }
            clear_plan_results(plan_);
            return false;
        }
        if (!apply_value_aliases(graph, match, plan_.status)) {
            if (diagnostics != nullptr) {
                diagnostics->unsupported_node_index = i;
                diagnostics->unsupported_node       = node;
                diagnostics->unsupported_message    = "invalid HRX value alias";
                diagnostics->match                  = std::move(match_diagnostics);
            }
            clear_plan_results(plan_);
            return false;
        }
        trace_fusion_storage(graph, target, match, i, covered_nodes, match_diagnostics);
        for (Dispatch & dispatch : match.initialization_dispatches) {
            plan_.initialization_dispatches.push_back(std::move(dispatch));
        }
        for (Dispatch & dispatch : match.dispatches) {
            plan_.dispatches.push_back(std::move(dispatch));
        }
        for (CommandPlanTransient & transient : match.transients) {
            plan_.transients.push_back(std::move(transient));
        }
        for (CommandPlanConstantInitialization & initialization : match.constant_initializations) {
            plan_.constant_initializations.push_back(std::move(initialization));
        }
        for (CommandPlanCompletionCounterRequest & request : match.completion_counter_requests) {
            plan_.completion_counter_requests.push_back(std::move(request));
        }
        if (!plan_.metadata.append(std::move(match.metadata), plan_.status)) {
            clear_plan_results(plan_);
            return false;
        }
        for (const size_t covered_node : match.covered_nodes) {
            covered_nodes[covered_node] = true;
        }
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!covered_nodes[i]) {
            if (can_elide_layout_alias_node(graph, nodes[i], covered_nodes)) {
                covered_nodes[i] = true;
                continue;
            }
            plan_.status.append(pending_diagnostics);
            const std::string message = unsupported_node_message(graph, i, nodes[i]);
            plan_.status.log("%s", message.c_str());
            if (diagnostics != nullptr) {
                DispatchMatch match;
                const ValueId next_plan_value(static_cast<int32_t>(graph.values().size() + plan_.transients.size() +
                                                                   plan_.completion_counter_requests.size()));
                DispatchMatchDiagnostics match_diagnostics;
                try_match_registration(graph, &nodes[i], i, covered_nodes, plan_, *registry, next_plan_value, match,
                                       &match_diagnostics);
                diagnostics->unsupported_node_index = i;
                diagnostics->unsupported_node       = &nodes[i];
                diagnostics->unsupported_message    = message;
                diagnostics->match                  = std::move(match_diagnostics);
            }
            clear_plan_results(plan_);
            return false;
        }
    }
    if (unsupported_surveyed) {
        clear_plan_results(plan_);
        return false;
    }
    return true;
}

bool DispatchScheduler::supports_node(const Graph & graph, const GraphNode * node, const DispatchTarget & target) {
    const DispatchRegistry * registry = find_dispatch_registry(target);
    if (registry == nullptr) {
        return false;
    }
    if (node == nullptr || !graph.has_index()) {
        return false;
    }
    size_t node_index = 0;
    if (!graph.index().node_index(node, node_index)) {
        return false;
    }
    const std::vector<bool> covered_nodes(graph.nodes().size(), false);
    DispatchMatch           match;
    const ValueId           next_plan_value(static_cast<int32_t>(graph.values().size()));
    const CommandPlan       plan;
    return try_match_registration(graph, node, node_index, covered_nodes, plan, *registry, next_plan_value, match,
                                  nullptr);
}

bool DispatchScheduler::can_schedule_graph(const Graph & graph, const DispatchTarget & target) {
    Graph             graph_copy = graph;
    DispatchScheduler scheduler;
    return scheduler.schedule_graph(graph_copy, target);
}

}  // namespace ggml::hrx
