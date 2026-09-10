#include "backend-context.h"
#include "dispatch/dispatch-scheduler.h"
#include "dispatch_registration/dispatch-llm-matmul.h"
#include "dispatch_registration/dispatch-routed-ffn.h"
#include "dispatch_registration/dispatch-qwen-preamble.h"
#include "dispatch_registration/dispatch-rmsnorm.h"
#include "dispatch_registration/dispatch-add.h"
#include "dispatch_registration/dispatch-copy.h"
#include "dispatch_registration/dispatch-gather-add.h"
#include "dispatch_registration/dispatch-elementwise.h"
#include "dispatch_registration/dispatch-gated-delta-net.h"
#include "dispatch_registration/dispatch-qwen4exp-rope.h"
#include "dispatch_registration/dispatch-swiglu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-hrx.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/command-program-executor.h"
#include "runtime/device-timing.h"
#include "runtime/graph-executor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

class ScopedHrxEnvironment {
  public:
    ScopedHrxEnvironment(const char * key, const char * value) : key(key) {
        const char * previous = std::getenv(key);
        had_previous = previous != nullptr;
        if (had_previous) {
            saved = previous;
        }
        set(value);
    }

    ~ScopedHrxEnvironment() {
        set(had_previous ? saved.c_str() : nullptr);
    }

  private:
    void set(const char * value) {
#ifdef _WIN32
        REQUIRE(_putenv_s(key, value != nullptr ? value : "") == 0);
#else
        REQUIRE((value != nullptr ? setenv(key, value, 1) : unsetenv(key)) == 0);
#endif
    }

    const char * key;
    bool        had_previous = false;
    std::string saved;
};

static constexpr float   kQwenRmsNormEps        = 0.000001f;
static constexpr int64_t kQwenFlashHeadSize     = 128;
static constexpr int64_t kQwenRouterExpertCount = 128;
static constexpr int64_t kQwenRouterRouteCount  = 8;
static constexpr int64_t kQwenHiddenSize        = 2048;
static constexpr int64_t kQwenMoeIntermediate   = 768;
static constexpr int64_t kQwenVocabularyCount   = 151936;
static constexpr float kDenseWmmaAbsTolerance = 3.0e-1f;
static constexpr float kDenseWmmaRelTolerance = 2.0e-2f;

static ggml::hrx::Value make_test_value(ggml::hrx::ValueId        id,
                                        ggml::hrx::ValueStorageId storage,
                                        ggml::hrx::ValueId        storage_root,
                                        ggml::hrx::ValueId        alias_source,
                                        size_t                    storage_offset,
                                        size_t                    storage_byte_count,
                                        ggml_type                 type,
                                        int64_t                   element_count) {
    ggml::hrx::Value value   = {};
    value.id                 = id;
    value.kind               = ggml::hrx::ValueKind::Transient;
    value.storage            = storage;
    value.storage_root       = storage_root;
    value.alias_source       = alias_source;
    value.storage_offset     = storage_offset;
    value.storage_byte_count = storage_byte_count;
    value.type               = type;
    value.ne                 = { element_count, 1, 1, 1 };
    value.nb                 = { ggml_type_size(type), ggml_type_size(type) * static_cast<size_t>(element_count),
                                 ggml_type_size(type) * static_cast<size_t>(element_count),
                                 ggml_type_size(type) * static_cast<size_t>(element_count) };
    value.element_count      = element_count;
    value.byte_count         = ggml_row_size(type, element_count);
    value.contiguous         = true;
    return value;
}

static std::vector<float> make_input(int64_t hidden_size, int64_t token_count) {
    std::vector<float> data(hidden_size * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 29) - 14) * 0.125f;
    }
    return data;
}

static std::vector<float> make_weight(int64_t hidden_size) {
    std::vector<float> data(hidden_size);
    for (int64_t i = 0; i < hidden_size; ++i) {
        data[i] = 0.5f + static_cast<float>(i % 17) * 0.03125f;
    }
    return data;
}

static std::vector<float> make_router_input(int64_t hidden_size, int64_t token_count) {
    std::vector<float> data(hidden_size * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 41) - 20) * 0.01f;
    }
    return data;
}

static std::vector<float> make_router_weight(int64_t hidden_size, int64_t expert_count) {
    std::vector<float> data(hidden_size * expert_count);
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        for (int64_t column = 0; column < hidden_size; ++column) {
            data[expert * hidden_size + column] = static_cast<float>(((expert + column) % 31) - 15) * 0.0025f;
        }
    }
    return data;
}

static std::vector<float> make_router_logits(int64_t token_count) {
    std::vector<float> data(kQwenRouterExpertCount * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>(i % kQwenRouterRouteCount);
    }
    return data;
}

static std::vector<float> make_flash_query(int64_t token_count) {
    std::vector<float> data(kQwenFlashHeadSize * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 37) - 18) * 0.01f;
    }
    return data;
}

static std::vector<ggml_fp16_t> make_flash_key_value(int64_t token_count, int offset) {
    std::vector<ggml_fp16_t> data(kQwenFlashHeadSize * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        const float value = static_cast<float>(((i + offset) % 31) - 15) * 0.015f;
        data[i]           = ggml_fp32_to_fp16(value);
    }
    return data;
}

static std::vector<ggml_fp16_t> make_flash_mask(int64_t query_token_count, int64_t key_value_token_count) {
    std::vector<ggml_fp16_t> data(query_token_count * key_value_token_count);
    for (int64_t query = 0; query < query_token_count; ++query) {
        for (int64_t key = 0; key < key_value_token_count; ++key) {
            const float value                         = key <= query + 1 ? 0.0f : -10000.0f;
            data[query * key_value_token_count + key] = ggml_fp32_to_fp16(value);
        }
    }
    return data;
}

static std::vector<float> rmsnorm_mul_reference(const std::vector<float> & input,
                                                const std::vector<float> & weight,
                                                int64_t                    hidden_size,
                                                int64_t                    token_count) {
    std::vector<float> output(input.size());
    for (int64_t token = 0; token < token_count; ++token) {
        float sum_squares = 0.0f;
        for (int64_t column = 0; column < hidden_size; ++column) {
            const float value = input[token * hidden_size + column];
            sum_squares += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden_size) + kQwenRmsNormEps);
        for (int64_t column = 0; column < hidden_size; ++column) {
            output[token * hidden_size + column] = input[token * hidden_size + column] * scale * weight[column];
        }
    }
    return output;
}

static std::vector<float> router_projection_reference(const std::vector<float> & input,
                                                      const std::vector<float> & weight,
                                                      int64_t                    hidden_size,
                                                      int64_t                    expert_count,
                                                      int64_t                    token_count) {
    std::vector<float> output(expert_count * token_count);
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t expert = 0; expert < expert_count; ++expert) {
            float sum = 0.0f;
            for (int64_t column = 0; column < hidden_size; ++column) {
                sum += input[token * hidden_size + column] * weight[expert * hidden_size + column];
            }
            output[token * expert_count + expert] = sum;
        }
    }
    return output;
}

static std::vector<float> router_top8_weights_reference(const std::vector<float> & logits, int64_t token_count) {
    std::vector<float> output(kQwenRouterRouteCount * token_count);
    for (int64_t token = 0; token < token_count; ++token) {
        bool    used[kQwenRouterExpertCount]    = {};
        int64_t selected[kQwenRouterRouteCount] = {};
        for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
            int64_t best_expert = -1;
            float   best_value  = -std::numeric_limits<float>::infinity();
            for (int64_t expert = 0; expert < kQwenRouterExpertCount; ++expert) {
                const float value = logits[token * kQwenRouterExpertCount + expert];
                if (!used[expert] &&
                    (best_expert < 0 || value > best_value || (value == best_value && expert < best_expert))) {
                    best_value  = value;
                    best_expert = expert;
                }
            }
            selected[route]   = best_expert;
            used[best_expert] = true;
        }

        float max_selected = -std::numeric_limits<float>::infinity();
        for (const int64_t expert : selected) {
            max_selected = std::max(max_selected, logits[token * kQwenRouterExpertCount + expert]);
        }
        float sum = 0.0f;
        for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
            const float value = std::exp(logits[token * kQwenRouterExpertCount + selected[route]] - max_selected);
            output[token * kQwenRouterRouteCount + route] = value;
            sum += value;
        }
        for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
            output[token * kQwenRouterRouteCount + route] /= sum;
        }
    }
    return output;
}

static std::vector<float> flash_attention_reference(const std::vector<float> &       query,
                                                    const std::vector<ggml_fp16_t> & key,
                                                    const std::vector<ggml_fp16_t> & value,
                                                    const std::vector<ggml_fp16_t> & mask,
                                                    int64_t                          query_token_count,
                                                    int64_t                          key_value_token_count) {
    std::vector<float> output(query_token_count * kQwenFlashHeadSize);
    const float        scale = 1.0f / std::sqrt(static_cast<float>(kQwenFlashHeadSize));
    for (int64_t query_token = 0; query_token < query_token_count; ++query_token) {
        std::vector<float> scores(key_value_token_count);
        float              max_score = -std::numeric_limits<float>::infinity();
        for (int64_t key_token = 0; key_token < key_value_token_count; ++key_token) {
            float dot = 0.0f;
            for (int64_t channel = 0; channel < kQwenFlashHeadSize; ++channel) {
                dot += query[query_token * kQwenFlashHeadSize + channel] *
                       ggml_fp16_to_fp32(key[key_token * kQwenFlashHeadSize + channel]);
            }
            const float score = dot * scale + ggml_fp16_to_fp32(mask[query_token * key_value_token_count + key_token]);
            scores[key_token] = score;
            max_score         = std::max(max_score, score);
        }

        float sum = 0.0f;
        for (float & score : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (int64_t channel = 0; channel < kQwenFlashHeadSize; ++channel) {
            float weighted_sum = 0.0f;
            for (int64_t key_token = 0; key_token < key_value_token_count; ++key_token) {
                const float probability = scores[key_token] / sum;
                weighted_sum += probability * ggml_fp16_to_fp32(value[key_token * kQwenFlashHeadSize + channel]);
            }
            output[query_token * kQwenFlashHeadSize + channel] = weighted_sum;
        }
    }
    return output;
}

static ggml_tensor * build_rmsnorm_mul_graph(ggml_context * ctx,
                                             ggml_tensor *  input,
                                             ggml_tensor *  weight,
                                             float          eps = kQwenRmsNormEps) {
    ggml_tensor * rms = ggml_rms_norm(ctx, input, eps);
    REQUIRE(rms != nullptr);
    ggml_tensor * output = ggml_mul(ctx, rms, weight);
    REQUIRE(output != nullptr);
    return output;
}

static ggml_tensor * build_qwen_flash_attention_graph(ggml_context * ctx,
                                                      ggml_tensor *  query,
                                                      ggml_tensor *  key,
                                                      ggml_tensor *  value,
                                                      ggml_tensor *  mask) {
    ggml_tensor * output = ggml_flash_attn_ext(ctx, query, key, value, mask,
                                               1.0f / std::sqrt(static_cast<float>(kQwenFlashHeadSize)), 0.0f, 0.0f);
    REQUIRE(output != nullptr);
    return output;
}

static ggml_tensor * build_qwen_router_top8_graph(ggml_context * ctx,
                                                  ggml_tensor *  logits,
                                                  ggml_tensor ** route_ids = nullptr,
                                                  int64_t expert_count = kQwenRouterExpertCount,
                                                  int64_t route_count = kQwenRouterRouteCount) {
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    REQUIRE(probs != nullptr);
    ggml_tensor * probs_reshaped = ggml_reshape_3d(ctx, probs, 1, expert_count, logits->ne[1]);
    REQUIRE(probs_reshaped != nullptr);
    ggml_tensor * argsort = ggml_argsort(ctx, probs, GGML_SORT_ORDER_DESC);
    REQUIRE(argsort != nullptr);
    ggml_tensor * topk = ggml_view_2d(ctx, argsort, route_count, logits->ne[1], argsort->nb[1], 0);
    REQUIRE(topk != nullptr);
    if (route_ids != nullptr) {
        *route_ids = topk;
    }
    ggml_tensor * selected = ggml_get_rows(ctx, probs_reshaped, topk);
    REQUIRE(selected != nullptr);
    ggml_tensor * selected_reshaped = ggml_reshape_2d(ctx, selected, route_count, logits->ne[1]);
    REQUIRE(selected_reshaped != nullptr);
    ggml_tensor * sum = ggml_sum_rows(ctx, selected_reshaped);
    REQUIRE(sum != nullptr);
    ggml_tensor * clamped_sum = ggml_clamp(ctx, sum, 1.0e-7f, std::numeric_limits<float>::infinity());
    REQUIRE(clamped_sum != nullptr);
    ggml_tensor * normalized = ggml_div(ctx, selected_reshaped, clamped_sum);
    REQUIRE(normalized != nullptr);
    ggml_tensor * output = ggml_reshape_3d(ctx, normalized, 1, route_count, logits->ne[1]);
    REQUIRE(output != nullptr);
    return output;
}

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    if (!resolved.found()) {
        std::fprintf(stderr, "%s\n", ggml::hrx::format_kernel_resolve_error(resolved, kernel_id).c_str());
    }
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

static std::vector<std::string> scheduled_kernel_sequence(ggml_cgraph * graph) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    if (!scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics)) {
        for (const std::string & error : scheduler.plan().status.errors()) {
            std::fprintf(stderr, "scheduler error: %s\n", error.c_str());
        }
        std::fprintf(stderr, "unsupported: %s\n", diagnostics.unsupported_message.c_str());
        for (const ggml::hrx::DispatchRegistrationAttempt & attempt : diagnostics.match.attempts) {
            std::fprintf(stderr, "  attempt %s matched=%d\n", attempt.name.c_str(), attempt.matched ? 1 : 0);
            if (!attempt.covered_nodes.empty()) {
                std::fprintf(stderr, "    covered:");
                for (size_t node : attempt.covered_nodes) {
                    std::fprintf(stderr, " %zu", node);
                }
                std::fprintf(stderr, "\n");
            }
            for (const std::string & error : attempt.errors) {
                std::fprintf(stderr, "    %s\n", error.c_str());
            }
        }
        std::abort();
    }
    REQUIRE(scheduler.plan().valid());

    std::vector<std::string> names;
    names.reserve(scheduler.plan().dispatches.size());
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        names.push_back(kernel_name_for_id(dispatch.kernel.kernel_id));
    }
    return names;
}

static size_t producer_index_for_tensor(const ggml::hrx::Graph & graph, const ggml_tensor * tensor) {
    const ggml::hrx::Value * value = graph.values().find_tensor(tensor);
    REQUIRE(value != nullptr);
    const ggml::hrx::GraphNode * producer = graph.index().producer(value->id);
    REQUIRE(producer != nullptr);
    size_t index = 0;
    REQUIRE(graph.index().node_index(producer, index));
    return index;
}

static ggml::hrx::ValueId next_plan_value(const ggml::hrx::Graph & graph, const ggml::hrx::CommandPlan & plan) {
    return ggml::hrx::ValueId(
        static_cast<int32_t>(graph.values().size() + plan.transients.size() + plan.completion_counter_requests.size()));
}

static void append_match_to_plan(ggml::hrx::CommandPlan &   plan,
                                 ggml::hrx::DispatchMatch & match,
                                 std::vector<bool> &        covered_nodes) {
    for (ggml::hrx::Dispatch & dispatch : match.initialization_dispatches) {
        plan.initialization_dispatches.push_back(std::move(dispatch));
    }
    for (ggml::hrx::Dispatch & dispatch : match.dispatches) {
        plan.dispatches.push_back(std::move(dispatch));
    }
    for (ggml::hrx::CommandPlanTransient & transient : match.transients) {
        plan.transients.push_back(std::move(transient));
    }
    for (ggml::hrx::CommandPlanConstantInitialization & initialization : match.constant_initializations) {
        plan.constant_initializations.push_back(std::move(initialization));
    }
    for (ggml::hrx::CommandPlanCompletionCounterRequest & request : match.completion_counter_requests) {
        plan.completion_counter_requests.push_back(std::move(request));
    }
    REQUIRE(plan.metadata.append(std::move(match.metadata), plan.status));
    plan.status.append(match.status);
    for (size_t covered_node : match.covered_nodes) {
        REQUIRE(covered_node < covered_nodes.size());
        REQUIRE(!covered_nodes[covered_node]);
        covered_nodes[covered_node] = true;
    }
}

static void match_dispatch_at_index(const ggml::hrx::Graph &            graph,
                                    const ggml::hrx::DispatchRegistry & registry,
                                    ggml::hrx::CommandPlan &            plan,
                                    std::vector<bool> &                 covered_nodes,
                                    size_t                              root_index,
                                    ggml::hrx::DispatchMatch &          match) {
    REQUIRE(root_index < graph.nodes().size());
    const ggml::hrx::DispatchMatchContext context = {
        graph, &graph.nodes()[root_index], root_index, covered_nodes, plan, next_plan_value(graph, plan),
    };
    ggml::hrx::DispatchMatchDiagnostics diagnostics;
    if (!registry.match(context, match, &diagnostics)) {
        std::fprintf(stderr, "manual matcher failed for node %zu %s\n", root_index,
                     ggml_op_name(graph.nodes()[root_index].op));
        for (const ggml::hrx::DispatchRegistrationAttempt & attempt : diagnostics.attempts) {
            std::fprintf(stderr, "  attempt %s matched=%d\n", attempt.name.c_str(), attempt.matched ? 1 : 0);
            for (const std::string & error : attempt.errors) {
                std::fprintf(stderr, "    %s\n", error.c_str());
            }
        }
        std::abort();
    }
    append_match_to_plan(plan, match, covered_nodes);
    REQUIRE(plan.valid());
}

static void require_kernel_subsequence(const std::vector<std::string> & sequence,
                                       const std::vector<std::string> & expected) {
    size_t sequence_index = 0;
    for (const std::string & name : expected) {
        while (sequence_index < sequence.size() && sequence[sequence_index] != name) {
            ++sequence_index;
        }
        if (sequence_index >= sequence.size()) {
            std::fprintf(stderr, "missing expected kernel: %s\nscheduled kernels:\n", name.c_str());
            for (const std::string & scheduled : sequence) {
                std::fprintf(stderr, "  %s\n", scheduled.c_str());
            }
            std::abort();
        }
        ++sequence_index;
    }
}

static void run_alternate_value_alias_lookup_checks() {
    constexpr int64_t element_count = 2048;
    const size_t      full_bytes    = ggml_row_size(GGML_TYPE_F32, element_count);
    const size_t      q8_bytes      = ggml_row_size(GGML_TYPE_Q8_1, element_count);

    ggml::hrx::Graph       graph;
    ggml::hrx::Status      status;
    ggml::hrx::CommandPlan plan;

    const ggml::hrx::ValueId        root(0);
    const ggml::hrx::ValueId        full_alias(1);
    const ggml::hrx::ValueId        partial_alias(2);
    const ggml::hrx::ValueId        q8_alternate(100);
    const ggml::hrx::ValueStorageId storage(0);

    status = graph.values().add_snapshot_storage({ storage, root, full_bytes });
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_test_value(root, storage, root, ggml::hrx::ValueId(), 0, full_bytes, GGML_TYPE_F32, element_count));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_test_value(full_alias, storage, root, root, 0, full_bytes, GGML_TYPE_F32, element_count));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_test_value(partial_alias, storage, root, root, 0, full_bytes, GGML_TYPE_F32, element_count / 2));
    REQUIRE(status.success());

    REQUIRE(plan.metadata.append_alternate_value({ root, q8_alternate, GGML_TYPE_Q8_1, q8_bytes, "q8" }, status));

    const ggml::hrx::CommandPlanAlternateValue * exact =
        ggml::hrx::find_alternate_value(graph, plan, root, GGML_TYPE_Q8_1, q8_bytes);
    REQUIRE(exact != nullptr);
    REQUIRE(exact->alternate_value == q8_alternate);

    const ggml::hrx::CommandPlanAlternateValue * through_full_alias =
        ggml::hrx::find_alternate_value(graph, plan, full_alias, GGML_TYPE_Q8_1, q8_bytes);
    REQUIRE(through_full_alias != nullptr);
    REQUIRE(through_full_alias->alternate_value == q8_alternate);

    const ggml::hrx::CommandPlanAlternateValue * through_partial_alias =
        ggml::hrx::find_alternate_value(graph, plan, partial_alias, GGML_TYPE_Q8_1, q8_bytes);
    REQUIRE(through_partial_alias == nullptr);
}

static void run_set_rows_trusted_producer_checks() {
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    REQUIRE(!ggml::hrx::is_trusted_host_index_producer(nullptr));

    // A leaf with no producer op, but never marked as a graph input (e.g. a resident weight
    // or scratch allocation), must not be trusted: nothing here proves host code, rather than
    // uninitialized or stale device memory, populated it before this command runs.
    ggml_tensor * plain_leaf = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 8);
    REQUIRE(plain_leaf != nullptr);
    REQUIRE(plain_leaf->op == GGML_OP_NONE);
    REQUIRE(!ggml::hrx::is_trusted_host_index_producer(plain_leaf));

    // A leaf explicitly marked as a graph input mirrors llama-kv-cache.cpp's k_idxs/v_idxs:
    // host code writes it via a raw CPU pointer before the graph runs, so it is trusted.
    ggml_tensor * host_input_leaf = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 8);
    REQUIRE(host_input_leaf != nullptr);
    ggml_set_input(host_input_leaf);
    REQUIRE(ggml::hrx::is_trusted_host_index_producer(host_input_leaf));

    // A tensor with a real producer op is never trusted, even if something also marked it as
    // an input -- e.g. the QSA/routing case (a view of a GPU-computed top-k selection). The
    // op check, not just the input flag, is what rules out GPU-computed producers.
    ggml_tensor * source          = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 8);
    ggml_tensor * view_of_gpu_op = ggml_view_1d(ctx, source, 4, 0);
    REQUIRE(source != nullptr);
    REQUIRE(view_of_gpu_op != nullptr);
    REQUIRE(view_of_gpu_op->op != GGML_OP_NONE);
    ggml_set_input(view_of_gpu_op);
    REQUIRE(!ggml::hrx::is_trusted_host_index_producer(view_of_gpu_op));

    for (const char * flag : { "0", "1", "1x" }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_TRUSTED_INDEX_VIEWS", flag);
        ggml_tensor * slice = ggml_view_1d(ctx, host_input_leaf, 6, sizeof(int64_t));
        ggml_tensor * nested = ggml_view_1d(ctx, slice, 2, sizeof(int64_t));
        const bool enabled = std::strcmp(flag, "1") == 0;
        REQUIRE(ggml::hrx::is_trusted_host_index_producer(slice) == enabled);
        REQUIRE(ggml::hrx::is_trusted_host_index_producer(nested) == enabled);
        ggml_tensor * empty = ggml_view_1d(ctx, host_input_leaf, 0, ggml_nbytes(host_input_leaf));
        REQUIRE(ggml::hrx::is_trusted_host_index_producer(empty) == enabled);
        REQUIRE(!ggml::hrx::is_trusted_host_index_producer(view_of_gpu_op));
        ggml_tensor * computed = ggml_add_inplace(ctx, host_input_leaf, source);
        ggml_set_input(computed);
        ggml_tensor * computed_view = ggml_view_1d(ctx, computed, 2, 0);
        REQUIRE(computed_view->view_src == host_input_leaf);
        REQUIRE(!ggml::hrx::is_trusted_host_index_producer(computed_view));
        ggml_tensor * retyped = ggml_view_1d(ctx, host_input_leaf, 2, 0);
        retyped->type = GGML_TYPE_I32;
        REQUIRE(!ggml::hrx::is_trusted_host_index_producer(retyped));
        ggml_tensor * strided = ggml_view_2d(ctx, host_input_leaf, 1, 2, 2 * sizeof(int64_t), 0);
        REQUIRE(!ggml::hrx::is_trusted_host_index_producer(strided));
        ggml_tensor * misaligned = ggml_view_1d(ctx, host_input_leaf, 2, 1);
        REQUIRE(!ggml::hrx::is_trusted_host_index_producer(misaligned));
        ggml_tensor * outside_parent = ggml_view_1d(ctx, nested, 1, 0);
        outside_parent->view_offs = ggml_nbytes(host_input_leaf);
        REQUIRE(!ggml::hrx::is_trusted_host_index_producer(outside_parent));
    }

    ggml_free(ctx);
}

static std::vector<float> make_pattern_f32(size_t element_count, int seed, float scale = 0.01f) {
    std::vector<float> data(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        const int value = static_cast<int>((i * 17 + static_cast<size_t>(seed) * 29) % 97) - 48;
        data[i]         = static_cast<float>(value) * scale;
    }
    return data;
}

static std::vector<int32_t> make_i32_mod_data(size_t element_count, int32_t modulo) {
    std::vector<int32_t> data(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        data[i] = static_cast<int32_t>(i % static_cast<size_t>(modulo));
    }
    return data;
}

static std::vector<int64_t> make_i64_mod_data(size_t element_count, int64_t modulo) {
    std::vector<int64_t> data(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        data[i] = static_cast<int64_t>(i % static_cast<size_t>(modulo));
    }
    return data;
}

static std::vector<ggml_fp16_t> make_pattern_f16(size_t element_count, int seed, float scale = 0.01f) {
    const std::vector<float> f32 = make_pattern_f32(element_count, seed, scale);
    std::vector<ggml_fp16_t> data(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        data[i] = ggml_fp32_to_fp16(f32[i]);
    }
    return data;
}

static std::vector<uint8_t> make_quantized_rows(ggml_type type, int64_t row_length, int64_t row_count, int seed) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    REQUIRE(traits != nullptr);
    REQUIRE(traits->from_float_ref != nullptr);
    const size_t         row_size = ggml_row_size(type, row_length);
    std::vector<uint8_t> data(static_cast<size_t>(row_count) * row_size);
    std::vector<float>   row(static_cast<size_t>(row_length));
    for (int64_t r = 0; r < row_count; ++r) {
        for (int64_t c = 0; c < row_length; ++c) {
            const int value             = static_cast<int>((r * 13 + c * 7 + seed * 31) % 101) - 50;
            row[static_cast<size_t>(c)] = static_cast<float>(value) * 0.005f;
        }
        traits->from_float_ref(row.data(), data.data() + static_cast<size_t>(r) * row_size, row_length);
    }
    return data;
}

static void set_tensor_bytes(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t byte_count) {
    REQUIRE(tensor != nullptr);
    REQUIRE(ggml_nbytes(tensor) == byte_count);
    ggml_backend_tensor_set(tensor, data, 0, byte_count);
    ggml_backend_synchronize(backend);
}

static void set_tensor_pair_bytes(ggml_backend_t cpu_backend,
                                  ggml_tensor *  cpu_tensor,
                                  ggml_backend_t hrx_backend,
                                  ggml_tensor *  hrx_tensor,
                                  const void *   data,
                                  size_t         byte_count) {
    set_tensor_bytes(cpu_backend, cpu_tensor, data, byte_count);
    set_tensor_bytes(hrx_backend, hrx_tensor, data, byte_count);
}

static std::vector<float> get_f32_tensor(ggml_backend_t backend, ggml_tensor * tensor) {
    REQUIRE(tensor != nullptr);
    const size_t       element_count = static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> data(element_count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16(element_count);
        ggml_backend_tensor_get(tensor, f16.data(), 0, f16.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < element_count; ++i) {
            data[i] = ggml_fp16_to_fp32(f16[i]);
        }
    } else {
        REQUIRE(false);
    }
    ggml_backend_synchronize(backend);
    return data;
}

static void require_close(const std::vector<float> & actual,
                          const std::vector<float> & expected,
                          float                      abs_tolerance,
                          float                      rel_tolerance = 0.0f) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff    = std::fabs(actual[i] - expected[i]);
        const float allowed = abs_tolerance + rel_tolerance * std::fabs(expected[i]);
        if (diff > allowed) {
            std::fprintf(stderr, "value mismatch at %zu: actual=%g expected=%g diff=%g allowed=%g\n", i, actual[i],
                         expected[i], diff, allowed);
            std::abort();
        }
    }
}

static ggml_backend_t init_cpu_backend() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(backend != nullptr);
    return backend;
}

struct AttentionPostprocessGraph {
    ggml_tensor * input               = nullptr;
    ggml_tensor * query_weight        = nullptr;
    ggml_tensor * key_weight          = nullptr;
    ggml_tensor * value_weight        = nullptr;
    ggml_tensor * query_norm_weight   = nullptr;
    ggml_tensor * key_norm_weight     = nullptr;
    ggml_tensor * positions           = nullptr;
    ggml_tensor * inverse_frequencies = nullptr;
    ggml_tensor * key_cache           = nullptr;
    ggml_tensor * value_cache         = nullptr;
    ggml_tensor * key_cache_indices   = nullptr;
    ggml_tensor * value_cache_indices = nullptr;
    ggml_tensor * attention_mask      = nullptr;
    ggml_tensor * query_raw           = nullptr;
    ggml_tensor * key_raw             = nullptr;
    ggml_tensor * value_raw           = nullptr;
    ggml_tensor * query_reshape       = nullptr;
    ggml_tensor * query_output        = nullptr;
    ggml_tensor * key_output          = nullptr;
    ggml_tensor * value_output        = nullptr;
};

static AttentionPostprocessGraph build_attention_postprocess_graph(ggml_context * ctx,
                                                                   int64_t        token_count,
                                                                   int64_t        query_head_count,
                                                                   int64_t        key_value_head_count,
                                                                   int64_t        cache_row_count) {
    AttentionPostprocessGraph graph;
    const int64_t             query_size     = query_head_count * kQwenFlashHeadSize;
    const int64_t             key_value_size = key_value_head_count * kQwenFlashHeadSize;

    graph.input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    graph.query_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, query_size);
    graph.key_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, key_value_size);
    graph.value_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, kQwenHiddenSize, key_value_size);
    REQUIRE(graph.input != nullptr);
    REQUIRE(graph.query_weight != nullptr);
    REQUIRE(graph.key_weight != nullptr);
    REQUIRE(graph.value_weight != nullptr);

    ggml_tensor * query_raw = ggml_mul_mat(ctx, graph.query_weight, graph.input);
    ggml_tensor * key_raw   = ggml_mul_mat(ctx, graph.key_weight, graph.input);
    ggml_tensor * value_raw = ggml_mul_mat(ctx, graph.value_weight, graph.input);
    REQUIRE(query_raw != nullptr);
    REQUIRE(key_raw != nullptr);
    REQUIRE(value_raw != nullptr);
    graph.query_raw = query_raw;
    graph.key_raw   = key_raw;
    graph.value_raw = value_raw;

    ggml_tensor * query_reshape = ggml_reshape_3d(ctx, query_raw, kQwenFlashHeadSize, query_head_count, token_count);
    ggml_tensor * key_reshape   = ggml_reshape_3d(ctx, key_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    ggml_tensor * value_reshape =
        ggml_reshape_3d(ctx, value_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    REQUIRE(query_reshape != nullptr);
    REQUIRE(key_reshape != nullptr);
    REQUIRE(value_reshape != nullptr);
    graph.query_reshape = query_reshape;

    graph.query_norm_weight   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize);
    graph.key_norm_weight     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize);
    graph.positions           = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    graph.inverse_frequencies = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize / 2);
    REQUIRE(graph.query_norm_weight != nullptr);
    REQUIRE(graph.key_norm_weight != nullptr);
    REQUIRE(graph.positions != nullptr);
    REQUIRE(graph.inverse_frequencies != nullptr);

    ggml_tensor * query_norm = ggml_rms_norm(ctx, query_reshape, kQwenRmsNormEps);
    ggml_tensor * query_mul  = ggml_mul(ctx, query_norm, graph.query_norm_weight);
    REQUIRE(query_norm != nullptr);
    REQUIRE(query_mul != nullptr);
    graph.query_output = ggml_rope_ext(ctx, query_mul, graph.positions, graph.inverse_frequencies, kQwenFlashHeadSize,
                                       GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    REQUIRE(graph.query_output != nullptr);

    ggml_tensor * key_norm = ggml_rms_norm(ctx, key_reshape, kQwenRmsNormEps);
    ggml_tensor * key_mul  = ggml_mul(ctx, key_norm, graph.key_norm_weight);
    REQUIRE(key_norm != nullptr);
    REQUIRE(key_mul != nullptr);
    ggml_tensor * key_rope = ggml_rope_ext(ctx, key_mul, graph.positions, graph.inverse_frequencies, kQwenFlashHeadSize,
                                           GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    REQUIRE(key_rope != nullptr);

    ggml_tensor * key_cache_rows   = ggml_reshape_2d(ctx, key_rope, key_value_size, token_count);
    ggml_tensor * value_cache_rows = ggml_reshape_2d(ctx, value_reshape, key_value_size, token_count);
    REQUIRE(key_cache_rows != nullptr);
    REQUIRE(value_cache_rows != nullptr);

    graph.key_cache           = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_size, cache_row_count);
    graph.value_cache         = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_size, cache_row_count);
    graph.key_cache_indices   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
    graph.value_cache_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
    REQUIRE(graph.key_cache != nullptr);
    REQUIRE(graph.value_cache != nullptr);
    REQUIRE(graph.key_cache_indices != nullptr);
    REQUIRE(graph.value_cache_indices != nullptr);
    // Match llama-kv-cache.cpp's build_input_k_idxs/build_input_v_idxs: real KV-cache slot
    // indices are host-populated graph-input leaves, which is what makes the SET_ROWS
    // trusted-producer bypass (see DispatchBinding::trusted) apply to them.
    ggml_set_input(graph.key_cache_indices);
    ggml_set_input(graph.value_cache_indices);

    graph.key_output   = ggml_set_rows(ctx, graph.key_cache, key_cache_rows, graph.key_cache_indices);
    graph.value_output = ggml_set_rows(ctx, graph.value_cache, value_cache_rows, graph.value_cache_indices);
    REQUIRE(graph.key_output != nullptr);
    REQUIRE(graph.value_output != nullptr);
    return graph;
}

static ggml_tensor * append_qwen_full_cache_flash_attention_consumer(ggml_context *              ctx,
                                                                     AttentionPostprocessGraph & graph,
                                                                     int64_t                     token_count,
                                                                     int64_t                     query_head_count,
                                                                     int64_t                     key_value_head_count,
                                                                     int64_t                     cache_row_count) {
    ggml_tensor * query_layout =
        ggml_reshape_3d(ctx, graph.query_output, kQwenFlashHeadSize, query_head_count, token_count);
    ggml_tensor * query_permute = ggml_permute(ctx, query_layout, 0, 2, 1, 3);
    ggml_tensor * key_cache_layout =
        ggml_reshape_3d(ctx, graph.key_cache, kQwenFlashHeadSize, key_value_head_count, cache_row_count);
    ggml_tensor * key_permute = ggml_permute(ctx, key_cache_layout, 0, 2, 1, 3);
    ggml_tensor * value_cache_layout =
        ggml_reshape_3d(ctx, graph.value_cache, kQwenFlashHeadSize, key_value_head_count, cache_row_count);
    ggml_tensor * value_permute = ggml_permute(ctx, value_cache_layout, 0, 2, 1, 3);
    graph.attention_mask        = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cache_row_count, token_count);
    REQUIRE(query_layout != nullptr);
    REQUIRE(query_permute != nullptr);
    REQUIRE(key_cache_layout != nullptr);
    REQUIRE(key_permute != nullptr);
    REQUIRE(value_cache_layout != nullptr);
    REQUIRE(value_permute != nullptr);
    REQUIRE(graph.attention_mask != nullptr);
    return build_qwen_flash_attention_graph(ctx, query_permute, key_permute, value_permute, graph.attention_mask);
}

struct QwenFlashAttentionLayoutGraph {
    ggml_tensor * query  = nullptr;
    ggml_tensor * key    = nullptr;
    ggml_tensor * value  = nullptr;
    ggml_tensor * mask   = nullptr;
    ggml_tensor * output = nullptr;
};

static QwenFlashAttentionLayoutGraph build_qwen_flash_attention_layout_graph(ggml_context * ctx,
                                                                             int64_t        query_token_count,
                                                                             int64_t        key_value_token_count) {
    QwenFlashAttentionLayoutGraph graph;
    constexpr int64_t             query_head_count     = 32;
    constexpr int64_t             key_value_head_count = 4;

    ggml_tensor * query_storage =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize, query_head_count, query_token_count);
    ggml_tensor * key_storage =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, key_value_head_count, key_value_token_count);
    ggml_tensor * value_storage =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, key_value_head_count, key_value_token_count);
    REQUIRE(query_storage != nullptr);
    REQUIRE(key_storage != nullptr);
    REQUIRE(value_storage != nullptr);

    graph.query = ggml_permute(ctx, query_storage, 0, 2, 1, 3);
    graph.key   = ggml_permute(ctx, key_storage, 0, 2, 1, 3);
    graph.value = ggml_permute(ctx, value_storage, 0, 2, 1, 3);
    graph.mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_token_count, query_token_count);
    REQUIRE(graph.query != nullptr);
    REQUIRE(graph.key != nullptr);
    REQUIRE(graph.value != nullptr);
    REQUIRE(graph.mask != nullptr);

    graph.output = build_qwen_flash_attention_graph(ctx, graph.query, graph.key, graph.value, graph.mask);
    return graph;
}

static AttentionPostprocessGraph build_decode_attention_qkv_graph(ggml_context * ctx) {
    constexpr int64_t         token_count          = 1;
    constexpr int64_t         query_head_count     = 32;
    constexpr int64_t         key_value_head_count = 4;
    constexpr int64_t         cache_row_count      = 1024;
    AttentionPostprocessGraph graph =
        build_attention_postprocess_graph(ctx, token_count, query_head_count, key_value_head_count, cache_row_count);

    ggml_tensor * hidden_state     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    ggml_tensor * attention_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
    REQUIRE(hidden_state != nullptr);
    REQUIRE(attention_weight != nullptr);
    ggml_tensor * attention_rms = ggml_rms_norm(ctx, hidden_state, kQwenRmsNormEps);
    REQUIRE(attention_rms != nullptr);
    graph.input = ggml_mul(ctx, attention_rms, attention_weight);
    REQUIRE(graph.input != nullptr);

    const int64_t key_value_size = key_value_head_count * kQwenFlashHeadSize;
    ggml_tensor * query_raw      = ggml_mul_mat(ctx, graph.query_weight, graph.input);
    ggml_tensor * key_raw        = ggml_mul_mat(ctx, graph.key_weight, graph.input);
    ggml_tensor * value_raw      = ggml_mul_mat(ctx, graph.value_weight, graph.input);
    REQUIRE(query_raw != nullptr);
    REQUIRE(key_raw != nullptr);
    REQUIRE(value_raw != nullptr);

    ggml_tensor * query_reshape = ggml_reshape_3d(ctx, query_raw, kQwenFlashHeadSize, query_head_count, token_count);
    ggml_tensor * key_reshape   = ggml_reshape_3d(ctx, key_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    ggml_tensor * value_reshape =
        ggml_reshape_3d(ctx, value_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    REQUIRE(query_reshape != nullptr);
    REQUIRE(key_reshape != nullptr);
    REQUIRE(value_reshape != nullptr);

    ggml_tensor * query_norm = ggml_rms_norm(ctx, query_reshape, kQwenRmsNormEps);
    ggml_tensor * query_mul  = ggml_mul(ctx, query_norm, graph.query_norm_weight);
    REQUIRE(query_norm != nullptr);
    REQUIRE(query_mul != nullptr);
    graph.query_output = ggml_rope_ext(ctx, query_mul, graph.positions, graph.inverse_frequencies, kQwenFlashHeadSize,
                                       GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    REQUIRE(graph.query_output != nullptr);

    ggml_tensor * key_norm = ggml_rms_norm(ctx, key_reshape, kQwenRmsNormEps);
    ggml_tensor * key_mul  = ggml_mul(ctx, key_norm, graph.key_norm_weight);
    REQUIRE(key_norm != nullptr);
    REQUIRE(key_mul != nullptr);
    ggml_tensor * key_rope = ggml_rope_ext(ctx, key_mul, graph.positions, graph.inverse_frequencies, kQwenFlashHeadSize,
                                           GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    REQUIRE(key_rope != nullptr);

    ggml_tensor * key_cache_rows   = ggml_reshape_2d(ctx, key_rope, key_value_size, token_count);
    ggml_tensor * value_cache_rows = ggml_reshape_2d(ctx, value_reshape, key_value_size, token_count);
    REQUIRE(key_cache_rows != nullptr);
    REQUIRE(value_cache_rows != nullptr);
    graph.key_output   = ggml_set_rows(ctx, graph.key_cache, key_cache_rows, graph.key_cache_indices);
    graph.value_output = ggml_set_rows(ctx, graph.value_cache, value_cache_rows, graph.value_cache_indices);
    REQUIRE(graph.key_output != nullptr);
    REQUIRE(graph.value_output != nullptr);
    return graph;
}

struct RoutedMoeGraph {
    ggml_tensor * logits       = nullptr;
    ggml_tensor * input        = nullptr;
    ggml_tensor * gate_weight  = nullptr;
    ggml_tensor * up_weight    = nullptr;
    ggml_tensor * down_weight  = nullptr;
    ggml_tensor * hidden_state = nullptr;
    ggml_tensor * norm_weight  = nullptr;
    ggml_tensor * output       = nullptr;
};

static RoutedMoeGraph build_routed_moe_graph(ggml_context * ctx,
                                             ggml_type      down_weight_type,
                                             bool           include_next_rmsnorm) {
    RoutedMoeGraph    graph;
    constexpr int64_t token_count = 1;
    graph.logits                  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
    REQUIRE(graph.logits != nullptr);
    ggml_tensor * route_ids     = nullptr;
    ggml_tensor * route_weights = build_qwen_router_top8_graph(ctx, graph.logits, &route_ids);
    REQUIRE(route_weights != nullptr);
    REQUIRE(route_ids != nullptr);

    graph.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenHiddenSize, 1, token_count);
    graph.gate_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate, kQwenRouterExpertCount);
    graph.up_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate, kQwenRouterExpertCount);
    graph.down_weight =
        ggml_new_tensor_3d(ctx, down_weight_type, kQwenMoeIntermediate, kQwenHiddenSize, kQwenRouterExpertCount);
    graph.hidden_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    REQUIRE(graph.input != nullptr);
    REQUIRE(graph.gate_weight != nullptr);
    REQUIRE(graph.up_weight != nullptr);
    REQUIRE(graph.down_weight != nullptr);
    REQUIRE(graph.hidden_state != nullptr);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, graph.gate_weight, graph.input, route_ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, graph.up_weight, graph.input, route_ids);
    REQUIRE(gate != nullptr);
    REQUIRE(up != nullptr);
    ggml_tensor * glu = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
    REQUIRE(glu != nullptr);
    ggml_tensor * down = ggml_mul_mat_id(ctx, graph.down_weight, glu, route_ids);
    REQUIRE(down != nullptr);
    ggml_tensor * weighted = ggml_mul(ctx, down, route_weights);
    REQUIRE(weighted != nullptr);

    std::vector<ggml_tensor *> route_views;
    route_views.reserve(kQwenRouterRouteCount);
    for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
        ggml_tensor * view = ggml_view_2d(ctx, weighted, kQwenHiddenSize, token_count, weighted->nb[2],
                                          static_cast<size_t>(route) * weighted->nb[1]);
        REQUIRE(view != nullptr);
        route_views.push_back(view);
    }

    ggml_tensor * reduced = route_views.front();
    for (size_t i = 1; i < route_views.size(); ++i) {
        reduced = ggml_add(ctx, reduced, route_views[i]);
        REQUIRE(reduced != nullptr);
    }

    ggml_tensor * residual = ggml_add(ctx, graph.hidden_state, reduced);
    REQUIRE(residual != nullptr);
    graph.output = residual;

    if (include_next_rmsnorm) {
        graph.norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
        REQUIRE(graph.norm_weight != nullptr);
        ggml_tensor * rms = ggml_rms_norm(ctx, residual, kQwenRmsNormEps);
        REQUIRE(rms != nullptr);
        graph.output = ggml_mul(ctx, rms, graph.norm_weight);
        REQUIRE(graph.output != nullptr);
    }

    return graph;
}

static void run_rmsnorm_support_checks() {
    ggml_backend_hrx_device_context device_context  = {};
    ggml_backend_hrx_context        backend_context = {};
    device_context.architecture                     = "gfx1151";
    backend_context.device                          = &device_context;
    const ggml::hrx::GraphExecutor executor(backend_context);

    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    REQUIRE(input != nullptr);
    REQUIRE(weight != nullptr);
    ggml_tensor * output = build_rmsnorm_mul_graph(ctx, input, weight);
    ggml_cgraph * graph  = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);
    const ggml::hrx::GraphSupportResult support = executor.can_execute(*graph);
    REQUIRE(support.supported);
    REQUIRE(support.status.success());

    ggml_tensor * wrong_eps_output = build_rmsnorm_mul_graph(ctx, input, weight, 1.0e-5f);
    ggml_cgraph * wrong_eps_graph  = ggml_new_graph(ctx);
    REQUIRE(wrong_eps_graph != nullptr);
    ggml_build_forward_expand(wrong_eps_graph, wrong_eps_output);
    const ggml::hrx::GraphSupportResult wrong_eps_support = executor.can_execute(*wrong_eps_graph);
    REQUIRE(!wrong_eps_support.supported);

    ggml_tensor * wrong_type_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 1);
    ggml_tensor * wrong_type_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 256);
    REQUIRE(wrong_type_input != nullptr);
    REQUIRE(wrong_type_weight != nullptr);
    ggml_tensor * wrong_type_output = build_rmsnorm_mul_graph(ctx, wrong_type_input, wrong_type_weight);
    ggml_cgraph * wrong_type_graph  = ggml_new_graph(ctx);
    REQUIRE(wrong_type_graph != nullptr);
    ggml_build_forward_expand(wrong_type_graph, wrong_type_output);
    const ggml::hrx::GraphSupportResult wrong_type_support = executor.can_execute(*wrong_type_graph);
    REQUIRE(!wrong_type_support.supported);

    ggml_tensor * wrong_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    REQUIRE(wrong_weight != nullptr);
    ggml_tensor * wrong_weight_output = build_rmsnorm_mul_graph(ctx, input, wrong_weight);
    ggml_cgraph * wrong_weight_graph  = ggml_new_graph(ctx);
    REQUIRE(wrong_weight_graph != nullptr);
    ggml_build_forward_expand(wrong_weight_graph, wrong_weight_output);
    const ggml::hrx::GraphSupportResult wrong_weight_support = executor.can_execute(*wrong_weight_graph);
    REQUIRE(!wrong_weight_support.supported);

    // 131072 rows (48 indexer heads * 2048 real prefill tokens) is the widened per-head cap;
    // one row over that must still fall back rather than silently become unlimited.
    ggml_tensor * row_cap_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 131073);
    ggml_tensor * row_cap_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    REQUIRE(row_cap_input != nullptr);
    REQUIRE(row_cap_weight != nullptr);
    ggml_tensor * row_cap_output = build_rmsnorm_mul_graph(ctx, row_cap_input, row_cap_weight);
    ggml_cgraph * row_cap_graph  = ggml_new_graph(ctx);
    REQUIRE(row_cap_graph != nullptr);
    ggml_build_forward_expand(row_cap_graph, row_cap_output);
    const ggml::hrx::GraphSupportResult row_cap_support = executor.can_execute(*row_cap_graph);
    REQUIRE(!row_cap_support.supported);

    ggml_free(ctx);
}

static void run_rmsnorm_mul_case(int64_t hidden_size, int64_t token_count) {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(hidden_size * token_count * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    REQUIRE(input != nullptr);
    REQUIRE(weight != nullptr);
    ggml_tensor * output = build_rmsnorm_mul_graph(ctx, input, weight);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph), { "qwen3_moe:qwen3_moe_rmsnorm_f32" });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float> input_data  = make_input(hidden_size, token_count);
    const std::vector<float> weight_data = make_weight(hidden_size);
    const std::vector<float> expected    = rmsnorm_mul_reference(input_data, weight_data, hidden_size, token_count);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 5.0e-4f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_router_projection_case(int64_t token_count) {
    static constexpr int64_t kHiddenSize  = 2048;
    static constexpr int64_t kExpertCount = 128;

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(
        (kHiddenSize * token_count + kHiddenSize * kExpertCount + kExpertCount * token_count) * sizeof(float) * 4 +
        1024 * 1024);
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHiddenSize, kExpertCount);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHiddenSize, token_count);
    REQUIRE(weight != nullptr);
    REQUIRE(input != nullptr);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph),
                               { "qwen3_moe:qwen3_moe_router_projection_f32_four_row_wave32" });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float> input_data  = make_router_input(kHiddenSize, token_count);
    const std::vector<float> weight_data = make_router_weight(kHiddenSize, kExpertCount);
    const std::vector<float> expected =
        router_projection_reference(input_data, weight_data, kHiddenSize, kExpertCount, token_count);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 1.0e-2f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_router_top8_case(int64_t token_count) {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size    = static_cast<size_t>(kQwenRouterExpertCount * token_count * sizeof(float) * 16 + 1024 * 1024);
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
    REQUIRE(logits != nullptr);
    ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph), { "qwen3_moe:qwen3_moe_router_top8_f32" });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float> logits_data = make_router_logits(token_count);
    const std::vector<float> expected    = router_top8_weights_reference(logits_data, token_count);

    ggml_backend_tensor_set(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 1.0e-5f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_qwen_flash_attention_case() {
    static constexpr int64_t kQueryTokenCount    = 2;
    static constexpr int64_t kKeyValueTokenCount = 4;

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize, kQueryTokenCount, 1);
    ggml_tensor * key   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, kKeyValueTokenCount, 1);
    ggml_tensor * value = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, kKeyValueTokenCount, 1);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kKeyValueTokenCount, kQueryTokenCount);
    REQUIRE(query != nullptr);
    REQUIRE(key != nullptr);
    REQUIRE(value != nullptr);
    REQUIRE(mask != nullptr);
    ggml_tensor * output = build_qwen_flash_attention_graph(ctx, query, key, value, mask);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph),
                               { "qwen3_moe:qwen3_moe_flash_attention_f32_f16_wmma" });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float>       query_data = make_flash_query(kQueryTokenCount);
    const std::vector<ggml_fp16_t> key_data   = make_flash_key_value(kKeyValueTokenCount, 3);
    const std::vector<ggml_fp16_t> value_data = make_flash_key_value(kKeyValueTokenCount, 11);
    const std::vector<ggml_fp16_t> mask_data  = make_flash_mask(kQueryTokenCount, kKeyValueTokenCount);
    const std::vector<float>       expected =
        flash_attention_reference(query_data, key_data, value_data, mask_data, kQueryTokenCount, kKeyValueTokenCount);

    ggml_backend_tensor_set(query, query_data.data(), 0, query_data.size() * sizeof(float));
    ggml_backend_tensor_set(key, key_data.data(), 0, key_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(value, value_data.data(), 0, value_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 5.0e-2f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_qwen_full_cache_prefill_flash_attention_scheduling_case(int64_t token_count) {
    constexpr int64_t kQueryHeadCount    = 32;
    constexpr int64_t kKeyValueHeadCount = 4;
    constexpr int64_t kFullCacheRowCount = 40960;
    const size_t      active_mask_byte_count =
        static_cast<size_t>(token_count) * static_cast<size_t>(token_count) * sizeof(ggml_fp16_t);

    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    AttentionPostprocessGraph attention =
        build_attention_postprocess_graph(ctx, token_count, kQueryHeadCount, kKeyValueHeadCount, kFullCacheRowCount);
    ggml_tensor * flash_output = append_qwen_full_cache_flash_attention_consumer(
        ctx, attention, token_count, kQueryHeadCount, kKeyValueHeadCount, kFullCacheRowCount);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, attention.key_output);
    ggml_build_forward_expand(graph, attention.value_output);
    ggml_build_forward_expand(graph, flash_output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    const ggml::hrx::DispatchRegistry * registry = ggml::hrx::find_dispatch_registry({ "gfx1151" });
    REQUIRE(registry != nullptr);

    std::vector<bool>        covered_nodes(imported.graph.nodes().size(), false);
    ggml::hrx::CommandPlan   plan;
    ggml::hrx::DispatchMatch postprocess_match;
    match_dispatch_at_index(imported.graph, *registry, plan, covered_nodes,
                            producer_index_for_tensor(imported.graph, attention.query_reshape), postprocess_match);
    ggml::hrx::DispatchMatch flash_match;
    match_dispatch_at_index(imported.graph, *registry, plan, covered_nodes,
                            producer_index_for_tensor(imported.graph, flash_output), flash_match);

    const ggml::hrx::Value * mask_value = imported.graph.values().find_tensor(attention.attention_mask);
    REQUIRE(mask_value != nullptr);
    const ggml::hrx::CommandPlanAlternateValue * compact_mask =
        ggml::hrx::find_alternate_value(plan, mask_value->id, GGML_TYPE_F16, active_mask_byte_count);
    REQUIRE(compact_mask != nullptr);

    const ggml::hrx::Dispatch * metadata_dispatch = nullptr;
    for (const ggml::hrx::Dispatch & dispatch : plan.initialization_dispatches) {
        if (kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:qwen_attention_metadata") {
            metadata_dispatch = &dispatch;
        }
    }
    REQUIRE(metadata_dispatch != nullptr);
    REQUIRE(metadata_dispatch->kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(metadata_dispatch->kernel.integer_parameters.at("context_capacity") == token_count);
    REQUIRE(metadata_dispatch->bindings.size() == 5);
    REQUIRE(metadata_dispatch->bindings[4].value == compact_mask->alternate_value);
    REQUIRE(metadata_dispatch->bindings[4].length == active_mask_byte_count);

    const ggml::hrx::Dispatch * flash_dispatch = nullptr;
    for (const ggml::hrx::Dispatch & dispatch : plan.dispatches) {
        if (kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:qwen3_moe_flash_attention_f32_f16_wmma") {
            flash_dispatch = &dispatch;
        }
    }
    REQUIRE(flash_dispatch != nullptr);
    REQUIRE(flash_dispatch->kernel.integer_parameters.at("query_token_count") == token_count);
    REQUIRE(flash_dispatch->kernel.integer_parameters.at("key_value_token_count") == token_count);
    REQUIRE(flash_dispatch->bindings.size() == 5);
    REQUIRE(flash_dispatch->bindings[3].value == compact_mask->alternate_value);
    REQUIRE(flash_dispatch->bindings[3].length == active_mask_byte_count);

    ggml_free(ctx);
}

static void run_qwen_decode_split_flash_attention_scheduling_case(int64_t query_token_count,
                                                                  int64_t key_value_token_count) {
    constexpr int64_t query_head_count     = 32;
    constexpr int64_t key_value_head_count = 4;
    constexpr int64_t hidden_size          = query_head_count * kQwenFlashHeadSize;
    const size_t      q8_row_bytes         = ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
    const size_t      q8_output_bytes      = static_cast<size_t>(query_token_count) * q8_row_bytes;
    const int64_t     key_value_capacity   = (key_value_token_count + 63) / 64 * 64;
    const int64_t     key_value_blocks     = key_value_capacity / 64;
    const size_t      partial_scalar_bytes =
        static_cast<size_t>(key_value_head_count * key_value_blocks * 16) * sizeof(float);
    const size_t partial_output_bytes =
        static_cast<size_t>(key_value_head_count * key_value_blocks * 16 * kQwenFlashHeadSize) * sizeof(ggml_fp16_t);

    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    QwenFlashAttentionLayoutGraph attention =
        build_qwen_flash_attention_layout_graph(ctx, query_token_count, key_value_token_count);

    ggml_cgraph * cgraph = ggml_new_graph(ctx);
    REQUIRE(cgraph != nullptr);
    ggml_build_forward_expand(cgraph, attention.output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*cgraph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.dispatches.size() == static_cast<size_t>(query_token_count));
    REQUIRE(plan.transients.size() == 4);
    REQUIRE(plan.completion_counter_requests.size() == 1);

    REQUIRE(plan.transients[0].size == partial_scalar_bytes);
    REQUIRE(plan.transients[1].size == partial_scalar_bytes);
    REQUIRE(plan.transients[2].size == partial_output_bytes);
    REQUIRE(plan.transients[3].size == q8_output_bytes);
    REQUIRE(plan.completion_counter_requests[0].count == key_value_head_count);

    const ggml::hrx::Value * query_value  = imported.graph.values().find_tensor(attention.query);
    const ggml::hrx::Value * mask_value   = imported.graph.values().find_tensor(attention.mask);
    const ggml::hrx::Value * output_value = imported.graph.values().find_tensor(attention.output);
    REQUIRE(query_value != nullptr);
    REQUIRE(mask_value != nullptr);
    REQUIRE(output_value != nullptr);
    const ggml::hrx::CommandPlanAlternateValue * q8_alternate =
        plan.metadata.find_alternate_value(output_value->id, GGML_TYPE_Q8_1, q8_output_bytes);
    REQUIRE(q8_alternate != nullptr);
    REQUIRE(q8_alternate->alternate_value == plan.transients[3].value);

    const size_t query_row_bytes  = static_cast<size_t>(hidden_size) * sizeof(float);
    const size_t mask_row_bytes   = static_cast<size_t>(key_value_token_count) * sizeof(ggml_fp16_t);
    const size_t output_row_bytes = query_row_bytes;
    for (int64_t row = 0; row < query_token_count; ++row) {
        const ggml::hrx::Dispatch & dispatch = plan.dispatches[static_cast<size_t>(row)];
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
                "qwen3_moe:qwen3_moe_flash_attention_decode_split_f32_f16_wmma_next_q8");
        REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") == key_value_token_count);
        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.attention.key_value_token_capacity") ==
                std::to_string(key_value_capacity));
        REQUIRE(dispatch.bindings.size() == 10);
        REQUIRE(dispatch.bindings[0].value == query_value->id);
        REQUIRE(dispatch.bindings[0].offset == static_cast<size_t>(row) * query_value->nb[1]);
        REQUIRE(dispatch.bindings[0].length == query_row_bytes);
        REQUIRE(dispatch.bindings[3].value == mask_value->id);
        REQUIRE(dispatch.bindings[3].offset == static_cast<size_t>(row) * mask_value->nb[1]);
        REQUIRE(dispatch.bindings[3].length == mask_row_bytes);
        REQUIRE(dispatch.bindings[8].value == output_value->id);
        REQUIRE(dispatch.bindings[8].offset == static_cast<size_t>(row) * output_value->nb[2]);
        REQUIRE(dispatch.bindings[8].length == output_row_bytes);
        REQUIRE(dispatch.bindings[9].value == q8_alternate->alternate_value);
        REQUIRE(dispatch.bindings[9].offset == static_cast<size_t>(row) * q8_row_bytes);
        REQUIRE(dispatch.bindings[9].length == q8_row_bytes);
    }

    ggml_free(ctx);
}

static void run_qwen_decode_attention_output_next_q8_scheduling_case(bool include_get_rows_selectors) {
    constexpr int64_t query_token_count     = 1;
    constexpr int64_t key_value_token_count = 512;
    constexpr int64_t attention_hidden_size = 4096;
    const size_t      attention_q8_bytes    = ggml_row_size(GGML_TYPE_Q8_1, attention_hidden_size);
    const size_t      next_q8_bytes         = ggml_row_size(GGML_TYPE_Q8_1, kQwenHiddenSize);

    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    QwenFlashAttentionLayoutGraph attention =
        build_qwen_flash_attention_layout_graph(ctx, query_token_count, key_value_token_count);
    ggml_tensor * attention_output = ggml_reshape_2d(ctx, attention.output, attention_hidden_size, query_token_count);
    ggml_tensor * output_weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, attention_hidden_size, kQwenHiddenSize);
    ggml_tensor * projection       = ggml_mul_mat(ctx, output_weight, attention_output);
    ggml_tensor * residual_input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, query_token_count);
    ggml_tensor * selected_projection = projection;
    ggml_tensor * selected_residual   = residual_input;
    ggml_tensor * row_indices         = nullptr;
    if (include_get_rows_selectors) {
        row_indices         = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        selected_projection = ggml_get_rows(ctx, projection, row_indices);
        selected_residual   = ggml_get_rows(ctx, residual_input, row_indices);
        REQUIRE(row_indices != nullptr);
        REQUIRE(selected_projection != nullptr);
        REQUIRE(selected_residual != nullptr);
    }
    ggml_tensor * residual    = ggml_add(ctx, selected_projection, selected_residual);
    ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
    ggml_tensor * rms         = ggml_rms_norm(ctx, residual, kQwenRmsNormEps);
    ggml_tensor * normalized  = ggml_mul(ctx, rms, norm_weight);
    REQUIRE(attention_output != nullptr);
    REQUIRE(output_weight != nullptr);
    REQUIRE(projection != nullptr);
    REQUIRE(residual_input != nullptr);
    REQUIRE(residual != nullptr);
    REQUIRE(norm_weight != nullptr);
    REQUIRE(rms != nullptr);
    REQUIRE(normalized != nullptr);

    ggml_cgraph * cgraph = ggml_new_graph(ctx);
    REQUIRE(cgraph != nullptr);
    ggml_build_forward_expand(cgraph, normalized);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*cgraph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.dispatches.size() == 2);
    REQUIRE(plan.completion_counter_requests.size() == 2);

    const ggml::hrx::Dispatch & projection_dispatch = plan.dispatches.back();
    REQUIRE(kernel_name_for_id(projection_dispatch.kernel.kernel_id) ==
            "qwen3_moe:qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8");
    REQUIRE(projection_dispatch.kernel.integer_parameters.at("token_count") == query_token_count);
    REQUIRE(projection_dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.input_size") ==
            std::to_string(attention_hidden_size));
    REQUIRE(projection_dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_size") ==
            std::to_string(kQwenHiddenSize));
    REQUIRE(projection_dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_accumulation") == "1");
    REQUIRE(projection_dispatch.bindings.size() == 7);

    const ggml::hrx::Value * flash_output_value   = imported.graph.values().find_tensor(attention.output);
    const ggml::hrx::Value * residual_input_value = imported.graph.values().find_tensor(residual_input);
    const ggml::hrx::Value * residual_value       = imported.graph.values().find_tensor(residual);
    const ggml::hrx::Value * normalized_value     = imported.graph.values().find_tensor(normalized);
    REQUIRE(flash_output_value != nullptr);
    REQUIRE(residual_input_value != nullptr);
    REQUIRE(residual_value != nullptr);
    REQUIRE(normalized_value != nullptr);
    const ggml::hrx::CommandPlanAlternateValue * attention_q8 =
        ggml::hrx::find_alternate_value(plan, flash_output_value->id, GGML_TYPE_Q8_1, attention_q8_bytes);
    const ggml::hrx::CommandPlanAlternateValue * next_q8 =
        ggml::hrx::find_alternate_value(plan, normalized_value->id, GGML_TYPE_Q8_1, next_q8_bytes);
    REQUIRE(attention_q8 != nullptr);
    REQUIRE(next_q8 != nullptr);
    REQUIRE(projection_dispatch.bindings[0].value == attention_q8->alternate_value);
    REQUIRE(projection_dispatch.bindings[2].value == residual_value->id);
    REQUIRE(projection_dispatch.bindings[4].value == normalized_value->id);
    REQUIRE(projection_dispatch.bindings[6].value == next_q8->alternate_value);

    const ggml::hrx::Value * aliased_residual = imported.graph.values().find(residual_value->id);
    REQUIRE(aliased_residual != nullptr);
    REQUIRE(aliased_residual->alias_source == residual_input_value->id);

    ggml_free(ctx);
}

static void run_add_f32_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params cpu_params = {};
    cpu_params.mem_size         = 256 * 1024;
    cpu_params.no_alloc         = true;
    ggml_init_params hrx_params = cpu_params;
    ggml_context *   cpu_ctx    = ggml_init(cpu_params);
    ggml_context *   hrx_ctx    = ggml_init(hrx_params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t element_count = 257;
    ggml_tensor *     cpu_a         = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     cpu_b         = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     cpu_output    = ggml_add(cpu_ctx, cpu_a, cpu_b);
    ggml_tensor *     hrx_a         = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     hrx_b         = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     hrx_output    = ggml_add(hrx_ctx, hrx_a, hrx_b);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "qwen3_moe:ggml_add_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> a = make_pattern_f32(element_count, 1, 0.125f);
    const std::vector<float> b = make_pattern_f32(element_count, 2, 0.25f);
    set_tensor_pair_bytes(cpu_backend, cpu_a, hrx_backend, hrx_a, a.data(), a.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_b, hrx_backend, hrx_b, b.data(), b.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 0.0f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_gather_add_f32_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t hidden_size        = 256;
    constexpr int64_t source_token_count = 11;
    constexpr int64_t output_token_count = 5;
    ggml_tensor *     cpu_a              = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor *     cpu_b              = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor *     cpu_ids            = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, output_token_count);
    ggml_tensor *     cpu_rows_a         = ggml_get_rows(cpu_ctx, cpu_a, cpu_ids);
    ggml_tensor *     cpu_rows_b         = ggml_get_rows(cpu_ctx, cpu_b, cpu_ids);
    ggml_tensor *     cpu_output         = ggml_add(cpu_ctx, cpu_rows_a, cpu_rows_b);

    ggml_tensor * hrx_a      = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor * hrx_b      = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor * hrx_ids    = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, output_token_count);
    ggml_tensor * hrx_rows_a = ggml_get_rows(hrx_ctx, hrx_a, hrx_ids);
    ggml_tensor * hrx_rows_b = ggml_get_rows(hrx_ctx, hrx_b, hrx_ids);
    ggml_tensor * hrx_output = ggml_add(hrx_ctx, hrx_rows_a, hrx_rows_b);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "qwen3_moe:ggml_gather_add_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float>   a   = make_pattern_f32(hidden_size * source_token_count, 3, 0.05f);
    const std::vector<float>   b   = make_pattern_f32(hidden_size * source_token_count, 4, 0.075f);
    const std::vector<int32_t> ids = { 9, 3, 7, 1, 5 };
    set_tensor_pair_bytes(cpu_backend, cpu_a, hrx_backend, hrx_a, a.data(), a.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_b, hrx_backend, hrx_b, b.data(), b.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int32_t));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 0.0f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_token_embedding_q4k_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t vocabulary_count = 64;
    constexpr int64_t hidden_size      = kQwenHiddenSize;
    constexpr int64_t token_count      = 7;
    ggml_tensor *     cpu_weight       = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_Q4_K, hidden_size, vocabulary_count);
    ggml_tensor *     cpu_ids          = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, token_count);
    ggml_tensor *     cpu_output       = ggml_get_rows(cpu_ctx, cpu_weight, cpu_ids);
    ggml_tensor *     hrx_weight       = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_Q4_K, hidden_size, vocabulary_count);
    ggml_tensor *     hrx_ids          = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, token_count);
    ggml_tensor *     hrx_output       = ggml_get_rows(hrx_ctx, hrx_weight, hrx_ids);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "qwen3_moe:qwen_token_embedding_q4k" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> weight = make_quantized_rows(GGML_TYPE_Q4_K, hidden_size, vocabulary_count, 5);
    const std::vector<int32_t> ids    = { 3, 17, 29, 41, 53, 7, 19 };
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int32_t));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_dense_matmul_cpu_reference_case(ggml_type    weight_type,
                                                const char * expected_kernel,
                                                int64_t      token_count,
                                                int64_t      output_size) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(32 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t input_size = kQwenHiddenSize;
    ggml_tensor *     cpu_weight = ggml_new_tensor_2d(cpu_ctx, weight_type, input_size, output_size);
    ggml_tensor *     cpu_input  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     cpu_output = ggml_mul_mat(cpu_ctx, cpu_weight, cpu_input);
    ggml_tensor *     hrx_weight = ggml_new_tensor_2d(hrx_ctx, weight_type, input_size, output_size);
    ggml_tensor *     hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     hrx_output = ggml_mul_mat(hrx_ctx, hrx_weight, hrx_input);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { expected_kernel });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> weight = make_quantized_rows(weight_type, input_size, output_size, 6);
    const std::vector<float>   input  = make_pattern_f32(input_size * token_count, 7, 0.01f);
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output),
                  kDenseWmmaAbsTolerance, kDenseWmmaRelTolerance);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void require_zero_device_timing_calls(const ggml::hrx::DeviceTimingTestSnapshot & snapshot) {
    REQUIRE(snapshot.init_attempts == 0);
    REQUIRE(snapshot.event_create_calls == 0);
    REQUIRE(snapshot.event_record_calls == 0);
    REQUIRE(snapshot.event_synchronize_calls == 0);
    REQUIRE(snapshot.event_elapsed_calls == 0);
    REQUIRE(snapshot.event_destroy_calls == 0);
    REQUIRE(snapshot.graph_measurements == 0);
}

static void run_device_timing_schedule_checks() {
    for (const char * flag : { static_cast<const char *>(nullptr), "", "0" }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_DEVICE_TIMING", flag);
        ggml::hrx::reset_device_timing_test_snapshot();
        ggml::hrx::device_timing_test_probe_flag_off_path();
        const auto snapshot = ggml::hrx::device_timing_test_snapshot();
        REQUIRE(!snapshot.env_enabled);
        require_zero_device_timing_calls(snapshot);
    }
}

static void run_device_timing_gpu_checks() {
    {
        ScopedHrxEnvironment environment("HRX_ENABLE_DEVICE_TIMING", "1");
        ggml::hrx::reset_device_timing_test_snapshot();
        run_rmsnorm_mul_case(256, 1);
        const auto snapshot = ggml::hrx::device_timing_test_snapshot();
        REQUIRE(snapshot.env_enabled);
        REQUIRE(snapshot.internals_available);
        REQUIRE(snapshot.init_attempts >= 1);
        REQUIRE(snapshot.event_create_calls >= 2);
        REQUIRE(snapshot.event_record_calls >= 2);
        REQUIRE(snapshot.event_synchronize_calls >= 1);
        REQUIRE(snapshot.event_elapsed_calls >= 1);
        REQUIRE(snapshot.graph_measurements >= 1);
        REQUIRE(snapshot.last_graph_ms >= 0.0);
        REQUIRE(snapshot.last_graph_ms < 60000.0);
    }
    {
        ScopedHrxEnvironment environment("HRX_ENABLE_DEVICE_TIMING", "0");
        ggml::hrx::reset_device_timing_test_snapshot();
        run_rmsnorm_mul_case(256, 1);
        const auto snapshot = ggml::hrx::device_timing_test_snapshot();
        REQUIRE(!snapshot.env_enabled);
        require_zero_device_timing_calls(snapshot);
    }
    std::fprintf(stderr, "Device timing checks passed (HIP graph timing on/off, no schedule-path overhead)\n");
}

static void run_q8_embedding_scheduling_checks() {
    for (const char * flag : { static_cast<const char *>(nullptr), "", "0", "true", "01", "1x", "1" }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_Q8_EMBEDDING", flag);
        for (int64_t tokens : { 1, 7, 32, 2048 }) {
            // ids marked as a graph input mirrors llama-graph.cpp's build_inp_embd() (inp->tokens);
            // unmarked mirrors any hypothetical GPU-computed producer, which must stay untrusted.
            for (bool trusted_producer : { false, true }) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                ggml_context * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2560, 248320);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
                if (trusted_producer) {
                    ggml_set_input(ids);
                }
                ggml_tensor * output = ggml_get_rows(ctx, weight, ids);
                ggml_cgraph * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, output);
                const bool enabled = flag != nullptr && std::strcmp(flag, "1") == 0;
                REQUIRE(ggml::hrx::supports_q8_embedding_dispatch(output) == enabled);
                auto imported = ggml::hrx::import_ggml_graph(*graph);
                REQUIRE(imported.valid());
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == enabled);
                if (enabled) {
                    const auto & plan = scheduler.plan();
                    REQUIRE(plan.dispatches.size() == 1);
                    REQUIRE(plan.transients.empty());
                    REQUIRE(plan.constant_initializations.empty());
                    const auto & dispatch = plan.dispatches[0];
                    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "hrx_owned:ggml_token_embedding_q8_0_f32");
                    REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == 2560);
                    REQUIRE(dispatch.kernel.integer_parameters.at("vocabulary_count") == 248320);
                    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tokens);
                    REQUIRE(dispatch.kernel.compile_parameters.empty());
                    REQUIRE(dispatch.bindings.size() == 3);
                    REQUIRE(dispatch.bindings[0].value == imported.graph.values().find_tensor(weight)->id);
                    REQUIRE(dispatch.bindings[0].length == ggml_nbytes(weight));
                    REQUIRE(dispatch.bindings[1].value == imported.graph.values().find_tensor(ids)->id);
                    REQUIRE(dispatch.bindings[1].length == ggml_nbytes(ids));
                    REQUIRE(dispatch.bindings[1].trusted == trusted_producer);
                    REQUIRE(dispatch.bindings[2].length == ggml_nbytes(output));
                    for (bool initialization : { false, true }) {
                        ggml::hrx::PreparedCommandProgram prepared;
                        ggml::hrx::PreparedCommand command;
                        command.kind = ggml::hrx::CommandKind::Kernel;
                        command.kernel.specialization = dispatch.kernel;
                        command.kernel.bindings.resize(dispatch.bindings.size());
                        for (size_t i = 0; i < dispatch.bindings.size(); ++i) {
                            command.kernel.bindings[i].binding.trusted = dispatch.bindings[i].trusted;
                        }
                        (initialization ? prepared.initialization_commands : prepared.commands).push_back(command);
                        ggml::hrx::RecordedCommandGraph recorded;
                        const auto replay = ggml::hrx::bind_and_launch_recorded_command_graph(
                            {}, {}, {}, prepared, recorded);
                        REQUIRE(!replay.success);
                        if (trusted_producer) {
                            REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::BuildFailed);
                        } else {
                            REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::Ineligible);
                            REQUIRE(replay.ineligible_reason == "q8_embedding_requires_index_validation");
                        }
                    }
                }
                ggml_free(ctx);
            }
        }
    }
    ScopedHrxEnvironment environment("HRX_ENABLE_Q8_EMBEDDING", "1");
    for (int malformed = 0; malformed < 19; ++malformed) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0,
                                                 malformed == 0 ? 32800 : 2560,
                                                 malformed == 1 ? 262145 : 65);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, malformed == 2 ? 2049 : 7);
        ggml_tensor * output = ggml_get_rows(ctx, weight, ids);
        if (malformed == 3) { weight->type = GGML_TYPE_Q4_K; }
        if (malformed == 4) { ids->type = GGML_TYPE_I64; }
        if (malformed == 5) { output->type = GGML_TYPE_F16; }
        if (malformed == 6) { weight->nb[1] += 34; }
        if (malformed == 7) { ids->nb[0] *= 2; }
        if (malformed == 8) { output->nb[1] += 4; }
        if (malformed == 9) { weight->ne[2] = 2; }
        if (malformed == 10) { ids->ne[1] = 2; }
        if (malformed == 11) { output->ne[2] = 2; }
        if (malformed == 12) { output->ne[0] -= 32; }
        if (malformed == 13) { weight->op = GGML_OP_DUP; }
        if (malformed == 14) { output->view_src = weight; }
        if (malformed == 15) { output->view_src = ids; }
        if (malformed == 16) {
            weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 32768, 262144);
            output = ggml_get_rows(ctx, weight, ids);
        }
        if (malformed == 17) { ids = ggml_view_1d(ctx, ids, 7, 0); output->src[1] = ids; }
        if (malformed == 18) { weight = ggml_view_2d(ctx, weight, 2560, 65, weight->nb[1], 0); output->src[0] = weight; }
        REQUIRE(!ggml::hrx::supports_q8_embedding_dispatch(output));
        // Malformed views/non-leaf nodes need not be well-formed GGML graphs.
        if (malformed < 13 || malformed == 16) {
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);
            auto imported = ggml::hrx::import_ggml_graph(*graph);
            if (imported.valid()) {
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
            }
        }
        ggml_free(ctx);
    }
}

static void run_q4_embedding_scheduling_checks() {
    ScopedHrxEnvironment q8("HRX_ENABLE_Q8_EMBEDDING", "1");
    for (const char * flag : { static_cast<const char *>(nullptr), "", "0", "true", "01", "1x", "1" }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_Q4_EMBEDDING", flag);
        for (int64_t tokens : { 1, 2, 3, 4, 8, 9 }) {
            // ids marked as a graph input mirrors qwen4exp.cpp's MTP draft embedding (inp->tokens);
            // unmarked mirrors any hypothetical GPU-computed producer, which must stay untrusted.
            for (bool trusted_producer : { false, true }) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                ggml_context * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2560, 248320);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
                if (trusted_producer) {
                    ggml_set_input(ids);
                }
                ggml_tensor * output = ggml_get_rows(ctx, weight, ids);
                const bool enabled = flag != nullptr && std::strcmp(flag, "1") == 0 && tokens <= 8;
                REQUIRE(ggml::hrx::supports_q4_embedding_dispatch(output) == enabled);
                REQUIRE(!ggml::hrx::supports_q8_embedding_dispatch(output));
                ggml_cgraph * graph = ggml_new_graph(ctx);
                // qwen4exp MTP build_norm(tok_embd, nextn.enorm) requires learned gamma.
                ggml_tensor * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560);
                ggml_tensor * normalized = ggml_mul(ctx, ggml_rms_norm(ctx, output, kQwenRmsNormEps), gamma);
                ggml_build_forward_expand(graph, normalized);
                auto imported = ggml::hrx::import_ggml_graph(*graph);
                REQUIRE(imported.valid());
                ggml::hrx::DispatchScheduler scheduler;
                ggml::hrx::DispatchScheduleDiagnostics diagnostics;
                const bool scheduled = scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics);
                if (scheduled != enabled) {
                    std::fprintf(stderr, "Q4 embedding scheduling: flag=%s T=%lld expected=%d scheduled=%d\n",
                                 flag == nullptr ? "<unset>" : flag, static_cast<long long>(tokens),
                                 enabled ? 1 : 0, scheduled ? 1 : 0);
                    for (const auto & error : scheduler.plan().status.errors()) {
                        std::fprintf(stderr, "scheduler error: %s\n", error.c_str());
                    }
                    std::fprintf(stderr, "unsupported: %s\n", diagnostics.unsupported_message.c_str());
                    for (const auto & attempt : diagnostics.match.attempts) {
                        std::fprintf(stderr, "  attempt %s matched=%d\n", attempt.name.c_str(), attempt.matched ? 1 : 0);
                    }
                }
                REQUIRE(scheduled == enabled);
                if (enabled) {
                    const auto & plan = scheduler.plan();
                    REQUIRE(plan.dispatches.size() == 2);
                    REQUIRE(kernel_name_for_id(plan.dispatches[1].kernel.kernel_id) == "qwen3_moe:qwen3_moe_rmsnorm_f32");
                    REQUIRE(plan.dispatches[1].bindings.size() == 3);
                    REQUIRE(plan.dispatches[1].bindings[0].value == imported.graph.values().find_tensor(output)->id);
                    REQUIRE(plan.dispatches[1].bindings[1].value == imported.graph.values().find_tensor(gamma)->id);
                    REQUIRE(plan.dispatches[1].bindings[2].value == imported.graph.values().find_tensor(normalized)->id);
                    const auto & dispatch = plan.dispatches.front();
                    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "hrx_owned:ggml_token_embedding_q4_k_f32");
                    REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == 2560);
                    REQUIRE(dispatch.kernel.integer_parameters.at("vocabulary_count") == 248320);
                    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tokens);
                    REQUIRE(dispatch.kernel.compile_parameters.empty());
                    REQUIRE(dispatch.bindings.size() == 3);
                    REQUIRE(dispatch.bindings[0].value == imported.graph.values().find_tensor(weight)->id);
                    REQUIRE(dispatch.bindings[0].length == ggml_nbytes(weight));
                    REQUIRE(dispatch.bindings[1].value == imported.graph.values().find_tensor(ids)->id);
                    REQUIRE(dispatch.bindings[1].length == ggml_nbytes(ids));
                    REQUIRE(dispatch.bindings[1].trusted == trusted_producer);
                    REQUIRE(dispatch.bindings[2].value == imported.graph.values().find_tensor(output)->id);
                    REQUIRE(dispatch.bindings[2].length == ggml_nbytes(output));
                    for (bool initialization : { false, true }) {
                        ggml::hrx::PreparedCommandProgram prepared;
                        ggml::hrx::PreparedCommand command;
                        command.kind = ggml::hrx::CommandKind::Kernel;
                        command.kernel.specialization = dispatch.kernel;
                        command.kernel.bindings.resize(dispatch.bindings.size());
                        for (size_t i = 0; i < dispatch.bindings.size(); ++i) {
                            command.kernel.bindings[i].binding.trusted = dispatch.bindings[i].trusted;
                        }
                        (initialization ? prepared.initialization_commands : prepared.commands).push_back(command);
                        ggml::hrx::RecordedCommandGraph recorded;
                        const auto replay = ggml::hrx::bind_and_launch_recorded_command_graph(
                            {}, {}, {}, prepared, recorded);
                        REQUIRE(!replay.success);
                        if (trusted_producer) {
                            REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::BuildFailed);
                        } else {
                            REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::Ineligible);
                            REQUIRE(replay.ineligible_reason == "q4_embedding_requires_index_validation");
                        }
                    }
                }
                // Original Qwen3 Q4 kernel remains selected regardless of the new flag.
                weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 65);
                output = ggml_get_rows(ctx, weight, ids);
                REQUIRE(!ggml::hrx::supports_q4_embedding_dispatch(output));
                graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, output);
                require_kernel_subsequence(scheduled_kernel_sequence(graph), { "qwen3_moe:qwen_token_embedding_q4k" });
                ggml_free(ctx);
            }
        }
    }
    ScopedHrxEnvironment environment("HRX_ENABLE_Q4_EMBEDDING", "1");
    ScopedHrxEnvironment q8_disabled("HRX_ENABLE_Q8_EMBEDDING", "0");
    for (int malformed = 0; malformed < 16; ++malformed) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K,
                                                 malformed == 0 ? 2816 : 2560,
                                                 malformed == 1 ? 262145 : 65);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ggml_tensor * output = ggml_get_rows(ctx, weight, ids);
        if (malformed == 2) { weight->type = GGML_TYPE_Q8_0; }
        if (malformed == 3) { ids->type = GGML_TYPE_I64; }
        if (malformed == 4) { output->type = GGML_TYPE_F16; }
        if (malformed == 5) { weight->nb[1] += 144; }
        if (malformed == 6) { ids->nb[0] *= 2; }
        if (malformed == 7) { output->nb[1] += 4; }
        if (malformed == 8) { weight->ne[2] = 2; }
        if (malformed == 9) { ids->ne[1] = 2; }
        if (malformed == 10) { output->ne[2] = 2; }
        if (malformed == 11) { output->ne[0] -= 256; }
        if (malformed == 12) { weight->op = GGML_OP_DUP; }
        if (malformed == 13) { output->view_src = weight; }
        if (malformed == 14) { output->src[1] = ggml_view_1d(ctx, ids, 4, 0); }
        if (malformed == 15) { output->src[0] = ggml_view_2d(ctx, weight, 2560, 65, weight->nb[1], 0); }
        REQUIRE(!ggml::hrx::supports_q4_embedding_dispatch(output));
        if (malformed < 12) {
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);
            auto imported = ggml::hrx::import_ggml_graph(*graph);
            if (imported.valid()) {
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
            }
        }
        ggml_free(ctx);
    }
}

static void run_set_rows_scheduling_checks() {
    // End-to-end: the trust flag match_set_rows_dispatch (dispatch-copy.cpp) computes for the
    // ids binding must actually reach CommandBinding::trusted and flip requires_validation's
    // replay-eligibility decision in bind_and_launch_recorded_command_graph.
    ScopedHrxEnvironment environment("HRX_ENABLE_SET_ROWS", "1");
    for (bool trusted_producer : { false, true }) {
        ggml_init_params set_rows_params = {};
        set_rows_params.mem_size = 1024 * 1024;
        set_rows_params.no_alloc = true;
        ggml_context * set_rows_ctx = ggml_init(set_rows_params);
        REQUIRE(set_rows_ctx != nullptr);
        ggml_tensor * cache  = ggml_new_tensor_2d(set_rows_ctx, GGML_TYPE_F32, 128, 64);
        ggml_tensor * source = ggml_new_tensor_2d(set_rows_ctx, GGML_TYPE_F32, 128, 4);
        ggml_tensor * ids    = ggml_new_tensor_1d(set_rows_ctx, GGML_TYPE_I64, 4);
        REQUIRE(cache != nullptr);
        REQUIRE(source != nullptr);
        REQUIRE(ids != nullptr);
        if (trusted_producer) {
            // Mirrors llama-kv-cache.cpp's k_idxs/v_idxs: a host-populated graph-input leaf.
            ggml_set_input(ids);
        }
        ggml_tensor * output = ggml_set_rows(set_rows_ctx, cache, source, ids);
        REQUIRE(output != nullptr);
        REQUIRE(ggml::hrx::supports_set_rows_dispatch(output));

        ggml_cgraph * graph = ggml_new_graph(set_rows_ctx);
        ggml_build_forward_expand(graph, output);
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.dispatches.size() == 1);
        const auto & dispatch = plan.dispatches[0];
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "hrx_owned:ggml_set_rows_f32");
        REQUIRE(dispatch.bindings.size() == 3);
        REQUIRE(dispatch.bindings[1].value == imported.graph.values().find_tensor(ids)->id);
        REQUIRE(dispatch.bindings[1].trusted == trusted_producer);

        for (bool initialization : { false, true }) {
            ggml::hrx::PreparedCommandProgram prepared;
            ggml::hrx::PreparedCommand command;
            command.kind = ggml::hrx::CommandKind::Kernel;
            command.kernel.specialization = dispatch.kernel;
            // Carry the scheduled trust decision into the prepared command, the same way
            // append_command() does for a real command program.
            command.kernel.bindings.resize(dispatch.bindings.size());
            for (size_t i = 0; i < dispatch.bindings.size(); ++i) {
                command.kernel.bindings[i].binding.trusted = dispatch.bindings[i].trusted;
            }
            (initialization ? prepared.initialization_commands : prepared.commands).push_back(command);
            ggml::hrx::RecordedCommandGraph recorded;
            const auto replay = ggml::hrx::bind_and_launch_recorded_command_graph(
                {}, {}, {}, prepared, recorded);
            REQUIRE(!replay.success);
            if (trusted_producer) {
                // Trust bypasses the ineligibility gate entirely; the empty execution context
                // then fails the very next check instead (still no device access required).
                REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::BuildFailed);
            } else {
                REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::Ineligible);
                REQUIRE(replay.ineligible_reason == "set_rows_requires_index_validation");
            }
        }
        ggml_free(set_rows_ctx);
    }
    std::fprintf(stderr,
                 "SET_ROWS scheduling checks passed (dispatch + trusted-producer replay-eligibility, no device initialization)\n");
}

static void run_f32_get_rows_scheduling_checks() {
    for (const char * flag : { static_cast<const char *>(nullptr), "", "0", "true", "01", "1x", "1" }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_F32_GET_ROWS", flag);
        for (int64_t tokens : { 1, 4, 16 }) {
            // ids marked as a graph input mirrors llama-graph.cpp's build_inp_out_ids()
            // (inp->out_ids); unmarked mirrors a hypothetical GPU-computed producer, which
            // must stay untrusted.
            for (bool trusted_producer : { false, true }) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                ggml_context * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 32);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
                if (trusted_producer) {
                    ggml_set_input(ids);
                }
                ggml_tensor * output = ggml_get_rows(ctx, source, ids);
                ggml_cgraph * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, output);
                // supports_f32_get_rows_dispatch() only inspects the first character (unlike the
                // exact "1" match used by the Q8/Q4 embedding flags), so "true" and "1x" count too.
                const bool enabled = flag != nullptr && flag[0] != '\0' && flag[0] != '0';
                REQUIRE(ggml::hrx::supports_f32_get_rows_dispatch(output) == enabled);
                auto imported = ggml::hrx::import_ggml_graph(*graph);
                REQUIRE(imported.valid());
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == enabled);
                if (enabled) {
                    const auto & plan = scheduler.plan();
                    REQUIRE(plan.dispatches.size() == 1);
                    const auto & dispatch = plan.dispatches[0];
                    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "hrx_owned:ggml_get_rows_f32");
                    REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == 8);
                    REQUIRE(dispatch.kernel.integer_parameters.at("vocabulary_count") == 32);
                    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tokens);
                    REQUIRE(dispatch.bindings.size() == 3);
                    REQUIRE(dispatch.bindings[0].value == imported.graph.values().find_tensor(source)->id);
                    REQUIRE(dispatch.bindings[0].length == ggml_nbytes(source));
                    REQUIRE(dispatch.bindings[1].value == imported.graph.values().find_tensor(ids)->id);
                    REQUIRE(dispatch.bindings[1].length == ggml_nbytes(ids));
                    REQUIRE(dispatch.bindings[1].trusted == trusted_producer);
                    REQUIRE(dispatch.bindings[2].value == imported.graph.values().find_tensor(output)->id);
                    REQUIRE(dispatch.bindings[2].length == ggml_nbytes(output));
                    for (bool initialization : { false, true }) {
                        ggml::hrx::PreparedCommandProgram prepared;
                        ggml::hrx::PreparedCommand command;
                        command.kind = ggml::hrx::CommandKind::Kernel;
                        command.kernel.specialization = dispatch.kernel;
                        command.kernel.bindings.resize(dispatch.bindings.size());
                        for (size_t i = 0; i < dispatch.bindings.size(); ++i) {
                            command.kernel.bindings[i].binding.trusted = dispatch.bindings[i].trusted;
                        }
                        (initialization ? prepared.initialization_commands : prepared.commands).push_back(command);
                        ggml::hrx::RecordedCommandGraph recorded;
                        const auto replay = ggml::hrx::bind_and_launch_recorded_command_graph(
                            {}, {}, {}, prepared, recorded);
                        REQUIRE(!replay.success);
                        if (trusted_producer) {
                            REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::BuildFailed);
                        } else {
                            REQUIRE(replay.event == ggml::hrx::HrxGraphReplayEvent::Ineligible);
                            REQUIRE(replay.ineligible_reason == "f32_get_rows_requires_index_validation");
                        }
                    }
                }
                ggml_free(ctx);
            }
        }
    }

    // A representative malformed battery: the scheduler must decline exactly when
    // supports_f32_get_rows_dispatch() does, regardless of the trust extension above.
    ScopedHrxEnvironment enabled_environment("HRX_ENABLE_F32_GET_ROWS", "1");
    for (int malformed = 0; malformed < 6; ++malformed) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 32);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ggml_tensor * output = ggml_get_rows(ctx, source, ids);
        if (malformed == 0) { source->type = GGML_TYPE_F16; }
        if (malformed == 1) { ids->type = GGML_TYPE_I64; }
        if (malformed == 2) { output->type = GGML_TYPE_F16; }
        if (malformed == 3) { source->nb[1] += 4; }
        if (malformed == 4) { ids->nb[0] *= 2; }
        if (malformed == 5) { output->nb[1] += 4; }
        REQUIRE(!ggml::hrx::supports_f32_get_rows_dispatch(output));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        if (imported.valid()) {
            ggml::hrx::DispatchScheduler scheduler;
            REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
        }
        ggml_free(ctx);
    }
    std::fprintf(stderr, "F32 GET_ROWS scheduling checks passed (flag gating, trusted-producer wiring, malformed rejection)\n");
}

static void run_trusted_index_view_reference_checks() {
    ScopedHrxEnvironment get_rows("HRX_ENABLE_F32_GET_ROWS", "1");
    for (bool enabled : { false, true }) {
        ScopedHrxEnvironment views("HRX_ENABLE_TRUSTED_INDEX_VIEWS", enabled ? "1" : "0");
        for (int64_t width : { 128, 30720 }) {
            for (int64_t tokens : { 1, 4, 16 }) {
                ggml_backend_t backend = ggml_backend_hrx_init(0);
                ggml_backend_t cpu = ggml_backend_cpu_init();
                REQUIRE(backend != nullptr && cpu != nullptr);
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                ggml_context * ctx = ggml_init(params);
                ggml_context * host_ctx = ggml_init(params);
                REQUIRE(ctx != nullptr && host_ctx != nullptr);
                ggml_tensor * ids_storage = ggml_new_tensor_1d(host_ctx, GGML_TYPE_I32, tokens + 2);
                ggml_set_input(ids_storage);
                ggml_tensor * slice = ggml_view_1d(host_ctx, ids_storage, tokens + 1, sizeof(int32_t));
                ggml_tensor * ids = ggml_view_1d(host_ctx, slice, tokens, sizeof(int32_t));
                ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 8);
                ggml_tensor * output = ggml_get_rows(ctx, source, ids);
                ggml_tensor * consumer = ggml_scale(ctx, output, 2.0f);
                ggml_cgraph * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, consumer);
                auto imported = ggml::hrx::import_ggml_graph(*graph);
                REQUIRE(imported.valid());
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
                bool found = false;
                for (const auto & dispatch : scheduler.plan().dispatches) {
                    if (ggml::hrx::is_f32_get_rows_kernel(dispatch.kernel.kernel_id)) {
                        REQUIRE(dispatch.bindings[1].trusted == enabled);
                        found = true;
                    }
                }
                REQUIRE(found);
                ggml_backend_buffer_t host_buffer = ggml_backend_alloc_ctx_tensors(host_ctx, cpu);
                ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                REQUIRE(host_buffer != nullptr && buffer != nullptr);
                std::vector<int32_t> indices(tokens + 2);
                std::vector<float> expected(width * tokens);
                const std::vector<float> dirty(width * tokens, -12345.0f);
                for (int replay = 0; replay < 4; ++replay) {
                    const auto values = make_pattern_f32(width * 8, replay);
                    for (size_t i = 0; i < indices.size(); ++i) {
                        indices[i] = static_cast<int32_t>((i + replay * 3) % 8);
                    }
                    ggml_backend_tensor_set(ids_storage, indices.data(), 0, indices.size() * sizeof(int32_t));
                    set_tensor_bytes(backend, source, values.data(), values.size() * sizeof(float));
                    set_tensor_bytes(backend, output, dirty.data(), dirty.size() * sizeof(float));
                    set_tensor_bytes(backend, consumer, dirty.data(), dirty.size() * sizeof(float));
                    for (int64_t token = 0; token < tokens; ++token) {
                        std::copy_n(values.begin() + indices[token + 2] * width, width,
                                    expected.begin() + token * width);
                    }
                    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    REQUIRE(get_f32_tensor(backend, output) == expected);
                    for (float & value : expected) { value *= 2.0f; }
                    REQUIRE(get_f32_tensor(backend, consumer) == expected);
                }
                if (!enabled) {
                    indices.back() = 8;
                    ggml_backend_tensor_set(ids_storage, indices.data(), 0, indices.size() * sizeof(int32_t));
                    set_tensor_bytes(backend, output, dirty.data(), dirty.size() * sizeof(float));
                    set_tensor_bytes(backend, consumer, dirty.data(), dirty.size() * sizeof(float));
                    REQUIRE(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
                    REQUIRE(get_f32_tensor(backend, output) == dirty);
                    REQUIRE(get_f32_tensor(backend, consumer) == dirty);
                }
                ggml_backend_buffer_free(buffer);
                ggml_backend_buffer_free(host_buffer);
                ggml_free(ctx);
                ggml_free(host_ctx);
                ggml_backend_free(backend);
                ggml_backend_free(cpu);
            }
        }
    }
    std::fprintf(stderr, "Trusted index view GPU checks passed (host slices, changed IDs/data, replay, validation control)\n");
}

static std::vector<uint8_t> make_q8_embedding_row(int64_t width, int32_t id) {
    std::vector<uint8_t> row(ggml_row_size(GGML_TYPE_Q8_0, width));
    for (int64_t block = 0; block < width / 32; ++block) {
        // Includes zero/negative scales and every signed byte, including -128 and 127.
        const ggml_fp16_t scale = ggml_fp32_to_fp16(
            static_cast<float>((id + block * 5) % 17 - 8) / 1024.0f);
        std::memcpy(row.data() + block * 34, &scale, sizeof(scale));
        for (int code = 0; code < 32; ++code) {
            row[block * 34 + 2 + code] = static_cast<uint8_t>(id * 13 + block * 37 + code);
        }
    }
    return row;
}

static std::vector<uint8_t> make_q4_embedding_row(int64_t width, int32_t id, int generation) {
    std::vector<uint8_t> row(ggml_row_size(GGML_TYPE_Q4_K, width));
    for (int64_t block = 0; block < width / 256; ++block) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(
            static_cast<float>((id + block * 5 + generation * 7) % 19 - 9) * 0.00037f);
        const ggml_fp16_t dmin = ggml_fp32_to_fp16(
            static_cast<float>((id + block * 3 + generation * 11) % 13 - 6) * 0.00029f);
        std::memcpy(row.data() + block * 144, &d, sizeof(d));
        std::memcpy(row.data() + block * 144 + 2, &dmin, sizeof(dmin));
        uint32_t state = 0x9e3779b9u ^ (static_cast<uint32_t>(id) * 37u) ^
                         (static_cast<uint32_t>(block) * 131u) ^ (static_cast<uint32_t>(generation) * 65537u);
        for (int byte = 4; byte < 144; ++byte) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            row[block * 144 + byte] = static_cast<uint8_t>(state);
        }
    }
    return row;
}

static void run_embedding_reference_case(int64_t width, int64_t vocabulary, int64_t tokens,
                                          ggml_type type = GGML_TYPE_Q8_0) {
    const bool q4 = type == GGML_TYPE_Q4_K;
    const char * flag = q4 ? "HRX_ENABLE_Q4_EMBEDDING" : "HRX_ENABLE_Q8_EMBEDDING";
    ScopedHrxEnvironment environment(flag, "1");
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, width, vocabulary);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_tensor * output = ggml_get_rows(ctx, weight, ids);
    ggml_tensor * gamma = q4 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width) : nullptr;
    ggml_tensor * consumer = q4 ? ggml_mul(ctx, ggml_rms_norm(ctx, output, kQwenRmsNormEps), gamma) :
                                 ggml_scale(ctx, output, 2.0f);
    REQUIRE(ggml_backend_supports_op(backend, output));
    {
        ScopedHrxEnvironment disabled(flag, "0");
        REQUIRE(!ggml_backend_supports_op(backend, output));
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, consumer);
    if (q4) {
        std::fprintf(stderr, "Q4 embedding reference: %s=1 T=%lld V=%lld (GET_ROWS -> RMS_NORM -> MUL gamma)\n",
                     flag, static_cast<long long>(tokens), static_cast<long long>(vocabulary));
    }
    require_kernel_subsequence(scheduled_kernel_sequence(graph),
        { q4 ? "hrx_owned:ggml_token_embedding_q4_k_f32" : "hrx_owned:ggml_token_embedding_q8_0_f32" });
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    const std::vector<float> gamma_values = q4 ? make_weight(width) : std::vector<float>();
    if (q4) {
        set_tensor_bytes(backend, gamma, gamma_values.data(), gamma_values.size() * sizeof(float));
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    REQUIRE(traits != nullptr && traits->to_float != nullptr);
    const std::vector<int32_t> selected = { 0, static_cast<int32_t>(vocabulary - 1), 3, 17, 3, 1 };
    std::vector<std::vector<float>> decoded;
    // Even the full-vocabulary case uploads only the rows used by this test, never downloads weights.
    for (int32_t id : selected) {
        const auto row = q4 ? make_q4_embedding_row(width, id, 0) : make_q8_embedding_row(width, id);
        ggml_backend_tensor_set(weight, row.data(), static_cast<size_t>(id) * row.size(), row.size());
        decoded.emplace_back(width);
        traits->to_float(row.data(), decoded.back().data(), width);
    }
    const std::vector<float> dirty(width * tokens, -12345.0f);
    std::vector<int32_t> token_ids(tokens);
    for (int replay = 0; replay < (q4 ? 4 : 3); ++replay) {
        // Q4 changes IDs first, then weights alone, then IDs again on the same cached graph.
        if (q4 && replay == 2) {
            for (size_t index = 0; index < selected.size(); ++index) {
                const auto row = make_q4_embedding_row(width, selected[index], replay);
                ggml_backend_tensor_set(weight, row.data(), static_cast<size_t>(selected[index]) * row.size(), row.size());
                traits->to_float(row.data(), decoded[index].data(), width);
            }
        }
        std::vector<float> expected(width * tokens);
        const int id_generation = q4 ? (replay == 0 ? 0 : (replay == 3 ? 2 : 1)) : replay;
        for (int64_t token = 0; token < tokens; ++token) {
            const size_t index = (static_cast<size_t>(token) + id_generation) % selected.size();
            token_ids[token] = selected[index];
            std::copy(decoded[index].begin(), decoded[index].end(), expected.begin() + token * width);
        }
        set_tensor_bytes(backend, ids, token_ids.data(), token_ids.size() * sizeof(int32_t));
        set_tensor_bytes(backend, output, dirty.data(), dirty.size() * sizeof(float));
        set_tensor_bytes(backend, consumer, dirty.data(), dirty.size() * sizeof(float));
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        REQUIRE(get_f32_tensor(backend, output) == expected);
        if (q4) {
            const auto normalized = rmsnorm_mul_reference(expected, gamma_values, width, tokens);
            require_close(get_f32_tensor(backend, consumer), normalized, 2.0e-4f, 2.0e-5f);
        } else {
            for (float & value : expected) { value *= 2.0f; }
            REQUIRE(get_f32_tensor(backend, consumer) == expected);
        }
    }
    for (int32_t invalid : { -1, static_cast<int32_t>(vocabulary),
                             std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max() }) {
        token_ids.back() = invalid;
        set_tensor_bytes(backend, ids, token_ids.data(), token_ids.size() * sizeof(int32_t));
        set_tensor_bytes(backend, output, dirty.data(), dirty.size() * sizeof(float));
        set_tensor_bytes(backend, consumer, dirty.data(), dirty.size() * sizeof(float));
        REQUIRE(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
        REQUIRE(get_f32_tensor(backend, output) == dirty);
        REQUIRE(get_f32_tensor(backend, consumer) == dirty);
    }
    // A failed cached execution must not poison a later valid replay.
    std::fill(token_ids.begin(), token_ids.end(), selected.front());
    set_tensor_bytes(backend, ids, token_ids.data(), token_ids.size() * sizeof(int32_t));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const auto recovered = get_f32_tensor(backend, output);
    for (int64_t token = 0; token < tokens; ++token) {
        REQUIRE(std::equal(decoded.front().begin(), decoded.front().end(), recovered.begin() + token * width));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_q8_embedding_reference_checks() {
    run_embedding_reference_case(2560, 248320, 1);
    run_embedding_reference_case(2560, 65, 7);
    run_embedding_reference_case(2560, 65, 32);
    run_embedding_reference_case(2560, 65, 2048);
    run_embedding_reference_case(32, 65, 7);
    run_embedding_reference_case(2592, 65, 7);  // Partial final 256-thread iteration.
}

static void run_q4_embedding_reference_checks() {
    for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
        run_embedding_reference_case(2560, 248320, tokens, GGML_TYPE_Q4_K);
    }
}

static void run_dense_q8_gemv_scheduling_checks() {
    const char * flags[] = { nullptr, "", "0", "true", "01", "1" };
    for (const char * flag : flags) {
        ScopedHrxEnvironment environment("HRX_ENABLE_Q8_GEMV", flag);
        for (int64_t tokens : { 1, 2, 32 }) {
            ggml_init_params params = {};
            params.mem_size         = 1024 * 1024;
            params.no_alloc         = true;
            ggml_context * ctx      = ggml_init(params);
            REQUIRE(ctx != nullptr);
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2560, 65);
            ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, tokens);
            ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
            ggml_cgraph * graph  = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);

            const bool use_gemv = tokens == 1 && flag != nullptr && std::strcmp(flag, "1") == 0;
            ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
            REQUIRE(imported.valid());
            ggml::hrx::DispatchScheduler scheduler;
            REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
            REQUIRE(scheduler.plan().dispatches.size() == 1);
            const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches[0];
            REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
                    (use_gemv ? "hrx_owned:ggml_dense_q8_0_gemv_f32" :
                                "qwen3_moe:qwen3_moe_dense_linear_q8_0_f16_wmma"));
            REQUIRE(dispatch.bindings.size() == 3);
            REQUIRE(dispatch.bindings[0].length == ggml_nbytes(input));
            REQUIRE(dispatch.bindings[1].length == ggml_nbytes(weight));
            REQUIRE(dispatch.bindings[2].length == ggml_nbytes(output));
            if (use_gemv) {
                REQUIRE(dispatch.kernel.integer_parameters.size() == 2);
                REQUIRE(dispatch.kernel.integer_parameters.at("input_size") == 2560);
                REQUIRE(dispatch.kernel.integer_parameters.at("output_size") == 65);
                REQUIRE(dispatch.kernel.compile_parameters.empty());
            } else {
                REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tokens);
                REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_accumulation") == "0");
            }
            ggml_free(ctx);
        }
    }

    // The optional kernel uses u32 byte offsets; larger existing shapes retain WMMA.
    ScopedHrxEnvironment environment("HRX_ENABLE_Q8_GEMV", "1");
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 32768, 262144);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32768, 1);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_cgraph * graph  = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    require_kernel_subsequence(scheduled_kernel_sequence(graph), { "qwen3_moe:qwen3_moe_dense_linear_q8_0_f16_wmma" });
    ggml_free(ctx);
}

static void run_dense_q8_gemv_raw_reference_case(int64_t input_size, int64_t output_size,
                                                bool use_gemv = true, int repetitions = 0) {
    ScopedHrxEnvironment environment("HRX_ENABLE_Q8_GEMV", use_gemv ? "1" : "0");
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * weight_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, input_size, output_size);
    ggml_tensor * input_tensor  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, 1);
    ggml_tensor * output_tensor = ggml_mul_mat(ctx, weight_tensor, input_tensor);
    ggml_cgraph * graph         = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output_tensor);
    require_kernel_subsequence(scheduled_kernel_sequence(graph),
                               { use_gemv ? "hrx_owned:ggml_dense_q8_0_gemv_f32" :
                                            "qwen3_moe:qwen3_moe_dense_linear_q8_0_f16_wmma" });
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const int64_t blocks = input_size / 32;
    REQUIRE(ggml_type_size(GGML_TYPE_Q8_0) == 34);
    std::vector<uint8_t> weight(ggml_nbytes(weight_tensor));
    std::vector<float> input(input_size);
    for (int64_t column = 0; column < input_size; ++column) {
        input[column] = static_cast<float>((column * 17 + column / 32) % 63 - 31) * 0.0013f;
    }
    for (int64_t row = 0; row < output_size; ++row) {
        for (int64_t block = 0; block < blocks; ++block) {
            const size_t base = static_cast<size_t>((row * blocks + block) * 34);
            const ggml_fp16_t scale = ggml_fp32_to_fp16(static_cast<float>((row * 3 + block * 5) % 15 - 7) / 256.0f);
            std::memcpy(weight.data() + base, &scale, sizeof(scale));
            for (int64_t q = 0; q < 32; ++q) {
                const int8_t code = static_cast<int8_t>((row * 29 + block * 13 + q * 7) % 256 - 128);
                std::memcpy(weight.data() + base + 2 + q, &code, sizeof(code));
            }
        }
    }

    // Decode the uploaded bytes independently; the CPU Q8 matmul quantizes its input.
    std::vector<float> expected(output_size);
    for (int64_t row = 0; row < output_size; ++row) {
        double sum = 0.0;
        for (int64_t column = 0; column < input_size; ++column) {
            const size_t base = static_cast<size_t>((row * blocks + column / 32) * 34);
            ggml_fp16_t scale;
            int8_t code;
            std::memcpy(&scale, weight.data() + base, sizeof(scale));
            std::memcpy(&code, weight.data() + base + 2 + column % 32, sizeof(code));
            const float decoded = ggml_fp16_to_fp32(scale) * static_cast<float>(code);
            sum += static_cast<double>(decoded) * input[column];
        }
        expected[row] = static_cast<float>(sum);
    }
    set_tensor_bytes(backend, weight_tensor, weight.data(), weight.size());
    set_tensor_bytes(backend, input_tensor, input.data(), input.size() * sizeof(float));
    // The kernel must overwrite a dirty destination rather than accumulate into it.
    const std::vector<float> initial(output_size, 123.0f);
    set_tensor_bytes(backend, output_tensor, initial.data(), initial.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const std::vector<float> actual = get_f32_tensor(backend, output_tensor);
    for (float value : actual) {
        REQUIRE(std::isfinite(value));
    }
    // Retain the existing half-precision WMMA gate; the new F32 GEMV has a tighter gate.
    const float abs_tolerance = use_gemv ? 3.0e-4f : kDenseWmmaAbsTolerance;
    const float rel_tolerance = use_gemv ? 1.0e-5f : kDenseWmmaRelTolerance;
    require_close(actual, expected, abs_tolerance, rel_tolerance);

    if (repetitions > 0) {
        using clock = std::chrono::steady_clock;
        std::vector<double> elapsed;
        for (int i = 0; i < repetitions; ++i) {
            const auto start = clock::now();
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);
            elapsed.push_back(std::chrono::duration<double, std::milli>(clock::now() - start).count());
        }
        require_close(get_f32_tensor(backend, output_tensor), expected, abs_tolerance, rel_tolerance);
        float max_error = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i) {
            max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
        }
        std::sort(elapsed.begin(), elapsed.end());
        std::fprintf(stderr, "Q8 %s %lld->%lld: warm graph wall median=%.4f ms p90=%.4f ms n=%d max_abs_error=%g\n",
                     use_gemv ? "GEMV" : "WMMA", static_cast<long long>(input_size),
                     static_cast<long long>(output_size), elapsed[elapsed.size() / 2],
                     elapsed[(elapsed.size() - 1) * 9 / 10], repetitions, max_error);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

struct MtpHcProjectionGraph {
    ggml_tensor * weight;
    ggml_tensor * input;
    ggml_tensor * output;
    ggml_tensor * norm_weight;
    ggml_tensor * normalized;
    ggml_cgraph * graph;
};

static MtpHcProjectionGraph build_mtp_hc_projection_graph(ggml_context * ctx, int64_t tokens) {
    MtpHcProjectionGraph result = {};
    result.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 5120, 2560);
    result.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5120, 4, tokens);
    result.output = ggml_mul_mat(ctx, result.weight, result.input);
    result.norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560);
    result.normalized = ggml_mul(ctx, ggml_rms_norm(ctx, result.output, kQwenRmsNormEps), result.norm_weight);
    result.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(result.graph, result.normalized);
    return result;
}

static void check_mtp_hc_projection_plan(const MtpHcProjectionGraph & tensors, int64_t tokens,
                                         bool f32_accum = false) {
    auto imported = ggml::hrx::import_ggml_graph(*tensors.graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
    const auto & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.dispatches.size() == 2);
    REQUIRE(plan.transients.empty());
    REQUIRE(plan.constant_initializations.empty());
    const auto & projection = plan.dispatches[0];
    REQUIRE(kernel_name_for_id(projection.kernel.kernel_id) == (f32_accum ?
            "hrx_owned:ggml_dense_q4k_f32_accum" : "qwen3_moe:qwen3_moe_dense_linear_q4k_f16_wmma"));
    REQUIRE(projection.kernel.integer_parameters.size() == 1);
    REQUIRE(projection.kernel.integer_parameters.at("token_count") == 4 * tokens);
    REQUIRE(projection.kernel.compile_parameters.at("qwen3_moe.workload.token_capacity") ==
            std::to_string(4 * tokens));
    REQUIRE(projection.kernel.compile_parameters.at("qwen3_moe.dense_quantized.input_size") == "5120");
    REQUIRE(projection.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_size") == "2560");
    REQUIRE(projection.kernel.compile_parameters.size() == (f32_accum ? 3 : 4));
    if (!f32_accum) {
        REQUIRE(projection.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_accumulation") == "0");
    }
    REQUIRE(projection.bindings.size() == 3);
    const ggml_tensor * bound[] = { tensors.input, tensors.weight, tensors.output };
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(projection.bindings[i].value == imported.graph.values().find_tensor(bound[i])->id);
        REQUIRE(projection.bindings[i].offset == 0);
        REQUIRE(projection.bindings[i].length == ggml_nbytes(bound[i]));
    }
    REQUIRE(projection.bindings[0].length == 5120 * 4 * tokens * sizeof(float));
    REQUIRE(projection.bindings[1].length == ggml_row_size(GGML_TYPE_Q4_K, 5120) * 2560);
    REQUIRE(projection.bindings[2].length == 2560 * 4 * tokens * sizeof(float));
    REQUIRE(plan.dispatches[1].bindings[0].value == projection.bindings[2].value);
    REQUIRE(tensors.output->ne[1] == 4 && tensors.output->ne[2] == tokens);
}

static void run_mtp_hc_projection_scheduling_checks(bool f32_accum = false) {
    const char * flags[] = { nullptr, "", "0", "true", "01", "1" };
    for (const char * flag : flags) {
        ScopedHrxEnvironment environment("HRX_ENABLE_MTP_HC_PROJECTION", flag);
        for (int64_t tokens : { 1, 2, 3, 4, 8, 9 }) {
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            auto * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            const auto tensors = build_mtp_hc_projection_graph(ctx, tokens);
            const bool enabled = flag != nullptr && std::strcmp(flag, "1") == 0 && tokens <= 8;
            REQUIRE(ggml::hrx::llm_mtp_hc_projection_supported(
                *tensors.weight, *tensors.input, *tensors.output) == enabled);
            // T=1 is also an ordinary [5120,4] 2D matmul and must keep working with the flag off.
            if (enabled || tokens == 1) {
                check_mtp_hc_projection_plan(tensors, tokens, f32_accum);
            } else {
                auto imported = ggml::hrx::import_ggml_graph(*tensors.graph);
                REQUIRE(imported.valid());
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
            }
            ggml_free(ctx);
        }
    }

    ScopedHrxEnvironment environment("HRX_ENABLE_MTP_HC_PROJECTION", "1");
    for (int invalid = 0; invalid < 16; ++invalid) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        const auto tensors = build_mtp_hc_projection_graph(ctx, 2);
        switch (invalid) {
            case 0: tensors.input->nb[0] *= 2; break;
            case 1: tensors.input->nb[1] += sizeof(float); break;
            case 2: tensors.input->nb[2] += sizeof(float); break;
            case 3: tensors.output->nb[1] += sizeof(float); break;
            case 4: tensors.output->nb[2] += sizeof(float); break;
            case 5: tensors.weight->nb[1] += ggml_type_size(GGML_TYPE_Q4_K); break;
            case 6: tensors.weight->ne[2] = 2; break;
            case 7: tensors.input->ne[3] = tensors.output->ne[3] = 2; break;
            case 8: tensors.input->ne[1] = tensors.output->ne[1] = 3; break;
            case 9: tensors.output->ne[0] = 2559; break;
            case 10: tensors.weight->type = GGML_TYPE_Q6_K; break;
            case 11: tensors.input->type = GGML_TYPE_F16; break;
            case 12: tensors.output->ne[2] = 3; break;
            case 13: tensors.weight->ne[3] = 2; break;
            case 14: tensors.output->nb[0] *= 2; break;
            case 15: tensors.input->nb[3] += sizeof(float); break;
        }
        REQUIRE(!ggml::hrx::llm_mtp_hc_projection_supported(*tensors.weight, *tensors.input, *tensors.output));
        // Import only the mutated projection, not its now-inconsistent RMS consumer.
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, tensors.output);
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        if (imported.valid()) {
            ggml::hrx::DispatchScheduler scheduler;
            REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
        }
        ggml_free(ctx);
    }
    // Ordinary dense projections do not depend on the MTP switch.
    ScopedHrxEnvironment disabled("HRX_ENABLE_MTP_HC_PROJECTION", "0");
    for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        auto * weight = ggml_new_tensor_2d(ctx, type, 2048, 128);
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 3);
        auto * output = ggml_mul_mat(ctx, weight, input);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        const auto sequence = scheduled_kernel_sequence(graph);
        REQUIRE(sequence.size() == 1);
        REQUIRE(sequence[0] == (f32_accum ?
            (type == GGML_TYPE_Q4_K ? "hrx_owned:ggml_dense_q4k_f32_accum" : "hrx_owned:ggml_dense_q6k_f32_accum") :
            (type == GGML_TYPE_Q4_K ? "qwen3_moe:qwen3_moe_dense_linear_q4k_f16_wmma" :
                                     "qwen3_moe:qwen3_moe_dense_linear_q6k_f16_wmma")));
        ggml_free(ctx);
    }
}

static void require_mtp_hc_projection_fixture_discriminates(const std::vector<float> & expected,
                                                           int64_t tokens, int replay) {
    constexpr size_t width = 2560;
    REQUIRE(expected.size() == width * 4 * static_cast<size_t>(tokens));
    size_t min_zero_rejections = width;
    size_t min_swap_rejections = width;
    for (int64_t token = 0; token < tokens; ++token) {
        for (size_t hc = 0; hc < 4; ++hc) {
            const size_t base = (static_cast<size_t>(token) * 4 + hc) * width;
            size_t zero_rejections = 0;
            for (size_t n = 0; n < width; ++n) {
                const float value = expected[base + n];
                REQUIRE(std::isfinite(value));
                const float allowed = kDenseWmmaAbsTolerance + kDenseWmmaRelTolerance * std::fabs(value);
                zero_rejections += std::fabs(value) > allowed;
            }
            // Require every HC column, including all later tokens, to reject zero under the real gate.
            REQUIRE(zero_rejections > 0);
            min_zero_rejections = std::min(min_zero_rejections, zero_rejections);
            for (size_t other_hc = 0; other_hc < 4; ++other_hc) {
                if (other_hc == hc) {
                    continue;
                }
                const size_t other_base = (static_cast<size_t>(token) * 4 + other_hc) * width;
                size_t swap_rejections = 0;
                for (size_t n = 0; n < width; ++n) {
                    const float value = expected[base + n];
                    const float allowed = kDenseWmmaAbsTolerance + kDenseWmmaRelTolerance * std::fabs(value);
                    swap_rejections += std::fabs(expected[other_base + n] - value) > allowed;
                }
                // All six HC-pair swaps per token are checked in both comparison directions.
                REQUIRE(swap_rejections > 0);
                min_swap_rejections = std::min(min_swap_rejections, swap_rejections);
            }
        }
    }
    std::fprintf(stderr, "MTP HC fixture T=%lld replay=%d elements=%zu: minimum rejecting elements/column "
                         "zero=%zu swap=%zu (atol=%g rtol=%g)\n",
                 static_cast<long long>(tokens), replay, expected.size(),
                 min_zero_rejections, min_swap_rejections, kDenseWmmaAbsTolerance, kDenseWmmaRelTolerance);
}

static void run_mtp_hc_projection_reference_checks() {
    ScopedHrxEnvironment environment("HRX_ENABLE_MTP_HC_PROJECTION", "1");
    ggml_backend_t cpu = init_cpu_backend();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(hrx != nullptr);
    const auto weight = make_quantized_rows(GGML_TYPE_Q4_K, 5120, 2560, 23);
    for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * cpu_ctx = ggml_init(params);
        auto * ctx = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr && ctx != nullptr);
        const auto reference = build_mtp_hc_projection_graph(cpu_ctx, tokens);
        const auto tensors = build_mtp_hc_projection_graph(ctx, tokens);
        check_mtp_hc_projection_plan(tensors, tokens);
        REQUIRE(ggml_backend_supports_op(hrx, tensors.output));
        {
            ScopedHrxEnvironment disabled("HRX_ENABLE_MTP_HC_PROJECTION", "0");
            REQUIRE(ggml_backend_supports_op(hrx, tensors.output) == (tokens == 1));
        }
        auto * cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, hrx);
        REQUIRE(cpu_buffer != nullptr && buffer != nullptr);
        set_tensor_pair_bytes(cpu, reference.weight, hrx, tensors.weight, weight.data(), weight.size());
        const std::vector<float> gamma(2560, 1.0f);
        set_tensor_pair_bytes(cpu, reference.norm_weight, hrx, tensors.norm_weight,
                              gamma.data(), gamma.size() * sizeof(float));
        for (int replay = 0; replay < 3; ++replay) {
            std::vector<float> input(5120 * 4 * tokens);
            for (int64_t column = 0; column < 4 * tokens; ++column) {
                uint32_t state = 0x9e3779b9u ^ static_cast<uint32_t>(column * 7919 + replay * 104729);
                for (int64_t k = 0; k < 5120; ++k) {
                    state = state * 1664525u + 1013904223u;
                    input[column * 5120 + k] = replay == 2 ? 0.0f :
                        static_cast<float>(static_cast<int>(state >> 16) - 32768) / 65536.0f;
                }
            }
            set_tensor_pair_bytes(cpu, reference.input, hrx, tensors.input,
                                  input.data(), input.size() * sizeof(float));
            const std::vector<float> dirty(2560 * 4 * tokens, 12345.0f);
            set_tensor_bytes(hrx, tensors.output, dirty.data(), dirty.size() * sizeof(float));
            REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
            REQUIRE(ggml_backend_graph_compute(hrx, tensors.graph) == GGML_STATUS_SUCCESS);
            const auto expected = get_f32_tensor(cpu, reference.output);
            const auto actual = get_f32_tensor(hrx, tensors.output);
            REQUIRE(expected.size() == static_cast<size_t>(2560 * 4 * tokens));
            REQUIRE(actual.size() == expected.size());
            for (float value : actual) {
                REQUIRE(std::isfinite(value));
            }
            require_close(actual, expected, kDenseWmmaAbsTolerance, kDenseWmmaRelTolerance);
            require_close(get_f32_tensor(hrx, tensors.normalized), get_f32_tensor(cpu, reference.normalized),
                          kDenseWmmaAbsTolerance, kDenseWmmaRelTolerance);
            if (replay != 2) {
                require_mtp_hc_projection_fixture_discriminates(expected, tokens, replay);
                for (int64_t column = 1; column < 4 * tokens; ++column) {
                    float separation = 0.0f;
                    for (int64_t n = 0; n < 2560; ++n) {
                        separation = std::max(separation, std::fabs(
                            expected[column * 2560 + n] - expected[(column - 1) * 2560 + n]));
                    }
                    REQUIRE(separation > 1.0f);
                }
            } else {
                require_close(actual, std::vector<float>(expected.size(), 0.0f), 0.0f);
            }
        }
        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(buffer);
        ggml_free(cpu_ctx);
        ggml_free(ctx);
    }
    // Capability negatives use real tensor layouts, without computing unsupported nodes.
    for (int invalid = 0; invalid < 7; ++invalid) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        const auto tensors = build_mtp_hc_projection_graph(ctx, invalid == 0 ? 9 : 2);
        if (invalid == 1) { tensors.input->nb[2] += sizeof(float); }
        if (invalid == 2) { tensors.output->nb[2] += sizeof(float); }
        if (invalid == 3) { tensors.weight->ne[2] = 2; }
        if (invalid == 4) { tensors.output->ne[0] = 2559; }
        if (invalid == 5) { tensors.weight->type = GGML_TYPE_Q6_K; }
        if (invalid == 6) { tensors.input->ne[3] = tensors.output->ne[3] = 2; }
        REQUIRE(!ggml_backend_supports_op(hrx, tensors.output));
        ggml_free(ctx);
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
}

static const char * dense_f32_kernel_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return "hrx_owned:ggml_dense_q4k_f32_accum";
        case GGML_TYPE_Q6_K: return "hrx_owned:ggml_dense_q6k_f32_accum";
        case GGML_TYPE_Q8_0: return "hrx_owned:ggml_dense_q8_0_f32_accum";
        default: REQUIRE(false); return "";
    }
}

static void run_dense_f32_accum_scheduling_checks() {
    ScopedHrxEnvironment gemv("HRX_ENABLE_Q8_GEMV", "0");
    for (const char * flag : { static_cast<const char *>(nullptr), "", "0", "true", "01", "1x", "1" }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_DENSE_F32_ACCUM", flag);
        const bool enabled = flag != nullptr && std::strcmp(flag, "1") == 0;
        for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 }) {
            for (int64_t columns : { 1, 3, 17, 33 }) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                auto * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                auto * weight = ggml_new_tensor_2d(ctx, type, 5120, 65);
                auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, columns);
                auto * output = ggml_mul_mat(ctx, weight, input);
                auto * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, output);
                auto imported = ggml::hrx::import_ggml_graph(*graph);
                REQUIRE(imported.valid());
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
                const auto & plan = scheduler.plan();
                REQUIRE(plan.valid() && plan.dispatches.size() == 1);
                REQUIRE(plan.transients.empty() && plan.constant_initializations.empty());
                const auto & dispatch = plan.dispatches[0];
                const char * legacy = type == GGML_TYPE_Q4_K ? "qwen3_moe:qwen3_moe_dense_linear_q4k_f16_wmma" :
                    type == GGML_TYPE_Q6_K ? "qwen3_moe:qwen3_moe_dense_linear_q6k_f16_wmma" :
                                            "qwen3_moe:qwen3_moe_dense_linear_q8_0_f16_wmma";
                REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == (enabled ? dense_f32_kernel_name(type) : legacy));
                REQUIRE(dispatch.kernel.integer_parameters.size() == 1);
                REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == columns);
                REQUIRE(dispatch.kernel.compile_parameters.size() == (enabled ? 3 : 4));
                REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.workload.token_capacity") == std::to_string(columns));
                REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.input_size") == "5120");
                REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_size") == "65");
                REQUIRE(dispatch.bindings.size() == 3);
                const ggml_tensor * bound[] = { input, weight, output };
                for (size_t i = 0; i < 3; ++i) {
                    REQUIRE(dispatch.bindings[i].value == imported.graph.values().find_tensor(bound[i])->id);
                    REQUIRE(dispatch.bindings[i].offset == 0);
                    REQUIRE(dispatch.bindings[i].length == ggml_nbytes(bound[i]));
                }
                if (enabled && type == GGML_TYPE_Q8_0 && columns == 1) {
                    ScopedHrxEnvironment both("HRX_ENABLE_Q8_GEMV", "1");
                    const auto sequence = scheduled_kernel_sequence(graph);
                    REQUIRE(sequence.size() == 1 && sequence[0] == dense_f32_kernel_name(type));
                }
                ggml_free(ctx);
            }
        }
    }
    ScopedHrxEnvironment enabled("HRX_ENABLE_DENSE_F32_ACCUM", "1");
    // Repeat the full HC flag/layout/batch rejection contract with the new arithmetic selected.
    run_mtp_hc_projection_scheduling_checks(true);
}

static float dense_round_f16(float value) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

static std::vector<float> dense_f16_operand_reference(ggml_type type, const std::vector<uint8_t> & weight,
                                                       const std::vector<float> & input,
                                                       int64_t width, int64_t rows, int64_t columns,
                                                       std::vector<float> * half_accumulation = nullptr) {
    const auto * traits = ggml_get_type_traits(type);
    REQUIRE(traits != nullptr && traits->to_float != nullptr);
    REQUIRE(weight.size() == ggml_row_size(type, width) * rows);
    REQUIRE(input.size() == static_cast<size_t>(width * columns));
    std::vector<float> operands(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        operands[i] = dense_round_f16(input[i]);
        REQUIRE(std::isfinite(operands[i]));
    }
    std::vector<float> output(rows * columns);
    if (half_accumulation != nullptr) {
        half_accumulation->resize(output.size());
    }
    // Only one decoded host reference row, never a runtime expanded weight tensor.
    std::vector<float> row(width);
    for (int64_t n = 0; n < rows; ++n) {
        traits->to_float(weight.data() + n * ggml_row_size(type, width), row.data(), width);
        for (float & value : row) {
            value = dense_round_f16(value);
            REQUIRE(std::isfinite(value));
        }
        for (int64_t column = 0; column < columns; ++column) {
            float sum = 0.0f;
            float half = 0.0f;
            for (int64_t k = 0; k < width; k += 16) {
                for (int64_t j = 0; j < 16; ++j) {
                    // The F16 product is exact in F32; only accumulation order may differ from WMMA.
                    const float product = row[k + j] * operands[column * width + k + j];
                    sum += product;
                    if (half_accumulation != nullptr) {
                        half += product;
                    }
                }
                if (half_accumulation != nullptr) {
                    half = dense_round_f16(half);
                }
            }
            output[column * rows + n] = sum;
            REQUIRE(std::isfinite(sum));
            if (half_accumulation != nullptr) {
                (*half_accumulation)[column * rows + n] = half;
            }
        }
    }
    return output;
}

static constexpr float kDenseF32AbsTolerance = 5.0e-4f;
static constexpr float kDenseF32RelTolerance = 3.0e-5f;

static void require_dense_f32_fixture_discriminates(const std::vector<float> & expected,
                                                     int64_t rows, int64_t columns) {
    REQUIRE(expected.size() == static_cast<size_t>(rows * columns));
    for (int64_t column = 0; column < columns; ++column) {
        bool rejects_zero = false;
        bool rejects_swapped_column = false;
        for (int64_t n = 0; n < rows; ++n) {
            const float value = expected[column * rows + n];
            const float allowed = kDenseF32AbsTolerance + kDenseF32RelTolerance * std::fabs(value);
            rejects_zero |= std::fabs(value) > allowed;
            rejects_swapped_column |= std::fabs(value - expected[((column + 1) % columns) * rows + n]) > allowed;
        }
        REQUIRE(rejects_zero);
        REQUIRE(columns == 1 || rejects_swapped_column);
    }
}

static void run_dense_f32_accum_reference_case(ggml_type type, int64_t width, int64_t rows,
                                               int64_t columns, bool hc, bool high_range, bool run_gpu) {
    ScopedHrxEnvironment enabled("HRX_ENABLE_DENSE_F32_ACCUM", "1");
    ScopedHrxEnvironment projection("HRX_ENABLE_MTP_HC_PROJECTION", "1");
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    auto * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    MtpHcProjectionGraph tensors = {};
    if (hc) {
        REQUIRE(type == GGML_TYPE_Q4_K && width == 5120 && rows == 2560 && columns % 4 == 0);
        tensors = build_mtp_hc_projection_graph(ctx, columns / 4);
        check_mtp_hc_projection_plan(tensors, columns / 4, true);
    } else {
        tensors.weight = ggml_new_tensor_2d(ctx, type, width, rows);
        tensors.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, columns);
        tensors.output = ggml_mul_mat(ctx, tensors.weight, tensors.input);
        tensors.graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(tensors.graph, tensors.output);
        const auto sequence = scheduled_kernel_sequence(tensors.graph);
        REQUIRE(sequence.size() == 1 && sequence[0] == dense_f32_kernel_name(type));
    }
    auto weight = make_quantized_rows(type, width, rows, 29);
    if (high_range) {
        const auto * traits = ggml_get_type_traits(type);
        std::vector<float> row(width);
        for (int64_t n = 0; n < rows; ++n) {
            std::fill(row.begin(), row.end(), static_cast<float>(1 + n % 7) * 0.125f);
            traits->from_float_ref(row.data(), weight.data() + n * ggml_row_size(type, width), width);
        }
    }
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    if (run_gpu) {
        backend = ggml_backend_hrx_init(0);
        REQUIRE(backend != nullptr && ggml_backend_supports_op(backend, tensors.output));
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer != nullptr);
        set_tensor_bytes(backend, tensors.weight, weight.data(), weight.size());
        if (hc) {
            const std::vector<float> gamma(rows, 1.0f);
            set_tensor_bytes(backend, tensors.norm_weight, gamma.data(), gamma.size() * sizeof(float));
        }
    }
    for (int replay = 0; replay < 3; ++replay) {
        std::vector<float> input(width * columns);
        for (int64_t column = 0; column < columns; ++column) {
            uint32_t state = 0x9e3779b9u ^ static_cast<uint32_t>(column * 7919 + replay * 104729);
            for (int64_t k = 0; k < width; ++k) {
                state = state * 1664525u + 1013904223u;
                float value = static_cast<float>(static_cast<int>(state >> 16) - 32768) / 16384.0f;
                if (high_range) {
                    value = static_cast<float>(4096 + column * 64);
                    if (replay == 1 && k >= width / 2) {
                        value = -value;
                        if (k >= width - 16) {
                            value += static_cast<float>(16 + column * 4);
                        }
                    }
                }
                input[column * width + k] = replay == 2 ? 0.0f : value;
            }
        }
        std::vector<float> half_accumulation;
        const auto expected = dense_f16_operand_reference(type, weight, input, width, rows, columns,
                                                           high_range ? &half_accumulation : nullptr);
        if (replay != 2) {
            require_dense_f32_fixture_discriminates(expected, rows, columns);
            if (hc) {
                require_mtp_hc_projection_fixture_discriminates(expected, columns / 4, replay);
            }
            if (high_range) {
                REQUIRE(half_accumulation.size() == expected.size());
                for (size_t i = 0; i < expected.size(); ++i) {
                    REQUIRE(!std::isfinite(half_accumulation[i]));
                    if (replay == 0) {
                        REQUIRE(std::fabs(expected[i]) > 65504.0f);
                        REQUIRE(!std::isfinite(dense_round_f16(expected[i])));
                    } else {
                        // Intermediate F16 overflow must be rejected even when the final answer fits F16.
                        REQUIRE(std::fabs(expected[i]) < 65504.0f);
                        REQUIRE(std::fabs(expected[i]) > 1.0f);
                    }
                }
            }
        }
        if (run_gpu) {
            set_tensor_bytes(backend, tensors.input, input.data(), input.size() * sizeof(float));
            const std::vector<float> dirty(rows * columns, std::numeric_limits<float>::quiet_NaN());
            set_tensor_bytes(backend, tensors.output, dirty.data(), dirty.size() * sizeof(float));
            REQUIRE(ggml_backend_graph_compute(backend, tensors.graph) == GGML_STATUS_SUCCESS);
            const auto actual = get_f32_tensor(backend, tensors.output);
            REQUIRE(actual.size() == static_cast<size_t>(rows * columns));
            for (float value : actual) {
                REQUIRE(std::isfinite(value));
            }
            require_close(actual, expected, replay == 2 ? 0.0f : kDenseF32AbsTolerance,
                          replay == 2 ? 0.0f : kDenseF32RelTolerance);
            if (hc) {
                const auto normalized = get_f32_tensor(backend, tensors.normalized);
                for (float value : normalized) {
                    REQUIRE(std::isfinite(value));
                }
                require_close(normalized, rmsnorm_mul_reference(expected, std::vector<float>(rows, 1.0f),
                              rows, columns), kDenseF32AbsTolerance, kDenseF32RelTolerance);
            }
        }
    }
    if (run_gpu) {
        std::vector<uint8_t> unchanged(weight.size());
        ggml_backend_tensor_get(tensors.weight, unchanged.data(), 0, unchanged.size());
        ggml_backend_synchronize(backend);
        REQUIRE(unchanged == weight);
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    if (run_gpu) {
        ggml_backend_free(backend);
    }
    std::fprintf(stderr, "Dense F32 accumulation %s K=%lld N=%lld columns=%lld HC=%d high/cancellation=%d "
                         "CPU operand oracle/replay%s passed\n", ggml_type_name(type),
                 (long long) width, (long long) rows, (long long) columns, hc, high_range,
                 run_gpu ? "/GPU" : "");
}

static void run_dense_f32_accum_reference_checks() {
    for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 }) {
        run_dense_f32_accum_reference_case(type, 256, 65, 33, false, false, true);
        run_dense_f32_accum_reference_case(type, 5120, 65, 17, false, false, true);
        run_dense_f32_accum_reference_case(type, 512, 65, 3, false, true, true);
    }
    for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
        run_dense_f32_accum_reference_case(GGML_TYPE_Q4_K, 5120, 2560, 4 * tokens, true, false, true);
    }
}

static void run_q8_narrow_scheduling_checks() {
    ScopedHrxEnvironment gemv("HRX_ENABLE_Q8_GEMV", "0");
    const char * flags[] = { nullptr, "", "0", "true", "01", "1" };
    for (const char * flag : flags) {
        ScopedHrxEnvironment environment("HRX_ENABLE_Q8_NARROW", flag);
        for (int64_t width : { 320, 640 }) {
            for (int64_t tokens : { 1, 2, 3, 4, 5, 6, 7, 8, 9, 512 }) {
                for (int64_t rows : { 3, 2560, 10240 }) {
                    ggml_init_params params = {};
                    params.mem_size = 1024 * 1024;
                    params.no_alloc = true;
                    ggml_context * ctx = ggml_init(params);
                    REQUIRE(ctx != nullptr);
                    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, rows);
                    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
                    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
                    ggml_cgraph * graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, output);
                    const bool enabled = flag != nullptr && std::strcmp(flag, "1") == 0 && tokens <= 8;
                    REQUIRE(ggml::hrx::llm_q8_narrow_supported(*weight, *input, *output) == enabled);
                    auto imported = ggml::hrx::import_ggml_graph(*graph);
                    REQUIRE(imported.valid());
                    ggml::hrx::DispatchScheduler scheduler;
                    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == enabled);
                    if (enabled) {
                        const auto & plan = scheduler.plan();
                        REQUIRE(plan.valid());
                        REQUIRE(plan.dispatches.size() == 2);
                        REQUIRE(plan.transients.size() == 1);
                        REQUIRE(plan.constant_initializations.empty());
                        const auto & pack = plan.dispatches[0];
                        const auto & dot = plan.dispatches[1];
                        REQUIRE(kernel_name_for_id(pack.kernel.kernel_id) == (tokens == 1 ?
                            "hrx_owned:ggml_q8_narrow_pack" : "hrx_owned:ggml_q8_narrow_pack_batched"));
                        REQUIRE(kernel_name_for_id(dot.kernel.kernel_id) == (tokens == 1 ?
                            "hrx_owned:ggml_q8_narrow_dot" : "hrx_owned:ggml_q8_narrow_dot_batched"));
                        REQUIRE(pack.kernel.integer_parameters.size() == (tokens == 1 ? 1 : 2));
                        REQUIRE(dot.kernel.integer_parameters.size() == (tokens == 1 ? 2 : 3));
                        if (tokens > 1) {
                            REQUIRE(pack.kernel.integer_parameters.at("token_count") == tokens);
                            REQUIRE(dot.kernel.integer_parameters.at("token_count") == tokens);
                        }
                        REQUIRE(pack.kernel.integer_parameters.at("input_size") == width);
                        REQUIRE(dot.kernel.integer_parameters.at("input_size") == width);
                        REQUIRE(dot.kernel.integer_parameters.at("output_size") == rows);
                        REQUIRE(pack.bindings.size() == 2);
                        REQUIRE(dot.bindings.size() == 3);
                        REQUIRE(pack.bindings[0].value == imported.graph.values().find_tensor(input)->id);
                        REQUIRE(pack.bindings[0].length == ggml_nbytes(input));
                        REQUIRE(pack.bindings[1].value == plan.transients[0].value);
                        REQUIRE(dot.bindings[0].value == plan.transients[0].value);
                        REQUIRE(pack.bindings[1].length == ggml_row_size(GGML_TYPE_Q8_0, width) * tokens);
                        REQUIRE(dot.bindings[0].length == pack.bindings[1].length);
                        REQUIRE(plan.transients[0].size == pack.bindings[1].length);
                        REQUIRE(dot.bindings[1].value == imported.graph.values().find_tensor(weight)->id);
                        REQUIRE(dot.bindings[1].length == ggml_nbytes(weight));
                        REQUIRE(dot.bindings[2].value == imported.graph.values().find_tensor(output)->id);
                        REQUIRE(dot.bindings[2].length == ggml_nbytes(output));
                    }
                    ggml_free(ctx);
                }
            }
        }
    }

    ScopedHrxEnvironment environment("HRX_ENABLE_Q8_NARROW", "1");
    for (int64_t tokens : { 1, 4 }) {
        for (int malformed = 0; malformed < 11; ++malformed) {
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            ggml_context * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, malformed == 0 ? 352 : 320,
                                                     malformed == 1 ? 10241 : 5);
            ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, weight->ne[0], tokens);
            ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
            if (malformed == 2) { weight->nb[1] += 34; }
            if (malformed == 3) { input->nb[0] *= 2; }
            if (malformed == 4) { output->nb[0] *= 2; }
            if (malformed == 5) { input->type = GGML_TYPE_F16; }
            if (malformed == 6) { output->ne[2] = 2; }
            if (malformed == 7) { input->nb[1] += sizeof(float); }
            if (malformed == 8) { output->nb[1] += sizeof(float); }
            if (malformed == 9) { weight->ne[2] = 2; }
            if (malformed == 10) { output->ne[1] += 1; }
            REQUIRE(!ggml::hrx::llm_q8_narrow_supported(*weight, *input, *output));
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);
            auto imported = ggml::hrx::import_ggml_graph(*graph);
            if (imported.valid()) {
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(!scheduler.schedule_graph(imported.graph, { "gfx1151" }));
            }
            ggml_free(ctx);
        }
    }

    // The activation's producer must execute before the pack, not be consumed by a later-root fusion.
    for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 320, 5);
        ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 320, tokens);
        ggml_tensor * input = ggml_add(ctx, source, source);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        const auto sequence = scheduled_kernel_sequence(graph);
        REQUIRE(sequence.size() == 3);
        REQUIRE(sequence[1] == (tokens == 1 ? "hrx_owned:ggml_q8_narrow_pack" : "hrx_owned:ggml_q8_narrow_pack_batched"));
        REQUIRE(sequence[2] == (tokens == 1 ? "hrx_owned:ggml_q8_narrow_dot" : "hrx_owned:ggml_q8_narrow_dot_batched"));
        ggml_free(ctx);
    }
}

static std::vector<uint8_t> q8_narrow_activation_reference(const std::vector<float> & input) {
    std::vector<uint8_t> packed(input.size() / 32 * 34);
    for (size_t block = 0; block < input.size() / 32; ++block) {
        float amax = 0.0f;
        for (size_t q = 0; q < 32; ++q) {
            amax = std::max(amax, std::fabs(input[block * 32 + q]));
        }
        const ggml_fp16_t d = ggml_fp32_to_fp16(amax / 127.0f);
        const float inverse = amax == 0.0f ? 0.0f : 127.0f / amax;
        std::memcpy(packed.data() + block * 34, &d, sizeof(d));
        for (size_t q = 0; q < 32; ++q) {
            const float x = input[block * 32 + q] * inverse;
            const float lower = std::floor(x);
            const float fraction = x - lower;
            const int rounded = static_cast<int>(lower) +
                                (fraction > 0.5f || (fraction == 0.5f && static_cast<int>(lower) % 2 != 0));
            const int8_t code = static_cast<int8_t>(rounded);
            std::memcpy(packed.data() + block * 34 + 2 + q, &code, sizeof(code));
        }
    }
    return packed;
}

static std::vector<float> q8_narrow_dot_reference(const std::vector<uint8_t> & weight,
                                                 const std::vector<uint8_t> & input, int64_t rows, int64_t tokens = 1) {
    const size_t blocks = input.size() / 34 / tokens;
    std::vector<float> expected(rows * tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (size_t block = 0; block < blocks; ++block) {
                const uint8_t * w = weight.data() + (row * blocks + block) * 34;
                const uint8_t * a = input.data() + (token * blocks + block) * 34;
                ggml_fp16_t wd, ad;
                std::memcpy(&wd, w, sizeof(wd));
                std::memcpy(&ad, a, sizeof(ad));
                int dot = 0;
                for (size_t q = 0; q < 32; ++q) {
                    int8_t wq, aq;
                    std::memcpy(&wq, w + 2 + q, 1);
                    std::memcpy(&aq, a + 2 + q, 1);
                    dot += static_cast<int>(wq) * static_cast<int>(aq);
                }
                sum += static_cast<float>(dot) * (ggml_fp16_to_fp32(wd) * ggml_fp16_to_fp32(ad));
            }
            expected[token * rows + row] = static_cast<float>(sum);
        }
    }
    return expected;
}

static void run_q8_narrow_reference_case(int64_t width, int64_t rows, bool basis, bool use_hrx,
                                        int64_t tokens = 1, bool alias = false) {
    ScopedHrxEnvironment environment("HRX_ENABLE_Q8_NARROW", "1");
    ScopedHrxEnvironment gemv("HRX_ENABLE_Q8_GEMV", "0");
    // init_by_type initializes the global backend registry, including GPU discovery.
    ggml_backend_t backends[2] = { ggml_backend_cpu_init(), use_hrx ? ggml_backend_hrx_init(0) : nullptr };
    REQUIRE(backends[0] != nullptr && (!use_hrx || backends[1] != nullptr));
    ggml_context * contexts[2] = {};
    ggml_backend_buffer_t buffers[2] = {};
    ggml_tensor * weights[2] = {};
    ggml_tensor * inputs[2] = {};
    ggml_tensor * outputs[2] = {};
    ggml_tensor * arenas[2] = {};
    ggml_cgraph * graphs[2] = {};
    ggml_tensor * single_input = nullptr;
    ggml_tensor * single_output = nullptr;
    ggml_cgraph * single_graph = nullptr;
    const int64_t guard = 32, shift = 8;
    const int64_t arena_count = 2 * guard + std::max(width * tokens, shift + rows * tokens);
    const int backend_count = use_hrx ? 2 : 1;
    for (int b = 0; b < backend_count; ++b) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        contexts[b] = ggml_init(params);
        REQUIRE(contexts[b] != nullptr);
        weights[b] = ggml_new_tensor_2d(contexts[b], GGML_TYPE_Q8_0, width, rows);
        if (alias) {
            arenas[b] = ggml_new_tensor_1d(contexts[b], GGML_TYPE_F32, arena_count);
            inputs[b] = ggml_view_2d(contexts[b], arenas[b], width, tokens, width * sizeof(float), guard * sizeof(float));
        } else {
            inputs[b] = ggml_new_tensor_2d(contexts[b], GGML_TYPE_F32, width, tokens);
        }
        outputs[b] = ggml_mul_mat(contexts[b], weights[b], inputs[b]);
        if (alias) {
            outputs[b]->view_src = arenas[b];
            outputs[b]->view_offs = (guard + shift) * sizeof(float);
        }
        graphs[b] = ggml_new_graph(contexts[b]);
        ggml_build_forward_expand(graphs[b], outputs[b]);
        if (b == 1 && tokens > 1) {
            single_input = ggml_new_tensor_2d(contexts[b], GGML_TYPE_F32, width, 1);
            single_output = ggml_mul_mat(contexts[b], weights[b], single_input);
            single_graph = ggml_new_graph(contexts[b]);
            ggml_build_forward_expand(single_graph, single_output);
        }
        buffers[b] = ggml_backend_alloc_ctx_tensors(contexts[b], backends[b]);
        REQUIRE(buffers[b] != nullptr);
        if (b == 1) {
            REQUIRE(ggml_backend_supports_op(backends[b], outputs[b]));
            require_kernel_subsequence(scheduled_kernel_sequence(graphs[b]), tokens == 1 ?
                std::vector<std::string>{ "hrx_owned:ggml_q8_narrow_pack", "hrx_owned:ggml_q8_narrow_dot" } :
                std::vector<std::string>{ "hrx_owned:ggml_q8_narrow_pack_batched", "hrx_owned:ggml_q8_narrow_dot_batched" });
            ScopedHrxEnvironment disabled("HRX_ENABLE_Q8_NARROW", "0");
            REQUIRE(!ggml_backend_supports_op(backends[b], outputs[b]));
        }
    }
    REQUIRE(ggml_type_size(GGML_TYPE_Q8_0) == 34);
    std::vector<uint8_t> weight(ggml_nbytes(weights[0]));
    const int64_t blocks = width / 32;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t block = 0; block < blocks; ++block) {
            const size_t base = (row * blocks + block) * 34;
            const ggml_fp16_t scale = ggml_fp32_to_fp16(
                basis ? 1.0f : static_cast<float>((row * 3 + block * 5) % 15 - 7) / 256.0f);
            std::memcpy(weight.data() + base, &scale, sizeof(scale));
            for (int64_t q = 0; q < 32; ++q) {
                const int8_t code = basis ? (row == block * 32 + q ? 1 : 0) :
                    static_cast<int8_t>((row * 29 + block * 13 + q * 7) % 256 - 128);
                std::memcpy(weight.data() + base + 2 + q, &code, 1);
            }
        }
    }
    for (int b = 0; b < backend_count; ++b) {
        set_tensor_bytes(backends[b], weights[b], weight.data(), weight.size());
    }
    for (int replay = 0; replay < 5; ++replay) {
        std::vector<float> input(width * tokens);
        for (int64_t token = 0; token < tokens; ++token) {
            for (int64_t column = 0; column < width; ++column) {
                const int64_t block = column / 32;
                const int64_t q = column % 32;
                if (replay == 1) {
                    const float step = (block % 2 == 0 ? 1.0f : 0.0625f);
                    input[token * width + column] = (q == 31 ? 127.0f : q == 30 ? -127.0f :
                        (q % 2 == 0 ? 1.0f : -1.0f) * (static_cast<float>((q / 2 + token) % 15) + 0.5f)) *
                        step * (token % 2 == 0 ? 1.0f : -2.0f);
                } else if (replay == 3 || (replay == 0 && block < 2)) {
                    input[token * width + column] = q % 2 == 0 ? 0.0f : -0.0f;
                } else {
                    input[token * width + column] =
                        static_cast<float>((column * 17 + block * 11 + replay * 7 + token * 23) % 253 - 126) *
                        (replay == 2 ? 0.0000013f : 0.0013f);
                }
            }
        }
        const auto packed = q8_narrow_activation_reference(input);
        if (replay == 1) {
            std::vector<uint8_t> scalar_reference(packed.size());
            ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(input.data(), scalar_reference.data(), input.size());
            REQUIRE(packed != scalar_reference);
            int8_t first_even, first_away;
            std::memcpy(&first_even, packed.data() + 2, 1);
            std::memcpy(&first_away, scalar_reference.data() + 2, 1);
            REQUIRE(first_even == 0 && first_away == 1);
        }
        const auto expected = q8_narrow_dot_reference(weight, packed, rows, tokens);
        for (int b = 0; b < backend_count; ++b) {
            if (alias) {
                const std::vector<float> canary(arena_count, 12345.0f);
                set_tensor_bytes(backends[b], arenas[b], canary.data(), canary.size() * sizeof(float));
            }
            const std::vector<float> dirty(rows * tokens, 123.0f);
            set_tensor_bytes(backends[b], outputs[b], dirty.data(), dirty.size() * sizeof(float));
            set_tensor_bytes(backends[b], inputs[b], input.data(), input.size() * sizeof(float));
            REQUIRE(ggml_backend_graph_compute(backends[b], graphs[b]) == GGML_STATUS_SUCCESS);
            const auto actual = get_f32_tensor(backends[b], outputs[b]);
            for (float value : actual) {
                REQUIRE(std::isfinite(value));
            }
            // Basis rows expose each packed code and f16 scale, including signed ties and zero blocks.
            // The CPU comparison pins the AVX semantics, rather than assuming from_float_ref is equivalent.
            require_close(actual, expected, basis ? 0.0f : 3.0e-5f, basis ? 0.0f : 2.0e-5f);
            if (alias) {
                const auto arena = get_f32_tensor(backends[b], arenas[b]);
                REQUIRE(std::memcmp(arena.data() + guard + shift, actual.data(), actual.size() * sizeof(float)) == 0);
                for (int64_t i = 0; i < arena_count; ++i) {
                    if (i >= guard + shift && i < guard + shift + rows * tokens) { continue; }
                    const float untouched = i >= guard && i < guard + width * tokens ? input[i - guard] : 12345.0f;
                    REQUIRE(std::memcmp(&arena[i], &untouched, sizeof(float)) == 0);
                }
            }
            if (b == 1 && tokens > 1) {
                for (int64_t token = 0; token < tokens; ++token) {
                    set_tensor_bytes(backends[b], single_input, input.data() + token * width, width * sizeof(float));
                    REQUIRE(ggml_backend_graph_compute(backends[b], single_graph) == GGML_STATUS_SUCCESS);
                    const auto single = get_f32_tensor(backends[b], single_output);
                    REQUIRE(std::memcmp(single.data(), actual.data() + token * rows, rows * sizeof(float)) == 0);
                }
            }
        }
    }
    for (int b = 0; b < backend_count; ++b) {
        std::vector<uint8_t> unchanged(weight.size());
        ggml_backend_tensor_get(weights[b], unchanged.data(), 0, unchanged.size());
        ggml_backend_synchronize(backends[b]);
        REQUIRE(unchanged == weight);
        ggml_backend_buffer_free(buffers[b]);
        ggml_free(contexts[b]);
        ggml_backend_free(backends[b]);
    }
}

static void run_q8_narrow_reference_checks(bool use_hrx = true) {
    for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
        for (int64_t width : { 320, 640 }) {
            run_q8_narrow_reference_case(width, width, true, use_hrx, tokens);
            for (int64_t rows : { 1, 3, 4, 5, 65 }) {
                run_q8_narrow_reference_case(width, rows, false, use_hrx, tokens);
            }
            run_q8_narrow_reference_case(width, 65, false, use_hrx, tokens, true);
        }
        run_q8_narrow_reference_case(320, 10240, false, use_hrx, tokens);
        run_q8_narrow_reference_case(640, 2560, false, use_hrx, tokens);
        std::fprintf(stderr, "Q8 narrow T=%lld: 16 cases, 80 CPU/reference%s replays passed\n",
                     (long long) tokens, use_hrx ? (tokens > 1 ? "/GPU/T1-bitwise" : "/GPU") : "");
    }
}

static void run_iq_expert_scheduling_checks(bool small_batch = false) {
    ScopedHrxEnvironment small_environment("HRX_ENABLE_MOE_SMALL_BATCH", small_batch ? "1" : nullptr);
    const char * flags[] = { nullptr, "0", "true", "1" };
    for (ggml_type type : { GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS }) {
        for (const char * flag : flags) {
            ScopedHrxEnvironment environment("HRX_ENABLE_IQ_EXPERTS", flag);
            for (int64_t tokens : { 1, 2, 3, 4, 8, 9, 512 }) {
                ggml_init_params params = {};
                params.mem_size         = 1024 * 1024;
                params.no_alloc         = true;
                ggml_context * ctx      = ggml_init(params);
                REQUIRE(ctx != nullptr);
                ggml_tensor * weight = ggml_new_tensor_3d(ctx, type, 2560, 640, 512);
                ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 1, tokens);
                ggml_tensor * ids_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 512, tokens);
                ggml_tensor * ids = ggml_view_2d(ctx, ids_storage, 10, tokens, 512 * sizeof(int32_t), 0);
                ggml_tensor * gate   = ggml_mul_mat_id(ctx, weight, input, ids);
                ggml_tensor * up     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 640, 10, tokens);
                ggml_tensor * output = ggml_swiglu_split(ctx, gate, up);
                ggml_cgraph * graph  = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, output);
                ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
                REQUIRE(imported.valid());
                ggml::hrx::DispatchScheduler scheduler;
                const bool enabled = flag != nullptr && std::strcmp(flag, "1") == 0 &&
                                     (tokens == 1 || (small_batch && tokens <= 8));
                REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == enabled);
                if (enabled) {
                    const auto & plan = scheduler.plan();
                    REQUIRE(plan.valid());
                    REQUIRE(plan.dispatches.size() == 3);
                    const auto & dispatch = plan.dispatches[0];
                    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
                            (type == GGML_TYPE_IQ3_XXS ? "hrx_owned:ggml_mul_mat_id_iq3_xxs_f32" :
                                                        "hrx_owned:ggml_mul_mat_id_iq4_xs_f32"));
                    REQUIRE(dispatch.kernel.integer_parameters.at("input_size") == 2560);
                    REQUIRE(dispatch.kernel.integer_parameters.at("output_size") == 640);
                    REQUIRE(dispatch.kernel.integer_parameters.at("expert_count") == 512);
                    REQUIRE(dispatch.kernel.integer_parameters.at("route_count") == 10);
                    REQUIRE(dispatch.bindings.size() == 5);
                    REQUIRE(dispatch.bindings[0].length == ggml_nbytes(weight));
                    REQUIRE(dispatch.bindings[1].length == ggml_nbytes(input));
                    REQUIRE(dispatch.bindings[2].length == ((tokens - 1) * 512 + 10) * sizeof(int32_t));
                    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tokens);
                    REQUIRE(dispatch.kernel.integer_parameters.at("route_id_stride") == 512);
                    REQUIRE(dispatch.bindings[3].length == 1168);
                    REQUIRE(dispatch.bindings[4].length == ggml_nbytes(gate));
                    REQUIRE(plan.constant_initializations.size() == 1);
                    REQUIRE(plan.constant_initializations[0].data.size() == 1168);
                    REQUIRE(plan.constant_initializations[0].value == dispatch.bindings[3].value);
                    REQUIRE(kernel_name_for_id(plan.dispatches[1].kernel.kernel_id) == "qwen3_moe:ggml_silu_f32");
                    REQUIRE(kernel_name_for_id(plan.dispatches[2].kernel.kernel_id) == "qwen3_moe:ggml_mul_f32");
                }
                ggml_free(ctx);
            }
        }
    }
}

static ggml_tensor * make_moe_ids_tensor(ggml_context * ctx, int64_t tokens, int64_t stride) {
    auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, stride, tokens);
    return ggml_view_2d(ctx, storage, 10, tokens, stride * sizeof(int32_t), 0);
}

static std::vector<int32_t> make_moe_ids(int64_t tokens) {
    const int32_t experts[] = { 511, 0, 257, 3, 127, 1, 511, 64, 255, 7 };
    std::vector<int32_t> ids(10 * tokens);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = experts[(i + i / 10 * 3) % 10];
    }
    return ids;
}

static void set_moe_ids(ggml_backend_t backend, ggml_tensor * tensor, const std::vector<int32_t> & ids) {
    const size_t stride = tensor->nb[1] / sizeof(int32_t);
    std::vector<int32_t> storage((tensor->ne[1] - 1) * stride + 10, -12345);
    for (size_t i = 0; i < ids.size(); ++i) {
        storage[i / 10 * stride + i % 10] = ids[i];
    }
    set_tensor_bytes(backend, tensor, storage.data(), storage.size() * sizeof(int32_t));
}

static void run_iq_expert_raw_reference_case(ggml_type type, bool random_codes = false,
                                           int64_t tokens = 1, int64_t stride = 10) {
    ScopedHrxEnvironment environment("HRX_ENABLE_IQ_EXPERTS", "1");
    ScopedHrxEnvironment small_environment("HRX_ENABLE_MOE_SMALL_BATCH", tokens > 1 ? "1" : nullptr);
    constexpr int64_t width = 2560;
    constexpr int64_t rows = 640;
    constexpr int64_t experts = 512;
    std::vector<int32_t> ids = make_moe_ids(tokens);
    constexpr int64_t routes = 10;
    const int64_t assignments = routes * tokens;
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * weight_tensor = ggml_new_tensor_3d(ctx, type, width, rows, experts);
    ggml_tensor * input_tensor  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, tokens);
    ggml_tensor * ids_tensor    = make_moe_ids_tensor(ctx, tokens, stride);
    ggml_tensor * gate_tensor   = ggml_mul_mat_id(ctx, weight_tensor, input_tensor, ids_tensor);
    ggml_tensor * up_tensor     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rows, routes, tokens);
    ggml_tensor * output_tensor = ggml_swiglu_split(ctx, gate_tensor, up_tensor);
    ggml_cgraph * graph         = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output_tensor);
    require_kernel_subsequence(scheduled_kernel_sequence(graph),
                               { type == GGML_TYPE_IQ3_XXS ? "hrx_owned:ggml_mul_mat_id_iq3_xxs_f32" :
                                                           "hrx_owned:ggml_mul_mat_id_iq4_xs_f32",
                                 "qwen3_moe:ggml_silu_f32", "qwen3_moe:ggml_mul_f32" });
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    const size_t block_bytes = ggml_type_size(type);
    REQUIRE(block_bytes == (type == GGML_TYPE_IQ3_XXS ? 98 : 136));
    const size_t row_bytes = ggml_row_size(type, width);
    std::vector<uint8_t> weight(ggml_nbytes(weight_tensor), 0);
    // Only selected experts need nonzero data; every uploaded block is still valid raw IQ.
    for (int32_t expert : ids) {
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t block = 0; block < width / 256; ++block) {
                const size_t base = (static_cast<size_t>(expert) * rows + row) * row_bytes + block * block_bytes;
                const ggml_fp16_t scale = ggml_fp32_to_fp16(
                    static_cast<float>((expert + row * 3 + block * 5) % 13 - 6) / 4096.0f);
                std::memcpy(weight.data() + base, &scale, sizeof(scale));
                uint32_t state = 0xa341316cu ^ (static_cast<uint32_t>(expert) * 0x9e3779b9u) ^
                                 (static_cast<uint32_t>(row) * 0x85ebca6bu) ^ static_cast<uint32_t>(block);
                for (size_t byte = sizeof(scale); byte < block_bytes; ++byte) {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    weight[base + byte] = random_codes ? static_cast<uint8_t>(state) :
                        static_cast<uint8_t>(expert * 29 + row * 13 + block * 7 + byte * 37);
                }
            }
        }
    }
    set_tensor_bytes(backend, weight_tensor, weight.data(), weight.size());
    for (int replay = 0; replay < 2; ++replay) {
    if (replay != 0) {
        std::rotate(ids.begin(), ids.begin() + 3, ids.end());
    }
    const std::vector<float> input = make_pattern_f32(width * tokens, 23 + replay * 8, 0.0013f);
    const std::vector<float> up = make_pattern_f32(rows * assignments, 29 + replay * 4, 0.013f);
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    REQUIRE(traits != nullptr && traits->to_float != nullptr);
    std::vector<float> decoded(width);
    std::vector<float> expected_gate(rows * assignments);
    std::vector<float> expected_output(rows * assignments);
    for (int64_t route = 0; route < assignments; ++route) {
        for (int64_t row = 0; row < rows; ++row) {
            const size_t base = (static_cast<size_t>(ids[route]) * rows + row) * row_bytes;
            traits->to_float(weight.data() + base, decoded.data(), width);
            double sum = 0.0;
            for (int64_t column = 0; column < width; ++column) {
                sum += static_cast<double>(decoded[column]) * input[(route / routes) * width + column];
            }
            const size_t index = static_cast<size_t>(route * rows + row);
            const float gate = static_cast<float>(sum);
            expected_gate[index] = gate;
            expected_output[index] = gate / (1.0f + std::exp(-gate)) * up[index];
        }
    }
    set_tensor_bytes(backend, input_tensor, input.data(), input.size() * sizeof(float));
    set_moe_ids(backend, ids_tensor, ids);
    set_tensor_bytes(backend, up_tensor, up.data(), up.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const auto actual_gate = get_f32_tensor(backend, gate_tensor);
    const auto actual_output = get_f32_tensor(backend, output_tensor);
    for (float value : actual_gate) {
        REQUIRE(std::isfinite(value));
    }
    for (float value : actual_output) {
        REQUIRE(std::isfinite(value));
    }
    require_close(actual_gate, expected_gate, 3.0e-4f, 1.0e-5f);
    require_close(actual_output, expected_output, 3.0e-4f, 1.0e-5f);

    // Invalid IDs are tested only against the explicit kernel error result, never a CPU weight lookup.
    std::vector<int32_t> invalid_ids = ids;
    invalid_ids[1] = -1;
    invalid_ids[assignments - 2] = static_cast<int32_t>(experts);
    set_moe_ids(backend, ids_tensor, invalid_ids);
    ggml_cgraph * invalid_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(invalid_graph, gate_tensor);
    REQUIRE(ggml_backend_graph_compute(backend, invalid_graph) == GGML_STATUS_SUCCESS);
    const auto invalid_output = get_f32_tensor(backend, gate_tensor);
    std::vector<float> valid_actual;
    std::vector<float> valid_expected;
    for (int64_t route = 0; route < assignments; ++route) {
        for (int64_t row = 0; row < rows; ++row) {
            const size_t index = static_cast<size_t>(route * rows + row);
            if (invalid_ids[route] < 0 || invalid_ids[route] >= experts) {
                if (!std::isnan(invalid_output[index])) {
                    std::fprintf(stderr, "IQ invalid route %d row %lld: expected NaN, got %g\n",
                                 invalid_ids[route], static_cast<long long>(row), invalid_output[index]);
                }
                REQUIRE(std::isnan(invalid_output[index]));
            } else {
                REQUIRE(std::isfinite(invalid_output[index]));
                valid_actual.push_back(invalid_output[index]);
                valid_expected.push_back(expected_gate[index]);
            }
        }
    }
    require_close(valid_actual, valid_expected, 3.0e-4f, 1.0e-5f);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

struct DecodeDownGraph {
    ggml_tensor * weight;
    ggml_tensor * input;
    ggml_tensor * ids;
    ggml_tensor * route_weights;
    ggml_tensor * output;
};

static DecodeDownGraph build_decode_down_graph(ggml_context * ctx, ggml_type type,
                                              int64_t tokens = 1, int64_t stride = 10) {
    DecodeDownGraph result = {};
    result.weight = ggml_new_tensor_3d(ctx, type, 640, 2560, 512);
    result.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 640, 10, tokens);
    result.ids = make_moe_ids_tensor(ctx, tokens, stride);
    result.route_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 10, tokens);
    ggml_tensor * down = ggml_mul_mat_id(ctx, result.weight, result.input, result.ids);
    ggml_tensor * weighted = ggml_mul(ctx, down, result.route_weights);
    result.output = ggml_view_2d(ctx, weighted, 2560, tokens, weighted->nb[2], 0);
    for (int64_t route = 1; route < 10; ++route) {
        auto * view = ggml_view_2d(ctx, weighted, 2560, tokens, weighted->nb[2], route * weighted->nb[1]);
        result.output = ggml_add(ctx, result.output, view);
    }
    return result;
}

struct Q4ExpertGraph {
    ggml_tensor * gate_weight;
    ggml_tensor * up_weight;
    ggml_tensor * down_weight;
    ggml_tensor * input;
    ggml_tensor * ids;
    ggml_tensor * route_weights;
    ggml_tensor * swiglu;
    ggml_tensor * output;
    ggml_tensor * standalone_input;
    ggml_tensor * standalone_output;
    ggml_cgraph * graph;
    ggml_cgraph * standalone_graph;
    ggml_tensor * logits_a;
    ggml_tensor * logits_b;
};

static ggml_tensor * q4_expert_fold(ggml_context * ctx, ggml_tensor * down, ggml_tensor * weights) {
    auto * weighted = ggml_mul(ctx, down, weights);
    auto * output = ggml_view_2d(ctx, weighted, 2560, down->ne[2], weighted->nb[2], 0);
    for (int64_t route = 1; route < 10; ++route) {
        auto * view = ggml_view_2d(ctx, weighted, 2560, down->ne[2], weighted->nb[2], route * weighted->nb[1]);
        output = ggml_add(ctx, output, view);
    }
    return output;
}

static Q4ExpertGraph build_q4_expert_graph(ggml_context * ctx, int64_t stride,
                                           bool router = false, ggml_context * weight_ctx = nullptr,
                                           ggml_context * host_input_ctx = nullptr) {
    Q4ExpertGraph result = {};
    if (weight_ctx == nullptr) {
        weight_ctx = ctx;
    }
    result.gate_weight = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_Q4_K, 2560, 640, 512);
    result.up_weight = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_Q4_K, 2560, 640, 512);
    result.down_weight = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_Q8_0, 640, 2560, 512);
    auto * input_ctx = host_input_ctx == nullptr ? ctx : host_input_ctx;
    result.input = ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, 2560, 1, 1);
    if (router) {
        REQUIRE(stride == 512);
        result.logits_a = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, 512, 1);
        if (host_input_ctx == nullptr) {
            result.logits_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
            ggml_set_input(result.logits_b);
        }
        ggml_set_input(result.logits_a);
        ggml_set_input(result.input);
        auto * logits = result.logits_b == nullptr ? result.logits_a : ggml_add(ctx, result.logits_a, result.logits_b);
        auto * probabilities = ggml_soft_max(ctx, logits);
        auto * sorted = ggml_argsort(ctx, probabilities, GGML_SORT_ORDER_DESC);
        result.ids = ggml_view_2d(ctx, sorted, 10, 1, sorted->nb[1], 0);
        auto * selected = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probabilities, 1, 512, 1), result.ids);
        auto * flat = ggml_reshape_2d(ctx, selected, 10, 1);
        auto * denominator = ggml_clamp(ctx, ggml_sum_rows(ctx, flat), 6.103515625e-5f,
                                        std::numeric_limits<float>::infinity());
        result.route_weights = ggml_reshape_3d(ctx, ggml_div(ctx, flat, denominator), 1, 10, 1);
    } else {
        result.ids = make_moe_ids_tensor(ctx, 1, stride);
        result.route_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 10, 1);
    }
    auto * up = ggml_mul_mat_id(ctx, result.up_weight, result.input, result.ids);
    auto * gate = ggml_mul_mat_id(ctx, result.gate_weight, result.input, result.ids);
    result.swiglu = ggml_swiglu_split(ctx, gate, up);
    if (!router) {
        ggml_set_output(result.swiglu);
    }
    result.output = q4_expert_fold(ctx, ggml_mul_mat_id(ctx, result.down_weight, result.swiglu, result.ids),
                                  result.route_weights);
    result.graph = ggml_new_graph(ctx);
    if (router) {
        ggml_set_output(result.output);
        ggml_build_forward_expand(result.graph, result.route_weights);
    }
    ggml_build_forward_expand(result.graph, result.output);
    result.standalone_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 640, 10, 1);
    result.standalone_output = q4_expert_fold(ctx,
        ggml_mul_mat_id(ctx, result.down_weight, result.standalone_input, result.ids), result.route_weights);
    result.standalone_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(result.standalone_graph, result.standalone_output);
    return result;
}

static void run_q4_expert_id_validation_checks() {
    using ggml::hrx::q4_routed_gate_up_id_byte_count;
    using ggml::hrx::validate_q4_routed_gate_up_id_bytes;
    for (int64_t tokens : { 1, 2, 3 }) {
        for (int64_t stride : { 10, 64, 512 }) {
            const size_t size = q4_routed_gate_up_id_byte_count(tokens, 10, stride, 512);
            REQUIRE(size == static_cast<size_t>((tokens - 1) * stride + 10) * sizeof(int32_t));
            std::vector<int32_t> ids(size / sizeof(int32_t), std::numeric_limits<int32_t>::min());
            for (int64_t token = 0; token < tokens; ++token) {
                const int32_t valid[] = { 511, 0, 128, 3, 127, 1, 511, 256, 255, 7 };
                std::copy_n(valid, 10, ids.begin() + token * stride);
            }
            REQUIRE(validate_q4_routed_gate_up_id_bytes(ids.data(), size, tokens, 10, stride, 512).valid);
            REQUIRE(!validate_q4_routed_gate_up_id_bytes(ids.data(), size - 1, tokens, 10, stride, 512).valid);
            REQUIRE(!validate_q4_routed_gate_up_id_bytes(nullptr, size, tokens, 10, stride, 512).valid);
            for (int64_t token : { int64_t(0), tokens - 1 }) {
                for (int64_t route : { int64_t(0), int64_t(9) }) {
                    const size_t index = static_cast<size_t>(token * stride + route);
                    const int32_t original = ids[index];
                    for (int32_t invalid : { -1, 512, std::numeric_limits<int32_t>::min(),
                                             std::numeric_limits<int32_t>::max() }) {
                        ids[index] = invalid;
                        const auto result = validate_q4_routed_gate_up_id_bytes(ids.data(), size, tokens, 10, stride, 512);
                        REQUIRE(!result.valid && result.invalid_id == invalid && result.token == token && result.route == route);
                    }
                    ids[index] = original;
                }
            }
            REQUIRE(validate_q4_routed_gate_up_id_bytes(ids.data(), size, tokens, 10, stride, 512).valid);
        }
    }
    REQUIRE(q4_routed_gate_up_id_byte_count(0, 10, 512, 512) == 0);
    REQUIRE(q4_routed_gate_up_id_byte_count(513, 10, 512, 512) == 0);
    REQUIRE(q4_routed_gate_up_id_byte_count(1, 17, 512, 512) == 0);
    REQUIRE(q4_routed_gate_up_id_byte_count(1, 10, 9, 512) == 0);
    REQUIRE(q4_routed_gate_up_id_byte_count(1, 10, 513, 512) == 0);
    REQUIRE(q4_routed_gate_up_id_byte_count(1, 10, 512, 513) == 0);
    REQUIRE(q4_routed_gate_up_id_byte_count(std::numeric_limits<int64_t>::max(), 10, 512, 512) == 0);
    for (const char * name : { "qwen3_moe_routed_gate_up_swiglu_q4k_q8",
                               "qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8" }) {
        const uint64_t id = ggml::hrx::kernel_catalog_id("qwen3_moe", name);
        REQUIRE(ggml::hrx::is_q4_routed_gate_up_kernel(id));
        for (bool initialization : { false, true }) {
            ggml::hrx::PreparedCommandProgram prepared;
            ggml::hrx::PreparedCommand command;
            command.kind = ggml::hrx::CommandKind::Kernel;
            command.kernel.specialization.kernel_id = id;
            (initialization ? prepared.initialization_commands : prepared.commands).push_back(command);
            ggml::hrx::RecordedCommandGraph recorded;
            const auto replay = ggml::hrx::bind_and_launch_recorded_command_graph({}, {}, {}, prepared, recorded);
            REQUIRE(!replay.success && replay.event == ggml::hrx::HrxGraphReplayEvent::Ineligible);
            REQUIRE(replay.ineligible_reason == "q4_routed_gate_up_requires_index_validation");
        }
    }
    REQUIRE(!ggml::hrx::is_q4_routed_gate_up_kernel(
        ggml::hrx::kernel_catalog_id("hrx_owned", "ggml_mul_mat_id_iq3_xxs_f32")));
    REQUIRE(!ggml::hrx::is_q4_routed_gate_up_kernel(
        ggml::hrx::kernel_catalog_id("hrx_owned", "ggml_mul_mat_id_iq4_xs_f32")));
    float host_data[4] = {};
    ggml_tensor tensor = {};
    ggml::hrx::CommandProgramBinding binding;
    binding.value = ggml::hrx::ValueId(0);
    binding.host_data = host_data;
    binding.length = binding.capacity = sizeof(host_data);
    const auto without_metadata = ggml::hrx::CommandProgramBindings::from_bindings({ binding });
    binding.tensor = &tensor;
    const auto with_metadata = ggml::hrx::CommandProgramBindings::from_bindings({ binding });
    REQUIRE(with_metadata.valid() && without_metadata.valid());
    REQUIRE(with_metadata.find(binding.value)->tensor == &tensor);
    REQUIRE(ggml::hrx::command_program_bindings_hash(with_metadata).value ==
            ggml::hrx::command_program_bindings_hash(without_metadata).value);
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(with_metadata).value ==
            ggml::hrx::command_program_bindings_fingerprint(without_metadata).value);
    std::vector<ggml::hrx::HostStagingGroup> groups;
    REQUIRE(ggml::hrx::plan_host_staging_groups(
        { { 0, 0x1172AB9700ULL, 24576 }, { 2, 0x1172AB9700ULL, 24576 },
          { 8, 0x1172AB9700ULL, 24576 }, { 9, 0x1172AB9700ULL, 24576 } }, groups).success());
    REQUIRE(groups.size() == 1 && groups[0].length == 24576 && groups[0].slices.size() == 4);
    for (const auto & slice : groups[0].slices) {
        REQUIRE(slice.address == groups[0].base);
    }
    for (uintptr_t shift : { uintptr_t{0}, uintptr_t{0x10000} }) {
        REQUIRE(ggml::hrx::plan_host_staging_groups(
            { { 2, 0x1070 + shift, 0x40 }, { 0, 0x1010 + shift, 0x40 },
              { 3, 0x10B0 + shift, 0x20 }, { 1, 0x1040 + shift, 0x40 } }, groups).success());
        REQUIRE(groups.size() == 2);
        REQUIRE(groups[0].base == 0x1000 + shift && groups[0].length == 0xB0);
        REQUIRE(groups[0].slices.size() == 3 && groups[1].slices.size() == 1);
        REQUIRE(groups[0].slices[0].address - groups[0].base == 0x10);
        REQUIRE(groups[0].slices[1].address - groups[0].base == 0x40);
        REQUIRE(groups[0].slices[2].address - groups[0].base == 0x70);
    }
    REQUIRE(!ggml::hrx::plan_host_staging_groups({ { 0, 0, 4 } }, groups).success() && groups.empty());
    REQUIRE(!ggml::hrx::plan_host_staging_groups({ { 0, 0x1000, 0 } }, groups).success());
    REQUIRE(!ggml::hrx::plan_host_staging_groups(
        { { 0, std::numeric_limits<uintptr_t>::max() - 4, 8 } }, groups).success());
    REQUIRE(!ggml::hrx::plan_host_staging_groups({ { 0, 0x1000, 4 }, { 0, 0x2000, 4 } }, groups).success());
    REQUIRE(ggml::hrx::plan_host_staging_groups({}, groups).success() && groups.empty());
    std::fprintf(stderr, "Host staging alias planner: exact/partial/transitive/adjacent/translation/invalid CPU-only checks passed\n");
    std::fprintf(stderr, "Q4 routed ID validation: valid/duplicate/bounds/strides/padding/replay CPU-only checks passed\n");
}

static void check_q4_expert_plan(const Q4ExpertGraph & tensors, int64_t stride) {
    ScopedHrxEnvironment iq_disabled("HRX_ENABLE_IQ_EXPERTS", "0");
    auto imported = ggml::hrx::import_ggml_graph(*tensors.graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
    const auto & plan = scheduler.plan();
    REQUIRE(plan.valid() && plan.dispatches.size() == 4);
    REQUIRE(kernel_name_for_id(plan.dispatches[0].kernel.kernel_id) == "qwen3_moe:ggml_quantize_q8_1_x4_f32");
    const auto & gate_up = plan.dispatches[1];
    REQUIRE(kernel_name_for_id(gate_up.kernel.kernel_id) ==
            "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8");
    REQUIRE(gate_up.kernel.integer_parameters.at("token_count") == 1);
    REQUIRE(gate_up.kernel.integer_parameters.at("route_count") == 10);
    REQUIRE(gate_up.kernel.integer_parameters.at("route_stride") == stride);
    REQUIRE(gate_up.kernel.integer_parameters.at("expert_count") == 512);
    REQUIRE(gate_up.kernel.integer_parameters.at("output_size") == 640);
    REQUIRE(gate_up.kernel.compile_parameters.at("qwen3_moe.routed_gate_up.input_size") == "2560");
    REQUIRE(gate_up.kernel.compile_parameters.at("qwen3_moe.routed_gate_up.expert_count") == "512");
    REQUIRE(gate_up.kernel.compile_parameters.at("qwen3_moe.routed_gate_up.route_count") == "10");
    REQUIRE(gate_up.kernel.compile_parameters.at("qwen3_moe.routed_gate_up.output_size") == "640");
    REQUIRE(gate_up.bindings.size() == 7);
    REQUIRE(gate_up.bindings[1].length == 10 * sizeof(int32_t));
    REQUIRE(gate_up.bindings[2].length == ggml_nbytes(tensors.gate_weight));
    REQUIRE(gate_up.bindings[3].length == ggml_nbytes(tensors.up_weight));
    REQUIRE(gate_up.bindings[4].value == imported.graph.values().find_tensor(tensors.swiglu)->id);
    REQUIRE(gate_up.bindings[4].length == 640 * 10 * sizeof(float));
    // Qwen30B needs 8*6=48 counters; qwen4exp needs 10*5=50, including route 9's last two groups.
    REQUIRE(plan.completion_counter_requests.size() == 1);
    REQUIRE(plan.completion_counter_requests[0].count == 50);
    REQUIRE(gate_up.bindings[5].value == plan.completion_counter_requests[0].value);
    REQUIRE(gate_up.bindings[5].length == 50 * sizeof(int32_t));
    REQUIRE(gate_up.bindings[6].length == 50 * 144);
    REQUIRE(kernel_name_for_id(plan.dispatches[2].kernel.kernel_id) == "qwen3_moe:ggml_zero_f32");
    const auto & down = plan.dispatches[3];
    REQUIRE(kernel_name_for_id(down.kernel.kernel_id) == "qwen3_moe:qwen3_moe_routed_down_q8_0_q8_1_x4");
    REQUIRE(down.bindings[0].value == gate_up.bindings[6].value);
    REQUIRE(down.bindings[0].length == gate_up.bindings[6].length);
    REQUIRE(down.bindings[1].length == 10 * sizeof(int32_t));
    REQUIRE(down.bindings[4].value == imported.graph.values().find_tensor(tensors.output)->id);
}

static void run_q4_expert_scheduling_checks() {
    for (int64_t stride : { 10, 64, 512 }) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        check_q4_expert_plan(build_q4_expert_graph(ctx, stride), stride);
        ggml_free(ctx);
    }
}

static std::vector<float> q4_expert_quantized_input(const std::vector<float> & input) {
    REQUIRE(input.size() % 32 == 0);
    std::vector<float> result(input.size());
    for (size_t base = 0; base < input.size(); base += 32) {
        float maximum = 0.0f;
        for (size_t j = 0; j < 32; ++j) {
            REQUIRE(std::isfinite(input[base + j]));
            maximum = std::max(maximum, std::fabs(input[base + j]));
        }
        const float d = maximum / 127.0f;
        const float inverse = d == 0.0f ? 0.0f : 1.0f / d;
        const float stored_d = dense_round_f16(d);
        for (size_t j = 0; j < 32; ++j) {
            result[base + j] = std::round(input[base + j] * inverse) * stored_d;
        }
    }
    return result;
}

static void require_q4_expert_close(const char * stage, const std::vector<float> & actual,
                                    const std::vector<float> & expected, float atol, float rtol) {
    REQUIRE(actual.size() == expected.size());
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(std::isfinite(actual[i]) && std::isfinite(expected[i]));
        max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    }
    std::fprintf(stderr, "Q4 experts %s: elements=%zu max_abs=%g\n", stage, actual.size(), max_error);
    require_close(actual, expected, atol, rtol);
}

static void run_q4_expert_reference_case(int64_t stride) {
    ScopedHrxEnvironment iq("HRX_ENABLE_IQ_EXPERTS", "1");
    auto * cpu = init_cpu_backend();
    auto * hrx = ggml_backend_hrx_init(0);
    REQUIRE(hrx != nullptr);
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    auto * cpu_ctx = ggml_init(params);
    auto * hrx_ctx = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
    const auto reference = build_q4_expert_graph(cpu_ctx, stride);
    const auto tensors = build_q4_expert_graph(hrx_ctx, stride);
    check_q4_expert_plan(tensors, stride);
    require_kernel_subsequence(scheduled_kernel_sequence(tensors.standalone_graph),
        { "qwen3_moe:ggml_quantize_q8_1_x4_f32", "qwen3_moe:ggml_zero_f32",
          "qwen3_moe:qwen3_moe_routed_down_q8_0_q8_1_x4" });
    auto * cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
    auto * hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
    REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
    ggml_backend_buffer_clear(cpu_buffer, 0);
    ggml_backend_buffer_clear(hrx_buffer, 0);

    struct Expert {
        int32_t id;
        std::vector<uint8_t> gate, up, down;
    };
    std::vector<int32_t> ids = { 511, 0, 128, 3, 127, 1, 511, 256, 255, 7 };
    std::vector<Expert> experts;
    for (int32_t id : ids) {
        if (std::find_if(experts.begin(), experts.end(), [id](const Expert & e) { return e.id == id; }) != experts.end()) {
            continue;
        }
        Expert e = { id, make_quantized_rows(GGML_TYPE_Q4_K, 2560, 640, id + 23),
                        make_quantized_rows(GGML_TYPE_Q4_K, 2560, 640, id + 97),
                        make_quantized_rows(GGML_TYPE_Q8_0, 640, 2560, id + 151) };
        ggml_tensor * cpu_weights[] = { reference.gate_weight, reference.up_weight, reference.down_weight };
        ggml_tensor * hrx_weights[] = { tensors.gate_weight, tensors.up_weight, tensors.down_weight };
        const std::vector<uint8_t> * data[] = { &e.gate, &e.up, &e.down };
        for (size_t i = 0; i < 3; ++i) {
            const size_t offset = static_cast<size_t>(id) * data[i]->size();
            ggml_backend_tensor_set(cpu_weights[i], data[i]->data(), offset, data[i]->size());
            ggml_backend_tensor_set(hrx_weights[i], data[i]->data(), offset, data[i]->size());
        }
        experts.push_back(std::move(e));
    }
    ggml_backend_synchronize(cpu);
    ggml_backend_synchronize(hrx);
    const auto * q4 = ggml_get_type_traits(GGML_TYPE_Q4_K);
    const auto * q8 = ggml_get_type_traits(GGML_TYPE_Q8_0);
    for (int replay = 0; replay < 4; ++replay) {
        if (replay != 0) {
            std::rotate(ids.begin(), ids.begin() + 3, ids.end());
        }
        std::vector<float> input(2560);
        for (size_t base = 0; base < input.size(); base += 32) {
            for (size_t pair = 0; pair < 16; ++pair) {
                const int code = pair == 0 ? 127 : static_cast<int>((base * 7 + pair * 31 + replay * 19) % 127);
                input[base + pair * 2] = static_cast<float>(code) / 128.0f;
                input[base + pair * 2 + 1] = -input[base + pair * 2];
            }
            if (replay == 1 || replay == 2) {
                input[base + 30] = 31.0f / 128.0f;
                input[base + 31] = 32.0f / 128.0f;
            }
        }
        if (replay == 3) {
            std::fill(input.begin(), input.end(), 0.0f);
        }
        // Shared per-32/per-256 amax and binary-exact codes/scales eliminate Q8_K
        // versus Q8_1 activation noise. Nonzero block sums also test Q4's minimum correction.
        require_close(q4_expert_quantized_input(input), input, 0.0f);
        std::vector<float> expected_swiglu(640 * 10), decoded(2560);
        for (size_t route = 0; route < ids.size(); ++route) {
            const auto & e = *std::find_if(experts.begin(), experts.end(),
                [&](const Expert & candidate) { return candidate.id == ids[route]; });
            for (size_t row = 0; row < 640; ++row) {
                double gate = 0.0, up = 0.0;
                q4->to_float(e.gate.data() + row * ggml_row_size(GGML_TYPE_Q4_K, 2560), decoded.data(), 2560);
                for (size_t k = 0; k < 2560; ++k) { gate += static_cast<double>(decoded[k]) * input[k]; }
                q4->to_float(e.up.data() + row * ggml_row_size(GGML_TYPE_Q4_K, 2560), decoded.data(), 2560);
                for (size_t k = 0; k < 2560; ++k) { up += static_cast<double>(decoded[k]) * input[k]; }
                const float g = static_cast<float>(gate);
                expected_swiglu[route * 640 + row] = g / (1.0f + std::exp(-g)) * static_cast<float>(up);
            }
            if (replay != 3) {
                float tail_peak = 0.0f;
                for (size_t row = 384; row < 640; ++row) {
                    tail_peak = std::max(tail_peak, std::fabs(expected_swiglu[route * 640 + row]));
                }
                REQUIRE(tail_peak > 0.01f);
            }
        }
        const int codes[] = { 31, 1, 7, 13, 5, 19, 3, 11, 17, 21 };
        std::vector<float> route_weights(10);
        for (size_t route = 0; route < 10; ++route) {
            route_weights[route] = replay == 2 ? (route == 9 ? 1.0f : 0.0f) :
                static_cast<float>(codes[(route + replay) % 10]) / 128.0f;
        }
        set_moe_ids(cpu, reference.ids, ids);
        set_moe_ids(hrx, tensors.ids, ids);
        set_tensor_pair_bytes(cpu, reference.input, hrx, tensors.input, input.data(), input.size() * sizeof(float));
        set_tensor_pair_bytes(cpu, reference.route_weights, hrx, tensors.route_weights,
                              route_weights.data(), route_weights.size() * sizeof(float));
        const std::vector<float> dirty_swiglu(640 * 10, std::numeric_limits<float>::quiet_NaN());
        const std::vector<float> dirty_output(2560, std::numeric_limits<float>::quiet_NaN());
        set_tensor_bytes(hrx, tensors.swiglu, dirty_swiglu.data(), dirty_swiglu.size() * sizeof(float));
        set_tensor_bytes(hrx, tensors.output, dirty_output.data(), dirty_output.size() * sizeof(float));
        REQUIRE(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx, tensors.graph) == GGML_STATUS_SUCCESS);
        const auto swiglu = get_f32_tensor(hrx, tensors.swiglu);
        require_q4_expert_close("SwiGLU CPU/raw", get_f32_tensor(cpu, reference.swiglu), expected_swiglu, 5.0e-4f, 3.0e-5f);
        require_q4_expert_close("SwiGLU GPU/raw", swiglu, expected_swiglu, 5.0e-4f, 3.0e-5f);
        const auto packed = q4_expert_quantized_input(swiglu);
        std::vector<float> expected_output(2560), down_row(640);
        for (size_t row = 0; row < 2560; ++row) {
            double folded = 0.0;
            for (size_t route = 0; route < ids.size(); ++route) {
                const auto & e = *std::find_if(experts.begin(), experts.end(),
                    [&](const Expert & candidate) { return candidate.id == ids[route]; });
                q8->to_float(e.down.data() + row * ggml_row_size(GGML_TYPE_Q8_0, 640), down_row.data(), 640);
                double dot = 0.0;
                for (size_t k = 0; k < 640; ++k) { dot += static_cast<double>(down_row[k]) * packed[route * 640 + k]; }
                folded += dot * route_weights[route];
            }
            expected_output[row] = static_cast<float>(folded);
        }
        const auto actual = get_f32_tensor(hrx, tensors.output);
        require_q4_expert_close("fused next-Q8 down/fold raw", actual, expected_output, 5.0e-4f, 3.0e-5f);
        require_q4_expert_close("full chain CPU", actual, get_f32_tensor(cpu, reference.output), 3.0e-2f, 1.0e-2f);
        set_tensor_bytes(hrx, tensors.standalone_input, swiglu.data(), swiglu.size() * sizeof(float));
        set_tensor_bytes(hrx, tensors.standalone_output, dirty_output.data(), dirty_output.size() * sizeof(float));
        REQUIRE(ggml_backend_graph_compute(hrx, tensors.standalone_graph) == GGML_STATUS_SUCCESS);
        require_q4_expert_close("standalone pack/down/fold raw", get_f32_tensor(hrx, tensors.standalone_output),
                                expected_output, 5.0e-4f, 3.0e-5f);
        if (replay == 3) {
            require_close(actual, std::vector<float>(2560, 0.0f), 0.0f);
            require_close(swiglu, std::vector<float>(640 * 10, 0.0f), 0.0f);
        }
    }
    const std::vector<float> untouched_swiglu(640 * 10, 777.0f);
    const std::vector<float> untouched_output(2560, 888.0f);
    for (int32_t invalid : { -1, 512, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max() }) {
        auto bad = ids;
        bad.back() = invalid;
        set_moe_ids(hrx, tensors.ids, bad);
        set_tensor_bytes(hrx, tensors.swiglu, untouched_swiglu.data(), untouched_swiglu.size() * sizeof(float));
        set_tensor_bytes(hrx, tensors.output, untouched_output.data(), untouched_output.size() * sizeof(float));
        REQUIRE(ggml_backend_graph_compute(hrx, tensors.graph) != GGML_STATUS_SUCCESS);
        require_close(get_f32_tensor(hrx, tensors.swiglu), untouched_swiglu, 0.0f);
        require_close(get_f32_tensor(hrx, tensors.output), untouched_output, 0.0f);
        set_moe_ids(hrx, tensors.ids, ids);
        REQUIRE(ggml_backend_graph_compute(hrx, tensors.graph) == GGML_STATUS_SUCCESS);
        require_close(get_f32_tensor(hrx, tensors.output), std::vector<float>(2560, 0.0f), 0.0f);
    }
    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::fprintf(stderr, "Q4 experts 2560/640/512 routes=10 T=1 stride=%lld CPU/raw/packed-down replay checks passed\n",
                 (long long) stride);
}

static void run_q4_router_reuse_reference_case(bool host_inputs = false) {
    ScopedHrxEnvironment iq("HRX_ENABLE_IQ_EXPERTS", "1");
    ggml_backend_t backends[] = { init_cpu_backend(), ggml_backend_hrx_init(0) };
    REQUIRE(backends[0] != nullptr && backends[1] != nullptr);
    ggml_context * contexts[2] = {};
    ggml_context * weight_contexts[2] = {};
    ggml_context * input_contexts[2] = {};
    ggml_backend_buffer_t weight_buffers[2] = {};
    ggml_backend_buffer_t input_buffers[2] = {};
    ggml_gallocr_t allocators[2] = {};
    Q4ExpertGraph tensors[2] = {};
    for (int b = 0; b < 2; ++b) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        contexts[b] = ggml_init(params);
        weight_contexts[b] = ggml_init(params);
        REQUIRE(contexts[b] != nullptr && weight_contexts[b] != nullptr);
        if (host_inputs) {
            input_contexts[b] = ggml_init(params);
            REQUIRE(input_contexts[b] != nullptr);
        }
        tensors[b] = build_q4_expert_graph(contexts[b], 512, true, weight_contexts[b], input_contexts[b]);
        if (host_inputs) {
            input_buffers[b] = ggml_backend_alloc_ctx_tensors(input_contexts[b], backends[0]);
            REQUIRE(input_buffers[b] != nullptr);
        }
        weight_buffers[b] = ggml_backend_alloc_ctx_tensors(weight_contexts[b], backends[b]);
        REQUIRE(weight_buffers[b] != nullptr);
        ggml_backend_buffer_clear(weight_buffers[b], 0);
        allocators[b] = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backends[b]));
        REQUIRE(allocators[b] != nullptr && ggml_gallocr_alloc_graph(allocators[b], tensors[b].graph));
    }
    require_kernel_subsequence(scheduled_kernel_sequence(tensors[1].graph),
        { "qwen3_moe:qwen3_moe_router_top8_f32", "qwen3_moe:ggml_quantize_q8_1_x4_f32",
          "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8",
          "qwen3_moe:qwen3_moe_routed_down_q8_0_q8_1_x4" });
    const auto materializes = [](const ggml_tensor * t) {
        return t->op != GGML_OP_NONE && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE &&
               t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_PERMUTE;
    };
    size_t reused_pairs = 0;
    for (int i = 0; i < ggml_graph_n_nodes(tensors[1].graph); ++i) {
        const auto * a = ggml_graph_node(tensors[1].graph, i);
        if (!materializes(a)) { continue; }
        for (int j = i + 1; j < ggml_graph_n_nodes(tensors[1].graph); ++j) {
            const auto * b = ggml_graph_node(tensors[1].graph, j);
            if (!materializes(b) || a->buffer != b->buffer) { continue; }
            const uintptr_t pa = reinterpret_cast<uintptr_t>(a->data);
            const uintptr_t pb = reinterpret_cast<uintptr_t>(b->data);
            reused_pairs += pa <= pb ? pb - pa < ggml_nbytes(a) : pa - pb < ggml_nbytes(b);
        }
    }
    REQUIRE(reused_pairs > 0);
    std::fprintf(stderr, "Q4 router/gallocr: host_inputs=%d reused materialized pairs=%zu IDs=%p weights=%p SwiGLU=%p fold=%p\n",
                 static_cast<int>(host_inputs), reused_pairs, tensors[1].ids->data, tensors[1].route_weights->data,
                 tensors[1].swiglu->data, tensors[1].output->data);
    std::vector<int32_t> order = { 511, 0, 128, 3, 127, 1, 64, 256, 255, 7 };
    for (int32_t id : order) {
        const auto gate = make_quantized_rows(GGML_TYPE_Q4_K, 2560, 640, id + 23);
        const auto up = make_quantized_rows(GGML_TYPE_Q4_K, 2560, 640, id + 97);
        const auto down = make_quantized_rows(GGML_TYPE_Q8_0, 640, 2560, id + 151);
        for (int b = 0; b < 2; ++b) {
            ggml_backend_tensor_set(tensors[b].gate_weight, gate.data(), id * gate.size(), gate.size());
            ggml_backend_tensor_set(tensors[b].up_weight, up.data(), id * up.size(), up.size());
            ggml_backend_tensor_set(tensors[b].down_weight, down.data(), id * down.size(), down.size());
        }
    }
    for (int replay = 0; replay < 3; ++replay) {
        std::rotate(order.begin(), order.begin() + 3, order.end());
        std::vector<float> logits(512, -8.0f), bias(512, 0.03125f), input(2560);
        for (size_t rank = 0; rank < order.size(); ++rank) {
            logits[order[rank]] = 5.0f - static_cast<float>(rank) * 0.25f;
        }
        for (size_t base = 0; base < input.size(); base += 32) {
            for (size_t pair = 0; pair < 16; ++pair) {
                const int code = pair == 0 ? 127 : static_cast<int>((base * 7 + pair * 31 + replay * 19) % 127);
                input[base + pair * 2] = static_cast<float>(code) / 128.0f;
                input[base + pair * 2 + 1] = -input[base + pair * 2];
            }
        }
        for (int b = 0; b < 2; ++b) {
            // Poison before restoring graph inputs: the allocator may legitimately
            // reuse an input's bytes for a later result. Only the GPU router may
            // publish the valid IDs consumed by the subsequent guarded command.
            set_moe_ids(backends[b], tensors[b].ids, std::vector<int32_t>(10, -1));
            set_tensor_bytes(backends[b], tensors[b].logits_a, logits.data(), logits.size() * sizeof(float));
            if (tensors[b].logits_b != nullptr) {
                set_tensor_bytes(backends[b], tensors[b].logits_b, bias.data(), bias.size() * sizeof(float));
            }
            set_tensor_bytes(backends[b], tensors[b].input, input.data(), input.size() * sizeof(float));
        }
        int32_t before[10];
        ggml_backend_tensor_get(tensors[1].ids, before, 0, sizeof(before));
        REQUIRE(!ggml::hrx::validate_q4_routed_gate_up_id_bytes(before, sizeof(before), 1, 10, 512, 512).valid);
        REQUIRE(ggml_backend_graph_compute(backends[0], tensors[0].graph) == GGML_STATUS_SUCCESS);
        if (host_inputs) {
            ggml::hrx::set_mtp_host_trace_scope("host-staging-fixture", replay);
        }
        REQUIRE(ggml_backend_graph_compute(backends[1], tensors[1].graph) == GGML_STATUS_SUCCESS);
        if (host_inputs) {
            ggml::hrx::set_mtp_host_trace_scope(nullptr, 0);
        }
        const auto expected = get_f32_tensor(backends[0], tensors[0].output);
        const auto actual = get_f32_tensor(backends[1], tensors[1].output);
        REQUIRE(*std::max_element(expected.begin(), expected.end()) > 0.01f ||
                *std::min_element(expected.begin(), expected.end()) < -0.01f);
        require_q4_expert_close("GPU router -> gate/up -> fold with gallocr reuse", actual, expected, 3.0e-2f, 1.0e-2f);
    }
    for (int b = 0; b < 2; ++b) {
        ggml_gallocr_free(allocators[b]);
        ggml_backend_buffer_free(weight_buffers[b]);
        ggml_backend_buffer_free(input_buffers[b]);
        ggml_free(contexts[b]);
        ggml_free(weight_contexts[b]);
        ggml_free(input_contexts[b]);
    }
    for (auto * backend : backends) {
        ggml_backend_free(backend);
    }
}

static void run_iq_down_pack_scheduling_checks(bool small_batch = false) {
    ScopedHrxEnvironment small_environment("HRX_ENABLE_MOE_SMALL_BATCH", small_batch ? "1" : nullptr);
    for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL }) {
        for (const char * flag : { "0", "1" }) {
          for (int64_t tokens : { 1, 2, 3, 4, 8, 9, 512 }) {
            ScopedHrxEnvironment environment("HRX_ENABLE_IQ_EXPERTS", flag);
            ggml_init_params params = {};
            params.mem_size         = 1024 * 1024;
            params.no_alloc         = true;
            ggml_context * ctx      = ggml_init(params);
            REQUIRE(ctx != nullptr);
            const auto tensors = build_decode_down_graph(ctx, type, tokens, 512);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, tensors.output);
            ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
            REQUIRE(imported.valid());
            ggml::hrx::DispatchScheduler scheduler;
            const bool enabled = std::strcmp(flag, "1") == 0 &&
                                 (tokens == 1 || (small_batch && tokens <= 8 && type != GGML_TYPE_Q5_1));
            REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == enabled);
            if (enabled) {
                const auto & plan = scheduler.plan();
                REQUIRE(plan.valid());
                REQUIRE(plan.dispatches.size() == 3);
                const auto & pack = plan.dispatches[0];
                const auto & zero = plan.dispatches[1];
                const auto & projection = plan.dispatches[2];
                REQUIRE(kernel_name_for_id(pack.kernel.kernel_id) == "qwen3_moe:ggml_quantize_q8_1_x4_f32");
                REQUIRE(kernel_name_for_id(zero.kernel.kernel_id) == "qwen3_moe:ggml_zero_f32");
                REQUIRE(kernel_name_for_id(projection.kernel.kernel_id) == (tokens > 1 ?
                        (type == GGML_TYPE_Q8_0 ? "hrx_owned:ggml_moe_small_down_q8_0" :
                                                  "hrx_owned:ggml_moe_small_down_iq4_nl") :
                        (type == GGML_TYPE_Q8_0 ? "qwen3_moe:qwen3_moe_routed_down_q8_0_q8_1_x4" :
                         type == GGML_TYPE_Q5_1 ? "qwen3_moe:qwen3_moe_routed_down_q5_1_q8_1_x4" :
                                                  "qwen3_moe:qwen3_moe_routed_down_iq4_nl_q8_1_x4")));
                REQUIRE(pack.kernel.integer_parameters.at("input_size") == 640);
                REQUIRE(pack.kernel.integer_parameters.at("token_count") == 10 * tokens);
                REQUIRE(pack.bindings[0].length == ggml_nbytes(tensors.input));
                REQUIRE(pack.bindings[1].length == 10 * tokens * 5 * 144);
                REQUIRE(projection.bindings[1].length == ((tokens - 1) * 512 + 10) * sizeof(int32_t));
                REQUIRE(pack.bindings[1].value == projection.bindings[0].value);
                REQUIRE(zero.bindings[0].value == projection.bindings[4].value);
                REQUIRE(pack.bindings[1].value != zero.bindings[0].value);
            }
            ggml_free(ctx);
          }
        }
    }
}

static void run_iq_down_pack_reference_case(ggml_type type, int64_t tokens = 1, int64_t stride = 10,
                                          bool with_residual = false, bool late_residual = false) {
    ScopedHrxEnvironment environment("HRX_ENABLE_IQ_EXPERTS", "1");
    ScopedHrxEnvironment small_environment("HRX_ENABLE_MOE_SMALL_BATCH", tokens > 1 ? "1" : nullptr);
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    auto tensors = build_decode_down_graph(ctx, type, tokens, stride);
    auto * graph = ggml_new_graph(ctx);
    ggml_tensor * residual_a = nullptr;
    ggml_tensor * residual_b = nullptr;
    ggml_tensor * residual = nullptr;
    if (with_residual) {
        residual_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, tokens);
        residual_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, tokens);
        residual = ggml_add(ctx, residual_a, residual_b);
        ggml_set_output(residual);
        if (!late_residual) {
            ggml_build_forward_expand(graph, residual);
        }
        tensors.output = ggml_add(ctx, tensors.output, residual);
    }
    ggml_build_forward_expand(graph, tensors.output);
    std::vector<std::string> expected_kernels =
                               { "qwen3_moe:ggml_quantize_q8_1_x4_f32",
                                 with_residual && !late_residual ? "qwen3_moe:ggml_copy_f32" : "qwen3_moe:ggml_zero_f32",
                                 tokens > 1 ?
                                 (type == GGML_TYPE_Q8_0 ? "hrx_owned:ggml_moe_small_down_q8_0" :
                                                          "hrx_owned:ggml_moe_small_down_iq4_nl") :
                                 (type == GGML_TYPE_Q8_0 ? "qwen3_moe:qwen3_moe_routed_down_q8_0_q8_1_x4" :
                                                          "qwen3_moe:qwen3_moe_routed_down_iq4_nl_q8_1_x4") };
    if (with_residual && !late_residual) {
        expected_kernels.push_back("qwen3_moe:ggml_copy_f32");
    }
    if (with_residual && late_residual) {
        expected_kernels.push_back("qwen3_moe:ggml_add_f32");
        expected_kernels.push_back("qwen3_moe:ggml_add_f32");
    }
    require_kernel_subsequence(scheduled_kernel_sequence(graph), expected_kernels);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    std::vector<int32_t> ids = make_moe_ids(tokens);
    const auto * traits = ggml_get_type_traits(type);
    REQUIRE(traits != nullptr && traits->to_float != nullptr);
    const size_t row_bytes = ggml_row_size(type, 640);
    std::vector<uint8_t> weight(ggml_nbytes(tensors.weight), 0);
    for (int32_t id : make_moe_ids(1)) {
        const auto expert = make_quantized_rows(type, 640, 2560, id);
        std::memcpy(weight.data() + id * 2560 * row_bytes, expert.data(), expert.size());
    }
    set_tensor_bytes(backend, tensors.weight, weight.data(), weight.size());
    for (int replay = 0; replay < 2; ++replay) {
    if (replay != 0) {
        std::rotate(ids.begin(), ids.begin() + 3, ids.end());
    }
    const auto input = make_pattern_f32(640 * 10 * tokens, 31 + replay * 6, 0.013f);
    std::vector<float> packed_input(input.size());
    for (size_t base = 0; base < input.size(); base += 32) {
        float maximum = 0.0f;
        for (size_t j = 0; j < 32; ++j) {
            maximum = std::max(maximum, std::fabs(input[base + j]));
        }
        const float scale = maximum / 127.0f;
        const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
        const float stored_scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
        for (size_t j = 0; j < 32; ++j) {
            packed_input[base + j] = std::round(input[base + j] * inverse) * stored_scale;
        }
    }
    std::vector<double> dots(2560 * 10 * tokens);
    std::vector<float> decoded(640);
    for (size_t route = 0; route < ids.size(); ++route) {
        const auto * expert = weight.data() + ids[route] * 2560 * row_bytes;
        for (size_t row = 0; row < 2560; ++row) {
            traits->to_float(expert + row * row_bytes, decoded.data(), 640);
            double sum = 0.0;
            for (size_t column = 0; column < 640; ++column) {
                sum += static_cast<double>(decoded[column]) * packed_input[route * 640 + column];
            }
            dots[route * 2560 + row] = sum;
        }
    }
    set_moe_ids(backend, tensors.ids, ids);
    set_tensor_bytes(backend, tensors.input, input.data(), input.size() * sizeof(float));
        std::vector<float> route_weights(10 * tokens);
        for (size_t route = 0; route < route_weights.size(); ++route) {
            route_weights[route] = static_cast<float>((route * 3 + replay * 7) % 13) / 64.0f;
        }
        std::vector<float> expected(2560 * tokens);
        std::vector<float> expected_residual(expected.size(), 0.0f);
        if (with_residual) {
            const auto a = make_pattern_f32(expected.size(), 13 + replay * 5, 0.021f);
            const auto b = make_pattern_f32(expected.size(), 17 + replay * 9, 0.037f);
            for (size_t i = 0; i < expected.size(); ++i) {
                expected_residual[i] = a[i] + b[i];
            }
            set_tensor_bytes(backend, residual_a, a.data(), a.size() * sizeof(float));
            set_tensor_bytes(backend, residual_b, b.data(), b.size() * sizeof(float));
        }
        for (size_t row = 0; row < expected.size(); ++row) {
            double sum = 0.0;
            for (size_t route = 0; route < 10; ++route) {
                const size_t assignment = row / 2560 * 10 + route;
                sum += dots[assignment * 2560 + row % 2560] * route_weights[assignment];
            }
            expected[row] = static_cast<float>(sum) + expected_residual[row];
        }
        set_tensor_bytes(backend, tensors.route_weights, route_weights.data(), route_weights.size() * sizeof(float));
        const std::vector<float> dirty(2560 * tokens, 123.0f);
        set_tensor_bytes(backend, tensors.output, dirty.data(), dirty.size() * sizeof(float));
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        const auto actual = get_f32_tensor(backend, tensors.output);
        for (float value : actual) {
            REQUIRE(std::isfinite(value));
        }
        require_close(actual, expected, 3.0e-4f, 1.0e-5f);
        if (with_residual) {
            require_close(get_f32_tensor(backend, residual), expected_residual, 0.0f);
        }
        if (tokens > 1 && replay == 1) {
            auto invalid_ids = ids;
            invalid_ids[0] = -1;
            invalid_ids.back() = 512;
            set_moe_ids(backend, tensors.ids, invalid_ids);
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            if (with_residual) {
                require_close(get_f32_tensor(backend, residual), expected_residual, 0.0f);
            }
            const auto invalid_output = get_f32_tensor(backend, tensors.output);
            for (size_t i = 0; i < invalid_output.size(); ++i) {
                if (i / 2560 == 0 || i / 2560 == static_cast<size_t>(tokens - 1)) {
                    REQUIRE(std::isnan(invalid_output[i]));
                } else {
                    REQUIRE(std::isfinite(invalid_output[i]));
                    REQUIRE(std::fabs(invalid_output[i] - expected[i]) <=
                            1.0e-5f + 3.0e-4f * std::fabs(expected[i]));
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::fprintf(stderr, "%s standalone down T=%lld stride=%lld residual=%d late=%d pack/input+IDs replay checks passed\n",
                 ggml_type_name(type), (long long) tokens, (long long) stride, with_residual, late_residual);
}

static void run_moe_small_batch_negative_checks() {
    ScopedHrxEnvironment iq("HRX_ENABLE_IQ_EXPERTS", "1");
    for (const char * flag : { "0", "true", "1" }) {
        ScopedHrxEnvironment small("HRX_ENABLE_MOE_SMALL_BATCH", flag);
        for (int variant = 0; variant < 10; ++variant) {
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            auto * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            auto tensors = build_decode_down_graph(ctx, GGML_TYPE_Q8_0, 3, 512);
            auto * graph = ggml_new_graph(ctx);
            ggml_tensor * residual = nullptr;
            if (variant == 1) {
                tensors.ids->nb[1] = 39; // misaligned and shorter than the active route row
            } else if (variant == 2) {
                tensors.input->nb[1] += 4;
            } else if (variant == 3) {
                tensors.route_weights->nb[1] += 4;
            } else if (variant == 4 || variant == 7) {
                // Force the route-weight producer after MUL_MAT_ID in the graph.
                tensors.route_weights->op = GGML_OP_ADD;
                tensors.route_weights->src[0] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 10, 3);
                tensors.route_weights->src[1] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 10, 3);
                if (variant == 7) {
                    ggml_build_forward_expand(graph, tensors.route_weights);
                }
            } else if (variant == 5) {
                tensors.ids->nb[1] = 513 * sizeof(int32_t);
            } else if (variant == 6) {
                // Duplicate one route view: same count/shape, incorrect fold semantics.
                auto * final_view = tensors.output->src[1];
                final_view->view_offs = 0;
                final_view->op_params[0] = 0;
            } else if (variant == 8 || variant == 9) {
                auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 3);
                auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 3);
                residual = ggml_add(ctx, a, b);
                if (variant == 9) {
                    ggml_set_output(residual);
                    ggml_build_forward_expand(graph, residual);
                }
                tensors.output = ggml_add(ctx, tensors.output, residual);
            }
            ggml_build_forward_expand(graph, tensors.output);
            auto imported = ggml::hrx::import_ggml_graph(*graph);
            const auto * output_value = imported.graph.values().find_tensor(tensors.output);
            const auto * residual_value = residual == nullptr ? nullptr : imported.graph.values().find_tensor(residual);
            const auto output_storage = output_value == nullptr ? ggml::hrx::ValueStorageId() : output_value->storage;
            const auto residual_storage = residual_value == nullptr ? ggml::hrx::ValueStorageId() : residual_value->storage;
            const bool expected = (variant == 0 || variant == 7 || variant == 9) &&
                                  std::strcmp(flag, "1") == 0;
            ggml::hrx::DispatchScheduler scheduler;
            const bool scheduled = imported.valid() && scheduler.schedule_graph(imported.graph, { "gfx1151" });
            if (scheduled != expected) {
                std::fprintf(stderr, "small MoE scheduling flag=%s variant=%d scheduled=%d expected=%d\n",
                             flag, variant, scheduled, expected);
                for (const auto & error : scheduler.plan().status.errors()) {
                    std::fprintf(stderr, "%s\n", error.c_str());
                }
            }
            REQUIRE(scheduled == expected);
            if (scheduled && variant == 9) {
                REQUIRE(output_value != nullptr && residual_value != nullptr);
                REQUIRE(output_value->kind == ggml::hrx::ValueKind::External);
                REQUIRE(residual_value->kind == ggml::hrx::ValueKind::External);
                REQUIRE(output_value->storage == output_storage);
                REQUIRE(residual_value->storage == residual_storage);
                REQUIRE(output_storage != residual_storage);
                const auto & plan = scheduler.plan();
                REQUIRE(plan.valid());
                REQUIRE(plan.dispatches.size() >= 4);
                const size_t n = plan.dispatches.size();
                const auto & copy_in = plan.dispatches[n - 3];
                const auto & projection = plan.dispatches[n - 2];
                const auto & copy_out = plan.dispatches[n - 1];
                REQUIRE(kernel_name_for_id(copy_in.kernel.kernel_id) == "qwen3_moe:ggml_copy_f32");
                REQUIRE(kernel_name_for_id(projection.kernel.kernel_id) == "hrx_owned:ggml_moe_small_down_q8_0");
                REQUIRE(kernel_name_for_id(copy_out.kernel.kernel_id) == "qwen3_moe:ggml_copy_f32");
                REQUIRE(copy_in.bindings[0].value == residual_value->id);
                REQUIRE(copy_out.bindings[1].value == output_value->id);
                REQUIRE(copy_in.bindings[1].value == projection.bindings[4].value);
                REQUIRE(copy_out.bindings[0].value == projection.bindings[4].value);
                REQUIRE(projection.bindings[4].value != projection.bindings[0].value);
                REQUIRE(projection.bindings[4].value != residual_value->id);
                REQUIRE(projection.bindings[4].value != output_value->id);
            }
            ggml_free(ctx);
        }
    }
}

static void run_moe_small_batch_capability_checks() {
    ScopedHrxEnvironment iq("HRX_ENABLE_IQ_EXPERTS", "1");
    ScopedHrxEnvironment small("HRX_ENABLE_MOE_SMALL_BATCH", "1");
    auto * backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    for (ggml_type type : { GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL }) {
        for (int64_t tokens : { 1, 2, 3, 4, 8, 9, 512 }) {
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            auto * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            const bool down = type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL;
            auto * weight = ggml_new_tensor_3d(ctx, type, down ? 640 : 2560, down ? 2560 : 640, 512);
            auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, down ? 640 : 2560, down ? 10 : 1, tokens);
            auto * ids = make_moe_ids_tensor(ctx, tokens, 512);
            auto * output = ggml_mul_mat_id(ctx, weight, input, ids);
            REQUIRE(ggml_backend_supports_op(backend, output) == (tokens <= 8));
            if (tokens == 512) {
                // Reproduce loader placement, which is NOT an actual executable T=512 graph.
                auto * probe_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, weight->ne[0], 10, 512);
                auto * probe_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 10, 512);
                auto * probe = ggml_mul_mat_id(ctx, weight, probe_input, probe_ids);
                weight->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), 0);
                REQUIRE(weight->buffer != nullptr);
                REQUIRE(ggml_backend_supports_op(backend, probe));
                REQUIRE(ggml_backend_supports_op(backend, weight));
                ggml_backend_buffer_free(weight->buffer);
                weight->buffer = nullptr;
                REQUIRE(!ggml_backend_supports_op(backend, probe));
            }
            if (tokens == 3) {
                for (int64_t invalid : { 0, -1, 9 }) {
                    input->ne[2] = invalid;
                    REQUIRE(!ggml_backend_supports_op(backend, output));
                }
                input->ne[2] = tokens;
                ids->nb[1] = 39;
                REQUIRE(!ggml_backend_supports_op(backend, output));
            }
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
}

static void run_moe_small_batch_reference_checks() {
    run_moe_small_batch_capability_checks();
    for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
        for (int64_t stride : { 10, 512 }) {
            run_iq_expert_raw_reference_case(GGML_TYPE_IQ3_XXS, true, tokens, stride);
            run_iq_expert_raw_reference_case(GGML_TYPE_IQ4_XS, true, tokens, stride);
            run_iq_down_pack_reference_case(GGML_TYPE_Q8_0, tokens, stride);
            run_iq_down_pack_reference_case(GGML_TYPE_IQ4_NL, tokens, stride);
        }
    }
    for (int64_t tokens : { 1, 3, 8 }) {
        run_iq_down_pack_reference_case(GGML_TYPE_Q8_0, tokens, 512, true);
        run_iq_down_pack_reference_case(GGML_TYPE_IQ4_NL, tokens, 512, true);
    }
}

static void run_endpoint_rmsnorm_q6k_q8_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 32 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t token_count = 1;
    ggml_tensor *     cpu_input   = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    ggml_tensor *     cpu_norm_w  = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, kQwenHiddenSize);
    ggml_tensor *     cpu_weight  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_Q6_K, kQwenHiddenSize, kQwenVocabularyCount);
    ggml_tensor *     cpu_norm    = build_rmsnorm_mul_graph(cpu_ctx, cpu_input, cpu_norm_w);
    ggml_tensor *     cpu_output  = ggml_mul_mat(cpu_ctx, cpu_weight, cpu_norm);

    ggml_tensor * hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    ggml_tensor * hrx_norm_w = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, kQwenHiddenSize);
    ggml_tensor * hrx_weight = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_Q6_K, kQwenHiddenSize, kQwenVocabularyCount);
    ggml_tensor * hrx_norm   = build_rmsnorm_mul_graph(hrx_ctx, hrx_input, hrx_norm_w);
    ggml_tensor * hrx_output = ggml_mul_mat(hrx_ctx, hrx_weight, hrx_norm);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(
        scheduled_kernel_sequence(hrx_graph),
        { "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4", "qwen3_moe:ggml_linear_q6k_q8_1_x4" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float>   input  = make_pattern_f32(kQwenHiddenSize * token_count, 8, 0.01f);
    const std::vector<float>   norm_w = make_weight(kQwenHiddenSize);
    const std::vector<uint8_t> weight = make_quantized_rows(GGML_TYPE_Q6_K, kQwenHiddenSize, kQwenVocabularyCount, 9);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_norm_w, hrx_backend, hrx_norm_w, norm_w.data(),
                          norm_w.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 6.0e-1f, 3.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_attention_postprocess_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 32 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t         token_count          = 2;
    constexpr int64_t         query_head_count     = 4;
    constexpr int64_t         key_value_head_count = 2;
    constexpr int64_t         cache_row_count      = 8;
    const int64_t             query_size           = query_head_count * kQwenFlashHeadSize;
    const int64_t             key_value_size       = key_value_head_count * kQwenFlashHeadSize;
    AttentionPostprocessGraph cpu = build_attention_postprocess_graph(cpu_ctx, token_count, query_head_count,
                                                                      key_value_head_count, cache_row_count);
    AttentionPostprocessGraph hrx = build_attention_postprocess_graph(hrx_ctx, token_count, query_head_count,
                                                                      key_value_head_count, cache_row_count);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    // The isolated postprocess fusion needs all three projections live at the query reshape.
    ggml_build_forward_expand(cpu_graph, cpu.query_raw);
    ggml_build_forward_expand(cpu_graph, cpu.key_raw);
    ggml_build_forward_expand(cpu_graph, cpu.value_raw);
    ggml_build_forward_expand(hrx_graph, hrx.query_raw);
    ggml_build_forward_expand(hrx_graph, hrx.key_raw);
    ggml_build_forward_expand(hrx_graph, hrx.value_raw);
    ggml_build_forward_expand(cpu_graph, cpu.query_output);
    ggml_build_forward_expand(cpu_graph, cpu.key_output);
    ggml_build_forward_expand(cpu_graph, cpu.value_output);
    ggml_build_forward_expand(hrx_graph, hrx.query_output);
    ggml_build_forward_expand(hrx_graph, hrx.key_output);
    ggml_build_forward_expand(hrx_graph, hrx.value_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph),
                               { "qwen3_moe:qwen3_moe_attention_postprocess_f32_f16" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float>   input     = make_pattern_f32(kQwenHiddenSize * token_count, 10, 0.01f);
    const std::vector<uint8_t> query_w   = make_quantized_rows(GGML_TYPE_Q4_K, kQwenHiddenSize, query_size, 11);
    const std::vector<uint8_t> key_w     = make_quantized_rows(GGML_TYPE_Q4_K, kQwenHiddenSize, key_value_size, 12);
    const std::vector<uint8_t> value_w   = make_quantized_rows(GGML_TYPE_Q6_K, kQwenHiddenSize, key_value_size, 13);
    const std::vector<float>   query_nw  = make_weight(kQwenFlashHeadSize);
    const std::vector<float>   key_nw    = make_pattern_f32(kQwenFlashHeadSize, 14, 0.02f);
    const std::vector<int32_t> positions = make_i32_mod_data(token_count, 1024);
    std::vector<float>         inv_freq(static_cast<size_t>(kQwenFlashHeadSize / 2));
    for (size_t i = 0; i < inv_freq.size(); ++i) {
        inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(kQwenFlashHeadSize));
    }
    const std::vector<ggml_fp16_t> key_cache(static_cast<size_t>(key_value_size * cache_row_count),
                                             ggml_fp32_to_fp16(0.0f));
    const std::vector<ggml_fp16_t> value_cache(static_cast<size_t>(key_value_size * cache_row_count),
                                               ggml_fp32_to_fp16(0.0f));
    const std::vector<int64_t>     cache_indices = make_i64_mod_data(token_count, cache_row_count);

    set_tensor_pair_bytes(cpu_backend, cpu.input, hrx_backend, hrx.input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.query_weight, hrx_backend, hrx.query_weight, query_w.data(), query_w.size());
    set_tensor_pair_bytes(cpu_backend, cpu.key_weight, hrx_backend, hrx.key_weight, key_w.data(), key_w.size());
    set_tensor_pair_bytes(cpu_backend, cpu.value_weight, hrx_backend, hrx.value_weight, value_w.data(), value_w.size());
    set_tensor_pair_bytes(cpu_backend, cpu.query_norm_weight, hrx_backend, hrx.query_norm_weight, query_nw.data(),
                          query_nw.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.key_norm_weight, hrx_backend, hrx.key_norm_weight, key_nw.data(),
                          key_nw.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.positions, hrx_backend, hrx.positions, positions.data(),
                          positions.size() * sizeof(int32_t));
    set_tensor_pair_bytes(cpu_backend, cpu.inverse_frequencies, hrx_backend, hrx.inverse_frequencies, inv_freq.data(),
                          inv_freq.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.key_cache, hrx_backend, hrx.key_cache, key_cache.data(),
                          key_cache.size() * sizeof(ggml_fp16_t));
    set_tensor_pair_bytes(cpu_backend, cpu.value_cache, hrx_backend, hrx.value_cache, value_cache.data(),
                          value_cache.size() * sizeof(ggml_fp16_t));
    set_tensor_pair_bytes(cpu_backend, cpu.key_cache_indices, hrx_backend, hrx.key_cache_indices, cache_indices.data(),
                          cache_indices.size() * sizeof(int64_t));
    set_tensor_pair_bytes(cpu_backend, cpu.value_cache_indices, hrx_backend, hrx.value_cache_indices,
                          cache_indices.data(), cache_indices.size() * sizeof(int64_t));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx.query_output), get_f32_tensor(cpu_backend, cpu.query_output), 2.0f,
                  5.0e-2f);
    require_close(get_f32_tensor(hrx_backend, hrx.key_output), get_f32_tensor(cpu_backend, cpu.key_output), 2.0f,
                  5.0e-2f);
    require_close(get_f32_tensor(hrx_backend, hrx.value_output), get_f32_tensor(cpu_backend, cpu.value_output), 2.0f,
                  5.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_routed_moe_cpu_reference_case(ggml_type down_weight_type, bool include_next_rmsnorm) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 64 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    RoutedMoeGraph cpu = build_routed_moe_graph(cpu_ctx, down_weight_type, include_next_rmsnorm);
    RoutedMoeGraph hrx = build_routed_moe_graph(hrx_ctx, down_weight_type, include_next_rmsnorm);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu.output);
    ggml_build_forward_expand(hrx_graph, hrx.output);

    std::vector<std::string> expected = {
        "qwen3_moe:qwen3_moe_router_top8_f32",
        "qwen3_moe:qwen3_moe_build_expert_table",
        "qwen3_moe:qwen3_moe_build_expert_partition_table",
        "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma",
        down_weight_type == GGML_TYPE_Q4_K ? "qwen3_moe:qwen3_moe_routed_down_q4k_f16_wmma_grouped" :
                                             "qwen3_moe:qwen3_moe_routed_down_q6k_f16_wmma_grouped",
        include_next_rmsnorm ? "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32" :
                               "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32",
    };
    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), expected);

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> logits = make_router_logits(1);
    const std::vector<float> input  = make_pattern_f32(kQwenHiddenSize, 15, 0.01f);
    const std::vector<float> hidden = make_pattern_f32(kQwenHiddenSize, 16, 0.02f);
    set_tensor_pair_bytes(cpu_backend, cpu.logits, hrx_backend, hrx.logits, logits.data(),
                          logits.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.input, hrx_backend, hrx.input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.hidden_state, hrx_backend, hrx.hidden_state, hidden.data(),
                          hidden.size() * sizeof(float));
    if (include_next_rmsnorm) {
        const std::vector<float> norm = make_weight(kQwenHiddenSize);
        set_tensor_pair_bytes(cpu_backend, cpu.norm_weight, hrx_backend, hrx.norm_weight, norm.data(),
                              norm.size() * sizeof(float));
    }

    {
        const std::vector<uint8_t> gate =
            make_quantized_rows(GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate * kQwenRouterExpertCount, 17);
        set_tensor_pair_bytes(cpu_backend, cpu.gate_weight, hrx_backend, hrx.gate_weight, gate.data(), gate.size());
    }
    {
        const std::vector<uint8_t> up =
            make_quantized_rows(GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate * kQwenRouterExpertCount, 18);
        set_tensor_pair_bytes(cpu_backend, cpu.up_weight, hrx_backend, hrx.up_weight, up.data(), up.size());
    }
    {
        const std::vector<uint8_t> down =
            make_quantized_rows(down_weight_type, kQwenMoeIntermediate, kQwenHiddenSize * kQwenRouterExpertCount, 19);
        set_tensor_pair_bytes(cpu_backend, cpu.down_weight, hrx_backend, hrx.down_weight, down.data(), down.size());
    }

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx.output), get_f32_tensor(cpu_backend, cpu.output), 2.5f, 1.0e-1f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_decode_routed_moe_scheduling_case(ggml_type down_weight_type, bool alias_gate_input = false) {
    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * hidden_state     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, 1);
    ggml_tensor * attention_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
    REQUIRE(hidden_state != nullptr);
    REQUIRE(attention_weight != nullptr);
    ggml_tensor * attention_rms = ggml_rms_norm(ctx, hidden_state, kQwenRmsNormEps);
    REQUIRE(attention_rms != nullptr);
    ggml_tensor * attention_prepared = ggml_mul(ctx, attention_rms, attention_weight);
    REQUIRE(attention_prepared != nullptr);
    ggml_tensor * moe_input = attention_prepared;
    if (alias_gate_input) {
        moe_input = ggml_reshape_2d(ctx, attention_prepared, kQwenHiddenSize, 1);
        REQUIRE(moe_input != nullptr);
    }

    ggml_tensor * router_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, kQwenRouterExpertCount);
    REQUIRE(router_weight != nullptr);
    ggml_tensor * logits = ggml_mul_mat(ctx, router_weight, attention_prepared);
    REQUIRE(logits != nullptr);
    ggml_tensor * route_ids     = nullptr;
    ggml_tensor * route_weights = build_qwen_router_top8_graph(ctx, logits, &route_ids);
    REQUIRE(route_ids != nullptr);
    REQUIRE(route_weights != nullptr);

    ggml_tensor * gate_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate, kQwenRouterExpertCount);
    ggml_tensor * up_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate, kQwenRouterExpertCount);
    ggml_tensor * down_weight =
        ggml_new_tensor_3d(ctx, down_weight_type, kQwenMoeIntermediate, kQwenHiddenSize, kQwenRouterExpertCount);
    REQUIRE(gate_weight != nullptr);
    REQUIRE(up_weight != nullptr);
    REQUIRE(down_weight != nullptr);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_weight, moe_input, route_ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, up_weight, moe_input, route_ids);
    REQUIRE(gate != nullptr);
    REQUIRE(up != nullptr);
    ggml_tensor * glu = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
    REQUIRE(glu != nullptr);
    ggml_tensor * down = ggml_mul_mat_id(ctx, down_weight, glu, route_ids);
    REQUIRE(down != nullptr);
    ggml_tensor * weighted = ggml_mul(ctx, down, route_weights);
    REQUIRE(weighted != nullptr);

    std::vector<ggml_tensor *> route_views;
    route_views.reserve(kQwenRouterRouteCount);
    for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
        ggml_tensor * view = ggml_view_2d(ctx, weighted, kQwenHiddenSize, 1, weighted->nb[2],
                                          static_cast<size_t>(route) * weighted->nb[1]);
        REQUIRE(view != nullptr);
        route_views.push_back(view);
    }

    ggml_tensor * reduced = route_views.front();
    for (size_t i = 1; i < route_views.size(); ++i) {
        reduced = ggml_add(ctx, reduced, route_views[i]);
        REQUIRE(reduced != nullptr);
    }
    ggml_tensor * residual = ggml_add(ctx, hidden_state, reduced);
    REQUIRE(residual != nullptr);
    ggml_tensor * next_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
    REQUIRE(next_norm_weight != nullptr);
    ggml_tensor * next_rms = ggml_rms_norm(ctx, residual, kQwenRmsNormEps);
    REQUIRE(next_rms != nullptr);
    ggml_tensor * output = ggml_mul(ctx, next_rms, next_norm_weight);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    std::vector<std::string> expected = {
        "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4",
        "qwen3_moe:qwen3_moe_router_projection_top8_fused_decode_f32",
        down_weight_type == GGML_TYPE_Q4_K ? "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8" :
                                             "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_q8",
        down_weight_type == GGML_TYPE_Q4_K ? "qwen3_moe:qwen3_moe_routed_down_q4k_q8_1_x4_next_q8" :
                                             "qwen3_moe:qwen3_moe_routed_down_q6k_f32_wave64_next_q8",
    };
    require_kernel_subsequence(scheduled_kernel_sequence(graph), expected);
    ggml_free(ctx);
}

static void run_decode_attention_qkv_scheduling_case() {
    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    AttentionPostprocessGraph graph  = build_decode_attention_qkv_graph(ctx);
    ggml_cgraph *             cgraph = ggml_new_graph(ctx);
    REQUIRE(cgraph != nullptr);
    ggml_build_forward_expand(cgraph, graph.query_output);
    ggml_build_forward_expand(cgraph, graph.key_output);
    ggml_build_forward_expand(cgraph, graph.value_output);

    require_kernel_subsequence(scheduled_kernel_sequence(cgraph),
                               { "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4",
                                 "qwen3_moe:qwen3_moe_attention_qkv_postprocess_fused_decode" });
    ggml_free(ctx);
}

static void run_dense_q8_gemv_reference_checks() {
    {
        ScopedHrxEnvironment environment("HRX_ENABLE_Q8_GEMV", "1");
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q8_0, "qwen3_moe:qwen3_moe_dense_linear_q8_0_f16_wmma", 2, 65);
    }
    for (int64_t width : { 256, 2560, 32768 }) {
        for (int64_t rows : { 1, 3, 4, 5, 65 }) {
            run_dense_q8_gemv_raw_reference_case(width, rows);
        }
    }
}

struct UniformRowAddGraph {
    ggml_tensor * storage[2];
    ggml_tensor * input[2];
    ggml_tensor * output;
    int64_t stride[2];
};

static UniformRowAddGraph build_uniform_row_add_graph(ggml_context * ctx, int64_t width, int64_t rows, int layout) {
    UniformRowAddGraph result = {};
    const int64_t strides[] = { 2 * width, 4 * width, width + 7 };
    for (int i = 0; i < 2; ++i) {
        result.stride[i] = strides[(layout + i) % 3];
        const int64_t prefix = (i + 1) * 16;
        result.storage[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                             prefix + (rows - 1) * result.stride[i] + width);
        auto * view = layout == 3 ?
            ggml_view_3d(ctx, result.storage[i], width, 2, rows / 2, result.stride[i] * sizeof(float),
                         2 * result.stride[i] * sizeof(float), prefix * sizeof(float)) :
            ggml_view_2d(ctx, result.storage[i], width, rows, result.stride[i] * sizeof(float), prefix * sizeof(float));
        result.input[i] = layout == i ? ggml_cont(ctx, view) : view;
    }
    result.output = ggml_add(ctx, result.input[0], result.input[1]);
    return result;
}

static void run_uniform_row_add_checks(bool execute) {
    ScopedHrxEnvironment hc("HRX_ENABLE_HC_SMALL_BATCH", nullptr);
    const std::vector<const char *> flags = execute ? std::vector<const char *>{ "1" } :
        std::vector<const char *>{ nullptr, "0", "true", "1" };
    ggml_backend_t cpu = execute ? init_cpu_backend() : nullptr;
    ggml_backend_t hrx = execute ? ggml_backend_hrx_init(0) : nullptr;
    if (execute) {
        REQUIRE(cpu != nullptr && hrx != nullptr);
    }
    for (const char * flag : flags) {
        ScopedHrxEnvironment glue("HRX_ENABLE_SMALL_BATCH_GLUE", flag);
        for (int64_t width : { 49, 64, 2560 }) {
            for (int64_t rows : { 2, 3, 4, 8 }) {
                for (int layout = 0; layout < 4; ++layout) {
                    if (layout == 3 && rows % 2 != 0) {
                        continue;
                    }
                    ggml_init_params params = {};
                    params.mem_size = 1024 * 1024;
                    params.no_alloc = true;
                    auto * ctx = ggml_init(params);
                    REQUIRE(ctx != nullptr);
                    const auto tensors = build_uniform_row_add_graph(ctx, width, rows, layout);
                    auto * graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, tensors.output);
                    auto imported = ggml::hrx::import_ggml_graph(*graph);
                    REQUIRE(imported.valid());
                    const bool expected = flag != nullptr && std::strcmp(flag, "1") == 0;
                    REQUIRE(ggml::hrx::supports_add_f32_dispatch(tensors.output) == expected);
                    REQUIRE(!ggml::hrx::supports_fused_moe_add(tensors.output));
                    ggml::hrx::DispatchScheduler scheduler;
                    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == expected);
                    if (expected) {
                        const auto & plan = scheduler.plan();
                        REQUIRE(plan.valid());
                        REQUIRE(plan.transients.size() == 2);
                        REQUIRE(plan.dispatches.size() >= 3);
                        const size_t n = plan.dispatches.size();
                        const auto & add = plan.dispatches[n - 1];
                        REQUIRE(kernel_name_for_id(add.kernel.kernel_id) == "qwen3_moe:ggml_add_f32");
                        REQUIRE(add.kernel.integer_parameters.at("element_count") == width * rows);
                        for (int i = 0; i < 2; ++i) {
                            const auto & copy = plan.dispatches[n - 3 + i];
                            REQUIRE(copy.bindings[1].value == add.bindings[i].value);
                            REQUIRE(copy.bindings[1].length == width * rows * sizeof(float));
                            REQUIRE(add.bindings[i].value != add.bindings[2].value);
                            if (layout != i) {
                                REQUIRE(kernel_name_for_id(copy.kernel.kernel_id) == "qwen3_moe:ggml_copy_rows_f32");
                                REQUIRE(copy.kernel.integer_parameters.at("row_length") == width);
                                REQUIRE(copy.kernel.integer_parameters.at("source_row_stride") == tensors.stride[i]);
                                REQUIRE(copy.kernel.integer_parameters.at("output_row_stride") == width);
                                REQUIRE(copy.bindings[0].length == ((rows - 1) * tensors.stride[i] + width) * sizeof(float));
                                const auto * source = imported.graph.values().find(copy.bindings[0].value);
                                REQUIRE(source != nullptr && source->storage_offset == (i + 1) * 16 * sizeof(float));
                            }
                        }
                        REQUIRE(add.bindings[0].value != add.bindings[1].value);
                    }
                    if (execute) {
                        REQUIRE(ggml_backend_supports_op(hrx, tensors.output));
                        {
                            ScopedHrxEnvironment disabled("HRX_ENABLE_SMALL_BATCH_GLUE", "0");
                            REQUIRE(!ggml_backend_supports_op(hrx, tensors.output));
                        }
                        auto * cpu_ctx = ggml_init(params);
                        REQUIRE(cpu_ctx != nullptr);
                        const auto reference = build_uniform_row_add_graph(cpu_ctx, width, rows, layout);
                        auto * cpu_graph = ggml_new_graph(cpu_ctx);
                        ggml_build_forward_expand(cpu_graph, reference.output);
                        auto * cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
                        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, hrx);
                        REQUIRE(cpu_buffer != nullptr && buffer != nullptr);
                        for (int replay = 0; replay < 2; ++replay) {
                            std::vector<float> expected_output(width * rows, 0.0f);
                            for (int i = 0; i < 2; ++i) {
                                auto logical = make_pattern_f32(width * rows, 17 + i * 6 + replay * 12, 0.031f);
                                std::vector<float> physical(ggml_nelements(tensors.storage[i]), -999.0f);
                                for (int64_t row = 0; row < rows; ++row) {
                                    for (int64_t col = 0; col < width; ++col) {
                                        const size_t index = row * width + col;
                                        logical[index] += static_cast<float>(row + i + replay) * 0.017f;
                                        physical[(i + 1) * 16 + row * tensors.stride[i] + col] = logical[index];
                                        expected_output[index] += logical[index];
                                    }
                                }
                                set_tensor_pair_bytes(cpu, reference.storage[i], hrx, tensors.storage[i],
                                                      physical.data(), physical.size() * sizeof(float));
                            }
                            REQUIRE(ggml_backend_graph_compute(cpu, cpu_graph) == GGML_STATUS_SUCCESS);
                            REQUIRE(ggml_backend_graph_compute(hrx, graph) == GGML_STATUS_SUCCESS);
                            const auto actual = get_f32_tensor(hrx, tensors.output);
                            require_close(actual, expected_output, 0.0f);
                            require_close(actual, get_f32_tensor(cpu, reference.output), 0.0f);
                            for (int i = 0; i < 2; ++i) {
                                require_close(get_f32_tensor(hrx, tensors.storage[i]),
                                              get_f32_tensor(cpu, reference.storage[i]), 0.0f);
                            }
                        }
                        ggml_backend_buffer_free(cpu_buffer);
                        ggml_backend_buffer_free(buffer);
                        ggml_free(cpu_ctx);
                    }
                    ggml_free(ctx);
                }
            }
        }
    }
    {
        ScopedHrxEnvironment glue("HRX_ENABLE_SMALL_BATCH_GLUE", "1");
        for (int variant = 0; variant < 8; ++variant) {
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            auto * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            const auto tensors = build_uniform_row_add_graph(ctx, variant == 6 ? 32769 : 64,
                                                            variant == 7 ? 10 : 4, 3);
            if (variant == 0) {
                tensors.input[0]->nb[0] = 2 * sizeof(float);
            } else if (variant == 1) {
                tensors.input[1]->nb[1] = 63 * sizeof(float);
            } else if (variant == 2) {
                tensors.input[0]->nb[2] += sizeof(float);
            } else if (variant == 3) {
                tensors.output->nb[1] += sizeof(float);
            } else if (variant == 4) {
                tensors.input[1]->nb[1] += 1;
            } else if (variant == 5) {
                tensors.storage[0]->ne[0] -= 1; // view byte span extends beyond its backing storage
            }
            REQUIRE(!ggml::hrx::supports_add_f32_dispatch(tensors.output));
            REQUIRE(!ggml::hrx::supports_fused_moe_add(tensors.output));
            if (execute) {
                REQUIRE(!ggml_backend_supports_op(hrx, tensors.output));
            }
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, tensors.output);
            auto imported = ggml::hrx::import_ggml_graph(*graph);
            ggml::hrx::DispatchScheduler scheduler;
            REQUIRE(!imported.valid() || !scheduler.schedule_graph(imported.graph, { "gfx1151" }));
            ggml_free(ctx);
        }
    }
    {
        ScopedHrxEnvironment glue("HRX_ENABLE_SMALL_BATCH_GLUE", nullptr);
        ScopedHrxEnvironment iq("HRX_ENABLE_IQ_EXPERTS", "1");
        ScopedHrxEnvironment moe("HRX_ENABLE_MOE_SMALL_BATCH", "1");
        for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL }) {
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            auto * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            const auto tensors = build_decode_down_graph(ctx, type, 3, 512);
            REQUIRE(!ggml::hrx::supports_add_f32_dispatch(tensors.output));
            REQUIRE(ggml::hrx::supports_fused_moe_add(tensors.output));
            if (execute) {
                REQUIRE(ggml_backend_supports_op(hrx, tensors.output));
            }
            auto * last = tensors.output->src[1];
            last->view_offs = 0; // repeating route zero is not a valid disjoint MoE fold
            REQUIRE(!ggml::hrx::supports_fused_moe_add(tensors.output));
            if (execute) {
                REQUIRE(!ggml_backend_supports_op(hrx, tensors.output));
            }
            ggml_free(ctx);
        }
    }
    if (execute) {
        ggml_backend_free(cpu);
        ggml_backend_free(hrx);
        std::fprintf(stderr, "Uniform row ADD W=49/64/2560 rows=2/3/4/8 strides=2x/4x/padded and 3D replay passed\n");
    }
}

struct RowBiasGraph {
    ggml_tensor * input;
    ggml_tensor * bias;
    ggml_tensor * output;
};

static RowBiasGraph build_row_bias_graph(ggml_context * ctx, int64_t width, int64_t tokens, bool inplace = false) {
    auto * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width * tokens + 16);
    auto * view = ggml_view_2d(ctx, storage, width, tokens, width * sizeof(float), 16 * sizeof(float));
    auto * input = ggml_reshape_2d(ctx, view, width, tokens);
    auto * bias_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width + 16);
    auto * bias = ggml_view_1d(ctx, bias_storage, width, 16 * sizeof(float));
    return { input, bias, inplace ? ggml_add_inplace(ctx, input, bias) : ggml_add(ctx, input, bias) };
}

static void run_small_batch_glue_scheduling_checks() {
    const char * flags[] = { nullptr, "0", "true", "1" };
    for (const char * flag : flags) {
        ScopedHrxEnvironment environment("HRX_ENABLE_SMALL_BATCH_GLUE", flag);
        for (int64_t width : { 1, 47, 48, 49, 2560, 32768, 32769 }) {
            for (int64_t tokens : { 1, 2, 3, 4, 8, 9, 512 }) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                auto * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                const auto tensors = build_row_bias_graph(ctx, width, tokens);
                const bool expected = tokens == 1 ||
                    (width <= 32768 && tokens <= 8 && flag != nullptr && std::strcmp(flag, "1") == 0);
                REQUIRE(ggml::hrx::supports_add_f32_dispatch(tensors.output) == expected);
                REQUIRE(ggml::hrx::is_broadcast_add_candidate(tensors.output) == (tokens != 1));
                auto * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, tensors.output);
                auto imported = ggml::hrx::import_ggml_graph(*graph);
                REQUIRE(imported.valid());
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == expected);
                if (expected && tokens > 1) {
                    const auto & plan = scheduler.plan();
                    REQUIRE(plan.valid());
                    REQUIRE(plan.dispatches.size() == static_cast<size_t>(tokens + 2));
                    REQUIRE(plan.transients.size() == 2);
                    const auto * input = imported.graph.values().find_tensor(tensors.input);
                    const auto * bias = imported.graph.values().find_tensor(tensors.bias);
                    const auto * output = imported.graph.values().find_tensor(tensors.output);
                    REQUIRE(input != nullptr && bias != nullptr && output != nullptr);
                    REQUIRE(input->storage_offset == 16 * sizeof(float));
                    REQUIRE(bias->storage_offset == 16 * sizeof(float));
                    for (size_t i = 0; i < 2; ++i) {
                        REQUIRE(kernel_name_for_id(plan.dispatches[i].kernel.kernel_id) == "qwen3_moe:ggml_copy_f32");
                    }
                    REQUIRE(plan.dispatches[0].bindings[0].value == input->id);
                    REQUIRE(plan.dispatches[1].bindings[0].value == bias->id);
                    const size_t row_bytes = width * sizeof(float);
                    REQUIRE(plan.dispatches[0].bindings[1].length == tokens * row_bytes);
                    REQUIRE(plan.dispatches[1].bindings[1].length == row_bytes);
                    for (int64_t token = 0; token < tokens; ++token) {
                        const auto & add = plan.dispatches[token + 2];
                        REQUIRE(kernel_name_for_id(add.kernel.kernel_id) == "qwen3_moe:ggml_add_f32");
                        REQUIRE(add.kernel.integer_parameters.at("element_count") == width);
                        REQUIRE(add.bindings[0].value == plan.dispatches[0].bindings[1].value);
                        REQUIRE(add.bindings[1].value == plan.dispatches[1].bindings[1].value);
                        REQUIRE(add.bindings[2].value == output->id);
                        REQUIRE(add.bindings[0].value != add.bindings[1].value);
                        REQUIRE(add.bindings[0].offset == token * row_bytes);
                        REQUIRE(add.bindings[1].offset == 0);
                        REQUIRE(add.bindings[2].offset == token * row_bytes);
                        for (const auto & binding : add.bindings) {
                            REQUIRE(binding.length == row_bytes);
                        }
                        REQUIRE(add.bindings[0].offset + row_bytes <= input->byte_count);
                        REQUIRE(add.bindings[2].offset + row_bytes <= output->byte_count);
                    }
                }
                ggml_free(ctx);
            }
        }
    }
    ScopedHrxEnvironment environment("HRX_ENABLE_SMALL_BATCH_GLUE", "1");
    for (int variant = 0; variant < 8; ++variant) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        auto tensors = build_row_bias_graph(ctx, 48, 3);
        if (variant == 0) {
            tensors.input->nb[1] += sizeof(float);
        } else if (variant == 1) {
            tensors.input->nb[0] = 2 * sizeof(float);
        } else if (variant == 2) {
            tensors.bias->nb[0] = 2 * sizeof(float);
        } else if (variant == 3) {
            tensors.output->nb[1] += sizeof(float);
        } else if (variant == 4) {
            tensors.bias->ne[0] = 1; // scalar broadcast is a different contract
        } else if (variant == 5) {
            std::swap(tensors.output->src[0], tensors.output->src[1]);
        } else if (variant == 6) {
            tensors.bias->type = GGML_TYPE_F16;
        } else {
            tensors.input->type = GGML_TYPE_F16;
        }
        REQUIRE(ggml::hrx::is_broadcast_add_candidate(tensors.output));
        REQUIRE(!ggml::hrx::supports_add_f32_dispatch(tensors.output));
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, tensors.output);
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(!imported.valid() || !scheduler.schedule_graph(imported.graph, { "gfx1151" }));
        ggml_free(ctx);
    }
}

static void run_small_batch_glue_reference_checks() {
    for (int64_t width : { 1, 47, 48, 49, 2560, 32768 }) {
        for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
            for (int inplace = 0; inplace < (width == 48 ? 2 : 1); ++inplace) {
                ScopedHrxEnvironment environment("HRX_ENABLE_SMALL_BATCH_GLUE", tokens == 1 ? nullptr : "1");
                auto * cpu = init_cpu_backend();
                auto * hrx = ggml_backend_hrx_init(0);
                REQUIRE(cpu != nullptr && hrx != nullptr);
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                auto * cc = ggml_init(params);
                auto * hc = ggml_init(params);
                REQUIRE(cc != nullptr && hc != nullptr);
                const auto ct = build_row_bias_graph(cc, width, tokens, inplace != 0);
                const auto ht = build_row_bias_graph(hc, width, tokens, inplace != 0);
                auto * cg = ggml_new_graph(cc);
                auto * hg = ggml_new_graph(hc);
                ggml_build_forward_expand(cg, ct.output);
                ggml_build_forward_expand(hg, ht.output);
                REQUIRE(ggml_backend_supports_op(hrx, ht.output));
                {
                    ScopedHrxEnvironment disabled("HRX_ENABLE_SMALL_BATCH_GLUE", "0");
                    REQUIRE(ggml_backend_supports_op(hrx, ht.output) == (tokens == 1));
                }
                if (tokens > 1) {
                    ht.bias->nb[0] *= 2;
                    REQUIRE(!ggml_backend_supports_op(hrx, ht.output));
                    ht.bias->nb[0] /= 2;
                    ht.input->nb[1] += sizeof(float);
                    REQUIRE(!ggml_backend_supports_op(hrx, ht.output));
                    ht.input->nb[1] -= sizeof(float);
                    ht.input->ne[1] = ht.output->ne[1] = 9;
                    REQUIRE(!ggml_backend_supports_op(hrx, ht.output));
                    ht.input->ne[1] = ht.output->ne[1] = tokens;
                }
                auto * cb = ggml_backend_alloc_ctx_tensors(cc, cpu);
                auto * hb = ggml_backend_alloc_ctx_tensors(hc, hrx);
                REQUIRE(cb != nullptr && hb != nullptr);
                for (int replay = 0; replay < 2; ++replay) {
                    const auto input = make_pattern_f32(width * tokens, 31 + replay * 8, 0.017f);
                    const auto bias = make_pattern_f32(width, 13 + replay * 6, 0.031f);
                    std::vector<float> oracle(input.size());
                    for (size_t i = 0; i < input.size(); ++i) {
                        oracle[i] = input[i] + bias[i % width];
                    }
                    set_tensor_pair_bytes(cpu, ct.input, hrx, ht.input, input.data(), input.size() * sizeof(float));
                    set_tensor_pair_bytes(cpu, ct.bias, hrx, ht.bias, bias.data(), bias.size() * sizeof(float));
                    REQUIRE(ggml_backend_graph_compute(cpu, cg) == GGML_STATUS_SUCCESS);
                    REQUIRE(ggml_backend_graph_compute(hrx, hg) == GGML_STATUS_SUCCESS);
                    const auto actual = get_f32_tensor(hrx, ht.output);
                    require_close(actual, oracle, 0.0f);
                    require_close(actual, get_f32_tensor(cpu, ct.output), 0.0f);
                    require_close(get_f32_tensor(hrx, ht.bias), bias, 0.0f);
                }
                ggml_backend_buffer_free(cb);
                ggml_backend_buffer_free(hb);
                ggml_free(cc);
                ggml_free(hc);
                ggml_backend_free(cpu);
                ggml_backend_free(hrx);
            }
        }
        std::fprintf(stderr, "Row-bias ADD W=%lld T=1/2/3/4/8 CPU/raw oracle and input+bias replay passed\n",
                     (long long) width);
    }
}

struct HcCollapseGraph {
    ggml_tensor * input;
    ggml_tensor * streams[4];
    ggml_tensor * output;
};

static HcCollapseGraph build_hc_collapse_graph(ggml_context * ctx, int64_t tokens, int mode = 0) {
    HcCollapseGraph result = {};
    auto * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 4, tokens + 1);
    result.input = ggml_view_3d(ctx, storage, 2560, 4, tokens, 2560 * sizeof(float),
                               10240 * sizeof(float), 16 * sizeof(float));
    for (int stream = 0; stream < 4; ++stream) {
        result.streams[stream] = ggml_view_2d(ctx, result.input, 2560, tokens,
                                            10240 * sizeof(float), stream * 2560 * sizeof(float));
    }
    result.output = mode == 2 ? result.streams[0] : ggml_cont(ctx, result.streams[0]);
    for (int stream = 1; stream < 4; ++stream) {
        result.output = mode == 1 ? ggml_add_inplace(ctx, result.output, result.streams[stream]) :
                                   ggml_add(ctx, result.output, result.streams[stream]);
    }
    return result;
}

static void run_hc_collapse_scheduling_checks() {
    ScopedHrxEnvironment generic("HRX_ENABLE_SMALL_BATCH_GLUE", nullptr);
    const char * flags[] = { nullptr, "0", "true", "1" };
    for (const char * flag : flags) {
        ScopedHrxEnvironment environment("HRX_ENABLE_HC_SMALL_BATCH", flag);
        for (int64_t tokens : { 1, 2, 3, 4, 8, 9, 512 }) {
            for (int mode = 0; mode < 3; ++mode) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                auto * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                const auto tensors = build_hc_collapse_graph(ctx, tokens, mode);
                auto * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, tensors.output);
                auto imported = ggml::hrx::import_ggml_graph(*graph);
                const bool expected = tokens == 1 ||
                    (tokens <= 8 && flag != nullptr && std::strcmp(flag, "1") == 0);
                REQUIRE(ggml::hrx::is_hc_collapse_add_candidate(tensors.output));
                REQUIRE(ggml::hrx::supports_add_f32_dispatch(tensors.output) == expected);
                ggml::hrx::DispatchScheduler scheduler;
                REQUIRE(imported.valid());
                REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == expected);
                if (expected) {
                    const auto & plan = scheduler.plan();
                    REQUIRE(plan.valid());
                    size_t adds = 0;
                    for (size_t i = 0; i < plan.dispatches.size(); ++i) {
                        const auto & add = plan.dispatches[i];
                        if (kernel_name_for_id(add.kernel.kernel_id) != "qwen3_moe:ggml_add_f32") {
                            continue;
                        }
                        ++adds;
                        REQUIRE(add.kernel.integer_parameters.at("element_count") == tokens * 2560);
                        for (const auto & binding : add.bindings) {
                            REQUIRE(binding.offset == 0 && binding.length == tokens * 2560 * sizeof(float));
                        }
                        if (tokens > 1) {
                            REQUIRE(i >= 2);
                            const auto & lhs = plan.dispatches[i - 2];
                            const auto & rhs = plan.dispatches[i - 1];
                            REQUIRE(add.bindings[0].value == lhs.bindings[1].value);
                            REQUIRE(add.bindings[1].value == rhs.bindings[1].value);
                            REQUIRE(add.bindings[0].value != add.bindings[1].value);
                            REQUIRE(add.bindings[0].value != add.bindings[2].value);
                            REQUIRE(add.bindings[1].value != add.bindings[2].value);
                            REQUIRE(kernel_name_for_id(rhs.kernel.kernel_id) == "qwen3_moe:ggml_copy_rows_f32");
                            REQUIRE(rhs.kernel.integer_parameters.at("source_row_stride") == 10240);
                            REQUIRE(rhs.kernel.integer_parameters.at("output_row_stride") == 2560);
                            REQUIRE(rhs.kernel.integer_parameters.at("row_length") == 2560);
                            REQUIRE(rhs.bindings[0].length == ((tokens - 1) * 10240 + 2560) * sizeof(float));
                            const auto * stream = imported.graph.values().find(rhs.bindings[0].value);
                            REQUIRE(stream != nullptr);
                            REQUIRE(stream->storage_offset == (16 + adds * 2560) * sizeof(float));
                        }
                    }
                    REQUIRE(adds == 3);
                    if (tokens > 1) {
                        REQUIRE(plan.transients.size() == 6);
                    }
                }
                ggml_free(ctx);
            }
        }
    }
    ScopedHrxEnvironment environment("HRX_ENABLE_HC_SMALL_BATCH", "1");
    for (int variant = 0; variant < 6; ++variant) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        auto tensors = build_hc_collapse_graph(ctx, 3);
        auto * rhs = tensors.streams[3];
        if (variant == 0) {
            rhs->nb[1] += sizeof(float);
        } else if (variant == 1) {
            rhs->nb[0] = 2 * sizeof(float);
        } else if (variant == 2) {
            rhs->ne[0] = 1280;
        } else if (variant == 3) {
            tensors.output->nb[1] += sizeof(float);
        } else if (variant == 4) {
            rhs->type = GGML_TYPE_F16;
        } else {
            rhs->nb[1] = 10240 * sizeof(float) - 1;
        }
        REQUIRE(ggml::hrx::is_hc_collapse_add_candidate(tensors.output));
        REQUIRE(!ggml::hrx::supports_add_f32_dispatch(tensors.output));
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, tensors.output);
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(!imported.valid() || !scheduler.schedule_graph(imported.graph, { "gfx1151" }));
        ggml_free(ctx);
    }
}

static void run_hc_collapse_reference_checks() {
    ScopedHrxEnvironment generic("HRX_ENABLE_SMALL_BATCH_GLUE", nullptr);
    for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_HC_SMALL_BATCH", tokens == 1 ? nullptr : "1");
        for (int mode = 0; mode < 3; ++mode) {
            auto * cpu = init_cpu_backend();
            auto * hrx = ggml_backend_hrx_init(0);
            REQUIRE(cpu != nullptr && hrx != nullptr);
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            auto * cpu_ctx = ggml_init(params);
            auto * hrx_ctx = ggml_init(params);
            REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
            const auto ct = build_hc_collapse_graph(cpu_ctx, tokens, mode);
            const auto ht = build_hc_collapse_graph(hrx_ctx, tokens, mode);
            auto * cg = ggml_new_graph(cpu_ctx);
            auto * hg = ggml_new_graph(hrx_ctx);
            ggml_build_forward_expand(cg, ct.output);
            ggml_build_forward_expand(hg, ht.output);
            REQUIRE(ggml_backend_supports_op(hrx, ht.output));
            {
                ScopedHrxEnvironment disabled("HRX_ENABLE_HC_SMALL_BATCH", "0");
                REQUIRE(ggml_backend_supports_op(hrx, ht.output) == (tokens == 1));
            }
            if (tokens > 1) {
                ht.streams[3]->nb[1] += sizeof(float);
                REQUIRE(!ggml_backend_supports_op(hrx, ht.output));
                ht.streams[3]->nb[1] -= sizeof(float);
            }
            auto * cb = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
            auto * hb = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
            REQUIRE(cb != nullptr && hb != nullptr);
            for (int replay = 0; replay < 2; ++replay) {
                auto input = make_pattern_f32(10240 * tokens, 23 + replay * 8, 0.017f);
                for (size_t i = 0; i < input.size(); ++i) {
                    input[i] += static_cast<float>(i / 2560) * 0.031f;
                }
                std::vector<float> oracle(2560 * tokens);
                for (int64_t token = 0; token < tokens; ++token) {
                    for (size_t col = 0; col < 2560; ++col) {
                        float sum = input[token * 10240 + col];
                        for (size_t stream = 1; stream < 4; ++stream) {
                            sum += input[token * 10240 + stream * 2560 + col];
                        }
                        oracle[token * 2560 + col] = sum;
                    }
                }
                set_tensor_pair_bytes(cpu, ct.input, hrx, ht.input, input.data(), input.size() * sizeof(float));
                REQUIRE(ggml_backend_graph_compute(cpu, cg) == GGML_STATUS_SUCCESS);
                REQUIRE(ggml_backend_graph_compute(hrx, hg) == GGML_STATUS_SUCCESS);
                const auto actual = get_f32_tensor(hrx, ht.output);
                for (float value : actual) {
                    REQUIRE(std::isfinite(value));
                }
                require_close(actual, oracle, 1.0e-6f, 1.0e-6f);
                require_close(actual, get_f32_tensor(cpu, ct.output), 1.0e-6f, 1.0e-6f);
                require_close(get_f32_tensor(hrx, ht.input), input, 0.0f);
            }
            ggml_backend_buffer_free(cb);
            ggml_backend_buffer_free(hb);
            ggml_free(cpu_ctx);
            ggml_free(hrx_ctx);
            ggml_backend_free(cpu);
            ggml_backend_free(hrx);
        }
        std::fprintf(stderr, "HC collapse T=%lld CONT+3 ADD, in-place, two-strided and replay checks passed\n",
                     (long long) tokens);
    }
}

struct HcNormGraph {
    ggml_tensor * input;
    ggml_tensor * gamma;
    ggml_tensor * rms;
    ggml_tensor * flat;
    ggml_tensor * output;
};

static HcNormGraph build_hc_norm_graph(ggml_context * ctx, int64_t tokens) {
    HcNormGraph result = {};
    auto * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10240 * tokens + 16);
    result.input = ggml_view_3d(ctx, storage, 2560, 4, tokens,
                               2560 * sizeof(float), 10240 * sizeof(float), 16 * sizeof(float));
    result.gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10240);
    result.rms = ggml_rms_norm(ctx, result.input, kQwenRmsNormEps);
    result.flat = ggml_reshape_2d(ctx, result.rms, 10240, tokens);
    result.output = ggml_mul(ctx, result.flat, result.gamma);
    return result;
}

static void run_hc_small_batch_scheduling_checks() {
    const char * flags[] = { nullptr, "0", "true", "1" };
    for (const char * flag : flags) {
        ScopedHrxEnvironment environment("HRX_ENABLE_HC_SMALL_BATCH", flag);
        for (int64_t tokens : { 1, 2, 3, 4, 8, 9, 512 }) {
            for (int variant = 0; variant < 8; ++variant) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                auto * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                auto tensors = build_hc_norm_graph(ctx, tokens);
                auto * graph = ggml_new_graph(ctx);
                if (variant == 1) {
                    tensors.input->nb[2] += sizeof(float);
                } else if (variant == 2) {
                    tensors.gamma->ne[0] = 2560; // shared gamma is not per-stream HC gamma
                } else if (variant == 3) {
                    const float epsilon = 0.00001f;
                    std::memcpy(tensors.rms->op_params, &epsilon, sizeof(epsilon));
                } else if (variant == 4) {
                    tensors.gamma->nb[0] = 2 * sizeof(float);
                } else if (variant == 5 || variant == 6) {
                    tensors.gamma->op = GGML_OP_ADD;
                    tensors.gamma->src[0] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10240);
                    tensors.gamma->src[1] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10240);
                    // Exercise readiness through a layout alias, not only a direct producer.
                    auto * alias = ggml_reshape_1d(ctx, tensors.gamma, 10240);
                    tensors.output->src[1] = alias;
                    if (variant == 6) {
                        ggml_build_forward_expand(graph, alias);
                    }
                } else if (variant == 7) {
                    tensors.flat->ne[0] = 2560;
                    tensors.flat->ne[1] = 4 * tokens;
                }
                ggml_build_forward_expand(graph, tensors.output);
                const bool enabled = tokens == 1 ||
                    (tokens <= 8 && flag != nullptr && std::strcmp(flag, "1") == 0);
                const bool expected = enabled && (variant == 0 || variant == 6);
                auto imported = ggml::hrx::import_ggml_graph(*graph);
                ggml::hrx::DispatchScheduler scheduler;
                const bool scheduled = imported.valid() && scheduler.schedule_graph(imported.graph, { "gfx1151" });
                if (scheduled != expected) {
                    std::fprintf(stderr, "HC scheduling T=%lld flag=%s variant=%d scheduled=%d expected=%d\n",
                                 (long long) tokens, flag == nullptr ? "unset" : flag, variant, scheduled, expected);
                    for (const auto & error : scheduler.plan().status.errors()) {
                        std::fprintf(stderr, "%s\n", error.c_str());
                    }
                }
                REQUIRE(scheduled == expected);
                if (variant == 0) {
                    REQUIRE(ggml::hrx::qwen4exp_hc_grouped_norm_rms_supported(tensors.rms) == enabled);
                } else if (variant == 1 || variant == 3) {
                    REQUIRE(!ggml::hrx::qwen4exp_hc_grouped_norm_rms_supported(tensors.rms));
                }
                if (scheduled) {
                    REQUIRE(scheduler.plan().valid());
                    const auto * input = imported.graph.values().find_tensor(tensors.input);
                    const auto * gamma = imported.graph.values().find_tensor(tensors.output->src[1]);
                    const auto * output = imported.graph.values().find_tensor(tensors.output);
                    REQUIRE(input != nullptr && gamma != nullptr && output != nullptr);
                    REQUIRE(input->storage_offset == 16 * sizeof(float));
                    size_t token = 0;
                    for (const auto & dispatch : scheduler.plan().dispatches) {
                        if (kernel_name_for_id(dispatch.kernel.kernel_id) !=
                            "qwen4exp:qwen38_hc_grouped_norm_decode") {
                            continue;
                        }
                        REQUIRE(dispatch.bindings.size() == 3);
                        REQUIRE(dispatch.bindings[0].value == input->id);
                        REQUIRE(dispatch.bindings[1].value == gamma->id);
                        REQUIRE(dispatch.bindings[2].value == output->id);
                        REQUIRE(dispatch.bindings[0].offset == token * 40960);
                        REQUIRE(dispatch.bindings[1].offset == 0);
                        REQUIRE(dispatch.bindings[2].offset == token * 40960);
                        for (size_t binding = 0; binding < 3; ++binding) {
                            REQUIRE(dispatch.bindings[binding].length == 40960);
                        }
                        REQUIRE(dispatch.bindings[0].offset + dispatch.bindings[0].length <= input->byte_count);
                        REQUIRE(dispatch.bindings[2].offset + dispatch.bindings[2].length <= output->byte_count);
                        ++token;
                    }
                    REQUIRE(token == static_cast<size_t>(tokens));
                }
                ggml_free(ctx);
            }
        }
    }
    ScopedHrxEnvironment environment("HRX_ENABLE_HC_SMALL_BATCH", "1");
    for (int64_t tokens : { -1, 0, 9, 512 }) {
        REQUIRE(!ggml::hrx::qwen4exp_hc_grouped_norm_token_count_supported(tokens));
    }
}

static void run_hc_small_batch_reference_checks() {
    for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_HC_SMALL_BATCH", tokens == 1 ? nullptr : "1");
        ggml_backend_t cpu = init_cpu_backend();
        ggml_backend_t hrx = ggml_backend_hrx_init(0);
        REQUIRE(cpu != nullptr && hrx != nullptr);
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * cpu_ctx = ggml_init(params);
        auto * hrx_ctx = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
        const auto cpu_tensors = build_hc_norm_graph(cpu_ctx, tokens);
        const auto hrx_tensors = build_hc_norm_graph(hrx_ctx, tokens);
        auto * cpu_graph = ggml_new_graph(cpu_ctx);
        auto * hrx_graph = ggml_new_graph(hrx_ctx);
        ggml_build_forward_expand(cpu_graph, cpu_tensors.output);
        ggml_build_forward_expand(hrx_graph, hrx_tensors.output);
        REQUIRE(ggml_backend_supports_op(hrx, hrx_tensors.rms));
        for (int64_t unsupported : { 9, 512 }) {
            hrx_tensors.input->ne[2] = unsupported;
            hrx_tensors.rms->ne[2] = unsupported;
            REQUIRE(!ggml_backend_supports_op(hrx, hrx_tensors.rms));
        }
        hrx_tensors.input->ne[2] = tokens;
        hrx_tensors.rms->ne[2] = tokens;
        hrx_tensors.input->nb[1] += sizeof(float);
        REQUIRE(!ggml_backend_supports_op(hrx, hrx_tensors.rms));
        hrx_tensors.input->nb[1] -= sizeof(float);
        {
            ScopedHrxEnvironment disabled("HRX_ENABLE_HC_SMALL_BATCH", "0");
            REQUIRE(ggml_backend_supports_op(hrx, hrx_tensors.rms) == (tokens == 1));
        }
        auto * cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
        auto * hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
        REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
        for (int replay = 0; replay < 2; ++replay) {
            auto input = make_pattern_f32(10240 * tokens, 31 + replay * 8, 0.013f);
            std::vector<float> gamma(10240);
            for (size_t i = 0; i < input.size(); ++i) {
                const size_t group = i / 2560;
                input[i] += static_cast<float>(group) * 0.017f;
                if ((group + replay) % 5 == 3) {
                    input[i] = 0.0f;
                }
            }
            for (size_t i = 0; i < gamma.size(); ++i) {
                gamma[i] = static_cast<float>(static_cast<int>(i % 67) - 33) * 0.021f +
                           static_cast<float>(i / 2560 + replay) * 0.13f;
            }
            std::vector<float> oracle(input.size());
            for (size_t base = 0; base < input.size(); base += 2560) {
                double sum = 0.0;
                for (size_t col = 0; col < 2560; ++col) {
                    sum += static_cast<double>(input[base + col]) * input[base + col];
                }
                const double inverse = 1.0 / std::sqrt(sum / 2560.0 + kQwenRmsNormEps);
                for (size_t col = 0; col < 2560; ++col) {
                    oracle[base + col] = static_cast<float>(input[base + col] * inverse * gamma[(base + col) % 10240]);
                }
            }
            set_tensor_pair_bytes(cpu, cpu_tensors.input, hrx, hrx_tensors.input,
                                  input.data(), input.size() * sizeof(float));
            set_tensor_pair_bytes(cpu, cpu_tensors.gamma, hrx, hrx_tensors.gamma,
                                  gamma.data(), gamma.size() * sizeof(float));
            const std::vector<float> dirty(input.size(), 123.0f);
            set_tensor_bytes(hrx, hrx_tensors.output, dirty.data(), dirty.size() * sizeof(float));
            REQUIRE(ggml_backend_graph_compute(cpu, cpu_graph) == GGML_STATUS_SUCCESS);
            REQUIRE(ggml_backend_graph_compute(hrx, hrx_graph) == GGML_STATUS_SUCCESS);
            const auto actual = get_f32_tensor(hrx, hrx_tensors.output);
            for (float value : actual) {
                REQUIRE(std::isfinite(value));
            }
            require_close(actual, get_f32_tensor(cpu, cpu_tensors.output), 3.0e-4f, 1.0e-5f);
            require_close(actual, oracle, 3.0e-4f, 1.0e-5f);
            require_close(get_f32_tensor(hrx, hrx_tensors.gamma), gamma, 0.0f);
        }
        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
        ggml_backend_free(cpu);
        ggml_backend_free(hrx);
        std::fprintf(stderr, "HC grouped RMSNorm T=%lld CPU/raw oracle and replay checks passed\n", (long long) tokens);
    }
}

struct SwigluGraph {
    ggml_tensor * a;
    ggml_tensor * b;
    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * output;
    ggml_cgraph * graph;
};

static SwigluGraph build_swiglu_graph(ggml_context * ctx, int64_t width, int64_t tokens,
                                      int alias = 0, bool reverse = false) {
    SwigluGraph result = {};
    if (alias == 0) {
        result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
        result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
        result.gate = result.a;
        result.up = result.b;
    } else {
        result.a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width * tokens + 32);
        result.b = alias == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width * tokens + 32) : result.a;
        result.gate = ggml_view_2d(ctx, result.a, width, tokens, width * sizeof(float), 3 * sizeof(float));
        result.up = alias == 2 ? result.gate :
            ggml_view_2d(ctx, result.b, width, tokens, width * sizeof(float), 11 * sizeof(float));
    }
    if (reverse) {
        std::swap(result.gate, result.up);
    }
    result.output = ggml_swiglu_split(ctx, result.gate, result.up);
    result.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

static void check_swiglu_plan(const SwigluGraph & tensors, bool enabled, ggml_backend_t backend = nullptr) {
    REQUIRE(ggml::hrx::supports_swiglu_f32_dispatch(tensors.output) == enabled);
    if (backend != nullptr) {
        REQUIRE(ggml_backend_supports_op(backend, tensors.output) == enabled);
    }
    auto imported = ggml::hrx::import_ggml_graph(*tensors.graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }) == enabled);
    if (!enabled) {
        return;
    }
    const auto & plan = scheduler.plan();
    REQUIRE(plan.valid() && plan.dispatches.size() == 1);
    REQUIRE(plan.transients.empty() && plan.constant_initializations.empty());
    const auto & dispatch = plan.dispatches[0];
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "hrx_owned:ggml_swiglu_split_f32");
    REQUIRE(dispatch.kernel.compile_parameters.empty());
    REQUIRE(dispatch.kernel.integer_parameters.size() == 1);
    REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == ggml_nelements(tensors.output));
    REQUIRE(dispatch.bindings.size() == 3);
    const ggml_tensor * bound[] = { tensors.gate, tensors.up, tensors.output };
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(dispatch.bindings[i].value == imported.graph.values().find_tensor(bound[i])->id);
        REQUIRE(dispatch.bindings[i].offset == 0);
        REQUIRE(dispatch.bindings[i].length == ggml_nbytes(bound[i]));
    }
}

static void run_swiglu_merged_ffn_scheduling_checks() {
    ScopedHrxEnvironment enabled("HRX_ENABLE_SWIGLU", "1");
    ScopedHrxEnvironment iq("HRX_ENABLE_IQ_EXPERTS", "1");
    ScopedHrxEnvironment narrow("HRX_ENABLE_Q8_NARROW", "1");
    // The shared Q8_0 640->2560 projection is currently a decode-only narrow route.
    for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS }) {
        for (int64_t tokens : { 1 }) {
            for (int ordering : { 0, 1, 2 }) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                auto * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 1, tokens);
                auto * ids = make_moe_ids_tensor(ctx, tokens, 512);
                auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 10, tokens);
                auto * gate_w = ggml_new_tensor_3d(ctx, type, 2560, 640, 512);
                auto * up_w = ggml_dup_tensor(ctx, gate_w);
                auto * gate = ggml_mul_mat_id(ctx, gate_w, input, ids);
                auto * up = ggml_mul_mat_id(ctx, up_w, input, ids);
                auto * glu = ggml_swiglu_split(ctx, gate, up);
                auto * down_w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 640, 2560, 512);
                auto * down = ggml_mul_mat_id(ctx, down_w, glu, ids);
                auto * routed = q4_expert_fold(ctx, down, weights);

                // Same shared branch and final ADD as qwen4exp::build_layer_ffn.
                auto * shared_input = ggml_reshape_2d(ctx, input, 2560, tokens);
                auto * shared_gate_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2560, 640);
                auto * shared_up_w = ggml_dup_tensor(ctx, shared_gate_w);
                auto * shared_glu = ggml_swiglu_split(ctx,
                    ggml_mul_mat(ctx, shared_gate_w, shared_input),
                    ggml_mul_mat(ctx, shared_up_w, shared_input));
                auto * shared_down_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 640, 2560);
                auto * shared_down = ggml_mul_mat(ctx, shared_down_w, shared_glu);
                auto * shared_scale_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
                auto * shared_scale = ggml_sigmoid(ctx, ggml_mul_mat(ctx, shared_scale_w, shared_input));
                auto * shared = ggml_mul(ctx, shared_down, shared_scale);
                auto * shared_view = ggml_view_2d(ctx, shared, 2560, tokens, shared->nb[1], 0);
                REQUIRE(shared_view->op == GGML_OP_VIEW && shared_view->src[0] == shared);
                auto * output = ggml_add(ctx, routed, shared_view);
                auto * graph = ggml_new_graph(ctx);
                if (ordering != 0) {
                    ggml_build_forward_expand(graph, shared_view);
                }
                ggml_build_forward_expand(graph, output);
                if (ordering == 2) {
                    ggml_build_forward_expand(graph, ggml_scale(ctx, shared_view, 2.0f));
                }
                auto imported = ggml::hrx::import_ggml_graph(*graph);
                REQUIRE(imported.valid());
                const auto id = [&](const ggml_tensor * tensor) {
                    return imported.graph.values().find_tensor(tensor)->id;
                };
                size_t down_node = 0, shared_node = 0;
                REQUIRE(imported.graph.index().node_index(imported.graph.index().producer(id(down)), down_node));
                REQUIRE(imported.graph.index().node_index(imported.graph.index().producer(id(shared)), shared_node));
                REQUIRE((down_node < shared_node) == (ordering == 0));
                ggml::hrx::DispatchScheduler scheduler;
                const bool scheduled = scheduler.schedule_graph(imported.graph, { "gfx1151" });
                if (!scheduled) {
                    std::fprintf(stderr, "merged SwiGLU type=%s T=%lld ordering=%d\n",
                                 ggml_type_name(type), (long long) tokens, ordering);
                    for (const auto & error : scheduler.plan().status.errors()) {
                        std::fprintf(stderr, "%s\n", error.c_str());
                    }
                }
                REQUIRE(scheduled);
                const auto & plan = scheduler.plan();
                REQUIRE(plan.valid());
                size_t down_dispatch = plan.dispatches.size();
                size_t shared_dispatch = plan.dispatches.size();
                size_t final_add = plan.dispatches.size();
                size_t q4_fusions = 0, iq_projections = 0;
                for (size_t i = 0; i < plan.dispatches.size(); ++i) {
                    const auto & dispatch = plan.dispatches[i];
                    const auto name = kernel_name_for_id(dispatch.kernel.kernel_id);
                    if (name == "qwen3_moe:qwen3_moe_routed_down_q8_0_q8_1_x4" ||
                        name == "hrx_owned:ggml_moe_small_down_q8_0") {
                        REQUIRE(down_dispatch == plan.dispatches.size());
                        down_dispatch = i;
                        if (ordering != 1) {
                            REQUIRE(dispatch.bindings[4].value == id(routed));
                            REQUIRE(i > 0 && kernel_name_for_id(plan.dispatches[i - 1].kernel.kernel_id) ==
                                             "qwen3_moe:ggml_zero_f32");
                        }
                    }
                    if (name == "hrx_owned:ggml_swiglu_split_f32") {
                        REQUIRE(dispatch.bindings[2].value == id(shared_glu));
                        shared_dispatch = i;
                    }
                    if (name == "qwen3_moe:ggml_add_f32" && dispatch.bindings[2].value == id(output)) {
                        REQUIRE(dispatch.bindings[0].value == id(routed));
                        REQUIRE(dispatch.bindings[1].value == id(shared_view));
                        final_add = i;
                    }
                    q4_fusions += name == "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8";
                    iq_projections += name == "hrx_owned:ggml_mul_mat_id_iq3_xxs_f32" ||
                                      name == "hrx_owned:ggml_mul_mat_id_iq4_xs_f32";
                }
                REQUIRE(down_dispatch < plan.dispatches.size() && shared_dispatch < plan.dispatches.size());
                REQUIRE(q4_fusions == (type == GGML_TYPE_Q4_K ? 1 : 0));
                REQUIRE(iq_projections == (type == GGML_TYPE_Q4_K ? 0 : 2));
                if (ordering == 1) {
                    REQUIRE(final_add == plan.dispatches.size());
                    REQUIRE(shared_dispatch < down_dispatch);
                } else {
                    REQUIRE(final_add > down_dispatch && final_add > shared_dispatch &&
                            final_add < plan.dispatches.size());
                    REQUIRE((down_dispatch < shared_dispatch) == (ordering == 0));
                }
                ggml_free(ctx);
            }
        }
    }
}

static uint64_t next_hrx_test_graph_uid() {
    static uint64_t next = 1;
    return next++;
}

static void run_swiglu_cache_contract_checks() {
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    auto * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    const auto tensors = build_swiglu_graph(ctx, 640, 3, 3);
    const auto & corpus = ggml::hrx::get_qwen_kernel_corpus();
    {
        ggml::hrx::GraphProgramCache cache;
        REQUIRE(tensors.graph->uid == 0);
        for (int i = 0; i < 2; ++i) {
            auto lookup = cache.get_or_build(*tensors.graph, corpus, "gfx1151");
            REQUIRE(lookup.valid() && lookup.uncached_program != nullptr);
            REQUIRE(cache.stats().builds == 0 && cache.stats().hits == 0);
        }
        tensors.graph->uid = next_hrx_test_graph_uid();
        ggml::hrx::GraphProgram * first = nullptr;
        for (uint64_t replay = 0; replay < 3; ++replay) {
            auto lookup = cache.get_or_build(*tensors.graph, corpus, "gfx1151");
            REQUIRE(lookup.valid() && lookup.uncached_program == nullptr);
            if (replay == 0) {
                first = lookup.program;
            }
            REQUIRE(lookup.program == first);
            REQUIRE(cache.stats().builds == 1 && cache.stats().hits == replay);
        }
    }
    ggml_free(ctx);
}

static void run_swiglu_scheduling_checks(ggml_backend_t backend = nullptr) {
    for (const char * flag : { static_cast<const char *>(nullptr), "", "0", "true", "01", "1x", "1" }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_SWIGLU", flag);
        const bool enabled = flag != nullptr && std::strcmp(flag, "1") == 0;
        for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
            for (int alias : { 0, 1, 2, 3 }) {
                ggml_init_params params = {};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                auto * ctx = ggml_init(params);
                REQUIRE(ctx != nullptr);
                const auto tensors = build_swiglu_graph(ctx, 640, tokens, alias, alias == 3);
                check_swiglu_plan(tensors, enabled, backend);
                // ggml ignores swapped for split inputs; it only selects halves in packed GLU.
                tensors.output->op_params[1] = 1;
                check_swiglu_plan(tensors, enabled, backend);
                ggml_free(ctx);
            }
        }
    }
    ScopedHrxEnvironment enabled("HRX_ENABLE_SWIGLU", "1");
    run_swiglu_cache_contract_checks();
    run_swiglu_merged_ffn_scheduling_checks();
    for (int invalid = 0; invalid < 22; ++invalid) {
        ggml_init_params params = {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        auto tensors = build_swiglu_graph(ctx, invalid == 0 ? 32769 : 640, invalid == 1 ? 9 : 3);
        if (invalid == 2) { tensors.gate->type = GGML_TYPE_F16; }
        if (invalid == 3) { tensors.up->type = GGML_TYPE_F16; }
        if (invalid == 4) { tensors.output->type = GGML_TYPE_F16; }
        if (invalid == 5) { tensors.gate->nb[0] *= 2; }
        if (invalid == 6) { tensors.up->nb[1] += sizeof(float); }
        if (invalid == 7) { tensors.output->nb[1] += sizeof(float); }
        if (invalid == 8) { tensors.up->ne[0] -= 1; }
        if (invalid == 9) { tensors.output->ne[2] = 2; }
        if (invalid == 10) { tensors.output->ne[3] = 2; }
        if (invalid == 11) { tensors.output->op_params[0] = GGML_GLU_OP_REGLU; }
        if (invalid == 12) { tensors.output->op_params[0] = GGML_GLU_OP_GEGLU; }
        if (invalid == 13) { tensors.output->op_params[0] = GGML_GLU_OP_SWIGLU_OAI; }
        if (invalid == 14) { tensors.output->op_params[0] = GGML_GLU_OP_SWIGLU_CLAMP; }
        if (invalid == 15) { tensors.output = ggml_swiglu(ctx, tensors.a); }
        if (invalid == 16) { tensors.output = ggml_swiglu_swapped(ctx, tensors.a); }
        if (invalid == 17) { tensors.output->view_src = tensors.a; }
        if (invalid == 18 || invalid == 19) {
            tensors.up = ggml_view_2d(ctx, tensors.b, 640, 3, 640 * sizeof(float), 0);
            tensors.up->view_offs = invalid == 18 ? sizeof(float) : 1;
            tensors.output->src[1] = tensors.up;
        }
        if (invalid == 20) { tensors.output->op_params[0] = GGML_GLU_OP_GEGLU_ERF; }
        if (invalid == 21) { tensors.output->op_params[0] = GGML_GLU_OP_GEGLU_QUICK; }
        tensors.graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(tensors.graph, tensors.output);
        check_swiglu_plan(tensors, false, backend);
        ggml_free(ctx);
    }
    // The opt-in must not replace either the routed IQ SiLU+MUL route or Q4 fused alternates.
    for (const char * flag : { "0", "1" }) {
        ScopedHrxEnvironment environment("HRX_ENABLE_SWIGLU", flag);
        // Use the complete qwen4exp 2560/640/512, routes=10 gate/up/down graph.
        run_q4_expert_scheduling_checks();
        ScopedHrxEnvironment iq("HRX_ENABLE_IQ_EXPERTS", "1");
        ScopedHrxEnvironment small("HRX_ENABLE_MOE_SMALL_BATCH", "1");
        for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
            ggml_init_params params = {};
            params.mem_size = 1024 * 1024;
            params.no_alloc = true;
            auto * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            auto * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 640, 10, tokens);
            auto * up = ggml_dup_tensor(ctx, gate);
            auto * glu = ggml_swiglu_split(ctx, gate, up);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, glu);
            const auto sequence = scheduled_kernel_sequence(graph);
            REQUIRE(sequence == std::vector<std::string>({
                "qwen3_moe:ggml_silu_f32", "qwen3_moe:ggml_mul_f32" }));
            ggml_free(ctx);
        }
    }
}

static void run_swiglu_reference_case(ggml_backend_t cpu, ggml_backend_t hrx, int64_t width,
                                      int64_t tokens, int alias = 0, bool reverse = false) {
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    auto * cpu_ctx = ggml_init(params);
    auto * hrx_ctx = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);
    const auto c = build_swiglu_graph(cpu_ctx, width, tokens, alias, reverse);
    const auto h = build_swiglu_graph(hrx_ctx, width, tokens, alias, reverse);
    // UID=0 intentionally bypasses both persistent graph and prepared-program reuse.
    h.graph->uid = next_hrx_test_graph_uid();
    check_swiglu_plan(h, true, hrx);
    auto cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
    auto hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
    REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
    ggml_backend_hrx_cache_stats before = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &before));
    ggml_backend_hrx_cache_stats first = {};
    for (int replay = 0; replay < 3; ++replay) {
        std::vector<float> a(ggml_nelements(c.a)), b(ggml_nelements(c.b));
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<float>(static_cast<int>((i * 7 + replay * 13) % 67) - 33) * 0.25f;
            b[i] = static_cast<float>(static_cast<int>((i * 11 + replay * 17) % 53) - 26) * 0.125f;
            if (i % 29 == 0) { a[i] = replay == 1 ? -80.0f : 80.0f; }
            if (i % 31 == 0) { a[i] = replay == 1 ? -1.0e-7f : 1.0e-7f; }
            if (i % 37 == 0) { a[i] = replay == 1 ? -0.0f : 0.0f; }
        }
        set_tensor_pair_bytes(cpu, c.a, hrx, h.a, a.data(), a.size() * sizeof(float));
        if (c.a != c.b) {
            set_tensor_pair_bytes(cpu, c.b, hrx, h.b, b.data(), b.size() * sizeof(float));
        }
        const auto gate = get_f32_tensor(cpu, c.gate);
        const auto up = get_f32_tensor(cpu, c.up);
        std::vector<float> expected(gate.size());
        bool ordering_discriminates = false;
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
            const float wrong = (up[i] / (1.0f + std::exp(-up[i]))) * gate[i];
            ordering_discriminates |= std::fabs(expected[i] - wrong) > 1.0e-3f;
        }
        REQUIRE(alias == 2 || width == 1 || ordering_discriminates);
        REQUIRE(ggml_backend_graph_compute(cpu, c.graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx, h.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu);
        ggml_backend_synchronize(hrx);
        const auto actual = get_f32_tensor(hrx, h.output);
        require_close(actual, expected, 2.0e-5f, 3.0e-6f);
        require_close(actual, get_f32_tensor(cpu, c.output), 2.0e-5f, 3.0e-6f);
        require_close(get_f32_tensor(hrx, h.a), a, 0.0f);
        require_close(get_f32_tensor(hrx, h.b), c.a == c.b ? a : b, 0.0f);
        ggml_backend_hrx_cache_stats stats = {};
        REQUIRE(ggml_backend_hrx_get_cache_stats(hrx, &stats));
        if (replay == 0) {
            REQUIRE(stats.graph_program_builds == before.graph_program_builds + 1);
            REQUIRE(stats.prepared_program_builds == before.prepared_program_builds + 1);
            REQUIRE(stats.graph_program_hits == before.graph_program_hits);
            REQUIRE(stats.prepared_program_hits == before.prepared_program_hits);
            first = stats;
        } else {
            REQUIRE(stats.graph_program_builds == first.graph_program_builds);
            REQUIRE(stats.prepared_program_builds == first.prepared_program_builds);
            REQUIRE(stats.graph_program_hits == first.graph_program_hits + replay);
            REQUIRE(stats.prepared_program_hits == first.prepared_program_hits + replay);
        }
    }
    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
}

static void run_swiglu_reference_checks() {
    ggml_backend_t cpu = init_cpu_backend();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(cpu != nullptr && hrx != nullptr);
    run_swiglu_scheduling_checks(hrx);
    ScopedHrxEnvironment enabled("HRX_ENABLE_SWIGLU", "1");
    for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
        run_swiglu_reference_case(cpu, hrx, 640, tokens);
        run_swiglu_reference_case(cpu, hrx, 640, tokens, 1, true);
    }
    for (int64_t width : { 1, 31, 255, 256, 257, 639, 641, 1025, 32768 }) {
        run_swiglu_reference_case(cpu, hrx, width, 3);
    }
    run_swiglu_reference_case(cpu, hrx, 640, 3, 2);
    run_swiglu_reference_case(cpu, hrx, 641, 3, 3, true);
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    run_iq_down_pack_reference_case(GGML_TYPE_Q8_0, 1, 512, true, true);
    run_iq_down_pack_reference_case(GGML_TYPE_IQ4_NL, 3, 512, true, true);
    std::fprintf(stderr, "SwiGLU checks passed (640 T=1/2/3/4/8, tails, CPU, ordering, aliases, replay)\n");
}

#include "test-hrx-f32-router.inc"
#include "test-hrx-recurrent-concat.inc"
#include "test-hrx-qsa-projections.inc"
#include "test-hrx-qsa-glue.inc"
#include "test-hrx-qsa-rope.inc"
#include "test-hrx-gdn-prefill.inc"
#include "test-hrx-iq-packets.inc"
#include "test-hrx-gdn-conv-prefill.inc"
#include "test-hrx-qsa-f16-gather.inc"
#include "test-hrx-device-rebind.inc"
#include "test-hrx-qsa-dense-bypass.inc"
#include "test-hrx-dense-f32-gemv.inc"
#include "test-hrx-gdn-norm-prefill.inc"
#include "test-hrx-qsa-mask.inc"

int main(int argc, char ** argv) {
    bool q8_selected = false;
    bool mtp_hc_projection_selected = false;
    bool dense_f32_accum_selected = false;
    bool dense_f32_gemv_selected = false;
    bool dense_rounded_gemv_selected = false;
    bool device_timing_selected = false;
    bool device_timing_schedule_selected = false;
    bool device_timing_schedule_only = argc > 1;
    bool q8_narrow_selected = false;
    bool q8_narrow_cpu_selected = false;
    bool q8_narrow_cpu_only = argc > 1;
    bool q8_embedding_selected = false;
    bool q4_embedding_selected = false;
    bool iq_selected = false;
    bool iq_packet4_selected = false;
    bool q4_experts_selected = false;
    bool q4_ids_selected = false;
    bool q4_ids_only = argc > 1;
    bool moe_small_batch_selected = false;
    bool hc_small_batch_selected = false;
    bool small_batch_glue_selected = false;
    bool swiglu_selected = false;
    bool swiglu_schedule_selected = false;
    bool swiglu_schedule_only = argc > 1;
    bool set_rows_trusted_producer_selected = false;
    bool set_rows_trusted_producer_only = argc > 1;
    bool f32_get_rows_scheduling_selected = false;
    bool f32_get_rows_scheduling_only = argc > 1;
    bool trusted_index_views_selected = false;
    bool f32_router_selected = false;
    bool recurrent_concat_selected = false;
    bool qsa_projections_selected = false;
    bool qsa_glue_selected = false;
    bool qsa_rope_selected = false;
    bool gdn_prefill_selected = false;
    bool gdn_conv_prefill_selected = false;
    bool gdn_norm_prefill_selected = false;
    bool qsa_f16_gather_selected = false;
    bool device_rebind_selected = false;
    bool qsa_dense_bypass_selected = false;
    bool qsa_dense_bypass_only = argc > 1;
    bool qsa_mask_selected = false;
    bool qsa_mask_schedule_selected = false;
    bool qsa_mask_schedule_only = argc > 1;
    bool q8_benchmark = false;
    for (int i = 1; i < argc; ++i) {
        qsa_mask_schedule_only = qsa_mask_schedule_only &&
            (std::strcmp(argv[i], "--qsa-mask-scheduling") == 0 ||
             std::strcmp(argv[i], "--qsa-dense-bypass") == 0);
        qsa_dense_bypass_only = qsa_dense_bypass_only && std::strcmp(argv[i], "--qsa-dense-bypass") == 0;
        q4_ids_only = q4_ids_only && std::strcmp(argv[i], "--q4-expert-ids") == 0;
        swiglu_schedule_only = swiglu_schedule_only && std::strcmp(argv[i], "--swiglu-schedule") == 0;
        device_timing_schedule_only =
            device_timing_schedule_only && std::strcmp(argv[i], "--device-timing-schedule") == 0;
        set_rows_trusted_producer_only =
            set_rows_trusted_producer_only && std::strcmp(argv[i], "--set-rows-trusted-producer") == 0;
        f32_get_rows_scheduling_only =
            f32_get_rows_scheduling_only && std::strcmp(argv[i], "--f32-get-rows-scheduling") == 0;
        q8_narrow_cpu_only = q8_narrow_cpu_only &&
            (std::strcmp(argv[i], "--q8-narrow-cpu") == 0 ||
             std::strcmp(argv[i], "--q4-expert-ids") == 0 || std::strcmp(argv[i], "--swiglu-schedule") == 0 ||
             std::strcmp(argv[i], "--device-timing-schedule") == 0);
        if (std::strcmp(argv[i], "--q8-gemv") == 0) {
            q8_selected = true;
        } else if (std::strcmp(argv[i], "--mtp-hc-projection") == 0) {
            mtp_hc_projection_selected = true;
        } else if (std::strcmp(argv[i], "--dense-f32-accum") == 0) {
            dense_f32_accum_selected = true;
        } else if (std::strcmp(argv[i], "--dense-f32-gemv") == 0) {
            dense_f32_gemv_selected = true;
        } else if (std::strcmp(argv[i], "--dense-rounded-gemv") == 0) {
            dense_rounded_gemv_selected = true;
        } else if (std::strcmp(argv[i], "--device-timing") == 0) {
            device_timing_selected = true;
        } else if (std::strcmp(argv[i], "--device-timing-schedule") == 0) {
            device_timing_schedule_selected = true;
        } else if (std::strcmp(argv[i], "--q8-narrow") == 0) {
            q8_narrow_selected = true;
        } else if (std::strcmp(argv[i], "--q8-narrow-cpu") == 0) {
            q8_narrow_cpu_selected = true;
        } else if (std::strcmp(argv[i], "--q8-embedding") == 0) {
            q8_embedding_selected = true;
        } else if (std::strcmp(argv[i], "--q4-embedding") == 0) {
            q4_embedding_selected = true;
        } else if (std::strcmp(argv[i], "--iq-experts") == 0) {
            iq_selected = true;
        } else if (std::strcmp(argv[i], "--iq-packets") == 0) {
            iq_packet4_selected = true;
        } else if (std::strcmp(argv[i], "--q4-experts") == 0) {
            q4_experts_selected = true;
        } else if (std::strcmp(argv[i], "--q4-expert-ids") == 0) {
            q4_ids_selected = true;
        } else if (std::strcmp(argv[i], "--moe-small-batch") == 0) {
            moe_small_batch_selected = true;
        } else if (std::strcmp(argv[i], "--hc-small-batch") == 0) {
            hc_small_batch_selected = true;
        } else if (std::strcmp(argv[i], "--small-batch-glue") == 0) {
            small_batch_glue_selected = true;
        } else if (std::strcmp(argv[i], "--swiglu") == 0) {
            swiglu_selected = true;
        } else if (std::strcmp(argv[i], "--swiglu-schedule") == 0) {
            swiglu_schedule_selected = true;
        } else if (std::strcmp(argv[i], "--set-rows-trusted-producer") == 0) {
            set_rows_trusted_producer_selected = true;
        } else if (std::strcmp(argv[i], "--f32-get-rows-scheduling") == 0) {
            f32_get_rows_scheduling_selected = true;
        } else if (std::strcmp(argv[i], "--trusted-index-views") == 0) {
            trusted_index_views_selected = true;
        } else if (std::strcmp(argv[i], "--f32-router") == 0) {
            f32_router_selected = true;
        } else if (std::strcmp(argv[i], "--recurrent-concat") == 0) {
            recurrent_concat_selected = true;
        } else if (std::strcmp(argv[i], "--qsa-projections") == 0) {
            qsa_projections_selected = true;
        } else if (std::strcmp(argv[i], "--qsa-glue") == 0) {
            qsa_glue_selected = true;
        } else if (std::strcmp(argv[i], "--qsa-rope") == 0) {
            qsa_rope_selected = true;
        } else if (std::strcmp(argv[i], "--gdn-prefill") == 0) {
            gdn_prefill_selected = true;
        } else if (std::strcmp(argv[i], "--gdn-conv-prefill") == 0) {
            gdn_conv_prefill_selected = true;
        } else if (std::strcmp(argv[i], "--gdn-norm-prefill") == 0) {
            gdn_norm_prefill_selected = true;
        } else if (std::strcmp(argv[i], "--qsa-f16-gather") == 0) {
            qsa_f16_gather_selected = true;
        } else if (std::strcmp(argv[i], "--device-rebind") == 0) {
            device_rebind_selected = true;
        } else if (std::strcmp(argv[i], "--qsa-dense-bypass") == 0) {
            qsa_dense_bypass_selected = true;
        } else if (std::strcmp(argv[i], "--qsa-mask") == 0) {
            qsa_mask_selected = true;
        } else if (std::strcmp(argv[i], "--qsa-mask-scheduling") == 0) {
            qsa_mask_schedule_selected = true;
        } else if (std::strcmp(argv[i], "--q8-benchmark") == 0) {
            q8_selected = true;
            q8_benchmark = true;
        } else {
            std::fprintf(stderr,
                "usage: %s [--q8-gemv] [--q8-narrow] [--q8-narrow-cpu] [--q8-embedding] [--q4-embedding]"
                " [--mtp-hc-projection] [--dense-f32-accum] [--dense-f32-gemv] [--dense-rounded-gemv]"
                " [--device-timing] [--device-timing-schedule]"
                " [--iq-experts] [--iq-packets] [--q4-experts] [--q4-expert-ids] [--moe-small-batch]"
                " [--hc-small-batch] [--small-batch-glue] [--swiglu] [--swiglu-schedule]"
                " [--set-rows-trusted-producer] [--f32-get-rows-scheduling] [--trusted-index-views]"
                " [--f32-router] [--recurrent-concat] [--qsa-projections] [--qsa-glue] [--qsa-rope]"
                " [--gdn-prefill] [--gdn-conv-prefill] [--gdn-norm-prefill] [--qsa-f16-gather] [--device-rebind]"
                " [--qsa-dense-bypass] [--qsa-mask] [--qsa-mask-scheduling] [--q8-benchmark]\n", argv[0]);
            return 1;
        }
    }
    // Legacy selectors keep their old-kernel assertions even when the caller enables this experiment.
    ScopedHrxEnvironment dense_default("HRX_ENABLE_DENSE_F32_ACCUM", "0");
    ScopedHrxEnvironment dense_gemv_default("HRX_ENABLE_DENSE_F32_GEMV", "0");
    ScopedHrxEnvironment dense_rounded_default("HRX_ENABLE_DENSE_ROUNDED_GEMV", "0");
    ScopedHrxEnvironment gdn_norm_default("HRX_ENABLE_GDN_NORM_PREFILL", "0");
    ScopedHrxEnvironment qsa_mask_default("HRX_ENABLE_QSA_MASK", "0");
    ScopedHrxEnvironment iq_default("HRX_ENABLE_IQ_PACKET4", "0");
    const bool selected_only = argc > 1;
    if (!selected_only || qsa_mask_selected || qsa_mask_schedule_selected) {
        run_qsa_mask_scheduling_checks();
    }
    if (qsa_dense_bypass_selected) {
        run_qsa_dense_bypass_checks();
    }
    if (qsa_dense_bypass_only) {
        std::fprintf(stderr, "QSA dense-bypass checks completed without device initialization\n");
        return 0;
    }
    if (qsa_mask_schedule_only) {
        std::fprintf(stderr, "QSA mask scheduling/snapshot checks passed without device initialization\n");
        return 0;
    }
    if (!selected_only || swiglu_selected || swiglu_schedule_selected) {
        run_swiglu_scheduling_checks();
    }
    if (!selected_only || device_timing_selected || device_timing_schedule_selected) {
        run_device_timing_schedule_checks();
    }
    if (!selected_only || set_rows_trusted_producer_selected || trusted_index_views_selected) {
        run_set_rows_trusted_producer_checks();
        run_set_rows_scheduling_checks();
    }
    if (!selected_only || f32_get_rows_scheduling_selected) {
        run_f32_get_rows_scheduling_checks();
    }
    if (swiglu_schedule_only) {
        std::fprintf(stderr, "SwiGLU scheduling checks passed (no device initialization)\n");
        return 0;
    }
    if (device_timing_schedule_only) {
        std::fprintf(stderr, "Device timing schedule checks passed (flag-off path performs no HIP calls)\n");
        return 0;
    }
    if (set_rows_trusted_producer_only) {
        std::fprintf(stderr, "SET_ROWS trusted-producer checks passed (no device initialization)\n");
        return 0;
    }
    if (f32_get_rows_scheduling_only) {
        std::fprintf(stderr, "F32 GET_ROWS scheduling checks passed (no device initialization)\n");
        return 0;
    }
    if (!selected_only || q4_experts_selected || q4_ids_selected) {
        run_q4_expert_id_validation_checks();
    }
    if (q4_ids_only) {
        REQUIRE(ggml_backend_hrx_shutdown());
        return 0;
    }
    if (q8_narrow_cpu_selected) {
        std::fprintf(stderr, "Q8 narrow CPU-only scheduling checks\n");
        run_q8_narrow_scheduling_checks();
        std::fprintf(stderr, "Q8 narrow CPU-only numerical checks\n");
        run_q8_narrow_reference_checks(false);
    }
    if (q8_narrow_cpu_selected && q8_narrow_cpu_only) {
        std::fprintf(stderr, "Q8 narrow T1..8 scheduling and CPU/reference/alias checks passed (no device initialization)\n");
        REQUIRE(ggml_backend_hrx_shutdown());
        return 0;
    }
    if (!selected_only || dense_f32_accum_selected) {
        run_dense_f32_accum_scheduling_checks();
        for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 }) {
            run_dense_f32_accum_reference_case(type, 512, 65, 3, false, true, false);
        }
    }
    if (!selected_only || dense_f32_gemv_selected) {
        run_dense_f32_gemv_scheduling_checks();
        run_dense_f32_gemv_reference_checks(false);
    }
    if (!selected_only || dense_rounded_gemv_selected) {
        run_dense_rounded_gemv_scheduling_checks();
        run_dense_rounded_gemv_reference_checks(false);
    }
    if (!selected_only || mtp_hc_projection_selected) {
        run_mtp_hc_projection_scheduling_checks();
    }
    if (!selected_only) {
        run_rmsnorm_support_checks();
        run_alternate_value_alias_lookup_checks();
    }
    if (!selected_only || q8_selected) {
        run_dense_q8_gemv_scheduling_checks();
    }
    if (!selected_only || q8_narrow_selected) {
        run_q8_narrow_scheduling_checks();
        run_q8_narrow_reference_case(320, 320, true, false);
        run_q8_narrow_reference_case(640, 640, true, false);
    }
    if (!selected_only || q8_embedding_selected) {
        run_q8_embedding_scheduling_checks();
    }
    if (!selected_only || q4_embedding_selected) {
        run_q4_embedding_scheduling_checks();
    }
    if (!selected_only || iq_selected) {
        run_iq_expert_scheduling_checks();
        run_iq_down_pack_scheduling_checks();
    }
    if (iq_packet4_selected) {
        run_iq_packet4_scheduling_checks();
    }
    if (!selected_only || q4_experts_selected) {
        run_q4_expert_scheduling_checks();
    }
    if (!selected_only || moe_small_batch_selected) {
        run_iq_expert_scheduling_checks(true);
        run_iq_down_pack_scheduling_checks(true);
        run_moe_small_batch_negative_checks();
    }
    if (!selected_only || hc_small_batch_selected) {
        run_hc_small_batch_scheduling_checks();
        run_hc_collapse_scheduling_checks();
    }
    if (!selected_only || small_batch_glue_selected) {
        run_small_batch_glue_scheduling_checks();
        run_uniform_row_add_checks(false);
    }

    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        REQUIRE(ggml_backend_hrx_shutdown());
        return 0;
    }
    if (selected_only) {
        if (swiglu_selected) {
            run_swiglu_reference_checks();
        }
        if (q4_experts_selected) {
            run_q4_expert_reference_case(10);
            run_q4_expert_reference_case(512);
            run_q4_router_reuse_reference_case();
            run_q4_router_reuse_reference_case(true);
        }
        if (dense_f32_accum_selected) {
            run_dense_f32_accum_reference_checks();
        }
        if (dense_f32_gemv_selected) {
            run_dense_f32_gemv_reference_checks();
            std::fprintf(stderr, "Dense raw-F32 GEMV scheduling/reference/cached replay checks passed\n");
        }
        if (dense_rounded_gemv_selected) {
            run_dense_rounded_gemv_reference_checks();
            std::fprintf(stderr, "Dense rounded GEMV scheduling/reference/cached replay checks passed\n");
        }
        if (device_timing_selected) {
            run_device_timing_gpu_checks();
        }
        if (mtp_hc_projection_selected) {
            run_mtp_hc_projection_reference_checks();
            std::fprintf(stderr, "MTP HC projection checks passed (Q4_K 5120->2560, 4*T columns, T=1/2/3/4/8)\n");
        }
        if (small_batch_glue_selected) {
            run_small_batch_glue_reference_checks();
            run_uniform_row_add_checks(true);
        }
        if (hc_small_batch_selected) {
            run_hc_small_batch_reference_checks();
            run_hc_collapse_reference_checks();
        }
        if (moe_small_batch_selected) {
            run_moe_small_batch_reference_checks();
            std::fprintf(stderr, "MoE small-batch checks passed (T=1/2/3/4/8, strides=10/512)\n");
        }
        if (q8_embedding_selected) {
            run_q8_embedding_reference_checks();
            std::fprintf(stderr, "Q8 embedding checks passed (raw rows, exact CPU decode, invalid-ID failure)\n");
        }
        if (trusted_index_views_selected) {
            run_trusted_index_view_reference_checks();
        }
        if (f32_router_selected) {
            run_f32_router_projection_checks();
        }
        if (recurrent_concat_selected) {
            run_recurrent_concat_checks();
        }
        if (qsa_projections_selected) {
            run_qsa_projection_checks();
        }
        if (qsa_glue_selected) {
            run_qsa_glue_checks();
        }
        if (qsa_rope_selected) {
            run_qsa_rope_checks();
        }
        if (gdn_prefill_selected) {
            run_gdn_prefill_scheduling_checks();
            run_gdn_prefill_reference_checks();
        }
        if (gdn_conv_prefill_selected) {
            run_gdn_conv_prefill_scheduling_checks();
            run_gdn_conv_prefill_reference_checks();
        }
        if (gdn_norm_prefill_selected) {
            run_gdn_norm_prefill_checks();
            std::fprintf(stderr, "GDN strided norm packing/reference/cached replay checks passed\n");
        }
        if (qsa_f16_gather_selected) {
            run_qsa_f16_gather_checks();
        }
        if (qsa_mask_selected) {
            run_qsa_mask_reference_checks();
        }
        if (device_rebind_selected) {
            run_device_binding_refresh_checks();
        }
        if (iq_packet4_selected) {
            run_iq_packet4_reference_checks();
            std::fprintf(stderr, "IQ packet4 scalar/CPU/replay checks passed (T1..8)\n");
        }
        if (q4_embedding_selected) {
            run_q4_embedding_reference_checks();
            std::fprintf(stderr, "Q4 embedding checks passed (MTP rows, exact CPU decode, changed-weight replay)\n");
        }
        if (q8_narrow_selected) {
            run_q8_narrow_reference_checks();
            std::fprintf(stderr, "Q8 narrow decode checks passed (K=320/640, T=1)\n");
        }
        if (q8_selected) {
            run_dense_q8_gemv_reference_checks();
            run_q8_narrow_reference_checks();
            std::fprintf(stderr, "Q8 GEMV checks passed\n");
        }
        if (q8_benchmark) {
            for (int64_t rows : { 2560, 10240 }) {
                for (int repeat = 0; repeat < 3; ++repeat) {
                    for (bool use_gemv : { repeat % 2 != 0, repeat % 2 == 0 }) {
                        run_dense_q8_gemv_raw_reference_case(2560, rows, use_gemv, 30);
                    }
                }
            }
        }
        if (iq_selected) {
            run_iq_expert_raw_reference_case(GGML_TYPE_IQ3_XXS);
            run_iq_expert_raw_reference_case(GGML_TYPE_IQ4_XS);
            run_iq_expert_raw_reference_case(GGML_TYPE_IQ3_XXS, true);
            run_iq_expert_raw_reference_case(GGML_TYPE_IQ4_XS, true);
            run_iq_down_pack_reference_case(GGML_TYPE_Q8_0);
            run_iq_down_pack_reference_case(GGML_TYPE_IQ4_NL);
            std::fprintf(stderr, "IQ expert checks passed\n");
        }
        REQUIRE(ggml_backend_hrx_shutdown());
        return 0;
    }

    run_add_f32_cpu_reference_case();
    run_swiglu_reference_checks();
    run_dense_f32_accum_reference_checks();
    run_dense_f32_gemv_reference_checks();
    run_dense_rounded_gemv_reference_checks();
    run_mtp_hc_projection_reference_checks();
    run_small_batch_glue_reference_checks();
    run_uniform_row_add_checks(true);
    run_hc_small_batch_reference_checks();
    run_hc_collapse_reference_checks();
    run_moe_small_batch_reference_checks();
    run_gather_add_f32_cpu_reference_case();
    run_trusted_index_view_reference_checks();
    run_f32_router_projection_checks();
    run_recurrent_concat_checks();
    run_qsa_projection_checks();
    run_qsa_glue_checks();
    run_qsa_rope_checks();
    run_gdn_prefill_scheduling_checks();
    run_gdn_prefill_reference_checks();
    run_gdn_norm_prefill_checks();
    run_token_embedding_q4k_cpu_reference_case();
    run_qsa_mask_reference_checks();
    run_q8_embedding_reference_checks();
    run_q4_embedding_reference_checks();
    run_dense_matmul_cpu_reference_case(GGML_TYPE_Q4_K, "qwen3_moe:qwen3_moe_dense_linear_q4k_f16_wmma", 2, 128);
    run_dense_matmul_cpu_reference_case(GGML_TYPE_Q6_K, "qwen3_moe:qwen3_moe_dense_linear_q6k_f16_wmma", 2, 128);
    run_dense_q8_gemv_reference_checks();
    run_q4_expert_reference_case(10);
    run_q4_expert_reference_case(512);
    run_q4_router_reuse_reference_case();
    run_q4_router_reuse_reference_case(true);
    run_iq_expert_raw_reference_case(GGML_TYPE_IQ3_XXS);
    run_iq_expert_raw_reference_case(GGML_TYPE_IQ4_XS);
    run_iq_expert_raw_reference_case(GGML_TYPE_IQ3_XXS, true);
    run_iq_expert_raw_reference_case(GGML_TYPE_IQ4_XS, true);
    run_iq_down_pack_reference_case(GGML_TYPE_Q8_0);
    run_iq_down_pack_reference_case(GGML_TYPE_IQ4_NL);
    run_endpoint_rmsnorm_q6k_q8_cpu_reference_case();
    run_attention_postprocess_cpu_reference_case();
    run_routed_moe_cpu_reference_case(GGML_TYPE_Q4_K, true);
    run_routed_moe_cpu_reference_case(GGML_TYPE_Q6_K, false);
    run_decode_attention_qkv_scheduling_case();
    run_decode_routed_moe_scheduling_case(GGML_TYPE_Q4_K);
    run_decode_routed_moe_scheduling_case(GGML_TYPE_Q6_K);
    run_decode_routed_moe_scheduling_case(GGML_TYPE_Q6_K, true);
    run_rmsnorm_mul_case(256, 1);
    run_rmsnorm_mul_case(256, 4);
    run_rmsnorm_mul_case(2048, 1);
    // Per-head norm shapes (qwen4exp indexer q_norm reshapes to [idx_dim=128, heads, real_tokens]
    // before RMS_NORM) collapse to a flat row count of heads * real_tokens here, which exceeds the
    // 2048 real-token cap well before any realistic prefill batch. 2496 reproduces the exact crash
    // shape observed at 48 indexer heads * 52 real tokens; 131072 exercises the new cap boundary.
    run_rmsnorm_mul_case(128, 2496);
    run_rmsnorm_mul_case(128, 131072);
    run_router_projection_case(4);
    run_router_top8_case(4);
    run_qwen_flash_attention_case();
    run_qwen_decode_split_flash_attention_scheduling_case(1, 512);
    run_qwen_decode_split_flash_attention_scheduling_case(4, 513);
    run_qwen_decode_attention_output_next_q8_scheduling_case(false);
    run_qwen_decode_attention_output_next_q8_scheduling_case(true);
    run_qwen_full_cache_prefill_flash_attention_scheduling_case(16);
    run_qwen_full_cache_prefill_flash_attention_scheduling_case(512);
    REQUIRE(ggml_backend_hrx_shutdown());
    return 0;
}
