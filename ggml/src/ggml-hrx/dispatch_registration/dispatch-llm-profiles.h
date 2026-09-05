#pragma once

#include <cstdint>

namespace ggml::hrx {

struct LlmMoeDispatchProfile {
    const char * name               = "";
    int64_t      hidden_size        = 0;
    int64_t      expert_hidden_size = 0;
    int64_t      expert_count       = 0;
    int64_t      route_count        = 0;
    int64_t      max_token_count    = 0;
    float        rms_norm_epsilon   = 0.0f;
};

constexpr LlmMoeDispatchProfile kLlmMoeQwen30BDispatchProfile = {
    "qwen30b", 2048, 768, 128, 8, 2048, 0.000001f,
};

// qwen4exp geometry (verified from the GGUF): hidden=2560, expert_hidden(expert_feed_forward_length)=640,
// expert_count=512, route_count(expert_used)=10, rms_eps=1e-6. NOTE: expert_hidden_size=640 is not a
// multiple of 256, unlike qwen30b's 768 -- that is why its down-projection expert weights ship as
// 32-block quants (Q5_1/Q8_0/IQ4_NL) instead of Q4_K/Q6_K; see hrx-moe-down-kernel-spec.md.
constexpr LlmMoeDispatchProfile kLlmMoeQwen4ExpDispatchProfile = {
    "qwen4exp", 2560, 640, 512, 10, 2048, 0.000001f,
};

// kActiveLlmMoeDispatchProfile is the single profile the routed-ffn dispatch matchers in
// dispatch-routed-ffn.cpp are compiled against (kRoutedFfnInputSize / kRoutedFfnExpertHiddenSize /
// etc. are `static constexpr` derived from it) -- it is a compile-time selection, NOT a per-model
// runtime choice. Do NOT switch this to kLlmMoeQwen4ExpDispatchProfile yet: doing so would change
// kRoutedFfnInputSize/kRoutedFfnExpertHiddenSize to qwen4exp's values and break every existing
// qwen30b (Q4_K/Q6_K) routed gate_up/down shape match, since qwen30b's actual weights would no longer
// match kRoutedFfnInputSize/kRoutedFfnExpertHiddenSize. Keep qwen30b active until the qwen4exp down
// kernels (Q5_1/Q8_0/IQ4_NL) are authored and a real per-profile (or per-quant) dispatch selection
// mechanism replaces this single-active-profile compile-time constant.
static constexpr const LlmMoeDispatchProfile & kActiveLlmMoeDispatchProfile = kLlmMoeQwen30BDispatchProfile;
static constexpr const LlmMoeDispatchProfile & kQwen30BMoeDispatchProfile   = kLlmMoeQwen30BDispatchProfile;
static constexpr const LlmMoeDispatchProfile & kQwen4ExpMoeDispatchProfile  = kLlmMoeQwen4ExpDispatchProfile;

constexpr bool is_llm_supported_query_length(const LlmMoeDispatchProfile & profile, int64_t query_length) {
    return query_length >= 1 && query_length <= profile.max_token_count;
}

constexpr bool is_llm_decode_query_length(int64_t query_length) {
    return query_length == 1;
}

constexpr bool is_llm_prefill_query_length(const LlmMoeDispatchProfile & profile, int64_t query_length) {
    return query_length > 1 && query_length <= profile.max_token_count;
}

}  // namespace ggml::hrx
