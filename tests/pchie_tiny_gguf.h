// A tiny hand-built Gemma 4 model file for loader and backend tests: 1 layer, d_model 32, one 32-wide head,
// d_ff 32, d_ple 32, 4 tokens. Tests change one field of good_model() and check that the change is refused.
#pragma once
#include "pchie/pchie_gguf.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace pchie_tiny {
using pchie::ggml_type_block;
using Bytes = std::vector<uint8_t>;
template <class T> inline void put(Bytes& b, T v) { uint8_t t[sizeof(T)]; std::memcpy(t, &v, sizeof(T)); b.insert(b.end(), t, t + sizeof(T)); }
inline void put_str(Bytes& b, const std::string& s) { put<uint64_t>(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }

// A tiny model: 1 layer, d_model 32, one 32-wide head, d_ff 32, d_ple 32, 4 tokens.
struct Model {
    std::vector<std::pair<std::string, double>> nums;   // gemma4.* and tokenizer scalars (written as f64)
    std::vector<std::string> tokens{"<pad>", "<eos>", "<bos>", "a"};
    std::vector<int32_t> swa_pattern{0};
    struct T { std::string name; std::vector<uint64_t> dims; uint32_t type; float fill = 0.0f; };   // fill: value of every F32 element
    std::vector<T> tensors;
    uint64_t align = 32;                                  // general.alignment (written only when not 32)
    void set(const std::string& k, double v) { for (auto& kv : nums) if (kv.first == k) { kv.second = v; return; } nums.push_back({k, v}); }
    T* find(const std::string& n) { for (auto& t : tensors) if (t.name == n) return &t; return nullptr; }
};

inline Model good_model() {
    Model m;
    auto s = [&](const char* k, double v) { m.set(std::string("gemma4.") + k, v); };
    s("block_count", 1); s("embedding_length", 32); s("feed_forward_length", 32); s("embedding_length_per_layer_input", 32);
    s("attention.head_count", 1); s("attention.head_count_kv", 1); s("attention.key_length", 32); s("attention.key_length_swa", 32);
    s("attention.value_length", 32); s("attention.value_length_swa", 32);
    s("rope.dimension_count", 32); s("rope.dimension_count_swa", 32); s("attention.sliding_window", 16);
    s("attention.shared_kv_layers", 0); s("attention.layer_norm_rms_epsilon", 1e-6); s("rope.freq_base", 10000.0);
    s("rope.freq_base_swa", 10000.0); s("final_logit_softcapping", 30.0);
    m.set("tokenizer.ggml.bos_token_id", 2); m.set("tokenizer.ggml.eos_token_id", 1);
    const uint32_t Q8 = 8, BF = 30, F32 = 0;
    m.tensors = {{"token_embd.weight", {32, 4}, Q8}, {"per_layer_token_embd.weight", {32, 4}, Q8},
                 {"per_layer_model_proj.weight", {32, 32}, BF}, {"per_layer_proj_norm.weight", {32}, F32},
                 {"output_norm.weight", {32}, F32}, {"rope_freqs.weight", {16}, F32, 1.0f}};
    for (const char* w : {"attn_q", "attn_output", "attn_k", "attn_v", "ffn_gate", "ffn_up", "ffn_down", "inp_gate", "proj"})
        m.tensors.push_back({std::string("blk.0.") + w + ".weight", {32, 32}, BF});
    for (const char* n : {"attn_norm", "attn_q_norm", "attn_k_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "post_norm"})
        m.tensors.push_back({std::string("blk.0.") + n + ".weight", {32}, F32});
    m.tensors.push_back({"blk.0.layer_output_scale.weight", {1}, F32});
    return m;
}

inline uint64_t tensor_bytes(const Model::T& t) {
    uint64_t be = 0, bb = 0, n = 1;
    for (uint64_t d : t.dims) n *= d;
    if (!ggml_type_block(t.type, be, bb)) return 64;
    return n / be * bb;
}

inline Bytes build(const Model& m) {
    Bytes b{'G', 'G', 'U', 'F'};
    put<uint32_t>(b, 3);
    put<uint64_t>(b, m.tensors.size());
    put<uint64_t>(b, m.nums.size() + 4 + (m.align != 32 ? 1 : 0));
    put_str(b, "general.architecture"); put<uint32_t>(b, 8); put_str(b, "gemma4");
    if (m.align != 32) { put_str(b, "general.alignment"); put<uint32_t>(b, 4); put<uint32_t>(b, uint32_t(m.align)); }
    for (const auto& kv : m.nums) { put_str(b, kv.first); put<uint32_t>(b, 12); put<double>(b, kv.second); }
    put_str(b, "gemma4.attention.sliding_window_pattern"); put<uint32_t>(b, 9); put<uint32_t>(b, 5); put<uint64_t>(b, m.swa_pattern.size());
    for (int32_t v : m.swa_pattern) put<int32_t>(b, v);
    put_str(b, "tokenizer.ggml.tokens"); put<uint32_t>(b, 9); put<uint32_t>(b, 8); put<uint64_t>(b, m.tokens.size());
    for (const auto& t : m.tokens) put_str(b, t);
    put_str(b, "tokenizer.ggml.merges"); put<uint32_t>(b, 9); put<uint32_t>(b, 8); put<uint64_t>(b, 0);
    uint64_t off = 0;
    for (const auto& t : m.tensors) {
        put_str(b, t.name);
        put<uint32_t>(b, uint32_t(t.dims.size()));
        for (uint64_t d : t.dims) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, off);
        off += (tensor_bytes(t) + m.align - 1) / m.align * m.align;
    }
    while (b.size() % m.align) b.push_back(0);
    const size_t data0 = b.size();
    b.insert(b.end(), size_t(off), uint8_t(0));
    uint64_t at = 0;
    for (const auto& t : m.tensors) {
        if (t.type == 0 && t.fill != 0.0f)
            for (uint64_t i = 0; i < tensor_bytes(t) / 4; ++i) std::memcpy(&b[data0 + size_t(at + 4 * i)], &t.fill, 4);
        at += (tensor_bytes(t) + m.align - 1) / m.align * m.align;
    }
    return b;
}

inline bool write_file(const Model& m, const std::string& path) {
    const Bytes b = build(m);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    return bool(f);
}
} // namespace pchie_tiny
