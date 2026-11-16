// Gemma 4 loader on untrusted input. A tiny, hand-built model file opens; a file whose sizes, weight shapes,
// weight types or token ids do not match what the forward pass will read is refused at load time with an
// error (no crash, no read outside a buffer or the mapping). An out-of-vocabulary token is refused by embed().
#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_gguf.h"
#include "pchie_tiny_gguf.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pchie;
static int passes = 0, fails = 0;
static void check(bool ok, const std::string& name) {
    std::printf("[G4LOAD] %-92s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    ok ? ++passes : ++fails;
}

using namespace pchie_tiny;

static bool load(const Model& m, const char* name, std::string* err_out = nullptr, std::function<void(Gemma4Model&)> after = nullptr) {
    const std::string p = std::string("g4_load_validate_") + name + ".gguf";
    write_file(m, p);
    bool ok = false;
    std::string err;
    {
        Gemma4Model g;
        ok = g.load(p, err);
        if (ok && after) after(g);
    }
    std::remove(p.c_str());
    std::printf("         %s -> %s\n", name, ok ? "(loaded)" : err.c_str());
    if (err_out) *err_out = err;
    return ok;
}
static void refused(const Model& m, const char* name, const std::string& label) {
    std::string err;
    const bool ok = load(m, name, &err);
    check(!ok && !err.empty(), label);
}

int main() {
    std::printf("[G4LOAD] Gemma 4 loader: sizes, weight shapes, types and token ids are checked before use\n");

    bool embed_ok = false, embed_refused = false, embed_neg_refused = false;
    check(load(good_model(), "good", nullptr, [&](Gemma4Model& g) {
        std::vector<float> x(32), ple(32);
        try { g.embed(3, x.data(), ple.data()); embed_ok = true; } catch (...) {}
        try { g.embed(4, x.data(), ple.data()); } catch (const std::out_of_range&) { embed_refused = true; }
        try { g.embed(-1, x.data(), ple.data()); } catch (const std::out_of_range&) { embed_neg_refused = true; }
    }), "a small well-formed model loads");
    check(embed_ok, "embed() reads a token inside the vocabulary");
    check(embed_refused, "embed() refuses token id == vocabulary size (would read past token_embd)");
    check(embed_neg_refused, "embed() refuses a negative token id");

    { Model m = good_model(); m.find("blk.0.attn_q.weight")->dims = {32, 64}; refused(m, "q_rows", "attn_q with more rows than heads x head size is refused"); }
    { Model m = good_model(); m.find("blk.0.ffn_down.weight")->dims = {64, 32}; refused(m, "down_cols", "ffn_down wider than feed_forward_length is refused (would read past the FF buffer)"); }
    { Model m = good_model(); m.find("blk.0.ffn_up.weight")->dims = {32, 64}; refused(m, "up_rows", "ffn_up taller than feed_forward_length is refused (would write past the FF buffer)"); }
    { Model m = good_model(); m.find("blk.0.attn_k.weight")->dims = {64, 32}; refused(m, "k_cols", "attn_k wider than embedding_length is refused"); }
    { Model m = good_model(); m.find("blk.0.proj.weight")->dims = {32, 64}; refused(m, "proj_rows", "per-layer proj taller than embedding_length is refused"); }
    { Model m = good_model(); m.find("per_layer_model_proj.weight")->dims = {32, 64}; refused(m, "ple_proj", "per_layer_model_proj with the wrong row count is refused"); }
    { Model m = good_model(); m.find("blk.0.attn_output.weight")->type = 0; refused(m, "f32_matrix", "a weight matrix in a type matvec cannot read (F32) is refused at load"); }
    { Model m = good_model(); m.find("token_embd.weight")->type = 30; refused(m, "embd_type", "token_embd that is not Q8_0 is refused (embed reads Q8_0 rows)"); }
    { Model m = good_model(); m.find("token_embd.weight")->dims = {64, 4}; refused(m, "embd_cols", "token_embd row width != embedding_length is refused"); }
    { Model m = good_model(); m.find("per_layer_token_embd.weight")->dims = {32, 3}; refused(m, "ple_rows", "per_layer_token_embd with fewer rows than the vocabulary is refused"); }
    { Model m = good_model(); m.tokens.push_back("b"); refused(m, "tok_more", "a tokenizer with more tokens than token_embd rows is refused"); }
    { Model m = good_model(); m.set("gemma4.block_count", 1e12); refused(m, "layers_huge", "block_count 1e12 is refused (not cast to int)"); }
    { Model m = good_model(); m.set("gemma4.block_count", -1); refused(m, "layers_neg", "negative block_count is refused"); }
    { Model m = good_model(); m.set("gemma4.embedding_length", 32.5); refused(m, "d_frac", "a fractional embedding_length is refused"); }
    { Model m = good_model(); m.set("gemma4.attention.head_count", 3); m.set("gemma4.attention.head_count_kv", 2); refused(m, "heads", "head_count not a multiple of head_count_kv is refused"); }
    { Model m = good_model(); m.set("gemma4.attention.head_count_kv", 0); refused(m, "kv_zero", "head_count_kv 0 is refused (no divide by zero)"); }
    { Model m = good_model(); m.set("gemma4.rope.dimension_count", 64); refused(m, "rope_wide", "rope dimension count larger than the head is refused"); }
    { Model m = good_model(); m.set("gemma4.rope.dimension_count_swa", 31); refused(m, "rope_odd", "odd rope dimension count is refused"); }
    { Model m = good_model(); m.set("gemma4.attention.shared_kv_layers", 1); refused(m, "shared_all", "shared_kv_layers == block_count is refused"); }
    { Model m = good_model(); m.swa_pattern = {0, 1}; refused(m, "pattern", "sliding_window_pattern length != block_count is refused"); }
    { Model m = good_model(); m.set("tokenizer.ggml.bos_token_id", 1e10); refused(m, "bos_huge", "bos_token_id 1e10 is refused (not cast to int32)"); }
    { Model m = good_model(); m.set("tokenizer.ggml.bos_token_id", 4); refused(m, "bos_oov", "bos_token_id outside the vocabulary is refused"); }
    { Model m = good_model(); m.set("tokenizer.ggml.padding_token_id", -7); refused(m, "pad_neg", "padding_token_id below -1 is refused"); }

    std::printf("[G4LOAD] %d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}