// Bolt PCHIE - gemma4 (E4B) CPU forward pass, tokenizer and greedy generator (clean-room).
// Spec: Archie's gemma4 (E4B) layer specification, checked against the GGUF metadata with pchie_gguf_dump.
#pragma once
#include "pchie/pchie_gguf.h"
#include "pchie/pchie_kv_pools.h"
#include "pchie/IJobDispatcher.h"
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pchie {

// Read-only memory map of the model file.
class MappedFile {
public:
    bool open(const std::string& path, std::string& err);
    ~MappedFile();
    const uint8_t* data() const noexcept { return base_; }
    uint64_t size() const noexcept { return size_; }
private:
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0;
    void* fh_ = nullptr; void* mh_ = nullptr; int fd_ = -1;
};

// Fixed-size worker pool. parallel_for hands out row chunks; each output element is
// produced by exactly one worker with a fixed inner order, so results never depend
// on the thread count.
class ThreadPool : public IJobDispatcher {
public:
    explicit ThreadPool(int n);
    ~ThreadPool() override;
    void parallel_for(int count, const std::function<void(int)>& fn) override;
    int size() const noexcept override { return n_; }
private:
    struct Impl; std::unique_ptr<Impl> impl_; int n_;
};

struct QTensor {
    const void* data = nullptr;
    uint32_t type = 0;
    uint64_t ne0 = 0, ne1 = 1;           // ne0 = input length (contiguous), ne1 = rows
};

struct Gemma4Config {
    int n_layer = 0, d_model = 0, d_ff = 0, d_ple = 0, n_vocab = 0;
    int n_head = 0, n_head_kv = 0, hd_global = 0, hd_swa = 0, n_rot_global = 0, n_rot_swa = 0;
    int window = 0, shared_kv = 0, n_layer_kv = 0;
    float eps = 1e-6f, rope_base = 0, rope_base_swa = 0, softcap = 0;
    std::vector<bool> is_swa;
    std::vector<int> kv_source;          // layer that owns the KV a layer reads
    std::vector<int> pool_slot;          // owned layer -> slot inside its pool (-1 if shared)
};

struct Gemma4Layer {
    QTensor wq, wk, wv, wo, gate, up, down, inp_gate, proj;
    const float *attn_norm, *q_norm, *k_norm, *post_attn_norm, *ffn_norm, *post_ffw_norm, *post_norm;
    float out_scale;
};

class Gemma4Tokenizer {
public:
    bool load(const GgufFile& g, std::string& err);
    // parse_special=true (default, TRUSTED harness framing only): control tokens like <|turn> are
    // matched literally, as before. parse_special=false (observation contract): for raw /
    // observation / untrusted text; no special is matched and any BPE piece whose id is special is
    // split to byte-fallback tokens, so the result never holds a special id (see is_special()).
    std::vector<int32_t> encode(const std::string& text, bool parse_special = true) const;
    // BOS policy: encode() (trusted), then, iff the GGUF sets
    // tokenizer.ggml.add_bos_token, exactly one `bos` at the front: an explicit leading "<bos>"
    // (e.g. a chat template) is never doubled. Empty text stays empty. Observation contract
    // Observation contract: any `bos` past index 0 (e.g. " <bos>Hi", "<bos><bos>") is dropped, so
    // whitespace or other code points before a literal "<bos>" cannot yield two BOS ids.
    // Bit-identical for every text whose encode() holds no BOS, or one BOS at index 0.
    // encode() itself stays the raw piece encoding (vocab-id lookups, legacy gate tools).
    std::vector<int32_t> encode_with_bos(const std::string& text) const;
    // Observation contract: true for control/unknown/unused ids, '<'-prefixed user-defined
    // specials (the set encode() matches), and bos/eos/unk/pad. Out-of-range ids -> false.
    bool is_special(int32_t id) const noexcept {
        return id >= 0 && size_t(id) < special_mask_.size() && special_mask_[size_t(id)] != 0; }
    size_t special_count() const noexcept;
    size_t vocab_size() const noexcept { return tokens_.size(); }
    const std::string& token_text(int32_t id) const;  // raw vocab string ("" if out of range)
    std::string decode(int32_t id) const;
    int32_t id_of(const std::string& tok) const;
    int32_t bos = 2, eos = 1;
    bool add_bos = false;  // GGUF tokenizer.ggml.add_bos_token (absent => false)
private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> index_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::vector<std::string> specials_;  // longest first
    std::vector<uint8_t> special_mask_;  // per id, 1 = special (observation contract)
    void bpe(const std::string& piece, std::vector<int32_t>& out, bool forbid_special = false) const;
};

class Gemma4Model {
public:
    bool load(const std::string& path, std::string& err);
    const Gemma4Config& cfg() const noexcept { return cfg_; }
    const GgufFile& gguf() const noexcept { return gguf_; }
    const Gemma4Tokenizer& tok() const noexcept { return tok_; }

    struct Branch {                      // one node's sequence state
        KVPagePool::Table swa, glob;
        uint32_t n_past = 0;
    };
    // Make the KV pools (max_ctx tokens per branch, `branches` concurrent branches).
    // keep_swa_history: keep sliding-window pages past the window (results unchanged) so a
    // branch can be re-forked at any earlier position (step-back across forks).
    void init_cache(uint32_t max_ctx, uint32_t branches, bool keep_swa_history = false);
    Branch new_branch();
    Branch fork(const Branch& b);
    void release(Branch& b);
    // New branch sharing b's KV pages (CoW) for positions < n_tokens; continues at n_tokens.
    Branch fork_at(const Branch& b, uint32_t n_tokens);
    // Run one token at position b.n_past; fills logits (n_vocab, soft-capped).
    void forward(Branch& b, int32_t token, std::vector<float>& logits, IJobDispatcher& pool);

    // --- pieces of forward(), shared with the CUDA backend (same math, same order) ---
    void begin_token(Branch& b);                                    // slide the window
    void embed(int32_t token, float* x, float* ple_raw) const;      // host Q8_0 rows: x scaled, PLE unscaled
    void ple_finish(const float* x, float* ple, IJobDispatcher& pool);  // BF16 projection + combine (in place)
    void layer(int l, Branch& b, float* x, const float* ple, IJobDispatcher& pool);
    // Batched forms (B branches, rows contiguous). Same bits per branch as B = 1.
    void forward_batch(Branch* const* bs, const int32_t* toks, int B, std::vector<float>* logits, IJobDispatcher& pool);
    void ple_finish_batch(const float* x, float* ple, int B, IJobDispatcher& pool);
    void layer_batch(int l, Branch* const* bs, int B, float* x, const float* ple, IJobDispatcher& pool);
    void head_batch(const float* x, int B, std::vector<float>* logits, IJobDispatcher& pool);
    void matvec_batch(const QTensor& w, const float* x, int B, float* y, IJobDispatcher& pool);
    void head(const float* x, std::vector<float>& logits, IJobDispatcher& pool); // final norm, tied LM head, softcap
    static void rope_table(int n_rot, float base, const float* ff, uint32_t pos, float* c, float* s);
    void rmsnorm(const float* x, const float* w, float* y, int n) const;

    const Gemma4Layer& layer_weights(int l) const { return layers_[size_t(l)]; }
    const QTensor& ple_proj() const { return ple_proj_; }
    const QTensor& tok_embd() const { return tok_embd_; }
    const float* ple_proj_norm() const { return ple_proj_norm_; }
    const float* out_norm() const { return out_norm_; }
    const float* rope_freqs() const { return rope_freqs_; }
    KVPagePool& pool_swa() { return *pool_swa_; }
    KVPagePool& pool_glob() { return *pool_glob_; }

private:
    void matvec(const QTensor& w, const float* x, float* y, IJobDispatcher& pool);
    void rope(float* v, int heads, int hd, int n_rot, float base, const float* ff, uint32_t pos) const;
    MappedFile map_;
    GgufFile gguf_;
    Gemma4Config cfg_;
    Gemma4Tokenizer tok_;
    std::vector<Gemma4Layer> layers_;
    QTensor tok_embd_, ple_embd_, ple_proj_;
    const float *ple_proj_norm_ = nullptr, *out_norm_ = nullptr, *rope_freqs_ = nullptr;
    std::unique_ptr<KVPagePool> pool_swa_, pool_glob_;
    // scratch
    std::vector<uint8_t> qbuf_;
    bool keep_swa_ = false;
};

} // namespace pchie
