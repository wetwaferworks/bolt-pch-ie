#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_kquants_det.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <chrono>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif
#include <stdexcept>
#include <thread>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace pchie {

// ---------------------------------------------------------------- MappedFile
bool MappedFile::open(const std::string& path, std::string& err) {
#if defined(_WIN32)
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { err = "cannot open " + path; return false; }
    LARGE_INTEGER sz; GetFileSizeEx(f, &sz);
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(f); err = "CreateFileMapping failed"; return false; }
    const void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(m); CloseHandle(f); err = "MapViewOfFile failed"; return false; }
    fh_ = f; mh_ = m; base_ = static_cast<const uint8_t*>(p); size_ = uint64_t(sz.QuadPart);
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { err = "cannot open " + path; return false; }
    struct stat st; fstat(fd_, &st);
    void* p = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) { err = "mmap failed"; return false; }
    base_ = static_cast<const uint8_t*>(p); size_ = uint64_t(st.st_size);
#endif
    return true;
}
MappedFile::~MappedFile() {
#if defined(_WIN32)
    if (base_) UnmapViewOfFile(base_);
    if (mh_) CloseHandle(mh_);
    if (fh_) CloseHandle(fh_);
#else
    if (base_) munmap(const_cast<uint8_t*>(base_), size_t(size_));
    if (fd_ >= 0) ::close(fd_);
#endif
}

// ---------------------------------------------------------------- ThreadPool
// Mutex-free (2026-11-11): phase-boundary fork/join; no mutex, no condvar.
//   FORK edge : the caller (sole owner of the pool) writes fn/count/next/remaining, then publishes a
//               new generation (acq_rel RMW). Workers acquire-load the generation, so everything
//               written before the publish is visible to them.
//   PHASE     : each row index is owned by exactly one thread (fetch_add claim); outputs are disjoint.
//   JOIN edge : each worker does an acq_rel decrement of `remaining`; the caller acquire-loads until 0,
//               so every worker write is visible before parallel_for returns. The next generation is
//               only published after this join, so fn/count are never rewritten while a worker reads.
//   Idle      : spin (pause) -> yield -> 50 us sleep backoff. Nothing is held across any wait.
// Per-row arithmetic is unchanged, so outputs are bit-identical to the old pool.
namespace {
inline void pchie_cpu_relax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}
}  // namespace
struct ThreadPool::Impl {
    std::vector<std::thread> workers;
    const std::function<void(int)>* fn = nullptr;   // written only before the FORK publish
    int count = 0;                                   // written only before the FORK publish
    std::atomic<int> next{0};
    std::atomic<int> remaining{0};                   // JOIN counter (workers still in the phase)
    std::atomic<uint64_t> gen{0};                    // FORK generation
    std::atomic<bool> stop{false};
    void run_items() { for (int i; (i = next.fetch_add(1, std::memory_order_relaxed)) < count;) (*fn)(i); }
};
ThreadPool::ThreadPool(int n) : impl_(new Impl), n_(std::max(1, n)) {
    for (int t = 1; t < n_; ++t)
        impl_->workers.emplace_back([this] {
            uint64_t seen = 0;
            unsigned idle = 0;
            for (;;) {
                if (impl_->stop.load(std::memory_order_acquire)) return;
                const uint64_t g = impl_->gen.load(std::memory_order_acquire);
                if (g == seen) {
                    ++idle;
                    if (idle < 4096) pchie_cpu_relax();
                    else if (idle < 8192) std::this_thread::yield();
                    else std::this_thread::sleep_for(std::chrono::microseconds(50));
                    continue;
                }
                idle = 0;
                seen = g;
                impl_->run_items();
                impl_->remaining.fetch_sub(1, std::memory_order_acq_rel);
            }
        });
}
ThreadPool::~ThreadPool() {
    impl_->stop.store(true, std::memory_order_release);
    for (auto& w : impl_->workers) w.join();
}
void ThreadPool::parallel_for(int count, const std::function<void(int)>& fn) {
    if (n_ == 1 || count == 1) { for (int i = 0; i < count; ++i) fn(i); return; }
    impl_->fn = &fn; impl_->count = count;
    impl_->next.store(0, std::memory_order_relaxed);
    impl_->remaining.store(n_ - 1, std::memory_order_relaxed);
    impl_->gen.fetch_add(1, std::memory_order_acq_rel);             // FORK edge
    impl_->run_items();
    unsigned spins = 0;
    while (impl_->remaining.load(std::memory_order_acquire) != 0) {  // JOIN edge
        if (++spins < 4096) pchie_cpu_relax(); else std::this_thread::yield();
    }
}

// ---------------------------------------------------------------- Tokenizer
bool Gemma4Tokenizer::load(const GgufFile& g, std::string& err) {
    auto it = g.meta_str_arr.find("tokenizer.ggml.tokens");
    if (it == g.meta_str_arr.end()) { err = "no tokenizer.ggml.tokens"; return false; }
    tokens_ = it->second;
    for (size_t i = 0; i < tokens_.size(); ++i) index_.emplace(tokens_[i], int32_t(i));
    auto mt = g.meta_str_arr.find("tokenizer.ggml.merges");
    if (mt == g.meta_str_arr.end()) { err = "no tokenizer.ggml.merges"; return false; }
    merge_rank_.reserve(mt->second.size());
    for (size_t i = 0; i < mt->second.size(); ++i) merge_rank_.emplace(mt->second[i], int32_t(i));
    special_mask_.assign(tokens_.size(), 0);
    auto tt = g.meta_num_arr.find("tokenizer.ggml.token_type");
    if (tt != g.meta_num_arr.end())
        for (size_t i = 0; i < tt->second.size() && i < tokens_.size(); ++i) {
            const int ty = int(tt->second[i]);
            if ((ty == 3 || ty == 4) && tokens_[i].size() > 1 && tokens_[i][0] == '<') specials_.push_back(tokens_[i]);
            // Observation contract: 2 unknown, 3 control, 5 unused, and the
            // '<'-prefixed user-defined set that encode() matches are never emitted raw.
            if (ty == 2 || ty == 3 || ty == 5 || (ty == 4 && tokens_[i].size() > 1 && tokens_[i][0] == '<'))
                special_mask_[i] = 1;
        }
    std::sort(specials_.begin(), specials_.end(), [](const std::string& a, const std::string& b) {
        return a.size() != b.size() ? a.size() > b.size() : a < b; });
    auto num = [&](const char* k, int32_t d) { auto f = g.meta_num.find(k); return f == g.meta_num.end() ? d : int32_t(f->second); };
    bos = num("tokenizer.ggml.bos_token_id", 2); eos = num("tokenizer.ggml.eos_token_id", 1);
    add_bos = num("tokenizer.ggml.add_bos_token", 0) != 0;  // BOS policy
    for (int32_t id : {bos, eos, num("tokenizer.ggml.unknown_token_id", 3), num("tokenizer.ggml.padding_token_id", 0)})
        if (id >= 0 && size_t(id) < special_mask_.size()) special_mask_[size_t(id)] = 1;
    return true;
}
size_t Gemma4Tokenizer::special_count() const noexcept {
    size_t n = 0;
    for (uint8_t m : special_mask_) n += m;
    return n;
}
static const std::string kEmptyTokenText;  // namespace-scope (no function-local static guard)
const std::string& Gemma4Tokenizer::token_text(int32_t id) const {
    return id >= 0 && size_t(id) < tokens_.size() ? tokens_[size_t(id)] : kEmptyTokenText;
}
int32_t Gemma4Tokenizer::id_of(const std::string& t) const { auto f = index_.find(t); return f == index_.end() ? -1 : f->second; }

void Gemma4Tokenizer::bpe(const std::string& piece, std::vector<int32_t>& out, bool forbid_special) const {
    // spaces -> U+2581, split into UTF-8 code points, then lowest-rank merges first.
    std::string s;
    for (char c : piece) { if (c == ' ') s += "\xE2\x96\x81"; else s += c; }
    std::vector<std::string> sym;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        n = std::min(n, s.size() - i);
        sym.push_back(s.substr(i, n)); i += n;
    }
    for (;;) {
        int best = -1; int32_t best_rank = INT32_MAX;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto f = merge_rank_.find(sym[i] + " " + sym[i + 1]);
            if (f != merge_rank_.end() && f->second < best_rank) { best_rank = f->second; best = int(i); }
        }
        if (best < 0) break;
        sym[size_t(best)] += sym[size_t(best) + 1];
        sym.erase(sym.begin() + best + 1);
    }
    for (auto& t : sym) {
        const int32_t id = id_of(t);
        if (id >= 0 && !(forbid_special && is_special(id))) { out.push_back(id); continue; }
        for (unsigned char c : t) {                       // byte fallback <0xXX>
            char buf[8]; std::snprintf(buf, sizeof buf, "<0x%02X>", c);
            const int32_t b = id_of(buf);
            out.push_back(b >= 0 ? b : 3);  // raw path: all 256 byte tokens exist (pinned by test)
        }
    }
}

std::vector<int32_t> Gemma4Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> out;
    if (!parse_special) {                                 // observation contract
        if (!text.empty()) bpe(text, out, /*forbid_special=*/true);
        return out;
    }
    size_t i = 0, start = 0;
    while (i < text.size()) {
        bool hit = false;
        if (text[i] == '<')
            for (const auto& sp : specials_)
                if (text.compare(i, sp.size(), sp) == 0) {
                    if (i > start) bpe(text.substr(start, i - start), out);
                    out.push_back(id_of(sp)); i += sp.size(); start = i; hit = true; break;
                }
        if (!hit) ++i;
    }
    if (start < text.size()) bpe(text.substr(start), out);
    return out;
}

// BOS policy: one leading BOS when the GGUF asks for it; never doubled.
// Observation contract: exactly one BOS, at index 0. Any BOS past index 0 (e.g.
// whitespace before a literal "<bos>", or "<bos><bos>") is dropped before the leading one is
// placed. No change for encode() results with no BOS or with a single BOS at index 0.
std::vector<int32_t> Gemma4Tokenizer::encode_with_bos(const std::string& text) const {
    std::vector<int32_t> out = encode(text);
    if (!add_bos) return out;
    bool had_bos = false;
    size_t w = 0;
    for (size_t r = 0; r < out.size(); ++r) {
        if (out[r] == bos) { had_bos = true; continue; }
        out[w++] = out[r];
    }
    out.resize(w);
    if (had_bos || !out.empty()) out.insert(out.begin(), bos);
    return out;
}

std::string Gemma4Tokenizer::decode(int32_t id) const {
    if (id < 0 || size_t(id) >= tokens_.size()) return "";
    const std::string& t = tokens_[size_t(id)];
    if (t.size() == 6 && t.compare(0, 3, "<0x") == 0 && t[5] == '>') return std::string(1, char(std::stoi(t.substr(3, 2), nullptr, 16)));
    std::string r;
    for (size_t i = 0; i < t.size();) {
        if (t.compare(i, 3, "\xE2\x96\x81") == 0) { r += ' '; i += 3; } else r += t[i++];
    }
    return r;
}

// ---------------------------------------------------------------- Model load
bool Gemma4Model::load(const std::string& path, std::string& err) {
    if (!gguf_.open(path, err)) return false;
    if (gguf_.meta_str["general.architecture"] != "gemma4") { err = "not gemma4"; return false; }
    if (!map_.open(path, err)) return false;
    if (!tok_.load(gguf_, err)) return false;
    auto N = [&](const std::string& k) -> double {
        auto f = gguf_.meta_num.find("gemma4." + k);
        if (f == gguf_.meta_num.end()) throw std::runtime_error("missing gemma4." + k);
        return f->second; };
    try {
        cfg_.n_layer = int(N("block_count")); cfg_.d_model = int(N("embedding_length"));
        cfg_.d_ff = int(N("feed_forward_length")); cfg_.d_ple = int(N("embedding_length_per_layer_input"));
        cfg_.n_head = int(N("attention.head_count")); cfg_.n_head_kv = int(N("attention.head_count_kv"));
        cfg_.hd_global = int(N("attention.key_length")); cfg_.hd_swa = int(N("attention.key_length_swa"));
        cfg_.n_rot_global = int(N("rope.dimension_count")); cfg_.n_rot_swa = int(N("rope.dimension_count_swa"));
        cfg_.window = int(N("attention.sliding_window")); cfg_.shared_kv = int(N("attention.shared_kv_layers"));
        cfg_.eps = float(N("attention.layer_norm_rms_epsilon"));
        cfg_.rope_base = float(N("rope.freq_base")); cfg_.rope_base_swa = float(N("rope.freq_base_swa"));
        cfg_.softcap = float(N("final_logit_softcapping"));
        if (int(N("attention.value_length")) != cfg_.hd_global || int(N("attention.value_length_swa")) != cfg_.hd_swa)
            throw std::runtime_error("value_length != key_length is not supported");
    } catch (const std::exception& e) { err = e.what(); return false; }
    const auto& pat = gguf_.meta_num_arr["gemma4.attention.sliding_window_pattern"];
    if (int(pat.size()) != cfg_.n_layer) { err = "sliding_window_pattern length"; return false; }
    cfg_.is_swa.resize(size_t(cfg_.n_layer));
    for (int l = 0; l < cfg_.n_layer; ++l) cfg_.is_swa[size_t(l)] = pat[size_t(l)] != 0.0;
    cfg_.n_layer_kv = cfg_.n_layer - cfg_.shared_kv;
    cfg_.kv_source.resize(size_t(cfg_.n_layer)); cfg_.pool_slot.assign(size_t(cfg_.n_layer), -1);
    int ns = 0, ng = 0;
    for (int l = 0; l < cfg_.n_layer; ++l) {
        if (l < cfg_.n_layer_kv) { cfg_.kv_source[size_t(l)] = l; cfg_.pool_slot[size_t(l)] = cfg_.is_swa[size_t(l)] ? ns++ : ng++; continue; }
        int src = -1;
        for (int j = cfg_.n_layer_kv - 1; j >= 0; --j) if (cfg_.is_swa[size_t(j)] == cfg_.is_swa[size_t(l)]) { src = j; break; }
        if (src < 0) { err = "no KV source layer"; return false; }
        cfg_.kv_source[size_t(l)] = src;
    }

    const uint8_t* base = map_.data() + gguf_.data_offset;
    auto T = [&](const std::string& name, QTensor& q) {
        const GgufTensorInfo* t = gguf_.find(name);
        if (!t) throw std::runtime_error("missing tensor " + name);
        q.data = base + t->offset; q.type = t->type; q.ne0 = t->dims[0]; q.ne1 = t->dims.size() > 1 ? t->dims[1] : 1;
    };
    auto F = [&](const std::string& name, uint64_t n) -> const float* {
        QTensor q; T(name, q);
        if (q.type != 0 || q.ne0 * q.ne1 != n) throw std::runtime_error("bad f32 tensor " + name);
        return static_cast<const float*>(q.data);
    };
    try {
        T("token_embd.weight", tok_embd_); T("per_layer_token_embd.weight", ple_embd_); T("per_layer_model_proj.weight", ple_proj_);
        cfg_.n_vocab = int(tok_embd_.ne1);
        ple_proj_norm_ = F("per_layer_proj_norm.weight", uint64_t(cfg_.d_ple));
        out_norm_ = F("output_norm.weight", uint64_t(cfg_.d_model));
        rope_freqs_ = F("rope_freqs.weight", uint64_t(cfg_.n_rot_global / 2));
        layers_.resize(size_t(cfg_.n_layer));
        for (int l = 0; l < cfg_.n_layer; ++l) {
            auto& L = layers_[size_t(l)];
            const std::string p = "blk." + std::to_string(l) + ".";
            const uint64_t hd = uint64_t(cfg_.is_swa[size_t(l)] ? cfg_.hd_swa : cfg_.hd_global);
            T(p + "attn_q.weight", L.wq); T(p + "attn_output.weight", L.wo);
            if (l < cfg_.n_layer_kv) { T(p + "attn_k.weight", L.wk); T(p + "attn_v.weight", L.wv); }
            T(p + "ffn_gate.weight", L.gate); T(p + "ffn_up.weight", L.up); T(p + "ffn_down.weight", L.down);
            T(p + "inp_gate.weight", L.inp_gate); T(p + "proj.weight", L.proj);
            const uint64_t D = uint64_t(cfg_.d_model);
            L.attn_norm = F(p + "attn_norm.weight", D); L.q_norm = F(p + "attn_q_norm.weight", hd);
            L.k_norm = l < cfg_.n_layer_kv ? F(p + "attn_k_norm.weight", hd) : nullptr;
            L.post_attn_norm = F(p + "post_attention_norm.weight", D); L.ffn_norm = F(p + "ffn_norm.weight", D);
            L.post_ffw_norm = F(p + "post_ffw_norm.weight", D); L.post_norm = F(p + "post_norm.weight", D);
            L.out_scale = F(p + "layer_output_scale.weight", 1)[0];
            if (L.wq.ne1 != uint64_t(cfg_.n_head) * hd) throw std::runtime_error("attn_q shape " + p);
        }
    } catch (const std::exception& e) { err = e.what(); return false; }
    return true;
}

// ---------------------------------------------------------------- KV branches
void Gemma4Model::init_cache(uint32_t max_ctx, uint32_t branches, bool keep_swa_history) {
    keep_swa_ = keep_swa_history;
    int ns = 0, ng = 0;
    for (int l = 0; l < cfg_.n_layer_kv; ++l) (cfg_.is_swa[size_t(l)] ? ns : ng)++;
    const uint32_t swa_pages = keep_swa_ ? (max_ctx + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS
                                         : (uint32_t(cfg_.window) + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS + 1;
    const uint32_t glob_pages = (max_ctx + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS;
    pool_swa_.reset(new KVPagePool(uint32_t(ns), uint32_t(cfg_.n_head_kv), uint32_t(cfg_.hd_swa), (swa_pages + 1) * branches + 1));
    pool_glob_.reset(new KVPagePool(uint32_t(ng), uint32_t(cfg_.n_head_kv), uint32_t(cfg_.hd_global), (glob_pages + 1) * branches + 1));
}
Gemma4Model::Branch Gemma4Model::new_branch() { return Branch{}; }
Gemma4Model::Branch Gemma4Model::fork(const Branch& b) {
    Branch n; n.swa = pool_swa_->fork(b.swa); n.glob = pool_glob_->fork(b.glob); n.n_past = b.n_past; return n;
}
void Gemma4Model::release(Branch& b) { pool_swa_->release(b.swa); pool_glob_->release(b.glob); b.n_past = 0; }

// ---------------------------------------------------------------- math helpers
void Gemma4Model::rmsnorm(const float* x, const float* w, float* y, int n) const {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) { const float p = x[i] * x[i]; ss = ss + p; }
    const float mean = ss / float(n);
    const float inv = 1.0f / std::sqrt(mean + cfg_.eps);
    for (int i = 0; i < n; ++i) y[i] = w ? (x[i] * inv) * w[i] : x[i] * inv;
}

// NeoX rotation over n_rot dims: pairs (i, i + n_rot/2), theta_i = pos * base^(-2i/n_rot) / ff[i].
void Gemma4Model::rope_table(int n_rot, float base, const float* ff, uint32_t pos, float* c, float* s) {
    for (int i = 0; i < n_rot / 2; ++i) {
        double th = double(pos) * std::pow(double(base), -2.0 * double(i) / double(n_rot));
        if (ff) th /= double(ff[i]);
        c[i] = float(std::cos(th)); s[i] = float(std::sin(th));
    }
}
void Gemma4Model::rope(float* v, int heads, int hd, int n_rot, float base, const float* ff, uint32_t pos) const {
    const int half = n_rot / 2;
    float c[512], s[512];
    rope_table(n_rot, base, ff, pos, c, s);
    for (int h = 0; h < heads; ++h)
        for (int i = 0; i < half; ++i) {
            float* x = v + size_t(h) * size_t(hd);
            const float x0 = x[i], x1 = x[i + half];
            const float a = x0 * c[i], b = x1 * s[i], d = x0 * s[i], e = x1 * c[i];
            x[i] = a - b; x[i + half] = d + e;
        }
}

void Gemma4Model::matvec(const QTensor& w, const float* x, float* y, IJobDispatcher& pool) {
    const int n = int(w.ne0), rows = int(w.ne1);
    constexpr int CH = 16;
    const int chunks = (rows + CH - 1) / CH;
    std::vector<uint8_t> q;
    if (w.type == 12 || w.type == 14) {
        q.resize(sizeof(BlockQ8_K) * size_t(n / QK_K));
        auto* xq = reinterpret_cast<BlockQ8_K*>(q.data());
        quantize_row_q8_K_deterministic(x, xq, n);
        const int nb = n / QK_K;
        if (w.type == 12) {
            const auto* W = static_cast<const BlockQ4_K*>(w.data);
            pool.parallel_for(chunks, [&](int c) { for (int r = c * CH; r < std::min(rows, (c + 1) * CH); ++r) y[r] = dot_q4_K_q8_K<true>(W + size_t(r) * nb, xq, nb); });
        } else {
            const auto* W = static_cast<const BlockQ6_K*>(w.data);
            pool.parallel_for(chunks, [&](int c) { for (int r = c * CH; r < std::min(rows, (c + 1) * CH); ++r) y[r] = dot_q6_K_q8_K<true>(W + size_t(r) * nb, xq, nb); });
        }
    } else if (w.type == 8) {
        q.resize(sizeof(BlockQ8_0) * size_t(n / 32));
        auto* xq = reinterpret_cast<BlockQ8_0*>(q.data());
        quantize_row_q8_0_deterministic(x, xq, n);
        const int nb = n / 32;
        const auto* W = static_cast<const BlockQ8_0W*>(w.data);
        pool.parallel_for(chunks, [&](int c) { for (int r = c * CH; r < std::min(rows, (c + 1) * CH); ++r) y[r] = dot_q8_0_q8_0<true>(W + size_t(r) * nb, xq, nb); });
    } else if (w.type == 30) {
        const auto* W = static_cast<const uint16_t*>(w.data);
        pool.parallel_for(chunks, [&](int c) { for (int r = c * CH; r < std::min(rows, (c + 1) * CH); ++r) y[r] = dot_bf16_f32(W + size_t(r) * n, x, n); });
    } else {
        throw std::runtime_error(std::string("matvec: unsupported type ") + ggml_type_name(w.type));
    }
}

// ---------------------------------------------------------------- forward
void Gemma4Model::begin_token(Branch& br) {
    const uint32_t t = br.n_past;
    if (!keep_swa_ && t >= uint32_t(cfg_.window)) pool_swa_->trim_before(br.swa, t - uint32_t(cfg_.window) + 1);
}

void Gemma4Model::embed(int32_t token, float* x, float* ple_raw) const {
    const int D = cfg_.d_model, P = cfg_.d_ple, NL = cfg_.n_layer;
    dequant_q8_0_row(static_cast<const BlockQ8_0W*>(tok_embd_.data) + size_t(token) * size_t(D / 32), x, D);
    const float emb_scale = std::sqrt(float(D));
    for (int i = 0; i < D; ++i) x[i] = x[i] * emb_scale;
    dequant_q8_0_row(static_cast<const BlockQ8_0W*>(ple_embd_.data) + size_t(token) * size_t(NL * P / 32), ple_raw, NL * P);
}

// One weight matrix against B activation rows (x stride n, y stride rows). Each output
// is the same dot as matvec() computes; the row loop is shared so weights are read once.
void Gemma4Model::matvec_batch(const QTensor& w, const float* x, int B, float* y, IJobDispatcher& pool) {
    if (B == 1) { matvec(w, x, y, pool); return; }
    const int n = int(w.ne0), rows = int(w.ne1);
    constexpr int CH = 16;
    const int chunks = (rows + CH - 1) / CH;
    std::vector<uint8_t> q;
    auto rows_loop = [&](auto&& dot) {
        pool.parallel_for(chunks, [&](int c) {
            for (int r = c * CH; r < std::min(rows, (c + 1) * CH); ++r)
                for (int b = 0; b < B; ++b) y[size_t(b) * rows + r] = dot(r, b);
        });
    };
    if (w.type == 12 || w.type == 14) {
        const int nb = n / QK_K;
        q.resize(sizeof(BlockQ8_K) * size_t(nb) * B);
        auto* xq = reinterpret_cast<BlockQ8_K*>(q.data());
        for (int b = 0; b < B; ++b) quantize_row_q8_K_deterministic(x + size_t(b) * n, xq + size_t(b) * nb, n);
        if (w.type == 12) { const auto* W = static_cast<const BlockQ4_K*>(w.data); rows_loop([&](int r, int b) { return dot_q4_K_q8_K<true>(W + size_t(r) * nb, xq + size_t(b) * nb, nb); }); }
        else              { const auto* W = static_cast<const BlockQ6_K*>(w.data); rows_loop([&](int r, int b) { return dot_q6_K_q8_K<true>(W + size_t(r) * nb, xq + size_t(b) * nb, nb); }); }
    } else if (w.type == 8) {
        const int nb = n / 32;
        q.resize(sizeof(BlockQ8_0) * size_t(nb) * B);
        auto* xq = reinterpret_cast<BlockQ8_0*>(q.data());
        for (int b = 0; b < B; ++b) quantize_row_q8_0_deterministic(x + size_t(b) * n, xq + size_t(b) * nb, n);
        const auto* W = static_cast<const BlockQ8_0W*>(w.data);
        rows_loop([&](int r, int b) { return dot_q8_0_q8_0<true>(W + size_t(r) * nb, xq + size_t(b) * nb, nb); });
    } else if (w.type == 30) {
        const auto* W = static_cast<const uint16_t*>(w.data);
        rows_loop([&](int r, int b) { return dot_bf16_f32(W + size_t(r) * n, x + size_t(b) * n, n); });
    } else {
        throw std::runtime_error(std::string("matvec_batch: unsupported type ") + ggml_type_name(w.type));
    }
}

void Gemma4Model::ple_finish(const float* x, float* ple, IJobDispatcher& pool) { ple_finish_batch(x, ple, 1, pool); }
void Gemma4Model::ple_finish_batch(const float* x, float* ple, int B, IJobDispatcher& pool) {
    const int D = cfg_.d_model, P = cfg_.d_ple, NL = cfg_.n_layer;
    std::vector<float> ctx(static_cast<size_t>(NL) * P * B);
    matvec_batch(ple_proj_, x, B, ctx.data(), pool);
    const float ple_scale = std::sqrt(float(P));
    const float proj_scale = 1.0f / std::sqrt(float(D)), comb = std::sqrt(0.5f);
    for (int s = 0; s < NL * B; ++s) {
        float* c = ctx.data() + size_t(s) * P;
        for (int i = 0; i < P; ++i) c[i] = c[i] * proj_scale;
        rmsnorm(c, ple_proj_norm_, c, P);
        float* p = ple + size_t(s) * P;
        for (int i = 0; i < P; ++i) { const float a = p[i] * ple_scale; p[i] = (a + c[i]) * comb; }
    }
}

void Gemma4Model::layer(int l, Branch& br, float* x, const float* ple, IJobDispatcher& pool) {
    Branch* bs[1] = {&br};
    layer_batch(l, bs, 1, x, ple, pool);
}

// Layer l for B branches (x: B x D, ple: B x NL x P). Every value is computed by the
// same op sequence as for B = 1, so a fork gives the same bits alone or batched.
void Gemma4Model::layer_batch(int l, Branch* const* bs, int B, float* X, const float* PLE, IJobDispatcher& pool) {
    const Gemma4Config& C = cfg_;
    const int D = C.d_model, P = C.d_ple, NL = C.n_layer, HQ = C.n_head, HKV = C.n_head_kv, FF = C.d_ff;
    const Gemma4Layer& L = layers_[size_t(l)];
    const bool swa = C.is_swa[size_t(l)];
    const int hd = swa ? C.hd_swa : C.hd_global;
    const int n_rot = swa ? C.n_rot_swa : C.n_rot_global;
    const float base = swa ? C.rope_base_swa : C.rope_base;
    const float* ff = swa ? nullptr : rope_freqs_;
    KVPagePool& kvp = swa ? *pool_swa_ : *pool_glob_;
    const int src = C.kv_source[size_t(l)];
    const uint32_t slot = uint32_t(C.pool_slot[size_t(src)]);
    const size_t QS = size_t(HQ) * hd, KS = size_t(HKV) * hd, BD = size_t(B) * D;
    std::vector<float> u(BD), tmp(BD), pg(static_cast<size_t>(B) * P), pg2(BD);
    std::vector<float> q(QS * B), k(KS * B), v(KS * B), att(QS * B);
    std::vector<float> g(static_cast<size_t>(B) * FF), up(static_cast<size_t>(B) * FF);

    // --- attention
    for (int b = 0; b < B; ++b) rmsnorm(X + size_t(b) * D, L.attn_norm, &u[size_t(b) * D], D);
    matvec_batch(L.wq, u.data(), B, q.data(), pool);
    for (int b = 0; b < B; ++b) {
        float* qb = &q[QS * b];
        for (int h = 0; h < HQ; ++h) rmsnorm(qb + size_t(h) * hd, L.q_norm, qb + size_t(h) * hd, hd);
        rope(qb, HQ, hd, n_rot, base, ff, bs[b]->n_past);
    }
    if (l < C.n_layer_kv) {
        matvec_batch(L.wk, u.data(), B, k.data(), pool);
        matvec_batch(L.wv, u.data(), B, v.data(), pool);
        for (int b = 0; b < B; ++b) {
            float* kb = &k[KS * b]; float* vb = &v[KS * b];
            const uint32_t t = bs[b]->n_past;
            KVPagePool::Table& tab = swa ? bs[b]->swa : bs[b]->glob;
            for (int h = 0; h < HKV; ++h) {
                rmsnorm(kb + size_t(h) * hd, L.k_norm, kb + size_t(h) * hd, hd);
                rmsnorm(vb + size_t(h) * hd, nullptr, vb + size_t(h) * hd, hd);
            }
            rope(kb, HKV, hd, n_rot, base, ff, t);
            std::memcpy(kvp.k_write(tab, slot, t), kb, sizeof(float) * KS);
            std::memcpy(kvp.v_write(tab, slot, t), vb, sizeof(float) * KS);
        }
    }
    const int group = HQ / HKV;
    pool.parallel_for(B * HQ, [&](int bh) {
        const int b = bh / HQ, h = bh % HQ, kh = h / group;
        const uint32_t t = bs[b]->n_past;
        const KVPagePool::Table& tab = swa ? bs[b]->swa : bs[b]->glob;
        const uint32_t j0 = (swa && t + 1 > uint32_t(C.window)) ? t + 1 - uint32_t(C.window) : 0;
        const float* qh = &q[QS * b + size_t(h) * hd];
        std::vector<float> sc(size_t(t - j0 + 1));
        float mx = -INFINITY;
        for (uint32_t j = j0; j <= t; ++j) {
            const float* kj = kvp.k_read(tab, slot, j) + size_t(kh) * hd;
            float s = 0.0f;
            for (int i = 0; i < hd; ++i) { const float p = qh[i] * kj[i]; s = s + p; }
            sc[j - j0] = s; mx = std::max(mx, s);
        }
        float den = 0.0f;
        for (auto& s : sc) { s = exp_deterministic(s - mx); den = den + s; }
        float* o = &att[QS * b + size_t(h) * hd];
        for (int i = 0; i < hd; ++i) o[i] = 0.0f;
        for (uint32_t j = j0; j <= t; ++j) {
            const float* vj = kvp.v_read(tab, slot, j) + size_t(kh) * hd;
            const float p = sc[j - j0] / den;
            for (int i = 0; i < hd; ++i) { const float m = p * vj[i]; o[i] = o[i] + m; }
        }
    });
    matvec_batch(L.wo, att.data(), B, tmp.data(), pool);
    for (int b = 0; b < B; ++b) {
        float* x = X + size_t(b) * D; float* tb = &tmp[size_t(b) * D];
        rmsnorm(tb, L.post_attn_norm, tb, D);
        for (int i = 0; i < D; ++i) x[i] = x[i] + tb[i];
        rmsnorm(x, L.ffn_norm, &u[size_t(b) * D], D);
    }
    // --- GeGLU FFN
    matvec_batch(L.gate, u.data(), B, g.data(), pool);
    matvec_batch(L.up, u.data(), B, up.data(), pool);
    for (size_t i = 0; i < g.size(); ++i) g[i] = gelu_tanh_deterministic(g[i]) * up[i];
    matvec_batch(L.down, g.data(), B, tmp.data(), pool);
    for (int b = 0; b < B; ++b) {
        float* x = X + size_t(b) * D; float* tb = &tmp[size_t(b) * D];
        rmsnorm(tb, L.post_ffw_norm, tb, D);
        for (int i = 0; i < D; ++i) x[i] = x[i] + tb[i];
    }
    // --- PLE gate
    matvec_batch(L.inp_gate, X, B, pg.data(), pool);
    for (int b = 0; b < B; ++b) {
        const float* pl = PLE + (size_t(b) * NL + size_t(l)) * P;
        float* pb = &pg[size_t(b) * P];
        for (int i = 0; i < P; ++i) pb[i] = gelu_tanh_deterministic(pb[i]) * pl[i];
    }
    matvec_batch(L.proj, pg.data(), B, pg2.data(), pool);
    for (int b = 0; b < B; ++b) {
        float* x = X + size_t(b) * D; float* p2 = &pg2[size_t(b) * D];
        rmsnorm(p2, L.post_norm, p2, D);
        for (int i = 0; i < D; ++i) x[i] = (x[i] + p2[i]) * L.out_scale;
    }
}

void Gemma4Model::head(const float* x, std::vector<float>& logits, IJobDispatcher& pool) { head_batch(x, 1, &logits, pool); }
void Gemma4Model::head_batch(const float* X, int B, std::vector<float>* logits, IJobDispatcher& pool) {
    const int D = cfg_.d_model, V = cfg_.n_vocab;
    std::vector<float> u(static_cast<size_t>(D) * B), y(static_cast<size_t>(V) * B);
    for (int b = 0; b < B; ++b) rmsnorm(X + size_t(b) * D, out_norm_, &u[size_t(b) * D], D);
    matvec_batch(tok_embd_, u.data(), B, y.data(), pool);
    const float cap = cfg_.softcap, icap = 1.0f / cfg_.softcap;
    for (int b = 0; b < B; ++b) {
        logits[b].resize(size_t(V));
        for (int i = 0; i < V; ++i) logits[b][size_t(i)] = cap * tanh_deterministic(y[size_t(b) * V + i] * icap);
    }
}

void Gemma4Model::forward(Branch& br, int32_t token, std::vector<float>& logits, IJobDispatcher& pool) {
    Branch* bs[1] = {&br};
    forward_batch(bs, &token, 1, &logits, pool);
}

void Gemma4Model::forward_batch(Branch* const* bs, const int32_t* toks, int B, std::vector<float>* logits, IJobDispatcher& pool) {
    const int D = cfg_.d_model, NP = cfg_.n_layer * cfg_.d_ple;
    std::vector<float> x(static_cast<size_t>(D) * B), ple(static_cast<size_t>(NP) * B);
    for (int b = 0; b < B; ++b) { begin_token(*bs[b]); embed(toks[b], &x[size_t(b) * D], &ple[size_t(b) * NP]); }
    ple_finish_batch(x.data(), ple.data(), B, pool);
    for (int l = 0; l < cfg_.n_layer; ++l) layer_batch(l, bs, B, x.data(), ple.data(), pool);
    for (int b = 0; b < B; ++b) bs[b]->n_past++;
    head_batch(x.data(), B, logits, pool);
}

Gemma4Model::Branch Gemma4Model::fork_at(const Branch& b, uint32_t n_tokens) {
    Branch n = fork(b);
    n.n_past = n_tokens;
    pool_swa_->truncate(n.swa, n_tokens);
    pool_glob_->truncate(n.glob, n_tokens);
    return n;
}

} // namespace pchie
