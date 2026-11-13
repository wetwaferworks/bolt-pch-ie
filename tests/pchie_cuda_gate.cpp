// CUDA gates: CUDA backend vs CPU path for the gemma4 forward pass.
//   Tier 1: two CUDA runs give the same run hash.
//   Tier 2: CUDA and CPU pick the same greedy tokens, the top-8 logit ranks match at
//           every step, and the tool counts logits whose bits differ (bonus: zero = bit-exact).
//   pchie_cuda_gate <model.gguf> <threads> [max_new] [gpu_layers|-1] [gpu_head 0|1] [reserve_mb] [prompt]
#include "pchie/pchie_cuda.h"
#include "pchie/pchie_tree.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
using namespace pchie;

struct Run { std::vector<int32_t> toks; std::vector<std::vector<float>> logits; uint64_t hash = 0; double pp_s = 0, tg_s = 0; int produced = 0; };

template <class F>
static Run run(Gemma4Model& m, const std::vector<int32_t>& ids, int max_new, F&& fwd) {
    Run r; auto br = m.new_branch(); std::vector<float> lg; uint64_t h = FNV_OFFSET;
    auto fold = [&](const std::vector<float>& v) { for (float f : v) { uint32_t u; std::memcpy(&u, &f, 4); h = hash_mix64(h, u); } r.logits.push_back(v); };
    const int32_t stop = m.tok().id_of("<turn|>");
    auto t0 = std::chrono::steady_clock::now();
    for (auto id : ids) fwd(br, id, lg);
    auto t1 = std::chrono::steady_clock::now();
    fold(lg);
    for (int s = 0; s < max_new; ++s) {
        int32_t best = 0;
        for (int32_t v = 1; v < int32_t(lg.size()); ++v) if (lg[size_t(v)] > lg[size_t(best)]) best = v;
        h = hash_mix64(h, uint64_t(best)); r.toks.push_back(best);
        if (best == m.tok().eos || best == stop) break;
        r.produced++;
        fwd(br, best, lg); fold(lg);
    }
    auto t2 = std::chrono::steady_clock::now();
    r.pp_s = std::chrono::duration<double>(t1 - t0).count(); r.tg_s = std::chrono::duration<double>(t2 - t1).count();
    r.hash = h; m.release(br);
    return r;
}

static std::vector<int32_t> top8(const std::vector<float>& v) {
    std::vector<int32_t> idx(v.size()); for (size_t i = 0; i < v.size(); ++i) idx[i] = int32_t(i);
    std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(), [&](int32_t a, int32_t b) { return v[size_t(a)] > v[size_t(b)] || (v[size_t(a)] == v[size_t(b)] && a < b); });
    idx.resize(8); return idx;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: pchie_cuda_gate <model.gguf> <threads> [max_new] [gpu_layers] [gpu_head] [reserve_mb] [prompt]\n"); return 2; }
    const int threads = std::atoi(argv[2]);
    const int max_new = argc > 3 ? std::atoi(argv[3]) : 48;
    const int gl = argc > 4 ? std::atoi(argv[4]) : -1;
    const bool gh = argc > 5 && std::atoi(argv[5]) != 0;
    const size_t reserve = argc > 6 ? size_t(std::atoll(argv[6])) : 256;
    const std::string user = argc > 7 ? argv[7] : "In two sentences, explain why the sky is blue.";
    Gemma4Model m; std::string err;
    if (!m.load(argv[1], err)) { std::printf("LOAD ERROR: %s\n", err.c_str()); return 1; }
    std::vector<int32_t> ids = m.tok().encode("<bos><|turn>user\n" + user + "<turn|>\n<|turn>model\n");
    m.init_cache(4096, 2);
    ThreadPool pool(threads);
    CudaInfo ci; if (!cuda_query(ci, err)) { std::printf("CUDA ERROR: %s\n", err.c_str()); return 1; }
    std::printf("GPU %s sm_%d free %zu MiB / %zu MiB before init\n", ci.name.c_str(), ci.sm, ci.free_bytes >> 20, ci.total_bytes >> 20);
    CudaBackend cb;
    if (!cb.init(m, gl, reserve, gh, err)) { std::printf("CUDA INIT ERROR: %s\n", err.c_str()); return 1; }
    CudaInfo c2; cuda_query(c2, err);
    std::printf("CUDA backend: %d/%d layers on GPU, gpu_lm_head=%d, allocated %.1f MiB (free after %zu MiB)\n",
                cb.gpu_layers(), m.cfg().n_layer, int(gh), cb.vram_bytes() / 1048576.0, c2.free_bytes >> 20);

    auto cpu = [&](Gemma4Model::Branch& b, int32_t t, std::vector<float>& lg) { m.forward(b, t, lg, pool); };
    auto gpu = [&](Gemma4Model::Branch& b, int32_t t, std::vector<float>& lg) { cb.forward(b, t, lg, pool); };
    Run g1 = run(m, ids, max_new, gpu);
    Run g2 = run(m, ids, max_new, gpu);
    Run c = run(m, ids, max_new, cpu);
    auto rep = [&](const char* n, const Run& r) {
        std::printf("%-6s hash=0x%016llX prompt %.2f tok/s  gen %d tok %.2f tok/s\n", n, (unsigned long long)r.hash,
                    ids.size() / r.pp_s, r.produced, r.produced / (r.tg_s > 0 ? r.tg_s : 1)); };
    rep("cuda#1", g1); rep("cuda#2", g2); rep("cpu", c);
    std::string text; for (auto t : g1.toks) text += m.tok().decode(t);
    std::printf("cuda text: %s\n", text.c_str());

    const bool t1 = g1.hash == g2.hash;
    const bool same_toks = g1.toks == c.toks;
    size_t steps = std::min(g1.logits.size(), c.logits.size()), diff_vals = 0, total = 0; int rank_ok = 0; uint32_t max_ulp = 0; int max_xor_bits = 0;
    for (size_t s = 0; s < steps; ++s) {
        const auto &a = g1.logits[s], &b = c.logits[s];
        for (size_t i = 0; i < a.size(); ++i) {
            uint32_t x, y; std::memcpy(&x, &a[i], 4); std::memcpy(&y, &b[i], 4); ++total;
            if (x != y) { ++diff_vals; const uint32_t u = x > y ? x - y : y - x; max_ulp = std::max(max_ulp, u);
                          int bits = 0; for (uint32_t z = x ^ y; z; z >>= 1) bits += int(z & 1u); max_xor_bits = std::max(max_xor_bits, bits); }
        }
        if (top8(a) == top8(b)) ++rank_ok;
    }
    std::printf("[PCHIE GATE] step4 tier1 cuda-vs-cuda hash %s (0x%016llX)\n", t1 ? "IDENTICAL" : "DIFFERENT", (unsigned long long)g1.hash);
    std::printf("[PCHIE GATE] step4 tier2 tokens %s (%zu vs %zu), top-8 rank match %d/%zu steps\n", same_toks ? "MATCH" : "DIFFER", g1.toks.size(), c.toks.size(), rank_ok, steps);
    std::printf("[PCHIE GATE] step4 logit bits: %zu of %zu logits differ (max ulp %u, max differing bits per value %d)\n", diff_vals, total, max_ulp, max_xor_bits);
    std::printf("[PCHIE GATE] step4 bit-exact CPU==CUDA: %s\n", (g1.hash == c.hash && diff_vals == 0) ? "YES" : "NO");
    return (t1 && same_toks && rank_ok == int(steps)) ? 0 : 1;
}
