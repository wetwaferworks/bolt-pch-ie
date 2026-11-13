// Fork-batching gates: fork batching over the variation tree, CPU and CUDA.
//   * F = 1/2/4/8 forks from one prompt, each run alone and in one batch (sorted by fork_id,
//     KV pages shared copy-on-write from the prompt). Per-fork hashes must match alone vs
//     batched, and CPU == CUDA.
//   * Step-back / step-forward: walk parent_node_id back from a fork head (also across forks,
//     through the shared root), re-fork the KV at that node and regenerate from the stored
//     step_seed; the tokens and logit digests must equal the recorded ones.
//   pchie_fork_batching <model.gguf> <threads> [steps] [gpu_layers|-1] [reserve_mb]
//   pchie_fork_batching <model.gguf> <threads> <steps> <gpu_layers> <reserve_mb> repeat <runs> <cpu|cuda>
//     Repeat test: prefill once, then <runs> identical F=8 batched fork runs; prints one run hash
//     per run (all per-fork hashes + tree digest) and the unique-hash count. Pass = 1 unique hash.
#include "pchie/pchie_cuda.h"
#include "pchie/pchie_det_math.h"
#include "pchie/pchie_rollout.h"
#include "pchie/pchie_tree.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
using namespace pchie;
using Br = Gemma4Model::Branch;
using Fwd = std::function<void(Br* const*, const int32_t*, int, std::vector<float>*)>;

static uint64_t logit_digest(const std::vector<float>& lg) {
    uint64_t h = FNV_OFFSET; for (float f : lg) { uint32_t u; std::memcpy(&u, &f, 4); h = hash_mix64(h, u); } return h;
}
// Deterministic top-40 / temperature 0.8 sampler; `state` is the step_seed and advances.
// Exact logit ties break by lowest token ID. Sampling runs on the host for both backends, so CPU and CUDA share this one sort.
static int32_t sample(const std::vector<float>& lg, uint64_t& state) {
    constexpr int K = 40; const float inv_t = 1.0f / 0.8f;
    std::vector<int32_t> idx(lg.size()); for (size_t i = 0; i < lg.size(); ++i) idx[i] = int32_t(i);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) { return lg[size_t(a)] > lg[size_t(b)] || (lg[size_t(a)] == lg[size_t(b)] && a < b); });
    float p[K]; float sum = 0.0f; const float mx = lg[size_t(idx[0])];
    for (int i = 0; i < K; ++i) { const float z = (lg[size_t(idx[size_t(i)])] - mx) * inv_t; p[i] = exp_deterministic(z); sum = sum + p[i]; }
    const float u = float(splitmix64(state) >> 40) * (1.0f / 16777216.0f) * sum;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) { acc = acc + p[i]; if (u < acc) return idx[size_t(i)]; }
    return idx[K - 1];
}

// Archie search ruling: each sample's PRNG seed is a pure function of logical tree coordinates,
// never of fork creation order or node allocation order:
//   seed = H(root_seed, parent_seq_pos, fork_slot, token_step)
// parent_seq_pos = sequence position of the token whose logits are sampled; fork_slot = the fork's
// slot index among the root's children; token_step = tokens already generated on this fork.
static uint64_t fork_seed(uint64_t root_seed, uint32_t parent_seq_pos, uint32_t fork_slot, uint32_t token_step) {
    return hash_mix64(hash_mix64(hash_mix64(hash_mix64(root_seed, 0x5EEDF0CC5EEDF0CCULL), parent_seq_pos), fork_slot), token_step);
}

struct ForkRun { std::map<uint32_t, uint64_t> hash; std::map<uint32_t, std::vector<int32_t>> toks; double gen_s = 0; };

struct Session {
    Gemma4Model& m; Fwd fwd; Br root; std::vector<float> root_logits; uint64_t base_seed; uint32_t root_pos;
    VariationTree tree; std::map<uint32_t, uint64_t> node_digest;   // node -> digest of the logits it was sampled from
    std::map<uint32_t, Br> live;                                     // fork_id -> branch (kept for step-back)
    std::map<uint32_t, uint32_t> slot;                               // fork_id -> fork slot index at the root

    // One step for a set of forks (sorted by fork_id). Each fork: feed its head token, sample the next.
    void step(std::vector<uint32_t>& fids, std::map<uint32_t, uint64_t>& seed, std::map<uint32_t, int32_t>& cur,
              ForkRun& r, std::vector<std::vector<float>>& lg) {
        std::vector<Br*> bs; std::vector<int32_t> tk;
        for (uint32_t f : fids) { bs.push_back(&live[f]); tk.push_back(cur[f]); }
        lg.resize(fids.size());
        fwd(bs.data(), tk.data(), int(bs.size()), lg.data());
        for (size_t i = 0; i < fids.size(); ++i) advance(fids[i], lg[i], seed, cur, r);
    }
    void advance(uint32_t f, const std::vector<float>& lg, std::map<uint32_t, uint64_t>& seed, std::map<uint32_t, int32_t>& cur, ForkRun& r) {
        (void)seed;
        const uint32_t k = uint32_t(r.toks[f].size());
        const uint64_t s = fork_seed(base_seed, root_pos + k, slot[f], k);
        uint64_t st = s;
        const uint64_t d = logit_digest(lg);
        const int32_t t = sample(lg, st);
        uint64_t& h = r.hash[f]; h = hash_mix64(hash_mix64(hash_mix64(h, d), uint64_t(t)), s);
        r.toks[f].push_back(t); cur[f] = t;
        tree.active_fork_id = f; tree.cursor_node_id = tree.forks[f].head_node_id;
        const uint32_t n = tree.append_generated_token(tree.nodes[tree.cursor_node_id].handshake_tick + 1, uint32_t(t), s, d, {});
        node_digest[n] = d;
    }
    // Make F forks at the root; fork_id -> slot index i is recorded for seed derivation.
    std::vector<uint32_t> make_forks(int F) {
        tree.init_root(root_pos, 0, base_seed); tree.nodes[0].seq_pos = root_pos; node_digest.clear();
        for (auto& kv : live) m.release(kv.second);
        live.clear();
        std::vector<uint32_t> fids;
        slot.clear(); for (int i = 0; i < F; ++i) { tree.cursor_node_id = 0; tree.active_fork_id = 0; const uint32_t f = tree.fork_at(root_pos); fids.push_back(f); slot[f] = uint32_t(i); live[f] = m.fork(root); }
        std::sort(fids.begin(), fids.end());
        return fids;
    }
    ForkRun run(int F, int steps, bool batched) {
        ForkRun r; std::vector<uint32_t> fids = make_forks(F);
        std::map<uint32_t, uint64_t> seed; std::map<uint32_t, int32_t> cur;
        for (uint32_t f : fids) { seed[f] = 0; r.hash[f] = FNV_OFFSET; advance(f, root_logits, seed, cur, r); }
        std::vector<std::vector<float>> lg;
        auto t0 = std::chrono::steady_clock::now();
        if (batched) { for (int s = 1; s < steps; ++s) step(fids, seed, cur, r, lg); }
        else for (uint32_t f : fids) { std::vector<uint32_t> one{f}; for (int s = 1; s < steps; ++s) step(one, seed, cur, r, lg); }
        r.gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return r;
    }
    // Regenerate `count` children of node n on fork f from n's KV and step_seed; compare with the record.
    bool regen_check(uint32_t f, uint32_t n, int count, int& matched) {
        const TokenRecord& nd = tree.nodes[n];
        Br b = (n == 0) ? m.fork(root) : m.fork_at(live[f], nd.seq_pos);   // KV for positions < seq_pos (node token not fed yet)
        // token_step of the node's first child = node depth below the root
        uint32_t k = 0; for (uint32_t a = n; tree.nodes[a].parent_node_id != INVALID_ID; a = tree.nodes[a].parent_node_id) ++k;
        std::vector<float> lg = root_logits;
        int32_t tok = int32_t(nd.token_id);
        tree.active_fork_id = f; tree.cursor_node_id = n;
        matched = 0; bool ok = true;
        for (int i = 0; i < count && ok; ++i) {
            if (!(n == 0 && i == 0)) { Br* bp = &b; std::vector<float>* lp = &lg; fwd(&bp, &tok, 1, lp); }
            const uint64_t d = logit_digest(lg);
            const uint64_t s = fork_seed(base_seed, root_pos + k, slot[f], k); ++k;
            uint64_t st = s;
            const int32_t t = sample(lg, st);
            const uint32_t child = tree.step_forward();                     // recorded child on this fork
            ok = child != INVALID_ID && tree.nodes[child].token_id == uint32_t(t) && node_digest[child] == d && tree.nodes[child].step_seed == s;
            if (ok) ++matched;
            tok = t;
        }
        m.release(b);
        return ok;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: pchie_fork_batching <model.gguf> <threads> [steps] [gpu_layers] [reserve_mb]\n"); return 2; }
    const int threads = std::atoi(argv[2]);
    const int steps = argc > 3 ? std::atoi(argv[3]) : 24;
    const int gl = argc > 4 ? std::atoi(argv[4]) : -1;
    const size_t reserve = argc > 5 ? size_t(std::atoll(argv[5])) : 256;
    Gemma4Model m; std::string err;
    if (!m.load(argv[1], err)) { std::printf("LOAD ERROR: %s\n", err.c_str()); return 1; }
    std::vector<int32_t> ids = m.tok().encode("<bos><|turn>user\nWrite one sentence about a lighthouse keeper.<turn|>\n<|turn>model\n");
    m.init_cache(160, 12, /*keep_swa_history=*/true);
    ThreadPool pool(threads);
    CudaBackend cb;
    if (!cb.init(m, gl, reserve, false, err, 8)) { std::printf("CUDA INIT ERROR: %s\n", err.c_str()); return 1; }
    CudaInfo ci; cuda_query(ci, err);
    std::printf("CUDA: %d/%d layers on GPU, max_batch 8, allocated %.1f MiB, free after init %zu MiB\n", cb.gpu_layers(), m.cfg().n_layer, cb.vram_bytes() / 1048576.0, ci.free_bytes >> 20);

    Fwd cpu = [&](Br* const* b, const int32_t* t, int B, std::vector<float>* l) { m.forward_batch(b, t, B, l, pool); };
    Fwd gpu = [&](Br* const* b, const int32_t* t, int B, std::vector<float>* l) { cb.forward_batch(b, t, B, l, pool); };
    if (argc > 8 && std::string(argv[6]) == "repeat") {
        const int runs = std::atoi(argv[7]); const bool use_gpu = std::string(argv[8]) == "cuda";
        Session S{m, use_gpu ? gpu : cpu, m.new_branch(), {}, 0x5EED5EED12345678ULL, uint32_t(ids.size() - 1)};
        for (size_t i = 0; i + 1 < ids.size(); ++i) { std::vector<float> lg; Br* bp = &S.root; S.fwd(&bp, &ids[i], 1, &lg); }
        { Br* bp = &S.root; std::vector<float>* lp = &S.root_logits; S.fwd(&bp, &ids.back(), 1, lp); }
        std::map<uint64_t, int> seen; double total = 0;
        for (int i = 0; i < runs; ++i) {
            ForkRun r = S.run(8, steps, true); total += r.gen_s;
            uint64_t h = hash_mix64(FNV_OFFSET, S.tree.tree_digest());
            for (auto& kv : r.hash) h = hash_mix64(hash_mix64(h, kv.first), kv.second);
            ++seen[h];
            std::printf("run %3d hash=0x%016llX (%.2fs)\n", i + 1, (unsigned long long)h, r.gen_s);
        }
        for (auto& kv : S.live) m.release(kv.second);
        m.release(S.root);
        std::printf("[PCHIE REPEAT] backend=%s threads=%d runs=%d steps=%d unique=%zu", use_gpu ? "cuda" : "cpu", threads, runs, steps, seen.size());
        for (auto& kv : seen) std::printf(" 0x%016llX x%d", (unsigned long long)kv.first, kv.second);
        std::printf("  total %.1fs\n[PCHIE GATE] repeat %s: %s\n", total, use_gpu ? "cuda" : "cpu", seen.size() == 1 ? "PASS" : "FAIL");
        return seen.size() == 1 ? 0 : 1;
    }
    bool all_ok = true;
    std::map<std::string, std::map<int, ForkRun>> res;
    for (int be = 0; be < 2; ++be) {
        const char* name = be ? "cuda" : "cpu";
        Session S{m, be ? gpu : cpu, m.new_branch(), {}, 0x5EED5EED12345678ULL, uint32_t(ids.size() - 1)};
        for (size_t i = 0; i + 1 < ids.size(); ++i) { std::vector<float> lg; Br* bp = &S.root; S.fwd(&bp, &ids[i], 1, &lg); }
        // root node = last prompt token, fed once; its logits seed every fork
        { Br* bp = &S.root; std::vector<float>* lp = &S.root_logits; S.fwd(&bp, &ids.back(), 1, lp); }
        for (int F : {1, 2, 4, 8}) {
            ForkRun a = S.run(F, steps, false), b = S.run(F, steps, true);
            const bool same = a.hash == b.hash;
            all_ok = all_ok && same;
            std::printf("[%s] F=%d alone %.2fs (%.2f tok/s)  batched %.2fs (%.2f tok/s)  speedup x%.2f  per-fork hashes %s\n", name, F,
                        a.gen_s, F * (steps - 1) / a.gen_s, b.gen_s, F * (steps - 1) / b.gen_s, a.gen_s / b.gen_s, same ? "IDENTICAL" : "DIFFERENT");
            for (auto& kv : b.hash) std::printf("    fork %u hash=0x%016llX\n", kv.first, (unsigned long long)kv.second);
            res[name][F] = b;
        }
        // step-back / step-forward on the F=8 batched tree (still live)
        {
            const uint32_t f = 3, back = 6;
            S.tree.active_fork_id = f; S.tree.cursor_node_id = S.tree.forks[f].head_node_id;
            uint32_t n = S.tree.cursor_node_id; for (uint32_t i = 0; i < back; ++i) n = S.tree.step_back();
            int got = 0; const bool ok1 = S.regen_check(f, n, int(back), got);
            std::printf("[%s] step-back %u on fork %u -> node %u (pos %u), regenerated %d/%u: %s\n", name, back, f, n, S.tree.nodes[n].seq_pos, got, back, ok1 ? "MATCH" : "MISMATCH");
            // across forks: from fork 5's head back to the shared root (mainline), then forward on fork 2
            S.tree.active_fork_id = 5; S.tree.cursor_node_id = S.tree.forks[5].head_node_id;
            uint32_t walked = 0; while (S.tree.nodes[S.tree.cursor_node_id].parent_node_id != INVALID_ID) { S.tree.step_back(); ++walked; }
            const bool at_root = S.tree.cursor_node_id == 0;
            const bool ok2 = S.regen_check(2, 0, steps, got);
            std::printf("[%s] across forks: fork 5 head -> root in %u step-backs (%s), forward on fork 2 regenerated %d/%d: %s\n", name, walked, at_root ? "at root" : "NOT ROOT", got, steps, ok2 ? "MATCH" : "MISMATCH");
            all_ok = all_ok && ok1 && ok2 && at_root;
            std::printf("[%s] tree digest 0x%016llX (%zu nodes)\n", name, (unsigned long long)S.tree.tree_digest(), S.tree.nodes.size());
        }
        for (auto& kv : S.live) m.release(kv.second);
        m.release(S.root);
    }
    bool xok = true;
    for (int F : {1, 2, 4, 8}) xok = xok && res["cpu"][F].hash == res["cuda"][F].hash;
    std::string t; for (auto x : res["cuda"][8].toks[1]) t += m.tok().decode(x);
    std::printf("fork 1 text: %s\n", t.c_str());
    std::printf("[PCHIE GATE] step5 alone==batched (F=1/2/4/8, CPU and CUDA) and step-back/forward: %s\n", all_ok ? "PASS" : "FAIL");
    std::printf("[PCHIE GATE] step5 CPU==CUDA per-fork hashes: %s\n", xok ? "PASS" : "FAIL");
    return (all_ok && xok) ? 0 : 1;
}
