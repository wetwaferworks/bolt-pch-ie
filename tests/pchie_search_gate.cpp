// Search gate: copy-on-write KV with the 33-page sliding ring, PUCT variation search with
// deterministic virtual loss, and engine snapshot/rollback on the public Timing Handshake keyed
// by variation-tree node IDs. Locked rules unchanged: top-40 at T = 0.8, ties to the lowest
// token ID, seeds from tree position (never node ID), physical page IDs never hashed.
//   pchie_search_gate <model.gguf> <threads> [gpu_layers|-1] [reserve_mb] [budget]
//   pchie_search_gate <model.gguf> <threads> <gpu_layers> <reserve_mb> <budget> repeat <runs> <cpu|cuda>
// Per-setup comparison (1/2/8/16 threads, CUDA) and gates 1-5 are run by tools/run-search-gate.ps1.
#include "pchie/pchie_cuda.h"
#include "pchie/pchie_det_math.h"
#include "pchie/pchie_host.h"
#include "pchie/pchie_mcts.h"
#include "pchie/pchie_rollout.h"
#include "pchie/pchie_tree.h"
#include "pchie/AcceptAllGate.h"
#include "pchie/CommitSelector.h"
#include "pchie/RootNoise.h"
#include "pchie/LogitStats.h"
#include "pchie/LogProbLeafEvaluator.h"
#include "pchie/TurnBoundaryTerminalPolicy.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
using namespace pchie;
using Br = Gemma4Model::Branch;
using Fwd = std::function<void(Br* const*, const int32_t*, int, std::vector<float>*)>;

static uint64_t logit_digest(const std::vector<float>& lg) {
    uint64_t h = FNV_OFFSET; for (float f : lg) { uint32_t u; std::memcpy(&u, &f, 4); h = hash_mix64(h, u); } return h;
}
// Same derivation as H(root_seed, parent_seq_pos, fork_slot, token_step). Here the slot is
// the edge's index among its parent's (token-sorted) candidates and token_step is the parent depth.
static uint64_t pos_seed(uint64_t root_seed, uint32_t parent_seq_pos, uint32_t slot, uint32_t step) {
    return hash_mix64(hash_mix64(hash_mix64(hash_mix64(root_seed, 0x5EEDF0CC5EEDF0CCULL), parent_seq_pos), slot), step);
}

// ---------- (1) synthetic CoW KV + 33-page sliding ring ----------
static bool kv_ring_test() {
    constexpr uint32_t W = 512, RING = (W + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS + 1; // 33
    KVPagePool pool(1, 1, 4, 512);
    auto val = [](uint32_t pos, uint32_t br) { return float((pos * 2654435761u + br * 40503u) & 0xFFFFu); };
    auto write = [&](KVPagePool::Table& t, uint32_t pos, uint32_t br) {
        if (pos >= W) pool.trim_before(t, pos - W + 1);
        float* k = pool.k_write(t, 0, pos); float* v = pool.v_write(t, 0, pos);
        for (int i = 0; i < 4; ++i) { k[i] = val(pos, br) + float(i); v[i] = -val(pos, br) - float(i); }
    };
    struct Snap { KVPagePool::Table t; uint32_t n; std::map<int32_t, uint64_t> cs; };
    std::vector<Snap> snaps; std::vector<KVPagePool::Table> kids;
    KVPagePool::Table root; uint32_t max_live = 0; bool ok = true;
    for (uint32_t pos = 0; pos < 1200; ++pos) {
        write(root, pos, 0);
        max_live = std::max(max_live, pool.table_live(root));
        if (pos % 41 == 40) {
            Snap s{pool.fork(root), pos + 1, {}};
            for (int32_t p : s.t.pages) if (p >= 0) s.cs[p] = pool.page_checksum(p);
            snaps.push_back(std::move(s));
            // a child of this snapshot diverges at once: writes into the shared partial page
            KVPagePool::Table c = pool.fork(root);
            for (uint32_t q = pos + 1; q < pos + 1 + 70; ++q) { write(c, q, 1 + uint32_t(snaps.size())); max_live = std::max(max_live, pool.table_live(c)); }
            kids.push_back(std::move(c));
        }
        if (pos == 700) {   // step-back on the root: re-fork at an earlier point and overwrite
            KVPagePool::Table b = pool.fork(snaps[3].t); pool.truncate(b, snaps[3].n);
            for (uint32_t q = snaps[3].n; q < snaps[3].n + 40; ++q) write(b, q, 999);
            kids.push_back(std::move(b));
        }
    }
    uint32_t bad_pages = 0, bad_vals = 0;
    for (const Snap& s : snaps) {
        for (auto& kv : s.cs) if (pool.page_checksum(kv.first) != kv.second) ++bad_pages;
        const uint32_t lo = s.n > W ? s.n - W : 0;
        for (uint32_t q = lo; q < s.n; ++q) if (pool.k_read(s.t, 0, q)[0] != val(q, 0)) ++bad_vals;
    }
    const uint64_t viol = pool.overwrite_violations(), cows = pool.cow_copies();
    for (auto& s : snaps) pool.release(s.t);
    for (auto& k : kids) pool.release(k);
    pool.release(root);
    ok = bad_pages == 0 && bad_vals == 0 && viol == 0 && max_live <= RING && pool.live_pages() == 0 && cows > 0;
    std::printf("[SEARCH] kv-ring: %zu ancestor snapshots, %llu CoW copies, max live pages per branch %u (cap %u), changed ancestor pages %u, bad reads %u, write-to-shared %llu, leaked pages %u: %s\n",
                snaps.size(), (unsigned long long)cows, max_live, RING, bad_pages, bad_vals, (unsigned long long)viol, pool.live_pages(), ok ? "PASS" : "FAIL");
    return ok;
}

// ---------- (3a) synthetic Timing Handshake ledger ----------
static bool ledger_test() {
    HostSession s(1000, 4, 0xABCDEF01ULL, stub_token_source);
    HandshakeLedger L(s.clock);
    L.snapshot(0, engine_digest_at(s.tree, 0, s.world));
    for (int i = 0; i < 24; ++i) { const uint32_t n = s.generate(); L.snapshot(n, engine_digest_at(s.tree, n, s.world)); }
    s.tree.cursor_node_id = 8; s.tree.active_fork_id = MAINLINE_FORK;
    s.tree.fork_at(s.tree.nodes[8].handshake_tick);
    for (int i = 0; i < 10; ++i) { const uint32_t n = s.generate(); L.snapshot(n, engine_digest_at(s.tree, n, s.world)); }
    bool ok = L.size() == s.tree.nodes.size();
    uint32_t checked = 0;
    for (uint32_t k = 0; k < s.tree.nodes.size(); ++k) {
        const uint32_t n = (k * 7u) % uint32_t(s.tree.nodes.size());     // fixed scrambled order
        uint64_t d = 0;
        ok = ok && L.rollback(n, d) && s.clock.now() == s.tree.nodes[n].handshake_tick && d == engine_digest_at(s.tree, n, s.world);
        ++checked;
    }
    const uint64_t t = L.commit_child(s.tree.nodes[5].handshake_tick);
    ok = ok && t == s.tree.nodes[5].handshake_tick + 4;
    std::printf("[SEARCH] handshake ledger: %zu snapshots, %u rollbacks by node ID: %s\n", L.size(), checked, ok ? "PASS" : "FAIL");
    return ok;
}

// ---------- (2)+(3b) PUCT search over the model ----------
struct MctsResult {
    uint64_t run_hash = 0, puct_digest = 0, tree_digest = 0;
    bool distinct = true, kv_intact = true, stepback = true, leak_free = true;
    uint32_t nodes = 0, rounds = 0, root_children = 0, terminal_hits = 0; uint64_t cows = 0, viol = 0; double gen_s = 0;
    std::vector<int32_t> best_path;
};

struct Mcts {
    Gemma4Model& m; Fwd fwd; Br& root; const std::vector<float>& root_logits; uint32_t root_pos; uint64_t base_seed;
    const LogProbLeafEvaluator& eval; const TurnBoundaryTerminalPolicy& stop; const AcceptAllGate& gate;   // concrete: no virtual calls in the hot path
    static void fill_log_probs(std::vector<PuctEdge>& e, const LogitStats& st) { for (auto& x : e) x.log_prob = st.LogProb(x.token); }
    static uint32_t to_q20(double v) { if (v < 0.0) v = 0.0; if (v > 1.0) v = 1.0; return uint32_t(v * double(PUCT_Q20)); }
    static constexpr uint32_t K_TICKS = 4;

    MctsResult run(int B, int budget, bool batched, bool check_stepback) {
        MctsResult R;
        const uint32_t live0 = m.pool_swa().live_pages() + m.pool_glob().live_pages();
        const uint64_t cow0 = m.pool_swa().cow_copies() + m.pool_glob().cow_copies();
        PuctSearch S; S.reset();
        VariationTree tree; tree.init_root(uint64_t(root_pos) * K_TICKS, 0, base_seed); tree.nodes[0].seq_pos = root_pos;
        TimingHandshake clock(tree.nodes[0].handshake_tick, K_TICKS); HandshakeLedger ledger(clock); FlatWorld world;
        ledger.snapshot(0, engine_digest_at(tree, 0, world));
        std::vector<Br> br; std::vector<uint64_t> dig; std::vector<std::vector<uint64_t>> cs;
        auto checksums = [&](const Br& b) { std::vector<uint64_t> v;
            for (int32_t p : b.swa.pages) if (p >= 0) v.push_back(m.pool_swa().page_checksum(p));
            for (int32_t p : b.glob.pages) if (p >= 0) v.push_back(m.pool_glob().page_checksum(p)); return v; };
        br.push_back(m.fork(root)); dig.push_back(logit_digest(root_logits)); cs.push_back(checksums(br[0]));
        std::vector<double> lps{0.0}; std::vector<std::vector<int32_t>> toks{{}};
        uint32_t terminal_hits = 0;
        { LogitStats st(root_logits); S.nodes[0].seed = base_seed;
          std::vector<PuctEdge> e0 = PuctSearch::priors_from_logits(root_logits, S.cfg.top_k, S.cfg.prior_temp, S.nodes[0].seed); fill_log_probs(e0, st);
          RootNoise(S.cfg.root_noise_num, S.cfg.root_noise_den).Apply(e0, S.nodes[0].seed);
          LeafContext c0; c0.depth = 0; c0.entropy = st.Entropy(); c0.gate_pass = gate.Pass({});
          S.expand(0, std::move(e0)); S.nodes[0].value_q20 = to_q20(eval.EvalLeaf(c0)); }
        S.nodes[0].tree_node = 0;
        int done = 0; auto t0 = std::chrono::steady_clock::now();
        std::vector<std::vector<float>> lg;
        while (done < budget) {
            std::vector<PuctPick> picks;
            bool stuck = false;
            while (int(picks.size()) < B && done + int(picks.size()) < budget) {
                PuctPick p; if (!S.select(p)) { stuck = true; break; }
                if (p.terminal) { S.backup(p, S.nodes[p.leaf].value_q20); ++done; ++terminal_hits; continue; }   // hard stop: back up at once
                picks.push_back(p);
            }
            if (picks.empty()) { if (stuck || done >= budget) break; continue; }
            std::set<uint32_t> leaves; std::set<std::pair<uint32_t, int32_t>> edges;
            for (auto& p : picks) { leaves.insert(p.leaf); edges.insert({p.parent, p.token}); }
            R.distinct = R.distinct && leaves.size() == picks.size() && edges.size() == picks.size();
            br.resize(S.nodes.size()); dig.resize(S.nodes.size()); cs.resize(S.nodes.size()); lps.resize(S.nodes.size()); toks.resize(S.nodes.size());
            std::vector<Br*> bs; std::vector<int32_t> tk;
            for (auto& p : picks) { br[p.leaf] = m.fork(br[p.parent]); bs.push_back(&br[p.leaf]); tk.push_back(p.token); }
            lg.assign(picks.size(), {});
            if (batched) fwd(bs.data(), tk.data(), int(bs.size()), lg.data());
            else for (size_t i = 0; i < picks.size(); ++i) fwd(&bs[i], &tk[i], 1, &lg[i]);
            for (size_t i = 0; i < picks.size(); ++i) {                       // slot order
                const PuctPick& p = picks[i];
                const uint64_t d = logit_digest(lg[i]);
                lps[p.leaf] = lps[p.parent] + S.nodes[p.parent].edges[S.nodes[p.leaf].edge_in].log_prob;
                toks[p.leaf] = toks[p.parent]; toks[p.leaf].push_back(p.token);
                LeafContext lc; lc.depth = S.nodes[p.leaf].depth; lc.sum_log_prob = lps[p.leaf]; lc.gate_pass = gate.Pass(toks[p.leaf]);
                uint32_t v = 0;
                if (stop.IsTerminal(p.token)) {                                  // Archie ruling 2: never expand
                    lc.entropy = 0.0; v = to_q20(eval.EvalLeaf(lc)); S.mark_terminal(p.leaf, v);
                } else {
                    LogitStats st(lg[i]);
                    std::vector<PuctEdge> e = PuctSearch::priors_from_logits(lg[i], S.cfg.top_k, S.cfg.prior_temp, S.nodes[p.leaf].seed); fill_log_probs(e, st);
                    lc.entropy = st.Entropy(); v = to_q20(eval.EvalLeaf(lc));    // Archie ruling 1
                    S.expand(p.leaf, std::move(e)); S.nodes[p.leaf].value_q20 = v;
                }
                const uint32_t ptn = S.nodes[p.parent].tree_node;
                const uint64_t tick = ledger.commit_child(tree.nodes[ptn].handshake_tick);
                const uint64_t seed = pos_seed(base_seed, tree.nodes[ptn].seq_pos, S.nodes[p.leaf].edge_in, S.nodes[p.parent].depth);
                const uint32_t tn = tree.append_child(ptn, S.nodes[p.leaf].edge_in, tick, uint32_t(p.token), seed, d);
                S.nodes[p.leaf].tree_node = tn;
                R.distinct = R.distinct && tn == p.leaf;                           // tree node ID == search node index
                ledger.snapshot(tn, engine_digest_at(tree, tn, world));
                dig[p.leaf] = d; cs[p.leaf] = checksums(br[p.leaf]);
                S.backup(p, v);
            }
            done += int(picks.size()); R.rounds++;
        }
        R.gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (size_t n = 0; n < br.size(); ++n) R.kv_intact = R.kv_intact && checksums(br[n]) == cs[n];
        for (auto& e : S.nodes[0].edges) if (e.child != 0xFFFFFFFFu) R.root_children++;
        // committed path (T = 0: most visited, ties to the lowest token)
        const CommitSelector commit(S.cfg.commit_temp);                          // ruling 5: commit-time choice
        for (uint32_t c = 0; S.nodes[c].expanded;) { const int ei = commit.Select(S, c);
            if (ei < 0) break; R.best_path.push_back(S.nodes[c].edges[size_t(ei)].token); c = S.nodes[c].edges[size_t(ei)].child; }
        if (check_stepback) {
            uint32_t L = 0; for (uint32_t n = 0; n < S.nodes.size(); ++n) if (S.nodes[n].depth > S.nodes[L].depth) L = n;
            // walk back by node ID through the tree; each rollback restores the handshake tick + engine digest
            tree.cursor_node_id = S.nodes[L].tree_node; std::vector<uint32_t> path{L};
            bool ok = true; uint64_t d = 0;
            while (tree.nodes[tree.cursor_node_id].parent_node_id != INVALID_ID) {
                const uint32_t n = tree.step_back(); path.push_back(n);
                ok = ok && ledger.rollback(n, d) && clock.now() == tree.nodes[n].handshake_tick && d == engine_digest_at(tree, n, world);
            }
            std::reverse(path.begin(), path.end());
            // regenerate the whole line from the root KV, one token at a time
            Br b = m.fork(root); std::vector<float> l;
            for (size_t i = 1; i < path.size(); ++i) {
                const uint32_t n = path[i]; Br* bp = &b; int32_t t = S.nodes[S.nodes[n].parent].edges[S.nodes[n].edge_in].token;
                fwd(&bp, &t, 1, &l);
                ok = ok && logit_digest(l) == dig[n];
            }
            m.release(b);
            // and from the parent's snapshot directly (rollback by node ID, then re-expand)
            const uint32_t P = S.nodes[L].parent; uint64_t pd = 0;
            ok = ok && ledger.rollback(S.nodes[P].tree_node, pd);
            Br c = m.fork(br[P]); Br* cp = &c; int32_t t = S.nodes[S.nodes[L].parent].edges[S.nodes[L].edge_in].token;
            fwd(&cp, &t, 1, &l); ok = ok && logit_digest(l) == dig[L] && ledger.commit_child(clock.now()) == tree.nodes[S.nodes[L].tree_node].handshake_tick;
            m.release(c);
            R.stepback = ok;
        }
        R.viol = m.pool_swa().overwrite_violations() + m.pool_glob().overwrite_violations();
        R.cows = m.pool_swa().cow_copies() + m.pool_glob().cow_copies() - cow0;
        for (auto& b : br) m.release(b);
        R.leak_free = m.pool_swa().live_pages() + m.pool_glob().live_pages() == live0;
        R.terminal_hits = terminal_hits; R.nodes = uint32_t(S.nodes.size()); R.puct_digest = S.digest(); R.tree_digest = tree.tree_digest();
        uint64_t h = hash_mix64(hash_mix64(FNV_OFFSET, R.puct_digest), R.tree_digest);
        for (uint64_t x : dig) h = hash_mix64(h, x);
        R.run_hash = h;
        return R;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: pchie_search_gate <model.gguf> <threads> [gpu_layers] [reserve_mb] [budget]\n"); return 2; }
    const int threads = std::atoi(argv[2]);
    const int gl = argc > 3 ? std::atoi(argv[3]) : -1;
    const size_t reserve = argc > 4 ? size_t(std::atoll(argv[4])) : 256;
    const int budget = argc > 5 ? std::atoi(argv[5]) : 32;
    const bool repeat = argc > 8 && std::string(argv[6]) == "repeat";
    bool ok = true;
    if (!repeat) { ok = kv_ring_test() && ok; ok = ledger_test() && ok; }
    Gemma4Model m; std::string err;
    if (!m.load(argv[1], err)) { std::printf("LOAD ERROR: %s\n", err.c_str()); return 1; }
    std::vector<int32_t> ids = m.tok().encode("<bos><|turn>user\nWrite one sentence about a lighthouse keeper.<turn|>\n<|turn>model\n");
    m.init_cache(160, 12, /*keep_swa_history=*/true);
    ThreadPool pool(threads);
    CudaBackend cb;
    CudaInfo c0; cuda_query(c0, err);
    if (!cb.init(m, gl, reserve, false, err, 8)) { std::printf("CUDA INIT ERROR: %s\n", err.c_str()); return 1; }
    CudaInfo ci; cuda_query(ci, err);
    std::printf("CUDA: %s, %d/%d layers on GPU, max_batch 8, allocated %.1f MiB, free %zu -> %zu MiB of %zu\n", ci.name.c_str(), cb.gpu_layers(), m.cfg().n_layer,
                cb.vram_bytes() / 1048576.0, c0.free_bytes >> 20, ci.free_bytes >> 20, ci.total_bytes >> 20);
    Fwd cpu = [&](Br* const* b, const int32_t* t, int B, std::vector<float>* l) { m.forward_batch(b, t, B, l, pool); };
    Fwd gpu = [&](Br* const* b, const int32_t* t, int B, std::vector<float>* l) { cb.forward_batch(b, t, B, l, pool); };
    const uint64_t seed = 0x5EED5EED12345678ULL;
    // alpha/beta: PROVISIONAL until the maintainer locks them; override with env PCHIE_ALPHA / PCHIE_BETA.
    const char* ea = std::getenv("PCHIE_ALPHA"); const char* eb = std::getenv("PCHIE_BETA");
    LogProbLeafEvaluator leaf_eval(ea ? std::atof(ea) : 0.7, eb ? std::atof(eb) : 0.0); AcceptAllGate gate;
    TurnBoundaryTerminalPolicy stop({m.tok().eos, m.tok().id_of("<turn|>"), m.tok().id_of("<end_of_turn>")});
    std::printf("[SEARCH] eval_leaf: logprob alpha=%.2f beta=%.2f; terminal stop tokens:", leaf_eval.Alpha(), leaf_eval.Beta());
    for (int32_t t : stop.StopTokens()) std::printf(" %d", t);
    std::printf("\n");
    auto prefill = [&](const Fwd& f, Br& root, std::vector<float>& rl) {
        for (size_t i = 0; i + 1 < ids.size(); ++i) { std::vector<float> lg; Br* bp = &root; f(&bp, &ids[i], 1, &lg); }
        Br* bp = &root; f(&bp, &ids.back(), 1, &rl);
    };
    if (repeat) {
        const int runs = std::atoi(argv[7]); const bool use_gpu = std::string(argv[8]) == "cuda";
        Br root = m.new_branch(); std::vector<float> rl; prefill(use_gpu ? gpu : cpu, root, rl);
        Mcts M{m, use_gpu ? gpu : cpu, root, rl, uint32_t(ids.size() - 1), seed, leaf_eval, stop, gate};
        std::map<uint64_t, int> seen; double total = 0;
        for (int i = 0; i < runs; ++i) {
            MctsResult r = M.run(8, budget, true, false); total += r.gen_s; ++seen[r.run_hash];
            std::printf("run %3d hash=0x%016llX (%.2fs)%s\n", i + 1, (unsigned long long)r.run_hash, r.gen_s, (r.distinct && r.kv_intact && r.leak_free) ? "" : " INVARIANT FAIL");
            ok = ok && r.distinct && r.kv_intact && r.leak_free;
        }
        std::printf("[PCHIE REPEAT] search backend=%s threads=%d runs=%d budget=%d unique=%zu", use_gpu ? "cuda" : "cpu", threads, runs, budget, seen.size());
        for (auto& kv : seen) std::printf(" 0x%016llX x%d", (unsigned long long)kv.first, kv.second);
        std::printf("  total %.1fs\n", total);
        ok = ok && seen.size() == 1;
        std::printf("[PCHIE GATE] search repeat %s: %s\n", use_gpu ? "cuda" : "cpu", ok ? "PASS" : "FAIL");
        m.release(root);
        return ok ? 0 : 1;
    }
    std::map<std::string, std::map<int, MctsResult>> res;
    for (int be = 0; be < 2; ++be) {
        const char* name = be ? "cuda" : "cpu";
        Br root = m.new_branch(); std::vector<float> rl; prefill(be ? gpu : cpu, root, rl);
        Mcts M{m, be ? gpu : cpu, root, rl, uint32_t(ids.size() - 1), seed, leaf_eval, stop, gate};
        for (int B : {1, 2, 4, 8}) {
            MctsResult a = M.run(B, budget, false, false), b = M.run(B, budget, true, B == 8);
            const bool same = a.run_hash == b.run_hash;
            const bool inv = a.distinct && b.distinct && a.kv_intact && b.kv_intact && a.leak_free && b.leak_free && a.viol == 0 && b.viol == 0;
            ok = ok && same && inv && (B != 8 || b.stepback);
            std::printf("[%s] B=%d solo %.2fs  batched %.2fs (x%.2f)  rounds %u  nodes %u  terminal hits %u  root children %u  CoW %llu  solo=0x%016llX batched=0x%016llX %s  distinct=%s kv_intact=%s leaks=%s\n",
                        name, B, a.gen_s, b.gen_s, a.gen_s / b.gen_s, b.rounds, b.nodes, b.terminal_hits, b.root_children, (unsigned long long)b.cows,
                        (unsigned long long)a.run_hash, (unsigned long long)b.run_hash, same ? "MATCH" : "DIFFERENT",
                        (a.distinct && b.distinct) ? "yes" : "NO", (a.kv_intact && b.kv_intact) ? "yes" : "NO", (a.leak_free && b.leak_free) ? "none" : "LEAK");
            std::printf("[SEARCH-HASH] %s B=%d run=0x%016llX puct=0x%016llX tree=0x%016llX\n", name, B, (unsigned long long)b.run_hash,
                        (unsigned long long)b.puct_digest, (unsigned long long)b.tree_digest);
            if (B == 8) {
                std::printf("[%s] step-back by node ID (rollback ticks + engine digest, regenerate line from root and from parent snapshot): %s\n", name, b.stepback ? "REPRODUCED" : "MISMATCH");
                std::string t; for (int32_t x : b.best_path) t += m.tok().decode(x);
                std::printf("[%s] most-visited line: %s\n", name, t.c_str());
            }
            res[name][B] = b;
        }
        m.release(root);
    }
    bool x = true; for (int B : {1, 2, 4, 8}) x = x && res["cpu"][B].run_hash == res["cuda"][B].run_hash;
    CudaInfo ce; cuda_query(ce, err);
    std::printf("CUDA free at end %zu MiB\n", ce.free_bytes >> 20);
    std::printf("[PCHIE GATE] search CPU==CUDA: %s\n", x ? "PASS" : "FAIL");
    std::printf("[PCHIE GATE] search in-process (kv ring, ledger, distinct picks, solo==batched, step-back): %s\n", ok ? "PASS" : "FAIL");
    return (ok && x) ? 0 : 1;
}
