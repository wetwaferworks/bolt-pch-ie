// Unit checks for Archie's search rulings (2026-10-29): leaf value, terminal stops,
// FPU / prior temperature / c_puct, virtual loss, per-node seed uses. No model needed.
#include "pchie/DetLog.h"
#include "pchie/FirstPlayUrgency.h"
#include "pchie/VirtualLossTracker.h"
#include "pchie/CommitSelector.h"
#include "pchie/NodeSeed.h"
#include "pchie/RootNoise.h"
#include "pchie/LogitStats.h"
#include "pchie/LogProbLeafEvaluator.h"
#include "pchie/TurnBoundaryTerminalPolicy.h"
#include "pchie/pchie_mcts.h"
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("[RULES] %-60s %s\n", what, ok ? "PASS" : "FAIL"); if (!ok) ++g_fail; }

static void ruling1_2() {
    bool ln_ok = true;
    for (double x : {1e-9, 0.1, 0.5, 1.0, 2.0, 3.14159, 1000.0, 262144.0}) ln_ok = ln_ok && std::fabs(DetLog::Ln(x) - std::log(x)) < 1e-12 * (1.0 + std::fabs(std::log(x)));
    check(ln_ok, "DetLog::Ln matches std::log to 1e-12");
    std::vector<float> lg{2.0f, 1.0f, 0.0f, -1.0f};
    LogitStats st(lg);
    double z = 0; for (float f : lg) z += std::exp(double(f) - 2.0);
    check(std::fabs(st.LogProb(0) - (-std::log(z))) < 1e-6, "LogitStats::LogProb is log-softmax");
    double h = 0; for (float f : lg) { const double p = std::exp(double(f) - 2.0) / z; h -= p * std::log(p); }
    check(std::fabs(st.Entropy() - h) < 1e-6, "LogitStats::Entropy matches -sum p ln p");
    LogProbLeafEvaluator ev(0.7, 0.1);
    LeafContext a; a.depth = 4; a.sum_log_prob = -2.0; a.entropy = 1.0;
    const double want = 1.0 / (1.0 + std::exp(-(-2.0 / std::pow(4.0, 0.7) - 0.1)));   // reference only (test side)
    check(std::fabs(ev.EvalLeaf(a) - want) < 1e-6, "EvalLeaf = sigmoid(sumlp / L^alpha - beta H)");
    LeafContext b = a; b.gate_pass = false;
    check(ev.EvalLeaf(b) == 0.0, "EvalLeaf gate G = 0 gives 0");
    LeafContext c = a; c.entropy = 3.0;
    check(ev.EvalLeaf(c) < ev.EvalLeaf(a), "higher entropy lowers the value");
    TurnBoundaryTerminalPolicy tp({106, 1, -1, 106});
    check(tp.IsTerminal(1) && tp.IsTerminal(106) && !tp.IsTerminal(2) && tp.StopTokens().size() == 2, "terminal policy: stop set, negatives dropped");
    // terminal node: selection returns it without creating a node; backup counts the visit
    PuctSearch S; S.reset();
    S.expand(0, {PuctEdge{1, PUCT_Q20 - 1, 0xFFFFFFFFu}});
    PuctPick p; S.select(p);
    S.mark_terminal(p.leaf, PUCT_Q20 / 2); S.backup(p, PUCT_Q20 / 2);
    const size_t n0 = S.nodes.size();
    PuctPick q; const bool got = S.select(q);
    check(got && q.terminal && q.leaf == p.leaf && S.nodes.size() == n0, "re-hit of a terminal node does not expand");
    S.backup(q, S.nodes[q.leaf].value_q20);
    check(S.nodes[p.leaf].n == 2 && S.nodes[0].vl == 0, "terminal value backed up at once");
}

// Bit pins (Archie amendment 2): our host math must give these exact bits on every build.
static uint64_t bits_of(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }
static void pin_math(bool print) {
    const double xs[] = {1e-9, 0.1, 0.5, 1.0, 2.0, 3.14159, 1000.0, 262144.0};
    static const uint64_t want[] = {
        0xC034B927F32BFFB8ULL,0xC0026BB1BBB55515ULL,0xBFE62E42FEFA39EFULL,0x0000000000000000ULL,0x3FE62E42FEFA39EFULL,0x3FF250CF662ABF9AULL,0x401BA18A998FFFA0ULL,0x4028F40B5ED9812DULL,0x3F35FA3DB7114E2BULL,0x3FD136561ABDF00AULL,0x3FE0000000000000ULL,0x3FE3EB2FD371BE79ULL,0x3FEE7B7CBCF0EBF0ULL,0x3FF9FDF8C0000000ULL,0x40051CB440000000ULL,0x402A73EC00000000ULL,0x3E21B48660000000ULL,0x3FD78B5640000000ULL,0x3FF0000000000000ULL,0x4001CDE860000000ULL,0x40D5829DC0000000ULL};
    uint64_t got[32]; int k = 0;
    for (double x : xs) got[k++] = bits_of(DetLog::Ln(x));
    for (double x : {-8.0, -1.0, 0.0, 0.5, 3.0}) got[k++] = bits_of(DetLog::Sigmoid(x));
    for (double x : {2.0, 4.0, 40.0}) got[k++] = bits_of(DetLog::Pow(x, 0.7));
    for (float x : {-20.0f, -1.0f, 0.0f, 0.8f, 10.0f}) got[k++] = bits_of(double(exp_deterministic(x)));
    if (print) { for (int i = 0; i < k; ++i) std::printf("0x%016llXULL,", (unsigned long long)got[i]); std::printf("\n"); return; }
    bool ok = sizeof(want) / sizeof(want[0]) == size_t(k);
    for (int i = 0; ok && i < k; ++i) ok = got[i] == want[i];
    check(ok, "Ln / Sigmoid / Pow / exp bit outputs pinned");
}

static void ruling3() {
    FirstPlayUrgency f(11, 50);
    check(f.Value(PUCT_Q20 / 2, 0) == PUCT_Q20 / 2, "FPU with no visited children = V(s)");
    const int64_t want = PUCT_Q20 / 2 - int64_t((11ull * (PUCT_Q20 / 2)) / 50);            // sqrt(0.25) = 0.5
    check(f.Value(PUCT_Q20 / 2, PUCT_Q20 / 4) == want, "FPU = V - c_fpu * sqrt(sum P visited)");
    SearchConfig c;
    check(c.prior_temp == 1.25f && c.c_puct_num * 2 >= c.c_puct_den * 3 && c.c_puct_num <= c.c_puct_den * 2 &&
          c.c_fpu_num * 5 >= c.c_fpu_den && c.c_fpu_num * 4 <= c.c_fpu_den, "config inside Archie ranges (tau 1.25, c_puct 1.5-2, c_fpu 0.2-0.25)");
    // drilling fix: peaked prior, visited child Q 0.7; the sibling must get a visit within a few selections
    PuctSearch S; S.reset();
    S.expand(0, {PuctEdge{1, uint32_t(PUCT_Q20 * 0.9), 0xFFFFFFFFu}, PuctEdge{2, uint32_t(PUCT_Q20 * 0.04), 0xFFFFFFFFu}});
    S.nodes[0].value_q20 = uint32_t(PUCT_Q20 * 0.7);
    bool sib = false;
    for (int i = 0; i < 64 && !sib; ++i) { PuctPick p; S.select(p); sib = p.token == 2 && p.parent == 0;
        S.expand(p.leaf, {PuctEdge{7, PUCT_Q20 - 1, 0xFFFFFFFFu}}); S.backup(p, uint32_t(PUCT_Q20 * 0.7)); }
    check(sib, "batch-1 search leaves the top-1 line within 64 picks (P 0.9 vs 0.04)");
}

static void ruling4() {
    VirtualLossTracker v(PUCT_Q20);
    check(v.EffectiveQ(3 * (PUCT_Q20 / 4), 1, 0) == 3 * (PUCT_Q20 / 4), "no in-flight visits: Q_eff = W / N");
    check(v.EffectiveQ(3 * (PUCT_Q20 / 4), 1, 1) == (3 * int64_t(PUCT_Q20 / 4) - int64_t(PUCT_Q20)) / 2, "Q_eff = (W - v N_virt) / (N + N_virt)");
    check(v.EffectiveQ(0, 0, 0) == 0, "empty edge: Q_eff = 0");
    // two co-batched picks from one root go to distinct leaves
    PuctSearch S; S.reset();
    S.expand(0, {PuctEdge{1, PUCT_Q20 / 2, 0xFFFFFFFFu}, PuctEdge{2, PUCT_Q20 / 2 - 1, 0xFFFFFFFFu}});
    PuctPick a, b; S.select(a); S.select(b);
    check(a.leaf != b.leaf && a.token != b.token, "virtual loss deflects co-batched picks");
}

static void ruling5() {
    check(NodeSeed::Splitmix64(0) == 0xE220A8397B1DCDAFULL, "splitmix64(0) reference value");
    check(NodeSeed::Child(7, 3, 2) == NodeSeed::Splitmix64(7ull ^ 3ull ^ 2ull), "child seed = splitmix64(parent ^ token ^ depth)");
    PuctSearch S; S.reset(); S.nodes[0].seed = 99;
    S.expand(0, {PuctEdge{5, PUCT_Q20 - 1, 0xFFFFFFFFu}}); PuctPick p; S.select(p);
    check(S.nodes[p.leaf].seed == NodeSeed::Child(99, 5, 1), "select stamps the tree-coordinate seed");
    std::vector<float> lg{1.0f, 3.0f, 3.0f, 3.0f, 0.0f};                 // exact ties at tokens 1, 2, 3
    auto a = PuctSearch::priors_from_logits(lg, 2, 1.25f, 11), b = PuctSearch::priors_from_logits(lg, 2, 1.25f, 11);
    bool same = a.size() == 2 && a[0].token == b[0].token && a[1].token == b[1].token;
    bool seeded = false; for (uint64_t sd = 0; sd < 64 && !seeded; ++sd) { auto c = PuctSearch::priors_from_logits(lg, 2, 1.25f, sd); seeded = c[0].token != a[0].token || c[1].token != a[1].token; }
    check(same && seeded, "exact logit ties break by node seed, reproducibly");
    std::vector<PuctEdge> e{PuctEdge{1, PUCT_Q20 / 2, 0xFFFFFFFFu}, PuctEdge{2, PUCT_Q20 / 2, 0xFFFFFFFFu}}, e2 = e, e3 = e;
    RootNoise(0, 4).Apply(e2, 5); RootNoise(1, 4).Apply(e3, 5);
    std::vector<PuctEdge> e4 = e; RootNoise(1, 4).Apply(e4, 5);
    check(e2[0].prior_q20 == e[0].prior_q20 && e3[0].prior_q20 != e[0].prior_q20 && e3[0].prior_q20 == e4[0].prior_q20, "root noise: off at eps 0, deterministic at eps > 0");
    PuctSearch T; T.reset(); T.nodes[0].seed = 1;
    T.expand(0, {PuctEdge{1, PUCT_Q20 / 2, 0xFFFFFFFFu}, PuctEdge{2, PUCT_Q20 / 2, 0xFFFFFFFFu}});
    for (int i = 0; i < 6; ++i) { PuctPick q; T.select(q); T.expand(q.leaf, {PuctEdge{9, PUCT_Q20 - 1, 0xFFFFFFFFu}}); T.backup(q, q.token == 1 ? PUCT_Q20 - 1 : 0); }
    const int g0 = CommitSelector(0.0).Select(T, 0), g1 = CommitSelector(1.0).Select(T, 0), g2 = CommitSelector(1.0).Select(T, 0);
    check(g0 >= 0 && T.nodes[T.nodes[0].edges[size_t(g0)].child].n >= T.nodes[T.nodes[0].edges[size_t(1 - g0)].child].n && g1 == g2 && g1 >= 0, "commit: T = 0 most visited, T > 0 seeded and reproducible");
}

int main(int argc, char**) {
    if (argc > 1) { pin_math(true); return 0; }
    pin_math(false);
    ruling3();
    ruling4();
    ruling5();
    ruling1_2();
    std::printf("[PCHIE GATE] search rules: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}