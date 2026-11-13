#include "pchie/pchie_mcts.h"
#include "pchie/pchie_det_math.h"
#include "pchie/pchie_tree.h"
#include "pchie/FirstPlayUrgency.h"
#include "pchie/NodeSeed.h"
#include <algorithm>
#include <stdexcept>

namespace pchie {

void PuctSearch::reset() { nodes.clear(); nodes.emplace_back(); }

uint64_t PuctSearch::isqrt64(uint64_t x) noexcept {
    uint64_t r = 0, b = 1ULL << 62;
    while (b > x) b >>= 2;
    while (b) { if (x >= r + b) { x -= r + b; r = (r >> 1) + b; } else r >>= 1; b >>= 2; }
    return r;
}

std::vector<PuctEdge> PuctSearch::priors_from_logits(const std::vector<float>& lg, int K, float temp, uint64_t node_seed) {
    if (lg.empty() || K <= 0) return {};
    if (K > int(lg.size())) K = int(lg.size());
    std::vector<int32_t> idx(lg.size());
    for (size_t i = 0; i < lg.size(); ++i) idx[i] = int32_t(i);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) {
        if (lg[size_t(a)] != lg[size_t(b)]) return lg[size_t(a)] > lg[size_t(b)];
        const uint64_t ka = NodeSeed::Splitmix64(node_seed ^ uint64_t(uint32_t(a))), kb = NodeSeed::Splitmix64(node_seed ^ uint64_t(uint32_t(b)));
        return ka != kb ? ka < kb : a < b; });
    const float inv_t = 1.0f / temp, mx = lg[size_t(idx[0])];
    std::vector<float> p(static_cast<size_t>(K)); float sum = 0.0f;
    for (int i = 0; i < K; ++i) { p[size_t(i)] = exp_deterministic((lg[size_t(idx[size_t(i)])] - mx) * inv_t); sum = sum + p[size_t(i)]; }
    std::vector<PuctEdge> e(static_cast<size_t>(K));
    for (int i = 0; i < K; ++i) { e[size_t(i)].token = idx[size_t(i)]; e[size_t(i)].prior_q20 = uint32_t((p[size_t(i)] / sum) * float(PUCT_Q20)); }
    std::sort(e.begin(), e.end(), [](const PuctEdge& a, const PuctEdge& b) { return a.token < b.token; });
    return e;
}

void PuctSearch::mark_terminal(uint32_t node, uint32_t v) {
    nodes[node].terminal = true; nodes[node].expanded = false; nodes[node].value_q20 = v; nodes[node].edges.clear();
}

void PuctSearch::expand(uint32_t node, std::vector<PuctEdge> edges) {
    nodes[node].edges = std::move(edges);
    nodes[node].expanded = true;
}

bool PuctSearch::select(PuctPick& out) {
    out.path.clear();
    uint32_t cur = 0;
    if (!nodes[0].expanded) throw std::logic_error("PUCT root not expanded");
    for (;;) {
        out.path.push_back(cur);
        const PuctNode& nd = nodes[cur];
        const uint64_t ntot = std::max<uint64_t>(1, uint64_t(nd.n) + nd.vl);
        const uint64_t sq_q10 = isqrt64(ntot << 20);                  // sqrt(N) in Q10
        uint64_t vis_p = 0;                                           // sum of priors of visited children
        for (const PuctEdge& e : nd.edges) if (e.child != 0xFFFFFFFFu) vis_p += e.prior_q20;
        const int64_t v_parent = nd.n ? nd.w / int64_t(nd.n) : int64_t(nd.value_q20);
        const int64_t fpu = FirstPlayUrgency(cfg.c_fpu_num, cfg.c_fpu_den).Value(v_parent, vis_p);
        int64_t best = INT64_MIN; int bi = -1;
        for (size_t i = 0; i < nd.edges.size(); ++i) {                // ascending token: first max wins ties
            const PuctEdge& e = nd.edges[i];
            int64_t q = fpu; uint64_t neff = 0;                         // unvisited: parent-relative FPU
            if (e.child != 0xFFFFFFFFu) {
                const PuctNode& c = nodes[e.child];
                if (!c.expanded && !c.terminal) continue;            // pending leaf in this batch: skip
                neff = uint64_t(c.n) + c.vl;
                q = vloss.EffectiveQ(c.w, c.n, c.vl);
            }
            const int64_t u = int64_t((uint64_t(cfg.c_puct_num) * e.prior_q20 * sq_q10) / (uint64_t(cfg.c_puct_den) * (1 + neff)) >> 10);
            const int64_t s = q + u;
            if (s > best) { best = s; bi = int(i); }
        }
        if (bi < 0) { // nothing selectable below: undo nothing (no vl applied yet) and fail
            out.path.clear(); return false;
        }
        PuctEdge& e = nodes[cur].edges[size_t(bi)];
        if (e.child == 0xFFFFFFFFu) {
            PuctNode leaf; leaf.parent = cur; leaf.edge_in = uint32_t(bi); leaf.depth = nodes[cur].depth + 1;
            leaf.seed = NodeSeed::Child(nodes[cur].seed, e.token, leaf.depth);
            const uint32_t id = uint32_t(nodes.size());
            e.child = id;
            const int32_t tok = e.token;
            nodes.push_back(std::move(leaf));
            out.path.push_back(id);
            for (uint32_t a : out.path) nodes[a].vl++;
            out.leaf = id; out.parent = cur; out.token = tok;
            return true;
        }
        if (nodes[e.child].terminal) {                                // hard stop: no expansion, value now
            out.path.push_back(e.child);
            for (uint32_t a : out.path) nodes[a].vl++;
            out.leaf = e.child; out.parent = cur; out.token = e.token; out.terminal = true;
            return true;
        }
        cur = e.child;
    }
}

void PuctSearch::backup(const PuctPick& p, uint32_t v) {
    for (uint32_t a : p.path) {
        if (nodes[a].vl == 0) throw std::logic_error("PUCT backup without virtual loss");
        nodes[a].vl--; nodes[a].n++; nodes[a].w += int64_t(v);
    }
}

uint64_t PuctSearch::digest() const noexcept {
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const PuctNode& n = nodes[i];
        h = hash_mix64(h, n.parent); h = hash_mix64(h, n.edge_in); h = hash_mix64(h, n.n);
        h = hash_mix64(h, uint64_t(n.w)); h = hash_mix64(h, n.tree_node); h = hash_mix64(h, n.terminal ? 1u : 0u); h = hash_mix64(h, n.seed);
        for (const PuctEdge& e : n.edges) { h = hash_mix64(h, uint32_t(e.token)); h = hash_mix64(h, e.prior_q20); h = hash_mix64(h, e.child); }
    }
    return h;
}

} // namespace pchie
