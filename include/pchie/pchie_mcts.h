// Bolt PCHIE - deterministic PUCT variation selection with virtual loss (clean-room).
// All search arithmetic is integer fixed point (Q20 values, Q10 square roots), so selection
// is bit-identical on every thread count and backend. Ties break by lowest token ID.
// Batched selection: picks are made one after another in slot order; each pick applies a
// virtual loss along its path and marks its new leaf pending, so the B picks of one batch
// are always distinct branches. Evaluation (solo or batched) never changes the picks.
#pragma once
#include "pchie/SearchConfig.h"
#include "pchie/VirtualLossTracker.h"
#include <cstdint>
#include <vector>

namespace pchie {

constexpr uint32_t PUCT_Q20 = 1u << 20;

struct PuctEdge { int32_t token = 0; uint32_t prior_q20 = 0; uint32_t child = 0xFFFFFFFFu; double log_prob = 0.0; };  // log_prob: ln P at T = 1 (leaf value input, not hashed)
struct PuctNode {
    uint32_t parent = 0xFFFFFFFFu, edge_in = 0, depth = 0;
    uint32_t tree_node = 0xFFFFFFFFu;    // VariationTree node ID (Timing Handshake key)
    uint32_t n = 0, vl = 0;              // completed visits, in-flight (virtual-loss) visits
    int64_t  w = 0;                      // sum of Q20 values
    uint64_t seed = 0;                   // splitmix64(parent_seed ^ token ^ depth); root = run seed (ruling 5)
    uint32_t value_q20 = 0;              // this node's own leaf value (terminal: the value backed up on every hit)
    bool     expanded = false;
    bool     terminal = false;           // end-of-turn / EOS / PCH delimiter: never expanded (Archie ruling 2)
    std::vector<PuctEdge> edges;         // sorted by token ID ascending
};
struct PuctPick { uint32_t leaf; uint32_t parent; int32_t token; std::vector<uint32_t> path; bool terminal = false; };  // terminal: re-hit, back up value_q20 now

class PuctSearch {
public:
    SearchConfig cfg;                    // c_puct, c_fpu, prior temperature, top-K (Archie ruling 3)
    VirtualLossTracker vloss{PUCT_Q20};  // v_loss = 1.0 normalized (Archie ruling 4)
    std::vector<PuctNode> nodes;

    void reset();                        // root = node 0, not yet expanded
    // Priors from logits: deterministic top-K at temperature T. Exact logit ties break by
    // splitmix64(node_seed ^ token), then by token ID (ruling 5).
    static std::vector<PuctEdge> priors_from_logits(const std::vector<float>& lg, int K, float temp, uint64_t node_seed = 0);
    void expand(uint32_t node, std::vector<PuctEdge> edges);
    void mark_terminal(uint32_t node, uint32_t value_q20);
    // One selection with virtual loss. Returns false if no selectable edge exists. A pick that
    // lands on a terminal node creates nothing; the caller backs up nodes[leaf].value_q20 at once.
    bool select(PuctPick& out);
    void backup(const PuctPick& p, uint32_t value_q20);
    uint64_t digest() const noexcept;    // order-fixed over nodes: token, parent, n, w, priors
    static uint64_t isqrt64(uint64_t x) noexcept;
};

} // namespace pchie
