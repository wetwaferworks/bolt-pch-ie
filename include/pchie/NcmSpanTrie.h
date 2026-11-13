// Bolt PCHIE - NcmSpanTrie: NCM concepts (including multi-token spans)
// become PUCT edges.
// Concrete final. Coordinator-thread only: no mutex, no atomic. Integer prior math after
// the one priors_from_logits call, so edges are bit-identical run to run.
//
// Default rules (logged for review; not a locked prior schema):
//  * Concept mass: priors_from_logits over the compact score vector (all concepts, K = n)
//    at `temp` with the node seed -- the same call PchStack::edges_from_ncm makes for
//    1-token concepts, so a set of distinct 1-token concepts yields the same root edges.
//  * Root edges: one edge per distinct first token; prior = sum of the masses of every
//    concept starting with that token.
//  * Mid-span node (prefix p of at least one longer concept): one edge per distinct next
//    token; prior = child subtree mass * 2^20 / total continuing mass (integer). Uniform if
//    the total is 0. A concept that ENDS at p while longer ones continue loses its "stop"
//    mass there (longest-span policy); the count is reported, not hidden.
//  * A prefix with no continuation is "complete": the driver expands it from the model.
#pragma once

#include "pchie/NcmConcept.h"
#include "pchie/pchie_mcts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pchie {

class NcmSpanTrie final {
 public:
  // Replaces the trie. Rejects (returns false, trie empty) on an empty set, a concept with
  // an empty / >16 span, a negative token, a NaN score, or temp <= 0.
  bool build(const std::vector<NcmConcept>& cs, float temp, uint64_t seed) {
    spans_.clear();
    mass_.clear();
    stop_dropped_ = 0;
    if (cs.empty() || !(temp > 0.0f)) return false;
    std::vector<float> lg;
    lg.reserve(cs.size());
    for (const NcmConcept& c : cs) {
      if (c.tokens().empty() || c.tokens().size() > kNcmMaxSpan || std::isnan(c.score())) return fail();
      for (int32_t t : c.tokens())
        if (t < 0) return fail();
      lg.push_back(static_cast<float>(c.score()));
    }
    const auto raw = PuctSearch::priors_from_logits(lg, int(lg.size()), temp, seed);
    if (raw.size() != cs.size()) return fail();
    spans_.reserve(cs.size());
    mass_.assign(cs.size(), 0u);
    for (const NcmConcept& c : cs) spans_.push_back(c.tokens());
    for (const PuctEdge& e : raw) {
      if (e.token < 0 || size_t(e.token) >= cs.size()) return fail();
      mass_[size_t(e.token)] = e.prior_q20;  // index -> concept mass
    }
    // Count prefixes where a concept ends but a longer one continues.
    for (size_t i = 0; i < spans_.size(); ++i)
      if (continues(spans_[i])) ++stop_dropped_;
    return true;
  }

  bool empty() const noexcept { return spans_.empty(); }
  size_t size() const noexcept { return spans_.size(); }
  size_t stop_mass_dropped() const noexcept { return stop_dropped_; }

  // True if some concept is strictly longer than `prefix` and starts with it.
  bool continues(const std::vector<int32_t>& prefix) const {
    for (const auto& s : spans_)
      if (s.size() > prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin())) return true;
    return false;
  }
  // True if some concept equals `prefix` exactly.
  bool ends(const std::vector<int32_t>& prefix) const {
    for (const auto& s : spans_)
      if (s == prefix) return true;
    return false;
  }

  // Edges below `prefix` (empty prefix = root), sorted by token ascending. Empty if the
  // prefix has no continuation.
  std::vector<PuctEdge> edges_at(const std::vector<int32_t>& prefix) const {
    std::vector<int32_t> toks;
    std::vector<uint64_t> mass;
    const size_t d = prefix.size();
    for (size_t i = 0; i < spans_.size(); ++i) {
      const auto& s = spans_[i];
      if (s.size() <= d || !std::equal(prefix.begin(), prefix.end(), s.begin())) continue;
      const int32_t t = s[d];
      auto it = std::lower_bound(toks.begin(), toks.end(), t);
      const size_t k = size_t(it - toks.begin());
      if (it == toks.end() || *it != t) {
        toks.insert(it, t);
        mass.insert(mass.begin() + std::ptrdiff_t(k), 0u);
      }
      mass[k] += mass_[i];
    }
    std::vector<PuctEdge> out(toks.size());
    if (toks.empty()) return out;
    uint64_t total = 0;
    for (uint64_t m : mass) total += m;
    for (size_t k = 0; k < toks.size(); ++k) {
      out[k].token = toks[k];
      uint64_t p;
      if (d == 0) p = mass[k];                                    // root: summed masses
      else if (total == 0) p = uint64_t(PUCT_Q20) / toks.size();  // degenerate: uniform
      else p = (mass[k] * uint64_t(PUCT_Q20)) / total;            // mid-span: renormalized
      out[k].prior_q20 = uint32_t(std::min<uint64_t>(p, 0xFFFFFFFFull));
    }
    return out;
  }

 private:
  bool fail() {
    spans_.clear();
    mass_.clear();
    stop_dropped_ = 0;
    return false;
  }
  std::vector<std::vector<int32_t>> spans_;  // input order
  std::vector<uint32_t> mass_;               // Q20 mass per concept
  size_t stop_dropped_ = 0;
};

}  // namespace pchie
