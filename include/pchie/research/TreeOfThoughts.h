// Tree of Thoughts (Yao et al., 2023) as an L3 extension. Little Nemo proposes a few short plans per
// node, a scorer rates each one, and only the best few survive to the next level (beam search).
// Proposals come out in a fixed order and get sequential ids. Scoring may run on many threads: each
// job writes only its own slot, and the join is the phase boundary. Survivors are ranked by
// (score high to low, id low to high), so the result is identical at every thread count.
#pragma once

#include "pchie/IJobDispatcher.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pchie {
namespace research {

struct Thought {
  uint32_t id = 0;
  uint32_t parent = 0;
  uint32_t depth = 0;
  std::string text;
  std::vector<uint8_t> moves;
  int32_t score_q20 = 0;
};

struct TotConfig {
  uint32_t breadth = 3;  // proposals kept per node
  uint32_t depth = 2;    // levels
  uint32_t beam = 3;     // survivors per level
};

struct TotResult {
  Thought best;
  std::vector<Thought> evaluated;  // every scored thought in id order
  uint32_t levels = 0;
  std::string digest;              // SHA-256 over ids, texts and scores
};

using ProposeFn = std::function<std::vector<Thought>(const Thought& parent, uint32_t breadth)>;
using EvaluateFn = std::function<int32_t(const Thought&)>;

class TreeOfThoughts {
 public:
  explicit TreeOfThoughts(TotConfig cfg = {}) : cfg_(cfg) {}
  /// evaluate must be pure (it may run on several threads at once). jobs may be null (serial).
  TotResult run(const Thought& root, const ProposeFn& propose, const EvaluateFn& evaluate,
                IJobDispatcher* jobs) const;

 private:
  TotConfig cfg_;
};

}  // namespace research
}  // namespace pchie
