// Bolt PCHIE - coordinator-thread CPU leaf path: PchGate + LogProbLeafEvaluator.
// New path only; does not replace the original AcceptAllGate path.
// Concrete final (no new virtuals). No mutex, no atomic.
#pragma once

#include "pchie/LeafContext.h"
#include "pchie/LogitStats.h"
#include "pchie/LogProbLeafEvaluator.h"
#include "pchie/PchGate.h"

#include <cstdint>
#include <vector>

namespace pchie {

class PchCpuLeafPath final {
 public:
  explicit PchCpuLeafPath(double alpha = 0.7, double beta = 0.0, PchGate gate = PchGate())
      : eval_(alpha, beta), gate_(std::move(gate)) {}

  // Fills last_context() and returns V in [0, 1]. G = 0 (reject) yields 0.
  // is_terminal: entropy forced to 0 (Archie ruling 2).
  // Gate split: `proposal` is what the gate sees (proposal tokens only); `depth` is the full
  // node depth the caller computes (unchanged meaning).
  double Evaluate(const ProposalTokens& proposal,
                  const std::vector<float>& logits,
                  uint32_t depth,
                  double sum_log_prob,
                  const float* hidden_state = nullptr,
                  size_t hidden_size = 0,
                  bool is_terminal = false) {
    last_ = LeafContext{};
    last_.depth = depth;
    last_.sum_log_prob = sum_log_prob;
    last_.gate_pass = gate_.Pass(proposal);
    last_.hidden_state = hidden_state;
    last_.hidden_size = hidden_size;
    if (is_terminal || logits.empty()) {
      last_.entropy = 0.0;
    } else {
      last_.entropy = LogitStats(logits).Entropy();
    }
    return eval_.EvalLeaf(last_);
  }

  const LeafContext& last_context() const { return last_; }
  PchGate& gate() { return gate_; }
  const PchGate& gate() const { return gate_; }
  const LogProbLeafEvaluator& evaluator() const { return eval_; }

 private:
  LogProbLeafEvaluator eval_;
  PchGate gate_;
  LeafContext last_;
};

}  // namespace pchie