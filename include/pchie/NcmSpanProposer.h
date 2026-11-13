// Bolt PCHIE - NcmSpanProposer: multi-token NCM spans from a live model.
// Concrete final. Coordinator-thread only: no mutex, no atomic.
// For the top `branches` root tokens (logit descending, ties lowest id) it forks the root,
// then extends greedily (argmax, ties lowest id) up to `span_len` tokens, stopping after a
// turn-boundary token. Score = raw root logit of the first token + sum of ln P of each
// continuation token (LogitStats, temperature 1). That keeps spans on the same softmax
// scale as NcmCandidateGenerator's raw-logit 1-token concepts they are merged with:
// exp(score) = exp(logit t1) * P(t2..tn | t1), i.e. a span's mass is its first token's
// mass times its continuation probability.
// Every fork is released before return; `root` is never mutated. model.logits() is
// overwritten, so callers pass a frozen copy of the root logits (same rule as PchStack).
// Default (logged for review): this is a separate proposer; NcmCandidateGenerator still
// emits 1-token concepts only.
#pragma once

#include "pchie/IModel.h"
#include "pchie/LogitStats.h"
#include "pchie/ModelBranch.h"
#include "pchie/NcmCandidateGenerator.h"
#include "pchie/NcmConcept.h"
#include "pchie/TurnBoundaryTerminalPolicy.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class NcmSpanProposer final {
 public:
  NcmSpanProposer(int branches = 4, int span_len = 3)
      : branches_(branches), span_len_(span_len) {}

  bool propose(IModel& m, const ModelBranch& root, const std::vector<float>& root_logits,
               uint32_t id_base, std::vector<NcmConcept>& out, std::string& err) const {
    out.clear();
    err.clear();
    if (branches_ <= 0 || span_len_ <= 0 || span_len_ > int(kNcmMaxSpan)) { err = "config"; return false; }
    if (root_logits.empty()) { err = "no root logits"; return false; }
    const TurnBoundaryTerminalPolicy stop({1, 106});
    const int B = std::min<int>(branches_, int(root_logits.size()));
    std::vector<int32_t> idx(root_logits.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = int32_t(i);
    std::partial_sort(idx.begin(), idx.begin() + B, idx.end(), [&](int32_t a, int32_t b) {
      const float la = root_logits[size_t(a)], lb = root_logits[size_t(b)];
      return la != lb ? la > lb : a < b;
    });
    for (int bi = 0; bi < B; ++bi) {
      std::vector<int32_t> toks{idx[size_t(bi)]};
      double score = double(root_logits[size_t(toks[0])]);
      if (!stop.IsTerminal(toks[0]) && span_len_ > 1) {
        ModelBranch w = m.fork(root);
        int32_t cur = toks[0];
        for (int s = 1; s < span_len_; ++s) {
          m.forward(w, cur);
          const std::vector<float> lg = m.logits();  // copy: LogitStats keeps a reference
          if (lg.empty()) break;
          int32_t best = 0;
          for (size_t i = 1; i < lg.size(); ++i)
            if (lg[i] > lg[size_t(best)]) best = int32_t(i);
          score += LogitStats(lg).LogProb(best);
          toks.push_back(best);
          cur = best;
          if (stop.IsTerminal(best)) break;
        }
        m.release(w);
      }
      out.emplace_back(id_base + uint32_t(bi), "span" + std::to_string(bi), std::move(toks), score);
    }
    return true;
  }

 private:
  int branches_;
  int span_len_;
};

}  // namespace pchie
