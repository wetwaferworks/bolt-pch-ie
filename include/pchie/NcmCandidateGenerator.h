// Bolt PCHIE - NcmCandidateGenerator.
// Concrete final. Coordinator-thread only: no mutex, no atomic.
// Order: score descending, then id ascending (stable). max_span = 16.
// NaN scores are rejected/skipped. +/-Inf and finite negatives are kept.
// propose_from_logits scores are raw logits (not log-softmax); each entry is a
// 1-token concept whose id equals the logit index. max_candidates <= 0 means all.
#pragma once

#include "pchie/NcmConcept.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace pchie {

constexpr uint32_t kNcmMaxSpan = 16;

class NcmCandidateGenerator final {
 public:
  bool add(NcmConcept c) {
    if (!accept(c)) return false;
    items_.push_back(std::move(c));
    sort_now();
    return true;
  }

  void clear() { items_.clear(); }

  // Replaces any previous candidates. Returns true even if the result is empty.
  bool propose_from_logits(const std::vector<float>& logits, int max_candidates = 0) {
    items_.clear();
    const size_t n = logits.size();
    // Token ids are int32_t; drop any index that would not fit.
    const size_t n_ok = n > size_t(std::numeric_limits<int32_t>::max())
                            ? size_t(std::numeric_limits<int32_t>::max()) + 1
                            : n;
    items_.reserve(n_ok);
    for (size_t i = 0; i < n_ok; ++i) {
      if (std::isnan(logits[i])) continue;
      NcmConcept c(static_cast<uint32_t>(i),
                   std::string(),
                   {static_cast<int32_t>(i)},
                   static_cast<double>(logits[i]));
      if (accept(c)) items_.push_back(std::move(c));
    }
    sort_now();
    if (max_candidates > 0 && items_.size() > static_cast<size_t>(max_candidates)) {
      items_.resize(static_cast<size_t>(max_candidates));
    }
    return true;
  }

  const std::vector<NcmConcept>& candidates() const { return items_; }
  uint32_t max_span() const noexcept { return kNcmMaxSpan; }

 private:
  static bool accept(const NcmConcept& c) {
    if (c.tokens().empty() || c.tokens().size() > kNcmMaxSpan) return false;
    if (std::isnan(c.score())) return false;
    return true;
  }

  void sort_now() {
    std::stable_sort(items_.begin(), items_.end(),
                     [](const NcmConcept& a, const NcmConcept& b) {
                       if (a.score() != b.score()) return a.score() > b.score();
                       return a.id() < b.id();
                     });
  }

  std::vector<NcmConcept> items_;
};

}  // namespace pchie