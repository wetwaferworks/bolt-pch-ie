// Bolt PCHIE - PchGate: deterministic structural/syntax check, G in {0,1}.
// Concrete final (Archie amendment 1: no virtual IGate). Coordinator-thread only:
// no mutex, no condition_variable, no atomic.
//
// Provisional reject set: DefaultStrictStandInRejectIds(). Real PCH syntax rules
// are an open review question for Archie. StrictStandInGate
// remains a labeled TEST STAND-IN and is not this class.
#pragma once

#include "pchie/StrictStandInGate.h"
#include "pchie/pchie_tree.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pchie {

// Gate split: the gate sees ONLY proposal
// tokens - never the trusted prompt, BOS, or chat-template tokens. ProposalTokens is the stack
// contract: it is built from the search's proposed edge tokens (or committed action proposals),
// never from a prompt. One class per file is relaxed here for a 10-line value type.
class ProposalTokens final {
 public:
  static ProposalTokens from_proposal(std::vector<int32_t> ids) { return ProposalTokens(std::move(ids)); }
  const std::vector<int32_t>& ids() const noexcept { return ids_; }
  size_t size() const noexcept { return ids_.size(); }
 private:
  explicit ProposalTokens(std::vector<int32_t> ids) : ids_(std::move(ids)) {}
  std::vector<int32_t> ids_;
};

class PchGate final {
 public:
  PchGate() : PchGate(DefaultStrictStandInRejectIds()) {}

  explicit PchGate(std::vector<int32_t> reject_ids) {
    for (int32_t t : reject_ids) {
      if (t >= 0) reject_ids_.push_back(t);
    }
    std::sort(reject_ids_.begin(), reject_ids_.end());
    reject_ids_.erase(std::unique(reject_ids_.begin(), reject_ids_.end()), reject_ids_.end());
    uint64_t h = FNV_OFFSET;
    for (int32_t id : reject_ids_) {
      h = hash_mix64(h, static_cast<uint64_t>(static_cast<uint32_t>(id)));
    }
    reject_set_hash_ = h ? h : 1ull;
  }

  // Return false iff any path token is in the reject set (first hit wins for LastRejectToken).
  // Empty path -> true (root).
  // Stack contract (gate split): proposal tokens only.
  bool Pass(const ProposalTokens& proposal) const noexcept { return PassRaw(proposal.ids()); }

 private:
  bool PassRaw(const std::vector<int32_t>& path) const noexcept {
    for (int32_t t : path) {
      if (std::binary_search(reject_ids_.begin(), reject_ids_.end(), t)) {
        ++reject_count_;
        last_reject_token_ = t;
        return false;
      }
    }
    return true;
  }

 public:
  uint64_t RejectCount() const noexcept { return reject_count_; }
  int32_t LastRejectToken() const noexcept { return last_reject_token_; }
  const std::vector<int32_t>& RejectIds() const noexcept { return reject_ids_; }
  uint64_t RejectSetHash() const noexcept { return reject_set_hash_; }
  bool IsBypass() const noexcept { return reject_ids_.empty(); }

 private:
  std::vector<int32_t> reject_ids_;
  mutable uint64_t reject_count_ = 0;
  mutable int32_t last_reject_token_ = -1;
  uint64_t reject_set_hash_ = 0;
};

}  // namespace pchie