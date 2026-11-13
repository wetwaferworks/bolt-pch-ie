// Bolt PCHIE - DeterministicJoin: the join-order / reduction-order invariant.
// Concrete final. No mutex, no atomic, no virtual.
//
// Contract: a published combine of parallel work is a deterministic fold by STABLE WORK IDENTITY,
// never by arrival (completion) order. Workers deposit (work_id, value) in any order, each into its
// own slot; after the phase boundary (thread join) the coordinator calls fold(), which visits the
// results in ascending work_id. Duplicate ids and missing ids fail closed (no silent partial fold).
// Floating-point folds are therefore bit-identical across worker counts and arrival orders.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pchie {

template <class T>
class DeterministicJoin final {
 public:
  /// expected work ids are 0 .. count-1 (the caller's stable enumeration of the work).
  explicit DeterministicJoin(size_t count) : slots_(count), filled_(count, 0) {}

  /// One deposit per work id, from any worker, in any order. Each id owns its slot: two workers
  /// never write the same slot. Returns false (and records the error) on an out-of-range or repeat id.
  bool deposit(uint64_t work_id, T value) {
    if (work_id >= slots_.size()) { err_ = "work id out of range"; return false; }
    if (filled_[size_t(work_id)]) { err_ = "duplicate work id"; return false; }
    slots_[size_t(work_id)] = std::move(value);
    filled_[size_t(work_id)] = 1;
    return true;
  }

  bool complete() const {
    for (uint8_t f : filled_) if (!f) return false;
    return err_.empty();
  }

  /// Coordinator only, after the phase boundary: fold in ascending work id. Fails closed when any
  /// id is missing or a deposit was refused.
  template <class Acc, class F>
  bool fold(Acc init, F&& f, Acc& out, std::string& err) const {
    if (!err_.empty()) { err = err_; return false; }
    for (size_t i = 0; i < filled_.size(); ++i)
      if (!filled_[i]) { err = "missing work id " + std::to_string(i); return false; }
    Acc acc = std::move(init);
    for (size_t i = 0; i < slots_.size(); ++i) acc = f(std::move(acc), uint64_t(i), slots_[i]);
    out = std::move(acc);
    return true;
  }

  size_t size() const noexcept { return slots_.size(); }

 private:
  std::vector<T> slots_;
  std::vector<uint8_t> filled_;
  std::string err_;
};

}  // namespace pchie
