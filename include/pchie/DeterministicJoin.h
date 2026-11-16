// Bolt PCHIE - DeterministicJoin: the join-order / reduction-order invariant.
// Concrete final. No mutex and no virtual; per-slot atomics only record ownership and errors.
//
// Contract: a published combine of parallel work is a deterministic fold by STABLE WORK IDENTITY,
// never by arrival (completion) order. Workers deposit (work_id, value) in any order, each into its
// own slot; after the phase boundary (thread join) the coordinator calls fold(), which visits the
// results in ascending work_id. Duplicate ids and missing ids fail closed (no silent partial fold).
// Floating-point folds are therefore bit-identical across worker counts and arrival orders.
#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
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
  explicit DeterministicJoin(size_t count)
      : slots_(count), filled_(std::make_unique<std::atomic<uint8_t>[]>(count)), count_(count) {
    for (size_t i = 0; i < count_; ++i) filled_[i].store(0, std::memory_order_relaxed);
  }
  DeterministicJoin(const DeterministicJoin&) = delete;
  DeterministicJoin& operator=(const DeterministicJoin&) = delete;

  /// One deposit per work id, from any worker, in any order. An atomic exchange claims the slot, so a
  /// duplicate id cannot race the first writer. Errors are atomic flags; only fold() builds strings.
  bool deposit(uint64_t work_id, T value) {
    if (work_id >= count_) { out_of_range_.store(true, std::memory_order_relaxed); return false; }
    if (filled_[size_t(work_id)].exchange(1, std::memory_order_acq_rel)) {
      duplicate_.store(true, std::memory_order_relaxed);
      return false;
    }
    slots_[size_t(work_id)] = std::move(value);
    return true;
  }

  bool complete() const {
    if (out_of_range_.load(std::memory_order_relaxed) || duplicate_.load(std::memory_order_relaxed)) return false;
    for (size_t i = 0; i < count_; ++i) if (!filled_[i].load(std::memory_order_acquire)) return false;
    return true;
  }

  /// Coordinator only, after the phase boundary: fold in ascending work id. Fails closed when any
  /// id is missing or a deposit was refused.
  template <class Acc, class F>
  bool fold(Acc init, F&& f, Acc& out, std::string& err) const {
    if (out_of_range_.load(std::memory_order_relaxed)) { err = "work id out of range"; return false; }
    if (duplicate_.load(std::memory_order_relaxed)) { err = "duplicate work id"; return false; }
    for (size_t i = 0; i < count_; ++i)
      if (!filled_[i].load(std::memory_order_acquire)) { err = "missing work id " + std::to_string(i); return false; }
    Acc acc = std::move(init);
    for (size_t i = 0; i < slots_.size(); ++i) acc = f(std::move(acc), uint64_t(i), slots_[i]);
    out = std::move(acc);
    return true;
  }

  size_t size() const noexcept { return slots_.size(); }

 private:
  std::vector<T> slots_;
  std::unique_ptr<std::atomic<uint8_t>[]> filled_;
  size_t count_ = 0;
  std::atomic<bool> out_of_range_{false};
  std::atomic<bool> duplicate_{false};
};

}  // namespace pchie
