// Bolt PCHIE B5 - coordinator watchdog for mid-search freeze and recovery.
// Concrete final. No mutex, no condition_variable, no atomic. Serial slots only
// (the LithoLattice threaded TimingHandshakeAuthority is not used).
// A forced-freeze hook is deterministic (not a wall-clock ceiling), so H_bolt
// stays stable. resetAfterFreeze clears the latch and bumps a generation so a
// stale or stuck reset cannot commit. Slot ids are unique; a reserved slot is
// never reissued (no duplicate slot).
#pragma once

#include <cstdint>
#include <vector>

namespace pchie {

struct FreezeInfo {
  uint32_t slot = 0;
  uint32_t generation = 0;
  uint32_t task_id = 0;
  int step = -1;
  int iter = -1;
  bool valid = false;
};

class FreezeWatchdog final {
 public:
  void reset() {
    slots_.clear();
    next_id_ = 1;
    generation_ = 1;
    frozen_ = false;
    last_ = FreezeInfo{};
  }

  uint32_t generation() const noexcept { return generation_; }
  bool frozen() const noexcept { return frozen_; }
  const FreezeInfo& last_freeze() const { return last_; }

  uint32_t allocate_slot() {
    const uint32_t id = next_id_;
    if (next_id_ < UINT32_MAX) ++next_id_;
    Slot s;
    s.id = id;
    slots_.push_back(s);
    return id;
  }

  bool begin_rollout(uint32_t slot) {
    Slot* s = find(slot);
    if (!s || s->in_flight || s->discarded || s->reset_stuck) return false;
    s->in_flight = true;
    s->generation = generation_;
    return true;
  }

  bool force_freeze(uint32_t slot, uint32_t task_id, int step, int iter) {
    Slot* s = find(slot);
    if (!s || !s->in_flight) return false;
    if (frozen_) return false;
    s->in_flight = false;
    s->discarded = true;
    frozen_ = true;
    last_.slot = slot;
    last_.generation = generation_;
    last_.task_id = task_id;
    last_.step = step;
    last_.iter = iter;
    last_.valid = true;
    return true;
  }

  void resetAfterFreeze() {
    frozen_ = false;
    if (generation_ < UINT32_MAX) ++generation_;
    for (Slot& s : slots_) {
      if (s.in_flight) s.in_flight = false;
    }
  }

  void mark_reset_stuck(uint32_t slot) {
    Slot* s = find(slot);
    if (!s) return;
    s->reset_stuck = true;
    s->discarded = true;
    s->in_flight = false;
  }

  void finish_reset(uint32_t slot) {
    Slot* s = find(slot);
    if (!s) return;
    s->reset_stuck = false;
    s->in_flight = false;
  }

  bool try_commit(uint32_t slot, uint32_t generation) {
    if (frozen_) return false;
    const Slot* s = find(slot);
    if (!s) return false;
    if (s->discarded || s->reset_stuck) return false;
    if (!s->in_flight) return false;
    if (generation != generation_) return false;
    return true;
  }

  void end_rollout(uint32_t slot) {
    Slot* s = find(slot);
    if (!s) return;
    if (s->discarded || s->reset_stuck) return;
    s->in_flight = false;
  }

  bool slot_discarded(uint32_t slot) const {
    const Slot* s = find(slot);
    return s && s->discarded;
  }

  bool slot_reserved(uint32_t slot) const {
    const Slot* s = find(slot);
    if (!s) return false;
    if (s->reset_stuck) return true;
    if (s->in_flight) return true;
    if (s->discarded && frozen_) return true;
    return false;
  }

  bool has_duplicate_slot() const {
    for (size_t i = 0; i < slots_.size(); ++i)
      for (size_t j = i + 1; j < slots_.size(); ++j)
        if (slots_[i].id == slots_[j].id) return true;
    return false;
  }

  int live_in_flight() const noexcept {
    int n = 0;
    for (const Slot& s : slots_)
      if (s.in_flight) ++n;
    return n;
  }

 private:
  struct Slot {
    uint32_t id = 0;
    uint32_t generation = 0;
    bool in_flight = false;
    bool discarded = false;
    bool reset_stuck = false;
  };

  Slot* find(uint32_t slot) {
    for (Slot& s : slots_)
      if (s.id == slot) return &s;
    return nullptr;
  }
  const Slot* find(uint32_t slot) const {
    for (const Slot& s : slots_)
      if (s.id == slot) return &s;
    return nullptr;
  }

  std::vector<Slot> slots_;
  uint32_t next_id_ = 1;
  uint32_t generation_ = 1;
  bool frozen_ = false;
  FreezeInfo last_{};
};

}  // namespace pchie
