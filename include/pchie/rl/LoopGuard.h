// Loop guard v2: the reasoning layer's standard loop recovery, used by every maze driver
// (HeadlessMaze for B1 and Stage A, StageBDoorMaze for Stage B, and MazeRl's L1/null decide path).
// On by default since 13 November 2026; set_enabled(false) (--loop-guard off) restores the old behavior exactly.
//
// Every committed state is counted by a key the maze supplies (cell plus whatever else defines the state).
// When a state is entered for the 4th time (a loop repeated more than three times) the guard triggers:
//  - the next decision is a forced new decision point with a reasoning check: a fresh model forward on a prompt
//    that names the loop (note()), a fresh search seed (reseed()), and 4x the search budget (budget());
//  - that decision may not turn 180 (the state just left is taboo at the root, while any other move exists);
//  - cautious mode stays on for the rest of the run (drivers with unsafe cells, such as pits, check each step's
//    landing before it is a candidate).
// The model still chooses through its own search; no path is forced. Counts restart after the forced decision.
// This is v2. v1 banned every loop cell, which also banned the forward crossing and kept re-looping.
#pragma once

#include "pchie/NodeSeed.h"

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace pchie {
namespace rl {

class LoopGuard {
 public:
  static constexpr int kTriggerEntry = 4;                    // the 4th entry of one state triggers
  static constexpr int kBudgetMultiplier = 4;                // reasoning-check search budget
  static constexpr uint64_t kSeedSalt = 0x4C4F4F5047554152ULL;  // "LOOPGUAR"

  void set_enabled(bool on) { enabled_ = on; if (!on) reset(); }
  bool enabled() const noexcept { return enabled_; }

  void reset() {
    force_ = caution_ = have_prev_ = have_last_ = false;
    prev_ = last_ = 0; triggers_ = 0;
    visits_.clear(); taboo_.clear(); note_.clear(); log_.clear();
  }

  // Call once after every committed move, with the key of the state the move reached.
  // x, y and task_name are for the log and the reasoning-check note only.
  void after_commit(uint64_t key, size_t committed_steps, int x, int y, const char* task_name) {
    if (!enabled_) return;
    char b[240];
    if (force_) {  // the forced decision point has made its move: restart the counts
      force_ = false;
      taboo_.clear();
      visits_.clear();
      std::snprintf(b, sizeof b, "loop guard: reasoning check #%d committed step %zu -> cell (%d,%d)", triggers_,
                    committed_steps, x, y);
      log_.push_back(b);
    }
    const int n = ++visits_[key];
    prev_ = have_last_ ? last_ : key;
    have_prev_ = have_last_;
    last_ = key;
    have_last_ = true;
    if (n >= kTriggerEntry) {
      ++triggers_;
      if (have_prev_) taboo_.insert(prev_);  // the forced decision may not turn 180
      caution_ = true;                       // cautious crossing from here on
      force_ = true;
      std::snprintf(b, sizeof b, "loop guard: TRIGGER #%d at committed step %zu, cell (%d,%d) entered %d times; %zu loop cells; forcing a new decision point + reasoning check; cautious crossing ON",
                    triggers_, committed_steps, x, y, n, taboo_.size());
      log_.push_back(b);
      std::snprintf(b, sizeof b, "%s LOOP at (%d,%d) x%d. Do not turn back. Check the next cell and cross carefully forward.",
                    task_name ? task_name : "", x, y, n);
      note_ = b;
    }
  }

  bool forcing() const noexcept { return enabled_ && force_; }    // the next decision is the reasoning check
  bool cautious() const noexcept { return enabled_ && caution_; } // check landings before a step is a candidate
  bool taboo(uint64_t key) const { return taboo_.count(key) != 0; }
  uint64_t reseed(uint64_t step_seed) const {
    return forcing() ? NodeSeed::Splitmix64(step_seed ^ kSeedSalt ^ uint64_t(triggers_)) : step_seed;
  }
  int budget(int base) const { return forcing() ? base * kBudgetMultiplier : base; }
  const std::string& note() const noexcept { return note_; }
  int triggers() const noexcept { return triggers_; }
  const std::vector<std::string>& log() const noexcept { return log_; }

  // Root candidates for the forced decision: drop moves into a taboo state, unless nothing would be left.
  // next_key(action, key&) returns false for a move that cannot be applied; such a move is dropped, as it always was.
  template <class A, class NextKey>
  std::vector<A> root_filter(const std::vector<A>& legal, NextKey next_key) const {
    if (!forcing() || taboo_.empty() || legal.empty()) return legal;
    std::vector<A> keep;
    for (const A& a : legal) {
      uint64_t k = 0;
      if (next_key(a, k) && taboo_.count(k) == 0) keep.push_back(a);
    }
    return keep.empty() ? legal : keep;
  }

 private:
  bool enabled_ = true;  // the default
  bool force_ = false, caution_ = false, have_prev_ = false, have_last_ = false;
  uint64_t prev_ = 0, last_ = 0;
  int triggers_ = 0;
  std::unordered_map<uint64_t, int> visits_;
  std::set<uint64_t> taboo_;
  std::string note_;
  std::vector<std::string> log_;
};

}  // namespace rl
}  // namespace pchie
