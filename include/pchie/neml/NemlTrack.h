// Bolt PCHIE - NEML per-event triggers for grid runs (headless maze and Stage B).
// Watches committed steps only (never the search) and fires the four salience triggers as they happen:
//   ForkPivot      the committed move differs from the move with the highest root prior, with 2+ moves open
//   DeadEnd        the step enters a new cell whose only open way is back (key and door cells excepted);
//                  recorded at the last junction the branch left, with the direction taken out of it
//   VisualRecovery the first new cell reached after a dead end
//   ExitFound      the step reaches the exit
// Read-only with respect to the run: it never changes an action, a prior, or a chain record.
#pragma once
#include "pchie/MazeGrid.h"
#include "pchie/neml/NemlLedger.h"
#include <cstdint>
#include <set>
#include <utility>

namespace pchie {
namespace neml {

inline uint8_t dir_of_action(int32_t tok) {
  switch (static_cast<MazeAction>(tok)) {
    case MazeAction::North: return kDirN;
    case MazeAction::South: return kDirS;
    case MazeAction::East: return kDirE;
    case MazeAction::West: return kDirW;
    default: return kDirNone;
  }
}

template <class G>
int open_moves(const G& g) {
  int n = 0;
  for (MazeAction a : g.legal_actions()) if (dir_of_action(int32_t(a)) != kDirNone) ++n;
  return n;
}

class GridTracker {
 public:
  template <class G>
  void begin(const G& g) {
    rec = Recorder{};
    visited_.clear();
    visited_.insert({g.x(), g.y()});
    have_branch_ = recovering_ = exit_seen = false;
  }
  // pre/post: the grid before and after the committed step. top_tok: token of the highest-prior root edge
  // (-1 if unknown). root_moves: open moves the decision chose among. special(post): key/door cell.
  template <class G, class Special>
  void step(const G& pre, int32_t act, const G& post, int32_t top_tok, int root_moves, int step_index, Special special) {
    const uint8_t d = dir_of_action(act);
    if (d != kDirNone) {
      if (open_moves(pre) >= 3 || (!have_branch_ && open_moves(pre) >= 2)) {
        have_branch_ = true; bx_ = pre.x(); by_ = pre.y(); bd_ = d;
      }
      if (root_moves >= 2 && top_tok >= 0 && dir_of_action(top_tok) != kDirNone && top_tok != act)
        rec.event(kForkPivot, step_index, 128, pre.x(), pre.y(), "pivot_from_top_prior", d);
      const bool fresh = visited_.insert({post.x(), post.y()}).second;
      if (fresh && open_moves(post) <= 1 && !post.at_exit() && !special(post)) {
        if (have_branch_) rec.event(kDeadEnd, step_index, 255, bx_, by_, "dead_end_branch", bd_);
        else rec.event(kDeadEnd, step_index, 255, post.x(), post.y(), "dead_end", kDirNone);
        recovering_ = true;
      } else if (fresh && recovering_) {
        rec.event(kVisualRecovery, step_index, 255, post.x(), post.y(), "recovered_new_cell");
        recovering_ = false;
      }
    }
    if (post.at_exit() && !exit_seen) { rec.event(kExitFound, step_index, 255, post.x(), post.y(), "exit"); exit_seen = true; }
  }
  Recorder rec;
  bool exit_seen = false;

 private:
  std::set<std::pair<int, int>> visited_;
  bool have_branch_ = false, recovering_ = false;
  int bx_ = 0, by_ = 0;
  uint8_t bd_ = kDirNone;
};

// Highest-prior root edge (ties -> lower token); -1 if none.
template <class Edges>
int32_t top_prior_token(const Edges& edges) {
  int32_t best = -1;
  uint32_t bp = 0;
  for (const auto& e : edges)
    if (best < 0 || e.prior_q20 > bp || (e.prior_q20 == bp && e.token < best)) { best = e.token; bp = e.prior_q20; }
  return best;
}

}  // namespace neml
}  // namespace pchie