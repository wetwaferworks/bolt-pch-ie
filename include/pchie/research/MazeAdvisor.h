// The research layers as one advisor for the grid-maze decide loop (pchie/rl/ResearchAdvisor.h).
//   Go-Explore (L7)        proposes its next move (return trip or exploration) as a root prior share.
//   Look-ahead             proposes the best first move inside the frozen dynamics model.
//   Step scorer            record: logs a score per root move; enforce: caps root priors (lower only).
//   Reflexion              at the end of the run, writes lessons from his own steps to the lesson ledger.
// It sees only rl::ResearchView. Positions are kept relative to where he first stood. Every decision, its
// proposals and the transition it caused go on the research side chain (--research-log).
#pragma once

#include "pchie/research/FrozenDynamics.h"
#include "pchie/research/GoExplore.h"
#include "pchie/research/Reflexion.h"
#include "pchie/research/ResearchFlags.h"
#include "pchie/research/SideLog.h"
#include "pchie/research/StepScorer.h"
#include "pchie/rl/ResearchAdvisor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pchie {
namespace research {

enum ProposalSource : uint8_t { kFromGoExplore = 1, kFromLookahead = 2 };

struct MazeAdvisorStats {
  uint64_t decisions = 0;
  uint64_t go_explore_proposals = 0;
  uint64_t lookahead_proposals = 0;
  uint64_t lookahead_unknown_state = 0;
  uint64_t scored_moves = 0;
  uint64_t transitions = 0;
  uint64_t lessons = 0;
};

class MazeResearchAdvisor final : public rl::IResearchAdvisor {
 public:
  explicit MazeResearchAdvisor(ResearchFlags flags);

  /// Loads the frozen dynamics (when the look-ahead is on). Call once before the run; false = fail closed.
  bool prepare(std::string& err);

  bool engaged() const override;
  bool advise(uint32_t task, const rl::ResearchView& view, const std::vector<int32_t>& legal,
              std::vector<rl::ResearchProposal>& proposals, std::vector<rl::ResearchCap>& caps,
              std::string& err) override;
  void decided(int32_t token, const rl::ResearchView& after) override;
  void end_run() override;

  const MazeAdvisorStats& stats() const { return st_; }
  const SideLog& log() const { return log_; }
  const std::vector<Transition>& transitions() const { return transitions_; }
  const std::string& error() const { return err_; }
  /// SHA-256 over the side chain head and the Go-Explore digest.
  std::string digest() const;

  int32_t go_explore_share_q20 = kQ20 / 2;
  int32_t lookahead_share_q20 = kQ20 / 2;

  /// Model state used by the look-ahead and by recorded transitions.
  static uint64_t model_state(int32_t x, int32_t y, bool have_key, bool opened);
  static int act_of_token(int32_t token);   // MazeAction id -> research action, -1 if none
  static int32_t token_of_act(int act);

 private:
  SelfObservation observe(const rl::ResearchView& v) const;
  void close_transition(const SelfObservation& now);

  ResearchFlags f_;
  GoExplorePolicy explore_;
  ExperienceMap map_;
  RuleStepScorer scorer_;
  FrozenDynamics model_;
  bool model_ready_ = false;
  SideLog log_;
  MazeAdvisorStats st_;
  std::string err_;

  bool have_origin_ = false;
  int32_t ox_ = 0, oy_ = 0;
  bool opened_ = false;
  uint8_t heading_ = kNorth;
  bool pending_ = false;
  int32_t pending_token_ = 0;
  SelfObservation pending_from_{};
  rl::ResearchView pending_after_{};
  bool pending_from_opened_ = false;
  SelfObservation last_obs_{};
  uint32_t last_task_ = 0;
  std::vector<rl::ResearchProposal> last_props_;
  std::vector<Transition> transitions_;
  RunTrace trace_;
};

}  // namespace research
}  // namespace pchie
