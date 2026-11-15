// Hook for the opt-in research layers (pchie/research, --research-* flags) inside the maze decide loop.
// A driver's MazeRl gets an advisor only when a research flag is on; official runners never set one, and a
// null or idle advisor leaves every decision, hash and chain exactly as before.
//
// What the advisor may do, per decision:
//  - read ResearchView: what Little Nemo sees from where he stands (his position as shown in his observation,
//    key state, the moves open to him, and pits, the key, a shut door or the exit one step away). Never the maze layout, never a distance.
//  - return proposals. A proposal is never a commit: it moves a share of root prior mass onto one root edge,
//    and only after the same legality and PchGate checks an L2 memory hit goes through (fail closed).
//  - return score caps (step scorer, enforce mode): a cap can only lower a root edge's prior, never raise it.
// The advisor keeps its own side-chain log; it never writes the official chain.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

struct ResearchView {
  int32_t x = 0;
  int32_t y = 0;
  bool have_key = false;
  bool key_here = false;
  bool door_near = false;   // a shut door next to him
  bool at_exit = false;
  uint8_t open_mask = 0;    // bit 0 North, 1 East, 2 South, 3 West: the move is open to him
  uint8_t danger_mask = 0;  // same bits: he sees a pit there
  uint8_t key_mask = 0;     // same bits: he sees the key one step away
  uint8_t door_mask = 0;    // same bits: he sees a shut door one step away
  uint8_t exit_mask = 0;    // same bits: he sees the exit one step away
};

struct ResearchProposal {
  int32_t token = 0;        // MazeAction id
  int32_t share_q20 = 0;    // share of root prior mass to move onto that edge
  uint8_t source = 0;       // which layer proposed it (advisor-defined)
};

struct ResearchCap {
  int32_t token = 0;
  int32_t score_q20 = 0;    // step score in [0, 2^20]; the prior becomes at most prior * (1 + score) / 2
};

class IResearchAdvisor {
 public:
  virtual ~IResearchAdvisor() = default;
  virtual bool engaged() const = 0;
  /// One decision. legal holds the root actions after legal-move masking. Return false to fail closed.
  virtual bool advise(uint32_t task, const ResearchView& view, const std::vector<int32_t>& legal,
                      std::vector<ResearchProposal>& proposals, std::vector<ResearchCap>& caps,
                      std::string& err) = 0;
  /// The action this decision picked (the driver commits it next, or refuses it) and what he would see
  /// once it is applied. The next advise() carries the real view; `after` matters only for the last step.
  virtual void decided(int32_t token, const ResearchView& after) = 0;
  virtual void end_run() {}
};

}  // namespace rl
}  // namespace pchie
