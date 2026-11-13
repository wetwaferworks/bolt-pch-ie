// L6. Labeled symbolic planner (peer) + thin verifier.
// The planner is a proposal source beside Little Nemo and NCM, never a hidden
// value oracle. The verifier can only downgrade or reject; it cannot pick a
// move or raise a prior. Start in record mode (log verdicts only).
// On the vision path it plans only over the L4 belief, never the world map.
#pragma once

#include "pchie/rl/BeliefState.h"
#include "pchie/rl/ForkScore.h"
#include "pchie/rl/RlCommon.h"

#include <string>
#include <vector>

namespace pchie {
namespace rl {

enum class ProposalSource : uint8_t { Model = 1, Ncm = 2, Symbolic = 3, Memory = 4, Human = 5 };

enum class Subgoal : uint8_t { ReachExit = 0, GoToDoor0 = 1, GoToDoor1 = 2, Explore = 3, EscapeLoop = 4 };

struct Proposal {
  ProposalSource src = ProposalSource::Symbolic;
  std::vector<int32_t> steps;
  int32_t prior_q20 = 0;
  const char* rationale_label = "";
};

class IProposalSource {
 public:
  virtual ~IProposalSource() = default;
  virtual std::vector<Proposal> propose(const BeliefState& belief, Subgoal subgoal) = 0;
  virtual const char* name() const = 0;
};

enum class Verdict : uint8_t { Coherent = 0, Incoherent = 1, Uncertain = 2, Escalate = 3 };

constexpr uint32_t kReasonLegalUnderBelief = 1u << 0;
constexpr uint32_t kReasonMovesIntoKnownWall = 1u << 1;
constexpr uint32_t kReasonMovesIntoKnownPit = 1u << 2;
constexpr uint32_t kReasonImmediateLoop = 1u << 3;
constexpr uint32_t kReasonRevisitsWithoutNewInfo = 1u << 4;
constexpr uint32_t kReasonExploresUnknown = 1u << 5;
constexpr uint32_t kReasonProgressTowardSubgoal = 1u << 6;
constexpr uint32_t kReasonIgnoresVisibleExit = 1u << 7;
constexpr uint32_t kReasonOverBudget = 1u << 8;
constexpr uint32_t kReasonUsesHiddenInfo = 1u << 9;

struct VerifierResult {
  Verdict v = Verdict::Uncertain;
  uint32_t reason_bits = 0;
  int32_t score_q20 = 0;
};

class IPlanVerifier {
 public:
  virtual ~IPlanVerifier() = default;
  virtual VerifierResult check(const DecisionTuple& tuple, const Proposal& prop,
                               const BeliefState& belief) = 0;
};

// Thin verifier. Legal-under-belief, no immediate 2-cycle, no move into
// ObservedWall / ObservedPit / Bumped. Cannot raise priors (see apply_verdict).
class ThinPlanVerifier final : public IPlanVerifier {
 public:
  VerifierResult check(const DecisionTuple& tuple, const Proposal& prop,
                       const BeliefState& belief) override {
    (void)tuple;
    VerifierResult r;
    r.v = Verdict::Coherent;
    r.reason_bits = kReasonLegalUnderBelief;
    r.score_q20 = kQ20;
    const std::vector<int32_t>& steps =
        !prop.steps.empty() ? prop.steps : std::vector<int32_t>{tuple_decision_placeholder(tuple)};
    (void)steps;
    const auto& plan = prop.steps;
    if (plan.size() >= 2) {
      // Immediate 2-cycle: A, opposite(A) as first two primitives (0/2 or 1/3).
      if (is_opposite(plan[0], plan[1])) {
        r.v = Verdict::Incoherent;
        r.reason_bits = kReasonImmediateLoop;
        r.score_q20 = 0;
        return r;
      }
    }
    // Simulate primitives over the belief without mutating it.
    Pose p = belief.pose();
    for (int32_t a : plan) {
      if (a < 0 || a > 3) continue;  // option ids are checked separately
      Heading h = Heading(a);
      const int32_t nx = p.x + dx_of(h);
      const int32_t ny = p.y + dy_of(h);
      const Cell c = belief.cell(nx, ny);
      if (c == Cell::Wall || c == Cell::Bumped) {
        r.v = Verdict::Incoherent;
        r.reason_bits = kReasonMovesIntoKnownWall;
        r.score_q20 = 0;
        return r;
      }
      if (c == Cell::Pit) {
        r.v = Verdict::Incoherent;
        r.reason_bits = kReasonMovesIntoKnownPit;
        r.score_q20 = 0;
        return r;
      }
      if (c == Cell::Unknown) r.reason_bits |= kReasonExploresUnknown;
      p.x = nx;
      p.y = ny;
      p.heading = h;
    }
    return r;
  }

 private:
  static bool is_opposite(int32_t a, int32_t b) {
    if (a < 0 || a > 3 || b < 0 || b > 3) return false;
    return ((a + 2) % 4) == b;
  }
  static int32_t tuple_decision_placeholder(const DecisionTuple& t) { return t.decision; }
};

// The verifier may only mask or downgrade. It never raises prior_q20.
inline int32_t apply_verdict(int32_t prior_q20, const VerifierResult& vr, VerifierMode mode) {
  if (prior_q20 < 0) prior_q20 = 0;
  if (mode == VerifierMode::Record) return prior_q20;  // log only; prior unchanged
  if (vr.v == Verdict::Incoherent) return 0;
  if (vr.v == Verdict::Uncertain || vr.v == Verdict::Escalate) {
    const int32_t down = vr.score_q20 < prior_q20 ? vr.score_q20 : prior_q20;
    return down < 0 ? 0 : down;
  }
  // Coherent: still cannot raise.
  return prior_q20;
}

inline bool verdict_raised_prior(int32_t prior_before, int32_t prior_after) {
  return prior_after > prior_before;
}

// Stub symbolic peer. Not for runs until the maintainer reviews the label schema / stubs.
// Proposes a single legal primitive under the belief, tagged source=symbolic.
class StubSymbolicPeer final : public IProposalSource {
 public:
  std::vector<Proposal> propose(const BeliefState& belief, Subgoal) override {
    Proposal p;
    p.src = ProposalSource::Symbolic;
    p.rationale_label = "STUB_AWAITING_JD_REVIEW";
    const ActionMask m = belief.mask();
    const Heading hs[4] = {Heading::N, Heading::E, Heading::S, Heading::W};
    for (Heading h : hs) {
      if (m.allowed(h)) {
        p.steps = {int32_t(h)};
        p.prior_q20 = kQ20 / 4;
        return {p};
      }
    }
    p.steps = {};
    p.prior_q20 = 0;
    return {p};
  }
  const char* name() const override { return "symbolic-stub"; }
};

}  // namespace rl
}  // namespace pchie
