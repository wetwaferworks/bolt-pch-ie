// Step scorer (process reward) used as the L6 verifier score. It judges one proposed step from what
// Little Nemo knows at that moment, not from the world. Scores are Q20 integers in [0, 2^20].
// Two scorers ship here: a hand-weighted rule scorer (the labeled baseline) and a frozen table
// scorer whose weights live in a text file pinned by SHA-256 (the slot a trained scorer fills).
// In record mode the score is only logged. In enforce mode it may lower a prior, never raise one.
#pragma once

#include "pchie/research/ExperienceMap.h"
#include "pchie/research/SelfObservation.h"

#include <array>
#include <cstdint>
#include <string>

namespace pchie {
namespace research {

constexpr int32_t kQ20 = 1 << 20;

enum StepFeature : int {
  kFeatUntried = 0,     // the edge has never been tried
  kFeatOpen,            // the edge is known open
  kFeatBlocked,         // he bumped into it before
  kFeatDanger,          // he sees a pit there
  kFeatReverse,         // straight back the way he came
  kFeatNewPlace,        // the place beyond has never been stood on
  kFeatCrowded,         // the place beyond was visited three or more times
  kFeatTowardGoal,      // shortens his known path to the current subgoal
  kFeatAwayFromGoal,    // lengthens it
  kFeatures
};

struct StepFeatures {
  std::array<int32_t, kFeatures> v{};
};

/// Build features for move m from his own map and observation. goal_delta: -1 closer, 0 unknown, +1 farther.
StepFeatures step_features(const ExperienceMap& map, const SelfObservation& o, int m, int goal_delta);

class IStepScorer {
 public:
  virtual ~IStepScorer() = default;
  virtual const char* name() const = 0;
  virtual int32_t score_q20(const StepFeatures& f) const = 0;
  virtual std::string artifact_sha256() const = 0;
};

struct StepWeights {
  int32_t bias = kQ20 / 2;
  std::array<int32_t, kFeatures> w{};
};

/// Clamp bias + sum(w * f) to [0, 2^20].
int32_t weighted_score(const StepWeights& sw, const StepFeatures& f);

class RuleStepScorer final : public IStepScorer {
 public:
  RuleStepScorer();
  const char* name() const override { return "rule"; }
  int32_t score_q20(const StepFeatures& f) const override { return weighted_score(w_, f); }
  std::string artifact_sha256() const override;
  const StepWeights& weights() const { return w_; }
  /// The weights in the frozen table text format.
  std::string table_text() const;

 private:
  StepWeights w_;
};

class FrozenTableStepScorer final : public IStepScorer {
 public:
  /// Fails closed (loaded() == false) when the text does not parse or its SHA-256 differs from the pin.
  FrozenTableStepScorer(const std::string& table_text, const std::string& pinned_sha256);
  bool loaded() const { return loaded_; }
  const std::string& error() const { return err_; }
  const char* name() const override { return "frozen_table"; }
  int32_t score_q20(const StepFeatures& f) const override { return loaded_ ? weighted_score(w_, f) : 0; }
  std::string artifact_sha256() const override { return sha_; }

 private:
  StepWeights w_;
  bool loaded_ = false;
  std::string sha_;
  std::string err_;
};

/// Enforce mode: the new prior is never above the old one.
int32_t enforce_prior_q20(int32_t prior_q20, int32_t score_q20);

const char* feature_name(int f);

}  // namespace research
}  // namespace pchie
