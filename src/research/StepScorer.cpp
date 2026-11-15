#include "pchie/research/StepScorer.h"

#include "pchie/research/Digest.h"

#include <cstdlib>
#include <sstream>

namespace pchie {
namespace research {

const char* feature_name(int f) {
  static const char* kNames[kFeatures] = {"untried", "open",     "blocked",      "danger",          "reverse",
                                          "new_place", "crowded", "toward_goal", "away_from_goal"};
  return (f >= 0 && f < kFeatures) ? kNames[f] : "?";
}

StepFeatures step_features(const ExperienceMap& map, const SelfObservation& o, int m, int goal_delta) {
  StepFeatures f;
  if (m < 0 || m >= kMoves) return f;
  const EdgeState e = map.edge(o.x, o.y, m);
  f.v[kFeatUntried] = e == EdgeState::Unknown;
  f.v[kFeatOpen] = e == EdgeState::Open;
  f.v[kFeatBlocked] = e == EdgeState::Blocked;
  f.v[kFeatDanger] = (o.danger_mask >> m) & 1;
  f.v[kFeatReverse] = m == reverse_move(o.heading);
  const Place* q = map.find(o.x + move_dx(m), o.y + move_dy(m));
  f.v[kFeatNewPlace] = (q == nullptr || q->visits == 0);
  f.v[kFeatCrowded] = (q != nullptr && q->visits >= 3);
  f.v[kFeatTowardGoal] = goal_delta < 0;
  f.v[kFeatAwayFromGoal] = goal_delta > 0;
  return f;
}

int32_t weighted_score(const StepWeights& sw, const StepFeatures& f) {
  int64_t s = sw.bias;
  for (int i = 0; i < kFeatures; ++i) s += int64_t(sw.w[size_t(i)]) * f.v[size_t(i)];
  if (s < 0) s = 0;
  if (s > kQ20) s = kQ20;
  return int32_t(s);
}

RuleStepScorer::RuleStepScorer() {
  w_.bias = kQ20 / 2;
  w_.w[kFeatUntried] = kQ20 / 8;
  w_.w[kFeatOpen] = 0;
  w_.w[kFeatBlocked] = -kQ20 / 2;
  w_.w[kFeatDanger] = -kQ20;
  w_.w[kFeatReverse] = -kQ20 / 16;
  w_.w[kFeatNewPlace] = kQ20 / 8;
  w_.w[kFeatCrowded] = -kQ20 / 8;
  w_.w[kFeatTowardGoal] = kQ20 / 4;
  w_.w[kFeatAwayFromGoal] = -kQ20 / 8;
}

std::string RuleStepScorer::table_text() const {
  std::ostringstream o;
  o << "pchie.step_scorer.v1\nbias " << w_.bias << '\n';
  for (int i = 0; i < kFeatures; ++i) o << feature_name(i) << ' ' << w_.w[size_t(i)] << '\n';
  return o.str();
}

std::string RuleStepScorer::artifact_sha256() const { return sha256_hex(table_text()); }

FrozenTableStepScorer::FrozenTableStepScorer(const std::string& text, const std::string& pinned) {
  sha_ = sha256_hex(text);
  if (sha_ != pinned) {
    err_ = "table does not match its pin";
    return;
  }
  std::istringstream in(text);
  std::string line;
  if (!std::getline(in, line) || line != "pchie.step_scorer.v1") {
    err_ = "bad header";
    return;
  }
  bool seen[kFeatures] = {};
  bool bias = false;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ls(line);
    std::string key;
    long long val = 0;
    if (!(ls >> key >> val) || val < -(int64_t(kQ20) * 4) || val > int64_t(kQ20) * 4) {
      err_ = "bad line";
      return;
    }
    if (key == "bias") {
      w_.bias = int32_t(val);
      bias = true;
      continue;
    }
    int idx = -1;
    for (int i = 0; i < kFeatures; ++i)
      if (key == feature_name(i)) idx = i;
    if (idx < 0 || seen[idx]) {
      err_ = "unknown or repeated feature";
      return;
    }
    seen[idx] = true;
    w_.w[size_t(idx)] = int32_t(val);
  }
  for (int i = 0; i < kFeatures; ++i)
    if (!seen[i]) {
      err_ = "missing feature";
      return;
    }
  if (!bias) {
    err_ = "missing bias";
    return;
  }
  loaded_ = true;
}

int32_t enforce_prior_q20(int32_t prior, int32_t score) {
  if (prior <= 0) return 0;
  if (score < 0) score = 0;
  if (score > kQ20) score = kQ20;
  // prior * (1 + score) / 2 for score in [0, 1]: at most the prior, at least half of it.
  const int64_t v = (int64_t(prior) * (int64_t(score) + kQ20)) >> 21;
  return int32_t(v < prior ? v : prior);
}

}  // namespace research
}  // namespace pchie
