// Opt-in reasoning-layer flags. Every flag defaults to off.
// Official runners must construct RlFlags{} (all off) or call official_defaults()
// and must never pass --rl-* arguments.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

enum class PriorSrc : uint8_t { Current = 0, Model = 1 };
enum class ValueSrc : uint8_t { Oracle = 0, Model = 1, Null = 2 };
enum class MinTruth : uint8_t { Sim = 0, Human = 1 };  // default Sim is a proposal; an open design question
enum class VerifierMode : uint8_t { Record = 0, Enforce = 1 };

struct RlFlags {
  // L1 model-driven PUCT
  bool puct_model = false;
  PriorSrc puct_prior = PriorSrc::Current;
  ValueSrc puct_value = ValueSrc::Oracle;

  // L2 query-before-plan
  bool qbp = false;
  bool qbp_write = false;
  MinTruth qbp_min_truth = MinTruth::Sim;  // proposed: sim for grid mazes; an open design question

  // L3 fork-and-score
  bool fork_score = false;
  uint32_t fork_horizon = 2;  // start at 2–3
  uint32_t fork_budget = 0;   // 0 = use existing fork-manager default

  // L4 belief
  bool belief = false;
  bool vision_only = false;  // implies belief and crutch audit

  // L5 options
  bool options = false;
  std::string options_set;  // e.g. "corridor,turn,backout,probe"

  // L6 symbolic peer + verifier
  bool symbolic_peer = false;
  int32_t symbolic_weight_q20 = 0;
  bool verifier = false;
  VerifierMode verifier_mode = VerifierMode::Record;  // start in record; an open design question

  bool any_on() const;
  // Stable, parseable summary used by identity tests (flags-off digest).
  std::string identity_line() const;
};

inline RlFlags official_defaults() { return RlFlags{}; }

// Parse argv. Unknown --rl-* flags fail closed (ok=false).
// Non --rl-* arguments are ignored (left for the host program).
// --rl-puct-model sets both prior and value to model.
// --rl-vision-only implies belief = true.
struct RlParseResult {
  RlFlags flags;
  bool ok = true;
  std::string err;
};

RlParseResult parse_rl_flags(int argc, char** argv);
RlParseResult parse_rl_flags(const std::vector<std::string>& args);

// Official-runner guard: true only when every flag is at its default-off value.
bool official_runner_clean(const RlFlags& f);

}  // namespace rl
}  // namespace pchie
