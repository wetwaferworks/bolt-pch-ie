// Opt-in research layers from the PCHIE research brief (Nov 2026). Every flag defaults to off.
// These sit beside the reasoning-layer flags (pchie/rl/RlFlags.h) and use their own --research-*
// prefix, so RlFlags, its identity line and every hash built from it stay exactly as they were.
// Official runners must construct ResearchFlags{} (all off) and never pass --research-* arguments.
//
//   --research-go-explore            L7 Go-Explore archive (return, then explore) over Nemo's own cells
//   --research-smt                   exact constraint peer beside L6 (built-in solver)
//   --research-smt-binary=PATH       also run the pinned Z3 binary at PATH (SHA-256 checked first)
//   --research-step-scorer[=record|enforce]  step scorer as the L6 verifier score (record by default)
//   --research-tree-of-thoughts      Tree of Thoughts over plan texts (L3 extension)
//   --research-tot-breadth=N         candidates per node (1..8, default 3)
//   --research-tot-depth=N           levels (1..6, default 2)
//   --research-reflexion             write failure lessons as receipts (never injected; that needs JD's go)
//   --research-lookahead[=N]         frozen dynamics look-ahead, depth N (1..6, default 3)
//   --research-ctx=1024|2048         context size for the 2048 test path (default 1024)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace research {

enum class ScorerMode : uint8_t { Record = 0, Enforce = 1 };

struct ResearchFlags {
  bool go_explore = false;
  bool smt = false;
  std::string smt_binary;
  bool step_scorer = false;
  ScorerMode step_scorer_mode = ScorerMode::Record;
  bool tree_of_thoughts = false;
  uint32_t tot_breadth = 3;
  uint32_t tot_depth = 2;
  bool reflexion = false;
  bool lookahead = false;
  uint32_t lookahead_depth = 3;
  uint32_t ctx = 1024;

  bool any_on() const;
  std::string identity_line() const;
};

inline ResearchFlags research_defaults() { return ResearchFlags{}; }

struct ResearchParseResult {
  ResearchFlags flags;
  bool ok = true;
  std::string err;
};

/// Unknown --research-* arguments fail closed; anything else is left for the host program.
ResearchParseResult parse_research_flags(const std::vector<std::string>& args);
ResearchParseResult parse_research_flags(int argc, char** argv);

/// True only when every research flag is at its default.
bool research_runner_clean(const ResearchFlags& f);

/// Lesson injection into Little Nemo's prompt is not wired. It stays off until JD gives his go.
constexpr bool kLessonInjectionAllowed = false;

}  // namespace research
}  // namespace pchie
