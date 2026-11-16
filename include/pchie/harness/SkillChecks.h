// Small skill-check suite for Little Nemo, after the lm-evaluation-harness idea: fixed questions with
// known answers, scored exactly, run before and after any change to his model (an adapter, a dreamed
// LoRA) so a regression shows up as a number. Scaffolding: the questions are generated deterministically
// here; the model bridge is any Generate function. Off the official path.
#pragma once

#include "pchie/IJobDispatcher.h"
#include "pchie/harness/Eval.h"
#include "pchie/harness/Task.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace harness {

/// compass, glyph_count, grid_lookup (first-word answers) and maze_rules (multiple choice).
std::vector<Task> skill_suite();

struct SkillLine {
  std::string task;
  int32_t mean_q16 = 0;
  uint32_t samples = 0;
  uint32_t errors = 0;
  std::string digest;
};

struct SkillReport {
  std::vector<SkillLine> lines;
  std::string digest;      // SHA-256 over every line
  std::string side_head;   // last task's side-chain head (all tasks share one chain file)
  bool ok = false;
  std::string err;
};

SkillReport run_skill_suite(const Generate& gen, IJobDispatcher* jobs, const std::string& side_log_path, const std::string& run_id);

struct SkillRegression {
  std::string task;
  int32_t before_q16 = 0;
  int32_t after_q16 = 0;
};

/// Tasks whose mean dropped by more than tolerance_q16 (a task missing after counts as a drop to zero).
std::vector<SkillRegression> regressions(const SkillReport& before, const SkillReport& after, int32_t tolerance_q16);

std::string report_text(const SkillReport& r);

}  // namespace harness
}  // namespace pchie
