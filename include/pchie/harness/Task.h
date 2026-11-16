// Harness tasks, after the Inspect AI pattern (task = samples + solver + scorer), rebuilt natively in C++.
// No Python runtime. A solver turns one sample into an answer and fills the transcript; a scorer grades
// the finished state. Scores are integers (Q16, 65536 = full marks) so every result is bit-exact.
// Scratch research only, off the official path: nothing here touches the official chain or goldens.
#pragma once

#include "pchie/harness/Transcript.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pchie {
namespace harness {

constexpr int32_t kFullScore = 65536;

struct Sample {
  std::string id;
  std::string input;                 // the question, or the game's instructions
  std::string target;                // the expected answer ("" when the scorer grades the state instead)
  uint64_t seed = 0;                 // game seed for game tasks
  std::vector<std::string> choices;  // multiple choice (A, B, C...) when not empty
};

struct TaskState {
  const Sample* sample = nullptr;
  std::string prompt;                // starts as the sample input; solvers may rewrite it
  std::string output;                // the final answer
  Transcript transcript;
  bool completed = false;
  uint32_t steps = 0;
  uint32_t step_budget = 0;
  bool success = false;              // game tasks: the goal was reached
  int32_t progress = 0;              // game tasks: graded after the episode, never shown to the agent
  int32_t progress_target = 0;
  std::vector<std::pair<std::string, std::string>> facts;  // extra results, in the order they were added
};

/// One model call: prompt in, text out. Must be deterministic (greedy). It may be called from several
/// threads at once when the eval runs samples in parallel; pass no dispatcher for a model that is not.
using Generate = std::function<std::string(const std::string& prompt)>;

struct Score {
  int32_t value_q16 = 0;
  std::string answer;
  std::string explanation;
};

class Solver {
 public:
  virtual ~Solver() = default;
  virtual std::string name() const = 0;
  /// False (with err) when the sample could not be run at all; a wrong answer is still true.
  virtual bool solve(TaskState& st, const Generate& gen, std::string& err) = 0;
};

class Scorer {
 public:
  virtual ~Scorer() = default;
  virtual std::string name() const = 0;
  virtual Score score(const TaskState& st) const = 0;
};

/// A fresh solver per sample, so samples never share solver state.
using SolverFactory = std::function<std::unique_ptr<Solver>()>;

struct Task {
  std::string name;
  std::string version = "1";
  std::vector<Sample> samples;
  SolverFactory solver;
  std::shared_ptr<const Scorer> scorer;
  uint32_t step_budget = 0;
};

// ---- built-in solvers ----
/// Replaces {input} in the template with the sample input (and {choices} with "A) ...\nB) ..." lines).
std::unique_ptr<Solver> prompt_template(std::string tmpl);
/// Calls the model with the current prompt and keeps the answer.
std::unique_ptr<Solver> generate();
/// Runs solvers in order; stops at the first one that fails or completes the sample.
std::unique_ptr<Solver> chain(std::vector<std::unique_ptr<Solver>> parts);

// ---- built-in scorers ----
std::shared_ptr<const Scorer> exact_match();
/// Trims, lower-cases and keeps the first word of the answer before comparing with the target.
std::shared_ptr<const Scorer> first_word_match();
/// Full marks when the target appears anywhere in the answer (case-insensitive).
std::shared_ptr<const Scorer> includes();
/// Multiple choice: the first A-Z letter of the answer against the target letter.
std::shared_ptr<const Scorer> choice_letter();

std::string trim_lower(const std::string& s);

}  // namespace harness
}  // namespace pchie
