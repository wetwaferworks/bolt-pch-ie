// Runs a harness task. Samples may run in parallel on the join dispatcher (std::thread::join is the
// phase boundary; no locks, no atomics). After the join, results and transcripts are written in sample
// order onto a research side chain (pchie.harness.*.v1 lines), so the side-chain head and the eval
// digest are identical at every thread count. Scratch research only: never the official chain.
#pragma once

#include "pchie/IJobDispatcher.h"
#include "pchie/harness/Task.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace harness {

struct SampleResult {
  std::string id;
  Score score;
  bool completed = false;
  bool success = false;
  int32_t progress = 0;
  int32_t progress_target = 0;
  uint32_t steps = 0;
  size_t events = 0;
  std::string transcript_sha256;
  std::string err;  // empty when the sample ran
};

struct EvalOptions {
  std::string run_id = "run";
  std::string side_log_path;          // empty: the chain head is still computed, nothing is written
  uint32_t step_budget = 0;           // 0: the task's own budget
  IJobDispatcher* jobs = nullptr;     // null or size 0: samples run one after another
};

struct EvalLog {
  bool ok = false;
  std::string err;
  std::string task, version, solver, scorer, run_id;
  std::vector<SampleResult> results;
  uint32_t completed = 0;
  uint32_t errors = 0;
  int32_t mean_q16 = 0;               // mean score, rounded down
  std::string side_head;
  size_t side_lines = 0;
  std::string digest;                 // SHA-256 over the canonical results
};

EvalLog run_eval(const Task& task, const Generate& gen, const EvalOptions& opt);

/// Canonical result text the digest is taken over (one line per sample).
std::string canonical_results(const EvalLog& log);

}  // namespace harness
}  // namespace pchie
