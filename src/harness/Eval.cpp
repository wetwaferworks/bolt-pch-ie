#include "pchie/harness/Eval.h"

#include "pchie/research/Digest.h"
#include "pchie/research/SideLog.h"

#include <exception>
#include <memory>
#include <sstream>

namespace pchie {
namespace harness {

namespace {

// Runs one sample into its own slot. Slots are only read after the join.
void run_one(const Task& task, const Generate& gen, const EvalOptions& opt, size_t i, SampleResult& r, Transcript& out) {
  TaskState st;
  st.sample = &task.samples[i];
  st.prompt = st.sample->input;
  st.step_budget = opt.step_budget ? opt.step_budget : task.step_budget;
  r.id = st.sample->id;
  try {
    std::unique_ptr<Solver> s = task.solver ? task.solver() : nullptr;
    std::string err;
    if (!s) {
      r.err = "task has no solver";
    } else if (!s->solve(st, gen, err)) {
      r.err = err.empty() ? "solver failed" : err;
    }
    if (r.err.empty() && task.scorer) r.score = task.scorer->score(st);
  } catch (const std::exception& e) {
    r.err = std::string("exception: ") + e.what();
  } catch (...) {
    r.err = "exception";
  }
  if (!r.err.empty()) {
    r.score = Score{};
    st.transcript.add(EventKind::Error, st.steps, r.err);
  } else {
    st.transcript.add(EventKind::Score, st.steps,
                      "value_q16=" + std::to_string(r.score.value_q16) + " answer=" + r.score.answer + " why=" + r.score.explanation);
  }
  r.completed = st.completed;
  r.success = st.success;
  r.progress = st.progress;
  r.progress_target = st.progress_target;
  r.steps = st.steps;
  r.events = st.transcript.size();
  r.transcript_sha256 = st.transcript.sha256();
  out = std::move(st.transcript);
}

std::string q(const std::string& s) { return "\"" + json_escape(s) + "\""; }

}  // namespace

std::string canonical_results(const EvalLog& log) {
  std::ostringstream o;
  o << "task=" << log.task << " version=" << log.version << " solver=" << log.solver << " scorer=" << log.scorer << "\n";
  for (const auto& r : log.results) {
    o << r.id << '|' << r.score.value_q16 << '|' << int(r.completed) << '|' << int(r.success) << '|' << r.progress << '/'
      << r.progress_target << '|' << r.steps << '|' << r.events << '|' << r.transcript_sha256 << '|' << json_escape(r.err) << '\n';
  }
  return o.str();
}

EvalLog run_eval(const Task& task, const Generate& gen, const EvalOptions& opt) {
  EvalLog log;
  log.task = task.name;
  log.version = task.version;
  log.run_id = opt.run_id;
  log.scorer = task.scorer ? task.scorer->name() : "-";
  {
    std::unique_ptr<Solver> probe = task.solver ? task.solver() : nullptr;
    log.solver = probe ? probe->name() : "-";
  }
  const size_t n = task.samples.size();
  log.results.assign(n, SampleResult{});
  std::vector<Transcript> transcripts(n);
  if (opt.jobs && opt.jobs->size() > 0 && n > 1) {
    opt.jobs->parallel_for(int(n), [&](int i) { run_one(task, gen, opt, size_t(i), log.results[size_t(i)], transcripts[size_t(i)]); });
  } else {
    for (size_t i = 0; i < n; ++i) run_one(task, gen, opt, i, log.results[i], transcripts[i]);
  }
  // Phase boundary: every worker has joined. Fold and write in sample order.
  int64_t sum = 0;
  for (const auto& r : log.results) {
    sum += r.score.value_q16;
    if (r.completed) ++log.completed;
    if (!r.err.empty()) ++log.errors;
  }
  log.mean_q16 = n ? int32_t(sum / int64_t(n)) : 0;
  log.digest = research::sha256_hex(canonical_results(log));

  research::SideLog side(opt.side_log_path);
  std::string err;
  bool ok = side.append("{\"format\":\"pchie.harness.eval.v1\",\"run\":" + q(log.run_id) + ",\"task\":" + q(log.task) +
                            ",\"version\":" + q(log.version) + ",\"solver\":" + q(log.solver) + ",\"scorer\":" + q(log.scorer) +
                            ",\"samples\":" + std::to_string(n),
                        err);
  for (size_t i = 0; ok && i < n; ++i) {
    const SampleResult& r = log.results[i];
    size_t k = 0;
    for (const auto& e : transcripts[i].events()) {
      ok = ok && side.append("{\"format\":\"pchie.harness.event.v1\",\"sample\":" + q(r.id) + ",\"i\":" + std::to_string(k++) +
                                 ",\"kind\":\"" + event_kind_name(e.kind) + "\",\"step\":" + std::to_string(e.step) +
                                 ",\"text\":" + q(e.text),
                             err);
    }
    ok = ok && side.append("{\"format\":\"pchie.harness.score.v1\",\"sample\":" + q(r.id) + ",\"value_q16\":" +
                               std::to_string(r.score.value_q16) + ",\"answer\":" + q(r.score.answer) + ",\"completed\":" +
                               (r.completed ? "true" : "false") + ",\"success\":" + (r.success ? "true" : "false") +
                               ",\"progress\":" + std::to_string(r.progress) + ",\"progress_target\":" +
                               std::to_string(r.progress_target) + ",\"steps\":" + std::to_string(r.steps) +
                               ",\"transcript_sha256\":\"" + r.transcript_sha256 + "\",\"err\":" + q(r.err),
                           err);
  }
  ok = ok && side.append("{\"format\":\"pchie.harness.summary.v1\",\"run\":" + q(log.run_id) + ",\"task\":" + q(log.task) +
                             ",\"mean_q16\":" + std::to_string(log.mean_q16) + ",\"completed\":" + std::to_string(log.completed) +
                             ",\"errors\":" + std::to_string(log.errors) + ",\"digest\":\"" + log.digest + "\"",
                         err);
  log.side_head = side.head();
  log.side_lines = side.lines();
  log.ok = ok;
  if (!ok) log.err = err;
  return log;
}

}  // namespace harness
}  // namespace pchie
