// Task / solver / scorer split, transcripts and the eval side chain.
#include "HarnessCheck.h"

#include "pchie/JoinJobDispatcher.h"
#include "pchie/harness/Eval.h"
#include "pchie/harness/Task.h"
#include "pchie/research/SideLog.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pchie;
using namespace pchie::harness;

namespace harness_test {

namespace {

// A stand-in model: answers "south" to the compass question, echoes everything else reversed.
std::string toy_model(const std::string& p) {
  if (p.find("opposite of north") != std::string::npos) return "South.";
  if (p.find("opposite of east") != std::string::npos) return "west";
  if (p.find("Which letter") != std::string::npos) return "B) the key";
  return std::string(p.rbegin(), p.rend());
}

Task compass_task(int n) {
  Task t;
  t.name = "compass";
  t.version = "1";
  for (int i = 0; i < n; ++i) {
    Sample s;
    s.id = "c" + std::to_string(i);
    s.input = (i % 2 == 0) ? "What is the opposite of north?" : "What is the opposite of east?";
    s.target = (i % 3 == 2) ? "north" : ((i % 2 == 0) ? "south" : "west");  // every third target is wrong on purpose
    t.samples.push_back(s);
  }
  t.solver = [] {
    std::vector<std::unique_ptr<Solver>> parts;
    parts.push_back(prompt_template("Answer in one word. {input}"));
    parts.push_back(generate());
    return chain(std::move(parts));
  };
  t.scorer = first_word_match();
  return t;
}

class ThrowingSolver final : public Solver {
 public:
  std::string name() const override { return "throws"; }
  bool solve(TaskState& st, const Generate&, std::string&) override {
    if (st.sample->id == "bad") throw std::runtime_error("boom");
    st.output = "ok";
    st.completed = true;
    return true;
  }
};

}  // namespace

void test_task_eval(const char* dir) {
  // Transcript canonical text and hash.
  {
    Transcript a, b;
    a.add(EventKind::Input, 0, "line \"one\"\n");
    a.add(EventKind::Model, 1, "two");
    b.add(EventKind::Input, 0, "line \"one\"\n");
    b.add(EventKind::Model, 1, "two");
    check(a.sha256() == b.sha256(), "transcript: same events, same hash");
    check(a.canonical() == "input|0|line \\\"one\\\"\\n\nmodel|1|two\n", "transcript: canonical text escapes quotes and newlines");
    b.add(EventKind::Note, 1, "");
    check(a.sha256() != b.sha256(), "transcript: an extra event changes the hash");
    check(json_escape(std::string("\x01", 1)) == "\\u0001", "transcript: control bytes are escaped");
  }
  // Scorers.
  {
    Sample s;
    s.target = "South";
    TaskState st;
    st.sample = &s;
    st.output = "  south, I think";
    check(first_word_match()->score(st).value_q16 == kFullScore, "scorer: first word match ignores case, spaces and punctuation");
    check(exact_match()->score(st).value_q16 == 0, "scorer: exact match is exact");
    check(includes()->score(st).value_q16 == kFullScore, "scorer: includes finds the target");
    Sample m;
    m.target = "b";
    st.sample = &m;
    st.output = "(B) the key";
    check(choice_letter()->score(st).value_q16 == kFullScore, "scorer: choice letter reads the first letter");
    st.output = "123";
    check(choice_letter()->score(st).value_q16 == 0, "scorer: no letter, no marks");
  }
  // Prompt template with choices.
  {
    Sample s;
    s.input = "Which letter opens the door?";
    s.choices = {"the lamp", "the key"};
    TaskState st;
    st.sample = &s;
    std::string err;
    auto p = prompt_template("{input}\n{choices}Answer:");
    check(p->solve(st, nullptr, err) && st.prompt == "Which letter opens the door?\nA) the lamp\nB) the key\nAnswer:",
          "solver: prompt template fills input and choices");
    auto g = generate();
    check(!g->solve(st, nullptr, err) && !err.empty(), "solver: generate without a model fails closed");
  }
  // An eval: scores, transcripts, side chain, and identity at every thread count.
  {
    const Task t = compass_task(24);
    std::string ref_digest, ref_head;
    bool same = true;
    for (int w : kWorkerCounts) {
      JoinJobDispatcher jobs(w);
      EvalOptions o;
      o.run_id = "compass-test";
      o.jobs = &jobs;
      o.side_log_path = std::string(dir) + "/harness_eval_w" + std::to_string(w) + ".jsonl";
      std::remove(o.side_log_path.c_str());
      const EvalLog L = run_eval(t, toy_model, o);
      std::string head, err;
      size_t lines = 0;
      const bool verified = research::SideLog::verify(o.side_log_path, head, lines, err);
      check(L.ok && verified && head == L.side_head && lines == L.side_lines, "eval: side chain verifies and matches the run");
      if (ref_digest.empty()) {
        ref_digest = L.digest;
        ref_head = L.side_head;
        // 24 samples, every third target wrong: 16 right.
        check(L.completed == 24 && L.errors == 0, "eval: every sample completes");
        check(L.mean_q16 == int32_t((int64_t(16) * kFullScore) / 24), "eval: mean score counts 16 of 24");
        check(L.results[0].events == 3 && L.results[0].score.value_q16 == kFullScore, "eval: sample 0 has input, model and score events");
        check(L.results[2].score.value_q16 == 0, "eval: a wrong target scores zero");
        check(L.solver == "chain(prompt_template,generate)" && L.scorer == "first_word_match", "eval: solver and scorer names are logged");
        check(lines == 1 + 24 * 3 + 24 + 1, "eval: header, events, scores and summary lines on the side chain");
      }
      same = same && L.digest == ref_digest && L.side_head == ref_head;
      std::remove(o.side_log_path.c_str());
    }
    check(same, "eval: digest and side-chain head identical at 1/2/4/8/12 workers");
    std::printf("  harness eval compass: digest=%.16s head=%.16s (1/2/4/8/12 workers)\n", ref_digest.c_str(), ref_head.c_str());
    // Serial (no dispatcher) gives the same answer, and no path still computes the head.
    EvalOptions o;
    o.run_id = "compass-test";
    const EvalLog L = run_eval(t, toy_model, o);
    check(L.ok && L.digest == ref_digest && L.side_head == ref_head, "eval: serial run, no file, same digest and head");
    // A changed answer moves the digest.
    const EvalLog L2 = run_eval(t, [](const std::string&) { return std::string("north"); }, o);
    check(L2.digest != ref_digest, "eval: different answers, different digest");
  }
  // Errors are contained per sample.
  {
    Task t;
    t.name = "errors";
    t.samples = {Sample{"good", "", "ok", 0, {}}, Sample{"bad", "", "ok", 0, {}}};
    t.solver = [] { return std::unique_ptr<Solver>(new ThrowingSolver()); };
    t.scorer = exact_match();
    JoinJobDispatcher jobs(2);
    EvalOptions o;
    o.jobs = &jobs;
    const EvalLog L = run_eval(t, nullptr, o);
    check(L.ok && L.errors == 1 && L.results[0].err.empty() && L.results[1].err == "exception: boom",
          "eval: a throwing sample is caught and logged, the rest still run");
    check(L.results[0].score.value_q16 == kFullScore && L.results[1].score.value_q16 == 0, "eval: the failed sample scores zero");
    Task none;
    none.name = "none";
    none.samples = {Sample{"x", "", "", 0, {}}};
    const EvalLog N = run_eval(none, nullptr, EvalOptions{});
    check(N.errors == 1 && N.results[0].err == "task has no solver", "eval: a task without a solver fails closed");
  }
}

}  // namespace harness_test
