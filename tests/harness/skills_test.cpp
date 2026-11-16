// Skill-check scaffolding: a perfect stand-in model, a damaged one, regression detection, thread identity.
#include "HarnessCheck.h"

#include "pchie/JoinJobDispatcher.h"
#include "pchie/harness/SkillChecks.h"

#include <cstdio>
#include <map>
#include <string>

using namespace pchie;
using namespace pchie::harness;

namespace harness_test {

void test_skills(const char* dir) {
  // A stand-in that knows every answer (looked up from the suite itself), built before any thread runs.
  std::map<std::string, std::string> key;
  size_t total = 0;
  for (const Task& t : skill_suite()) {
    for (const Sample& s : t.samples) {
      std::unique_ptr<Solver> solver = t.solver();
      TaskState st;
      st.sample = &s;
      st.prompt = s.input;
      std::string err;
      // Run only the prompt step: a generator that records the prompt and returns the target.
      std::string seen;
      (void)solver->solve(st, [&](const std::string& p) { seen = p; return s.target; }, err);
      key[seen] = s.target;
      ++total;
    }
  }
  check(total == 12 + 12 + 9 + 6, "skills: 39 questions across four tasks");
  check(key.size() == total, "skills: every prompt is distinct");
  const Generate perfect = [&key](const std::string& p) {
    const auto it = key.find(p);
    return it == key.end() ? std::string("?") : it->second;
  };
  const Generate damaged = [&key](const std::string& p) {
    if (p.find("How many #") != std::string::npos) return std::string("0");
    const auto it = key.find(p);
    return it == key.end() ? std::string("?") : it->second;
  };
  std::string ref;
  bool same = true;
  SkillReport good;
  for (int w : kWorkerCounts) {
    JoinJobDispatcher jobs(w);
    const SkillReport r = run_skill_suite(perfect, &jobs, std::string(dir) + "/harness_skills_w" + std::to_string(w), "skills");
    if (ref.empty()) { ref = r.digest; good = r; }
    same = same && r.ok && r.digest == ref;
    for (const auto& l : r.lines) std::remove((std::string(dir) + "/harness_skills_w" + std::to_string(w) + "." + l.task + ".jsonl").c_str());
  }
  check(same, "skills: suite digest identical at 1/2/4/8/12 workers");
  bool full = good.lines.size() == 4;
  for (const auto& l : good.lines) full = full && l.mean_q16 == kFullScore && l.errors == 0;
  check(full, "skills: the perfect stand-in scores full marks on every task");
  const SkillReport bad = run_skill_suite(damaged, nullptr, "", "skills");
  const auto reg = regressions(good, bad, kFullScore / 20);
  check(reg.size() == 1 && reg[0].task == "glyph_count" && reg[0].after_q16 < kFullScore, "skills: damage to one skill shows as exactly one regression");
  check(regressions(good, good, 0).empty(), "skills: no change, no regression");
  check(report_text(good).find("compass") != std::string::npos, "skills: the report names each task");
  std::printf("  skills perfect: digest=%.16s; damaged glyph_count=%d/65536\n", ref.c_str(), reg.empty() ? -1 : reg[0].after_q16);
}

}  // namespace harness_test
