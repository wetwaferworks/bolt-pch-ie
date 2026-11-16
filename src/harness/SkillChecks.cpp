#include "pchie/harness/SkillChecks.h"

#include "pchie/NodeSeed.h"
#include "pchie/research/Digest.h"

#include <cstdio>
#include <memory>

namespace pchie {
namespace harness {

namespace {

const char* kDir[4] = {"north", "east", "south", "west"};

SolverFactory one_word_solver() {
  return [] {
    std::vector<std::unique_ptr<Solver>> p;
    p.push_back(prompt_template("Answer with one word.\n{input}\nAnswer:"));
    p.push_back(generate());
    return chain(std::move(p));
  };
}

Task compass() {
  Task t;
  t.name = "compass";
  for (int d = 0; d < 4; ++d) {
    t.samples.push_back(Sample{"opp_" + std::string(kDir[d]), std::string("What direction is the opposite of ") + kDir[d] + "?",
                               kDir[(d + 2) & 3], 0, {}});
    t.samples.push_back(Sample{"right_" + std::string(kDir[d]),
                               std::string("You face ") + kDir[d] + " and turn right. Which way do you face now?", kDir[(d + 1) & 3], 0, {}});
    t.samples.push_back(Sample{"left_" + std::string(kDir[d]),
                               std::string("You face ") + kDir[d] + " and turn left. Which way do you face now?", kDir[(d + 3) & 3], 0, {}});
  }
  t.solver = one_word_solver();
  t.scorer = first_word_match();
  return t;
}

Task glyph_count() {
  Task t;
  t.name = "glyph_count";
  uint64_t rng = NodeSeed::Splitmix64(0x5C111C0DEULL);
  for (int i = 0; i < 12; ++i) {
    std::string row;
    int walls = 0;
    const int len = 8 + i % 5;
    for (int c = 0; c < len; ++c) {
      rng = NodeSeed::Splitmix64(rng);
      const bool wall = (rng % 3) == 0;
      row.push_back(wall ? '#' : '.');
      walls += wall ? 1 : 0;
    }
    t.samples.push_back(Sample{"row_" + std::to_string(i), "How many # are in this row? " + row, std::to_string(walls), 0, {}});
  }
  t.solver = one_word_solver();
  t.scorer = first_word_match();
  return t;
}

Task grid_lookup() {
  Task t;
  t.name = "grid_lookup";
  const char* grid[3] = {"#K.", ".@#", "D.E"};
  const std::string pic = std::string(grid[0]) + "\n" + grid[1] + "\n" + grid[2];
  auto word = [](char c) -> std::string {
    switch (c) {
      case '#': return "wall";
      case 'K': return "key";
      case 'D': return "door";
      case 'E': return "exit";
      case '@': return "you";
      default: return "floor";
    }
  };
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      t.samples.push_back(Sample{"r" + std::to_string(r + 1) + "c" + std::to_string(c + 1),
                                 "Grid (# wall, . floor, K key, D door, E exit, @ you):\n" + pic + "\nWhat is at row " +
                                     std::to_string(r + 1) + ", column " + std::to_string(c + 1) + "? Give its name.",
                                 word(grid[r][c]), 0, {}});
  t.solver = one_word_solver();
  t.scorer = first_word_match();
  return t;
}

Task maze_rules() {
  Task t;
  t.name = "maze_rules";
  t.samples = {
      Sample{"door", "What opens a shut door?", "B", 0, {"a lamp", "the key", "a pit", "waiting"}},
      Sample{"pit", "A pit is directly ahead. What should you do?", "C", 0, {"walk forward", "pick it up", "go around it", "use the key on it"}},
      Sample{"key", "A key is at your feet. What do you do?", "A", 0, {"pick it up", "walk past", "turn around", "wait"}},
      Sample{"exit", "Where does the maze end?", "D", 0, {"at the start", "at the key", "at the door", "at the exit"}},
      Sample{"wall", "You walk into a wall. What happens?", "B", 0, {"you pass through", "you stay where you are", "you fall", "the maze ends"}},
      Sample{"loop", "You are back where you were three moves ago. What is best?", "A", 0, {"try a way you have not tried", "repeat the same moves", "stop forever", "walk into the wall"}},
  };
  t.solver = [] {
    std::vector<std::unique_ptr<Solver>> p;
    p.push_back(prompt_template("{input}\n{choices}Answer with the letter only.\nAnswer:"));
    p.push_back(generate());
    return chain(std::move(p));
  };
  t.scorer = choice_letter();
  return t;
}

}  // namespace

std::vector<Task> skill_suite() { return {compass(), glyph_count(), grid_lookup(), maze_rules()}; }

SkillReport run_skill_suite(const Generate& gen, IJobDispatcher* jobs, const std::string& side_log_path, const std::string& run_id) {
  SkillReport r;
  r.ok = true;
  std::string all;
  for (const Task& t : skill_suite()) {
    EvalOptions o;
    o.run_id = run_id + ":" + t.name;
    o.jobs = jobs;
    o.side_log_path = side_log_path.empty() ? std::string() : side_log_path + "." + t.name + ".jsonl";
    const EvalLog L = run_eval(t, gen, o);
    SkillLine s;
    s.task = t.name;
    s.mean_q16 = L.mean_q16;
    s.samples = uint32_t(L.results.size());
    s.errors = L.errors;
    s.digest = L.digest;
    r.lines.push_back(s);
    all += s.task + "|" + std::to_string(s.mean_q16) + "|" + std::to_string(s.samples) + "|" + std::to_string(s.errors) + "|" + s.digest + "\n";
    r.side_head = L.side_head;
    if (!L.ok) { r.ok = false; r.err = L.err; }
  }
  r.digest = research::sha256_hex(all);
  return r;
}

std::vector<SkillRegression> regressions(const SkillReport& before, const SkillReport& after, int32_t tol) {
  std::vector<SkillRegression> out;
  for (const auto& b : before.lines) {
    int32_t a = 0;
    for (const auto& x : after.lines)
      if (x.task == b.task) a = x.mean_q16;
    if (b.mean_q16 - a > tol) out.push_back(SkillRegression{b.task, b.mean_q16, a});
  }
  return out;
}

std::string report_text(const SkillReport& r) {
  std::string s;
  char line[160];
  for (const auto& l : r.lines) {
    std::snprintf(line, sizeof line, "%-12s %3d.%02d%%  (%u samples, %u errors)\n", l.task.c_str(), int((int64_t(l.mean_q16) * 100) / 65536),
                  int(((int64_t(l.mean_q16) * 10000) / 65536) % 100), l.samples, l.errors);
    s += line;
  }
  s += "digest " + r.digest + "\n";
  return s;
}

}  // namespace harness
}  // namespace pchie
