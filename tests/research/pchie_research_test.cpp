// Research layers (Go-Explore, constraint peer, step scorer, Tree of Thoughts, Reflexion, frozen
// dynamics look-ahead, context budget). Everything here is opt-in; this test also proves the flags-off
// side: research defaults are clean and the reasoning-layer flags parse exactly as before.
// Little Nemo's policies only see SelfObservation (his own odometry and what is at his feet or next
// to him); the maze grid is used only to apply his actions and to grade results.
//
//   pchie_research_test --unit             fast checks (CTest)
//   pchie_research_test --unit --z3=PATH --z3-sha=HEX [--z3-lib=PATH --z3-lib-sha=HEX]
//                                         also compare with the pinned Z3 binary (and its library)
//   pchie_research_test --report           a longer side-by-side over 64 mazes (prints a table)
#include "pchie/JoinJobDispatcher.h"
#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/StageBDoorKey.h"
#include "pchie/rl/RlFlags.h"
#include "pchie/research/ConstraintSolver.h"
#include "pchie/research/ContextBudget.h"
#include "pchie/research/Digest.h"
#include "pchie/research/ExperienceMap.h"
#include "pchie/research/FrozenDynamics.h"
#include "pchie/research/GoExplore.h"
#include "pchie/research/Reflexion.h"
#include "pchie/research/ResearchFlags.h"
#include "pchie/research/StepScorer.h"
#include "pchie/research/TreeOfThoughts.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace pchie;
using namespace pchie::research;
using pchie::rl::RlParseResult;
using pchie::rl::official_defaults;
using pchie::rl::parse_rl_flags;

namespace {

int g_fail = 0;
int g_pass = 0;

void check(bool ok, const char* what) {
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::printf("FAIL: %s\n", what);
  }
}

const int kWorkerCounts[] = {1, 2, 4, 8, 12};

// ---- a maze seen only through Little Nemo's own senses ----

class SelfView {
 public:
  explicit SelfView(MazeGrid g) : g_(std::move(g)), sx_(g_.x()), sy_(g_.y()) {}

  SelfObservation observe() const {
    SelfObservation o;
    o.x = g_.x() - sx_;
    o.y = g_.y() - sy_;
    o.heading = heading_;
    o.have_key = g_.have_key();
    o.key_here = g_.on_key();
    o.door_near = !g_.door_open() && g_.adjacent_to_door();
    o.at_exit = g_.at_exit();
    for (int m = 0; m < kMoves; ++m)
      if (g_.cell(g_.x() + move_dx(m), g_.y() + move_dy(m)) == MazeCell::Pit) o.danger_mask |= uint8_t(1u << m);
    return o;
  }

  // Applies one research action; returns whether it succeeded.
  bool act(int a) {
    static const MazeAction kMap[kActs] = {MazeAction::North, MazeAction::East, MazeAction::South,
                                           MazeAction::West,  MazeAction::Pickup, MazeAction::Use};
    if (a < 0 || a >= kActs) return false;
    const bool ok = g_.apply(kMap[a]);
    if (ok && a < kMoves) heading_ = uint8_t(a);
    return ok;
  }

  const MazeGrid& grid() const { return g_; }

 private:
  MazeGrid g_;
  int sx_, sy_;
  uint8_t heading_ = kNorth;
};

struct Episode {
  bool reached_exit = false;
  bool fell = false;
  uint32_t steps = 0;
  std::string digest;
  RunTrace trace;
  std::vector<Transition> transitions;
};

// Model state: his place, whether he holds the key, and whether he remembers opening the door.
uint64_t dyn_state(const SelfObservation& o, bool opened) {
  return (place_key(o.x, o.y) << 2) | (uint64_t(o.have_key) << 1) | uint64_t(opened);
}

int32_t reward_for(const SelfObservation& before, const SelfObservation& after) {
  if (after.at_exit) return 1 << 20;
  if (!before.have_key && after.have_key) return 1 << 18;
  return 0;
}

Episode run_go_explore(uint64_t seed, uint32_t budget, GoExploreConfig cfg = {}) {
  SelfView w(MazeGrid(seed, MazeGrid::kDefaultWidth, MazeGrid::kDefaultHeight));
  GoExplorePolicy p(cfg);
  Episode e;
  e.trace.run_id = "go_explore_seed_" + std::to_string(seed);
  e.trace.step_budget = budget;
  bool opened = false;
  for (uint32_t s = 0; s < budget; ++s) {
    const SelfObservation o = w.observe();
    const int a = p.decide(o);
    if (a < 0) {
      e.reached_exit = true;
      break;
    }
    const bool opened_before = opened;
    const bool ok = w.act(a);
    p.outcome(a, ok);
    if (a == kUse && ok) opened = true;
    const SelfObservation after = w.observe();
    StepRecord r;
    r.x = o.x;
    r.y = o.y;
    r.action = a;
    r.succeeded = ok;
    r.fell = w.grid().in_pit();
    e.trace.steps.push_back(r);
    e.transitions.push_back(Transition{dyn_state(o, opened_before), uint8_t(a), dyn_state(after, opened), reward_for(o, after)});
    ++e.steps;
    if (w.grid().in_pit()) {
      e.fell = true;
      break;
    }
  }
  if (!e.reached_exit && w.grid().at_exit()) e.reached_exit = true;
  e.trace.reached_exit = e.reached_exit;
  e.digest = p.digest();
  return e;
}

// Baseline without the archive: always the least-tried move here (compass order on ties).
Episode run_least_tried(uint64_t seed, uint32_t budget) {
  SelfView w(MazeGrid(seed, MazeGrid::kDefaultWidth, MazeGrid::kDefaultHeight));
  ExperienceMap map;
  Episode e;
  bool used = false;
  for (uint32_t s = 0; s < budget; ++s) {
    const SelfObservation o = w.observe();
    map.stand(o);
    if (o.at_exit) {
      e.reached_exit = true;
      break;
    }
    int a = -1;
    if (o.key_here && !o.have_key) a = kPickup;
    else if (o.door_near && o.have_key && !used) a = kUse;
    else {
      const Place* here = map.find(o.x, o.y);
      uint32_t best = 0;
      for (int m = 0; m < kMoves; ++m) {
        if ((o.danger_mask >> m) & 1) continue;
        if (map.edge(o.x, o.y, m) == EdgeState::Blocked) continue;
        const uint32_t t = here ? here->tries[m] : 0;
        if (a < 0 || t < best) {
          a = m;
          best = t;
        }
      }
    }
    if (a < 0) break;
    const bool ok = w.act(a);
    if (a < kMoves) map.tried(o.x, o.y, a, ok);
    if (a == kUse) {
      used = true;
      if (ok) map.forget_door_blocks();
    }
    ++e.steps;
    if (w.grid().in_pit()) {
      e.fell = true;
      break;
    }
  }
  return e;
}

// ---- sections ----

void test_flags() {
  const ResearchFlags d;
  check(!d.any_on(), "research defaults are all off");
  check(research_runner_clean(d), "research defaults are runner-clean");
  const std::string want =
      "research.v1 go_explore=0 smt=0 smt_binary=- step_scorer=0 step_scorer_mode=record tot=0 tot_breadth=3 "
      "tot_depth=2 reflexion=0 lookahead=0 lookahead_depth=3 ctx=1024";
  check(d.identity_line() == want, "research identity line is pinned");
  check(!kLessonInjectionAllowed, "lesson injection stays off");

  const std::vector<std::string> on = {"--research-go-explore", "--research-smt-binary=z3", "--research-step-scorer=enforce",
                                       "--research-tree-of-thoughts", "--research-tot-breadth=4", "--research-tot-depth=3",
                                       "--research-reflexion", "--research-lookahead=4", "--research-ctx=2048"};
  const ResearchParseResult r = parse_research_flags(on);
  check(r.ok, "research flags parse");
  check(r.flags.go_explore && r.flags.smt && r.flags.smt_binary == "z3" && r.flags.step_scorer &&
            r.flags.step_scorer_mode == ScorerMode::Enforce && r.flags.tree_of_thoughts && r.flags.tot_breadth == 4 &&
            r.flags.tot_depth == 3 && r.flags.reflexion && r.flags.lookahead && r.flags.lookahead_depth == 4 &&
            r.flags.ctx == 2048,
        "research flags carry their values");
  check(!research_runner_clean(r.flags), "any research flag makes a run not runner-clean");
  check(!parse_research_flags({"--research-nonsense"}).ok, "unknown research flag fails closed");
  check(!parse_research_flags({"--research-ctx=4096"}).ok, "bad ctx fails closed");
  check(!parse_research_flags({"--research-tot-depth=0"}).ok, "bad depth fails closed");
  check(parse_research_flags({"--rl-l1", "--maze", "b1"}).ok, "non-research arguments are left alone");

  // Reasoning-layer flags are untouched by research arguments.
  const RlParseResult base = parse_rl_flags(std::vector<std::string>{});
  RlParseResult mixed = parse_rl_flags(on);
  check(base.ok && mixed.ok, "rl flags ignore research arguments");
  check(base.flags.identity_line() == official_defaults().identity_line(), "rl defaults identity unchanged");
  check(mixed.flags.identity_line() == official_defaults().identity_line(), "research arguments never move rl identity");
  std::printf("  rl identity: %s\n", official_defaults().identity_line().c_str());
}

void test_experience_map() {
  ExperienceMap m;
  SelfObservation o;
  m.stand(o);
  m.tried(0, 0, kEast, true);
  o.x = 1;
  m.stand(o);
  m.tried(1, 0, kEast, false);
  m.tried(1, 0, kSouth, true);
  o.y = 1;
  m.stand(o);
  bool ok = false;
  const auto p = m.path(1, 1, 0, 0, ok);
  check(ok && p.size() == 2 && p[0] == kNorth && p[1] == kWest, "path walks only known open edges");
  check(m.edge(1, 0, kEast) == EdgeState::Blocked && m.edge(1, 1, kNorth) == EdgeState::Open,
        "edges are recorded both ways");
  m.path(0, 0, 5, 5, ok);
  check(!ok, "no path to a place he has never been");
  int32_t fx = 0, fy = 0;
  check(m.nearest_frontier(1, 1, fx, fy) && fx == 1 && fy == 1, "nearest frontier is where he stands");
}

void test_go_explore() {
  const Episode a = run_go_explore(1, 4000);
  const Episode b = run_go_explore(1, 4000);
  check(a.digest == b.digest && a.steps == b.steps, "go-explore repeats exactly");
  check(a.reached_exit, "go-explore reaches the exit on the B1 maze (seed 1)");
  check(!a.fell, "go-explore never steps into a pit it can see");
  std::printf("  go-explore seed 1: steps=%u exit=%d digest=%.16s\n", a.steps, int(a.reached_exit), a.digest.c_str());

  // 64 mazes in parallel; folded in seed order, so the fold is identical at every worker count.
  std::string ref;
  for (int workers : kWorkerCounts) {
    std::vector<std::string> slot(64);
    std::vector<uint32_t> steps(64);
    JoinJobDispatcher jobs(workers);
    jobs.parallel_for(64, [&](int i) {
      const Episode e = run_go_explore(uint64_t(100 + i), 4000);
      slot[size_t(i)] = e.digest + ":" + std::to_string(e.steps) + ":" + std::to_string(int(e.reached_exit));
      steps[size_t(i)] = e.steps;
    });
    std::string fold;
    for (const auto& s : slot) fold += s + "\n";
    const std::string sha = sha256_hex(fold);
    if (ref.empty()) ref = sha;
    char what[96];
    std::snprintf(what, sizeof what, "go-explore 64-maze fold identical at %d workers", workers);
    check(sha == ref, what);
  }
  std::printf("  go-explore 64-maze fold: %.16s (1/2/4/8/12 workers)\n", ref.c_str());
}

void test_constraint_solver(const std::string& z3, const std::string& z3_sha, const std::vector<PinnedFile>& libs) {
  // Gate doors: the built-in answer agrees with brute force over the grader for every port setting.
  int agree = 0, total = 0;
  for (int s = 0; s <= 8; ++s)
    for (int wv = 0; wv <= 4; ++wv)
      for (int ev = 0; ev <= 1; ++ev)
        for (int goal = 0; goal <= 4; ++goal) {
          TraceL1Task t;
          t.id = "probe";
          t.south = s;
          t.west = wv;
          t.east = ev;
          t.goal = goal;
          const GateAnswer a = GateDoorSolver::solve(t);
          const bool div = (s / 2) == goal;
          const bool mux = (ev > 0 ? wv : s) == goal;
          ++total;
          if (a.solvable == (div || mux) && (!a.solvable || a.component == (div ? "dividor" : "multiplexor"))) ++agree;
        }
  check(agree == total, "gate door solver matches the grader on every probe");
  TraceL1Task d0;
  check(StageBDoorKey::task_for_door(0, d0), "stage B door 0 has a task");
  const GateAnswer g0 = GateDoorSolver::solve(d0);
  check(g0.solvable, "stage B door 0 is solvable");
  std::printf("  stage B door 0 (%s): %s\n", d0.id, g0.component.c_str());

  // Gated graph: start 0 - 1 - [door 0] - 2 = goal; the key for door 0 is behind node 3 off node 0.
  GatedGraph g;
  g.nodes = 4;
  g.start = 0;
  g.goal = 2;
  g.doors = 1;
  g.edges = {{0, 1, -1}, {1, 2, 0}, {0, 3, -1}};
  g.keys = {{3, 0}};
  const GatedPlan p = GatedGraphSolver::solve(g);
  check(p.reachable && GatedGraphSolver::describe(p) == "go3 key0 go0 go1 open0 go2", "key before door, shortest plan");
  GatedGraph locked = g;
  locked.keys.clear();
  check(!GatedGraphSolver::solve(locked).reachable, "no key: the goal is unreachable");

  if (z3.empty()) {
    std::printf("  pinned Z3: skipped (no --z3)\n");
    return;
  }
  const PinnedZ3 solver(PinnedSolver{z3, z3_sha, 20000000, libs});
  const PinnedZ3 wrong(PinnedSolver{z3, std::string(64, '0'), 20000000, libs});
  check(wrong.check("(check-sat)\n", "research_z3_probe.smt2").verdict == SmtVerdict::Refused,
        "a binary that does not match its pin is refused");
  const int L = int(p.actions.size());
  const SmtResult at_l = solver.check(GatedGraphSolver::smt2(g, L), "research_z3_graph.smt2");
  const SmtResult below = solver.check(GatedGraphSolver::smt2(g, L - 1), "research_z3_graph.smt2");
  const SmtResult no_key = solver.check(GatedGraphSolver::smt2(locked, 12), "research_z3_graph.smt2");
  check(at_l.verdict == SmtVerdict::Sat, "z3: reachable at the exact optimum");
  check(below.verdict == SmtVerdict::Unsat, "z3: unreachable one action below the optimum");
  check(no_key.verdict == SmtVerdict::Unsat, "z3: unreachable without the key");
  int z_agree = 0, z_total = 0;
  for (int s = 0; s <= 6; s += 6)
    for (int wv = 0; wv <= 3; wv += 3)
      for (int ev = 0; ev <= 1; ++ev)
        for (int goal = 0; goal <= 3; ++goal) {
          TraceL1Task t;
          t.south = s;
          t.west = wv;
          t.east = ev;
          t.goal = goal;
          const SmtResult r = solver.check(GateDoorSolver::smt2(t), "research_z3_gate.smt2");
          ++z_total;
          if ((r.verdict == SmtVerdict::Sat) == GateDoorSolver::solve(t).solvable) ++z_agree;
        }
  check(z_agree == z_total, "z3 agrees with the built-in gate solver");
  std::printf("  pinned Z3: graph %s/%s/%s, gate agreement %d/%d\n", verdict_name(at_l.verdict),
              verdict_name(below.verdict), verdict_name(no_key.verdict), z_agree, z_total);
  std::remove("research_z3_probe.smt2");
  std::remove("research_z3_graph.smt2");
  std::remove("research_z3_gate.smt2");
}

void test_step_scorer() {
  RuleStepScorer rule;
  const std::string text = rule.table_text();
  FrozenTableStepScorer frozen(text, rule.artifact_sha256());
  check(frozen.loaded(), "frozen table loads with the right pin");
  FrozenTableStepScorer bad_pin(text, std::string(64, 'a'));
  check(!bad_pin.loaded() && bad_pin.score_q20(StepFeatures{}) == 0, "frozen table fails closed on a wrong pin");
  std::string tampered = text;
  tampered[tampered.size() - 2] = '9';
  FrozenTableStepScorer tamper(tampered, rule.artifact_sha256());
  check(!tamper.loaded(), "an edited table fails its old pin");

  ExperienceMap m;
  SelfObservation o;
  m.stand(o);
  m.tried(0, 0, kEast, false);
  o.danger_mask = uint8_t(1u << kSouth);
  const int32_t blocked = rule.score_q20(step_features(m, o, kEast, 0));
  const int32_t pit = rule.score_q20(step_features(m, o, kSouth, 0));
  const int32_t fresh = rule.score_q20(step_features(m, o, kWest, -1));
  check(pit < blocked && blocked < fresh, "rule scorer: pit < bumped wall < fresh step toward the goal");
  check(frozen.score_q20(step_features(m, o, kWest, -1)) == fresh, "frozen table matches the rule scorer");
  bool never_raises = true;
  for (int32_t prior : {0, 1, 1000, kQ20 / 3, kQ20})
    for (int32_t s : {-5, 0, kQ20 / 2, kQ20, kQ20 * 2})
      if (enforce_prior_q20(prior, s) > prior) never_raises = false;
  check(never_raises, "enforce mode never raises a prior");
  std::printf("  step scorer: rule table sha %.16s\n", rule.artifact_sha256().c_str());
}

void test_tree_of_thoughts() {
  // Plans on the map he knows after a short Go-Explore walk; the goal is to head back toward his start.
  SelfView w(MazeGrid(3, MazeGrid::kDefaultWidth, MazeGrid::kDefaultHeight));
  GoExplorePolicy walk;
  for (int s = 0; s < 40; ++s) {
    const SelfObservation o = w.observe();
    const int a = walk.decide(o);
    if (a < 0) break;
    walk.outcome(a, w.act(a));
  }
  const ExperienceMap& map = walk.map();
  const SelfObservation here = w.observe();
  const int32_t gx = 0, gy = 0;
  RuleStepScorer rule;
  auto sim = [&](const Thought& t, int32_t& x, int32_t& y) {
    x = here.x;
    y = here.y;
    for (uint8_t m : t.moves) {
      x += move_dx(m);
      y += move_dy(m);
    }
  };
  ProposeFn propose = [&](const Thought& parent, uint32_t breadth) {
    std::vector<Thought> out;
    int32_t x, y;
    sim(parent, x, y);
    static const char* kWord[kMoves] = {"N", "E", "S", "W"};
    for (int m = 0; m < kMoves && out.size() < breadth; ++m) {
      if (map.edge(x, y, m) == EdgeState::Blocked) continue;
      Thought t = parent;
      t.moves.push_back(uint8_t(m));
      t.text = parent.text + (parent.text.empty() ? "" : " ") + kWord[m];
      out.push_back(t);
    }
    return out;
  };
  EvaluateFn evaluate = [&](const Thought& t) {
    int32_t x = here.x, y = here.y;
    int64_t total = 0;
    SelfObservation o = here;
    for (uint8_t m : t.moves) {
      o.x = x;
      o.y = y;
      const int before = std::abs(gx - x) + std::abs(gy - y);
      const int after = std::abs(gx - x - move_dx(m)) + std::abs(gy - y - move_dy(m));
      total += rule.score_q20(step_features(map, o, m, after < before ? -1 : (after > before ? 1 : 0)));
      x += move_dx(m);
      y += move_dy(m);
      o.heading = m;
    }
    const int start = std::abs(gx - here.x) + std::abs(gy - here.y);
    const int end = std::abs(gx - x) + std::abs(gy - y);
    return int32_t(total / int64_t(t.moves.empty() ? 1 : t.moves.size()) + int64_t(start - end) * (kQ20 / 4));
  };
  TreeOfThoughts tot(TotConfig{3, 3, 3});
  const TotResult serial = tot.run(Thought{}, propose, evaluate, nullptr);
  check(serial.levels == 3 && !serial.best.moves.empty(), "tree of thoughts runs three levels");
  int32_t bx = 0, by = 0;
  sim(serial.best, bx, by);
  check(std::abs(bx) + std::abs(by) < std::abs(here.x) + std::abs(here.y), "the best plan heads toward the goal");
  bool same = true;
  for (int workers : kWorkerCounts) {
    JoinJobDispatcher jobs(workers);
    const TotResult r = tot.run(Thought{}, propose, evaluate, &jobs);
    same = same && r.digest == serial.digest && r.best.id == serial.best.id;
  }
  check(same, "tree of thoughts identical at 1/2/4/8/12 workers");
  std::printf("  tree of thoughts: %zu thoughts, best \"%s\" q20=%d digest=%.16s\n", serial.evaluated.size(),
              serial.best.text.c_str(), serial.best.score_q20, serial.digest.c_str());
}

void test_reflexion() {
  RunTrace t;
  t.run_id = "unit";
  t.step_budget = 12;
  auto push = [&](int32_t x, int32_t y, int a, bool ok, bool fell = false) {
    StepRecord r;
    r.x = x;
    r.y = y;
    r.action = a;
    r.succeeded = ok;
    r.fell = fell;
    t.steps.push_back(r);
  };
  push(0, 0, kEast, false);
  push(0, 0, kEast, false);
  push(0, 0, kEast, false);
  push(0, 0, kUse, false);
  push(0, 0, kUse, false);
  push(0, 0, kSouth, true);
  push(0, 1, kNorth, true);
  push(0, 0, kWest, true);
  push(-1, 0, kSouth, true, true);
  for (int i = 0; i < 3; ++i) push(-1, 1, kNorth, false);
  const auto lessons = Reflexion::reflect(t);
  std::vector<std::string> kinds;
  for (const auto& l : lessons) kinds.push_back(Reflexion::kind_name(l.kind));
  std::string joined;
  for (const auto& k : kinds) joined += k + " ";
  check(joined == "pit bump bump loop door_stall timeout ", "reflexion finds pit, bumps, loop, door stall and timeout");
  check(lessons.size() > 1 && lessons[1].text == "Moving east from (0, 0) is blocked; I bumped it 3 times.",
        "lesson text is the fixed template");
  const std::string block = Reflexion::prompt_block(lessons, 120);
  check(block.size() <= 120, "prompt block respects its cap");

  const std::string path = "research_lessons_test.jsonl";
  std::remove(path.c_str());
  LessonLedger ledger(path);
  std::string err, head;
  size_t lines = 0;
  check(ledger.append("unit", lessons, err), "lessons append to the ledger");
  check(ledger.append("unit2", lessons, err), "a second run appends");
  check(ledger.verify(head, lines, err) && lines == lessons.size() * 2, "ledger side chain verifies");
  {
    FILE* f = std::fopen(path.c_str(), "r+b");
    if (f) {
      std::fseek(f, 40, SEEK_SET);
      std::fputc('X', f);
      std::fclose(f);
    }
  }
  check(!ledger.verify(head, lines, err), "an edited ledger line breaks the chain");
  std::remove(path.c_str());

  const Episode e = run_go_explore(1, 60);
  const auto real = Reflexion::reflect(e.trace);
  check(!real.empty() && real.back().kind == LessonKind::Timeout, "a cut-short real run ends with a timeout lesson");
}

void test_frozen_dynamics() {
  // Learn from one Go-Explore run on seed 1, freeze, then plan the same maze with look-ahead only.
  const Episode learn = run_go_explore(1, 4000);
  DynamicsTrainer trainer;
  for (const auto& t : learn.transitions) trainer.add(t);
  const FrozenDynamics model = trainer.freeze();
  DynamicsTrainer again;
  for (const auto& t : learn.transitions) again.add(t);
  check(again.freeze().sha256() == model.sha256(), "frozen model hash repeats");

  LookaheadConfig cfg;
  cfg.depth = 4;
  cfg.untried_value_q20 = -(1 << 16);  // exploit: the model is all he has here
  SelfView w(MazeGrid(1, MazeGrid::kDefaultWidth, MazeGrid::kDefaultHeight));
  uint32_t steps = 0;
  uint64_t nodes = 0;
  bool exit = false, opened = false;
  for (; steps < 400; ++steps) {
    const SelfObservation o = w.observe();
    if (o.at_exit) {
      exit = true;
      break;
    }
    const LookaheadChoice c = model.best_action(dyn_state(o, opened), cfg);
    nodes += c.nodes;
    if (c.action < 0) break;
    if (w.act(c.action) && c.action == kUse) opened = true;
  }
  const int shortest = MazeOracle::shortest_length(MazeGrid(1, MazeGrid::kDefaultWidth, MazeGrid::kDefaultHeight));
  std::printf("  frozen dynamics: %zu entries sha %.16s; look-ahead depth 4 reached exit=%d in %u steps "
              "(oracle shortest %d, exploration run %u), %llu nodes\n",
              model.size(), model.sha256().c_str(), int(exit), steps, shortest, learn.steps,
              static_cast<unsigned long long>(nodes));
  check(exit, "look-ahead in the frozen model reaches the exit it learned");
  check(steps < learn.steps, "look-ahead beats the exploration run that taught it");
}

void test_context_budget() {
  const ContextPlan a = context_plan(1024);
  const ContextPlan b = context_plan(2048);
  check(a.fits && b.fits, "both context layouts fit their budget");
  check(!context_plan(4096).fits, "other sizes are refused");
  check(b.view_tokens > a.view_tokens, "2048 carries more view than 1024");
  std::printf("  %s\n  %s\n", describe(a).c_str(), describe(b).c_str());
}

void report() {
  std::printf("seed  shortest  go_explore  least_tried\n");
  uint64_t ge_total = 0, lt_total = 0;
  int ge_ok = 0, lt_ok = 0;
  for (int i = 0; i < 64; ++i) {
    const uint64_t seed = uint64_t(100 + i);
    const int shortest = MazeOracle::shortest_length(MazeGrid(seed, MazeGrid::kDefaultWidth, MazeGrid::kDefaultHeight));
    const Episode ge = run_go_explore(seed, 4000);
    const Episode lt = run_least_tried(seed, 4000);
    ge_ok += ge.reached_exit;
    lt_ok += lt.reached_exit;
    ge_total += ge.steps;
    lt_total += lt.steps;
    std::printf("%4llu  %8d  %10u%s  %11u%s\n", static_cast<unsigned long long>(seed), shortest, ge.steps,
                ge.reached_exit ? " " : "x", lt.steps, lt.reached_exit ? " " : "x");
  }
  std::printf("exits: go_explore %d/64, least_tried %d/64; total steps %llu vs %llu\n", ge_ok, lt_ok,
              static_cast<unsigned long long>(ge_total), static_cast<unsigned long long>(lt_total));
}

}  // namespace

int main(int argc, char** argv) {
  bool unit = false, rep = false;
  std::string z3, z3_sha;
  std::vector<PinnedFile> libs;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--unit") unit = true;
    else if (a == "--report") rep = true;
    else if (a.rfind("--z3=", 0) == 0) z3 = a.substr(5);
    else if (a.rfind("--z3-sha=", 0) == 0) z3_sha = a.substr(9);
    else if (a.rfind("--z3-lib=", 0) == 0) libs.push_back(PinnedFile{a.substr(9), ""});
    else if (a.rfind("--z3-lib-sha=", 0) == 0 && !libs.empty()) libs.back().sha256_hex = a.substr(13);
  }
  if (!unit && !rep) unit = true;
  const auto t0 = std::chrono::steady_clock::now();
  if (unit) {
    test_flags();
    test_experience_map();
    test_go_explore();
    test_constraint_solver(z3, z3_sha, libs);
    test_step_scorer();
    test_tree_of_thoughts();
    test_reflexion();
    test_frozen_dynamics();
    test_context_budget();
  }
  if (rep) report();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("research: %d passed, %d failed (%.0f ms)\n", g_pass, g_fail, ms);
  return g_fail ? 1 : 0;
}
