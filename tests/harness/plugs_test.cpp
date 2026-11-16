// The single plug interface: every built-in game through the same socket, episodes that replay exactly,
// grading kept away from the agent, and game tasks through the eval at every thread count.
#include "HarnessCheck.h"

#include "pchie/JoinJobDispatcher.h"
#include "pchie/MazeGrid.h"
#include "pchie/harness/Eval.h"
#include "pchie/harness/GamePlug.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

using namespace pchie;
using namespace pchie::harness;

namespace harness_test {

namespace {

// A small crafting agent for the Craftax-like grid. It sees only the 5x5 window and the status text.
class CraftAgent final : public IAgentPlug {
 public:
  std::string name() const override { return "craft_test"; }
  void reset(const PlugInfo&) override { k_ = 0; }
  int act(const PlugObservation& o, std::string& why) override {
    const int wood = number_after(o.text, "Wood ");
    const bool pick = o.text.find("pickaxe yes") != std::string::npos;
    const int ach = number_after(o.text, "Achievements ");
    const bool table = ach >= 2 || pick;
    const int face = facing(o.text);
    auto g = [&](int x, int y) { return o.glyphs[size_t((y + 2) * 6 + (x + 2))]; };
    static const int dx[4] = {0, 1, 0, -1}, dy[4] = {-1, 0, 1, 0};
    if (!table && wood >= 3) {  // a table costs 2 wood, the pickaxe 1 more
      if (g(dx[face], dy[face]) == '.') { why = "place table"; return 6; }
      for (int d = 0; d < 4; ++d)
        if (g(dx[d], dy[d]) == '.') { why = "step to open ground"; return 1 + d; }
    }
    if (table && !pick && wood >= 1) {
      for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
          if (g(x, y) == 'B') { why = "make pickaxe"; return 7; }
    }
    const char want = pick ? 'S' : 'T';
    for (int d = 0; d < 4; ++d)
      if (g(dx[d], dy[d]) == want) {
        if (d == face) { why = "use"; return 5; }
        why = "turn";
        return 1 + d;  // blocked, so this only turns him
      }
    // Walk toward the nearest wanted glyph in view (BFS over grass), otherwise sweep.
    int best = -1, bestd = 99;
    for (int y = -2; y <= 2; ++y)
      for (int x = -2; x <= 2; ++x)
        if (g(x, y) == want) {
          const int d = (x < 0 ? -x : x) + (y < 0 ? -y : y);
          if (d < bestd) {
            bestd = d;
            best = (y < 0 && g(0, -1) == '.') ? 0 : (x > 0 && g(1, 0) == '.') ? 1 : (y > 0 && g(0, 1) == '.') ? 2 : (x < 0 && g(-1, 0) == '.') ? 3 : -1;
          }
        }
    if (best >= 0) { why = "approach"; return 1 + best; }
    why = "sweep";
    const int order[4] = {1, 2, 3, 0};
    const int d = order[(k_++ / 3) % 4];
    return 1 + d;
  }

 private:
  static int number_after(const std::string& t, const std::string& key) {
    const size_t p = t.find(key);
    return p == std::string::npos ? 0 : std::atoi(t.c_str() + p + key.size());
  }
  static int facing(const std::string& t) {
    static const char* n[4] = {"Facing north", "Facing east", "Facing south", "Facing west"};
    for (int i = 0; i < 4; ++i)
      if (t.find(n[i]) != std::string::npos) return i;
    return 1;
  }
  uint32_t k_ = 0;
};

// An agent that tries to cheat: it fails the test if any observation ever carries grading words.
class PeekAgent final : public IAgentPlug {
 public:
  bool saw_grade = false;
  std::string name() const override { return "peek"; }
  void reset(const PlugInfo& i) override { n_ = int(i.actions.size()); }
  int act(const PlugObservation& o, std::string&) override {
    if (o.text.find("progress") != std::string::npos || o.text.find("distance") != std::string::npos ||
        o.text.find("layout") != std::string::npos)
      saw_grade = true;
    return int(k_++ % uint32_t(n_));
  }

 private:
  int n_ = 1;
  uint32_t k_ = 0;
};

}  // namespace

void test_plugs(const char* dir) {
  // Registry.
  {
    const auto names = plug_names();
    bool all = names.size() == 4;
    for (const auto& n : names) {
      std::string err;
      auto p = make_plug(n, err);
      all = all && p && p->info().name == n && !p->info().actions.empty();
    }
    check(all, "plugs: every registered plug builds and names itself");
    std::string err;
    check(!make_plug("nethack", err) && !err.empty(), "plugs: an unknown plug is refused");
    check(make_plug("litholattice.testbed.stub", err)->info().stub, "plugs: the LithoLattice testbed is marked as a stub");
    check(make_plug("lithokraft.maze.b1", err)->info().actions.size() == 6, "plugs: the maze speaks the research action order");
  }
  // Go-Explore through the socket on the B1 maze of record (seed 1).
  {
    std::string err;
    auto g = make_plug("lithokraft.maze.b1", err);
    for (bool sense : {false, true}) {
      auto a = go_explore_agent(sense);
      Transcript t1, t2;
      const EpisodeResult r1 = run_episode(*g, *a, 1, 4000, &t1);
      const EpisodeResult r2 = run_episode(*g, *a, 1, 4000, &t2);
      check(r1.success && r1.grade.success && r1.grade.progress == r1.grade.target && r1.grade.target == 50,
            sense ? "plugs: Go-Explore (sensing open moves) reaches the B1 exit, 50 steps of distance closed"
                  : "plugs: Go-Explore (research walk) reaches the B1 exit, 50 steps of distance closed");
      check(r1.actions_sha256 == r2.actions_sha256 && t1.sha256() == t2.sha256() && r1.final_state == r2.final_state,
            "plugs: a maze episode replays exactly (actions, transcript, final state)");
      std::printf("  plug b1 seed 1 go_explore sense=%d: steps=%u failed=%u reward=%lld actions=%.16s\n", int(sense), r1.steps,
                  r1.failed_actions, (long long)r1.reward, r1.actions_sha256.c_str());
    }
  }
  // Seeds that are really different mazes, and seeds that fall back to the same one.
  {
    std::string err;
    auto g = make_plug("lithokraft.maze.b1", err);
    auto a = cycle_agent({"north"});
    const EpisodeResult s2 = run_episode(*g, *a, 2, 1, nullptr);
    const EpisodeResult s3 = run_episode(*g, *a, 3, 1, nullptr);
    const EpisodeResult s6 = run_episode(*g, *a, 6, 1, nullptr);
    check(s2.grade.note == s3.grade.note, "plugs: B1 seeds 2 and 3 are the same fallback layout");
    check(s2.grade.note != s6.grade.note, "plugs: B1 seed 6 is a different layout");
  }
  // Grading never reaches the agent.
  {
    bool clean = true;
    for (const auto& n : plug_names()) {
      std::string err;
      auto g = make_plug(n, err);
      PeekAgent p;
      (void)run_episode(*g, p, 7, 60, nullptr);
      clean = clean && !p.saw_grade;
    }
    check(clean, "plugs: no observation carries grading (progress, distance, layout)");
  }
  // Stage A: falls are episode ends; the safe distance is graded.
  {
    std::string err;
    auto g = make_plug("lithokraft.maze.stagea", err);
    auto a = go_explore_agent();
    const EpisodeResult r = run_episode(*g, *a, 64, 300, nullptr);
    check(r.grade.target == 470 && r.grade.progress >= 0 && r.grade.progress <= r.grade.target, "plugs: Stage A seed 64 grades against the safe distance (470)");
    check(r.grade.note.find("fell") == std::string::npos, "plugs: Go-Explore never falls in a Stage A pit it can see");
    std::printf("  plug stagea seed 64 go_explore: steps=%u progress=%d/%d\n", r.steps, r.grade.progress, r.grade.target);
  }
  // Testbed stub and the Craftax-like grid.
  {
    std::string err;
    auto t = make_plug("litholattice.testbed.stub", err);
    auto fwd = cycle_agent({"forward"});
    const EpisodeResult r = run_episode(*t, *fwd, 3, 50, nullptr);
    check(r.success && r.steps == 8 && r.grade.progress == 8, "plugs: testbed stub corridor (seed 3, 9 cells) is walked in 8 steps");
    auto c = make_plug("craftax_like.grid", err);
    const PlugObservation o = c->reset(5);
    check(o.glyph_cols == 5 && o.glyph_rows == 5 && o.glyphs.size() == 29 && o.glyphs[14] == '@', "plugs: Craftax-like view is 5x5 with him in the middle");
    int wins = 0, progressed = 0;
    for (uint64_t s = 1; s <= 16; ++s) {
      CraftAgent ca;
      const EpisodeResult e = run_episode(*c, ca, s, 400, nullptr);
      wins += int(e.success);
      progressed += int(e.grade.progress >= 1);
    }
    check(progressed >= 12, "plugs: the crafting test agent collects wood on most Craftax-like seeds");
    check(wins >= 1, "plugs: the crafting test agent finishes at least one Craftax-like seed (wood, table, pickaxe, stone)");
    std::printf("  plug craftax_like 16 seeds: finished=%d collected_wood=%d\n", wins, progressed);
    auto bad = cycle_agent({"fly"});
    const EpisodeResult b = run_episode(*c, *bad, 1, 5, nullptr);
    check(!b.err.empty() && b.steps == 0, "plugs: an action outside the list stops the episode");
  }
  // Game tasks through the eval: 8 B1 seeds with Go-Explore, identical at every thread count.
  {
    const std::vector<uint64_t> seeds = {6, 7, 10, 17, 19, 33, 39, 65};
    const Task t = game_task("lithokraft.maze.b1", seeds, 400, [] { return go_explore_agent(); });
    std::string ref, ref_head;
    bool same = true;
    EvalLog first;
    for (int w : kWorkerCounts) {
      JoinJobDispatcher jobs(w);
      EvalOptions o;
      o.run_id = "plug-b1";
      o.jobs = &jobs;
      o.side_log_path = std::string(dir) + "/harness_plug_w" + std::to_string(w) + ".jsonl";
      std::remove(o.side_log_path.c_str());
      const EvalLog L = run_eval(t, nullptr, o);
      if (ref.empty()) { ref = L.digest; ref_head = L.side_head; first = L; }
      same = same && L.ok && L.digest == ref && L.side_head == ref_head;
      std::remove(o.side_log_path.c_str());
    }
    check(same, "plugs: 8-seed B1 game task identical at 1/2/4/8/12 workers");
    uint32_t wins = 0;
    for (const auto& r : first.results) wins += r.success ? 1u : 0u;
    check(first.errors == 0 && first.completed == 8, "plugs: every game sample completes");
    check(wins == 8 && first.mean_q16 == kFullScore, "plugs: Go-Explore finishes all 8 distinct B1 seeds within 400 steps");
    std::printf("  plug task b1 8 seeds: finished=%u mean_q16=%d digest=%.16s\n", wins, first.mean_q16, ref.c_str());
    // Unfinished episodes earn half marks scaled by progress.
    const Task short_t = game_task("lithokraft.maze.b1", {6}, 3, [] { return go_explore_agent(); });
    const EvalLog S = run_eval(short_t, nullptr, EvalOptions{});
    check(S.results[0].score.value_q16 < kFullScore / 2 + 1 && !S.results[0].success, "plugs: an unfinished episode scores at most half");
  }
}

}  // namespace harness_test
