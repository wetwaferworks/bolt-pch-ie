// Loop guard v2, the shared reasoning-layer guard (include/pchie/rl/LoopGuard.h). Fast, no model.
// Part 1: the guard on its own. Part 2: the guard inside the maze drivers (B1 and Stage A through HeadlessMaze,
// Stage B through StageBDoorMaze, and MazeRl's decide path), on by default, with --loop-guard off giving the old results.
#include <cstdio>
#include <set>
#include <string>
#include <vector>
#include "pchie/HeadlessMaze.h"
#include "pchie/ScaledPitMaze.h"
#include "pchie/Sha256.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/StageBDoorFactory.h"
#include "pchie/StageBDoorMaze.h"
#include "pchie/rl/LoopGuard.h"
#include "pchie/rl/MazeL1.h"
#include "pchie/rl/RlCommon.h"

using namespace pchie;
using rl::LoopGuard;
static int passes = 0, fails = 0;
static void check(bool ok, const std::string& what) { std::printf("[LG] %-98s %s\n", what.c_str(), ok ? "PASS" : "FAIL"); ok ? ++passes : ++fails; }
static std::string hx(const std::vector<uint8_t>& v) { return rl::to_hex(v); }
static std::string sha_actions(const std::vector<int32_t>& a) { std::vector<uint8_t> b; for (int32_t x : a) rl::append_i32_le(b, x); return hx(Sha256::hash(b.data(), b.size())); }
static int unique_cells(const MazeGrid& start, const std::vector<int32_t>& acts) {
  std::set<std::pair<int, int>> s; MazeGrid w = start; s.insert({w.x(), w.y()});
  for (int32_t a : acts) { (void)w.apply(static_cast<MazeAction>(a)); s.insert({w.x(), w.y()}); }
  return int(s.size());
}

static void guard_alone() {
  const uint64_t A = 10, B = 11, C = 12;
  { LoopGuard g; size_t step = 0; const uint64_t seq[] = {B, A, B, A, B, A};
    for (uint64_t k : seq) g.after_commit(k, ++step, int(k), 0, "GoToKeyA");
    check(!g.forcing() && g.triggers() == 0, "a state entered 3 times is not a loop yet");
    g.after_commit(B, ++step, int(B), 0, "GoToKeyA");
    check(g.forcing() && g.cautious() && g.triggers() == 1, "2-cell bounce: the 4th entry triggers, forces a reasoning check and turns cautious mode on");
    check(g.taboo(A) && !g.taboo(B) && !g.taboo(C), "the forced decision may not turn 180 (only the cell just left is taboo)");
    check(g.log().size() == 1 && g.log()[0].find("TRIGGER #1 at committed step 7, cell (11,0) entered 4 times; 1 loop cells") != std::string::npos, "the trigger is logged with step, cell and entry count");
    check(g.note() == "GoToKeyA LOOP at (11,0) x4. Do not turn back. Check the next cell and cross carefully forward.", "the reasoning-check note names the loop for the model");
    check(g.budget(8) == 32 && g.reseed(99) == NodeSeed::Splitmix64(99 ^ LoopGuard::kSeedSalt ^ 1ULL), "the reasoning check gets 4x budget and a fresh seed");
    const std::vector<uint64_t> moves = {A, C};
    const auto kept = g.root_filter(moves, [](uint64_t m, uint64_t& k) { k = m; return true; });
    check(kept.size() == 1 && kept[0] == C, "breaks the bounce: the move back into the loop is dropped at the root");
    const std::vector<uint64_t> only_back = {A};
    check(g.root_filter(only_back, [](uint64_t m, uint64_t& k) { k = m; return true; }) == only_back, "a dead end may still turn back (the filter never leaves no move)");
    g.after_commit(C, ++step, int(C), 0, "GoToKeyA");
    check(!g.forcing() && g.cautious() && !g.taboo(A) && g.log().size() == 2 && g.log()[1].find("reasoning check #1 committed step 8 -> cell (12,0)") != std::string::npos,
          "after the forced move: counts restart, the taboo clears, cautious mode stays on");
    check(g.reseed(99) == 99 && g.budget(8) == 8, "after the forced move the seed and budget are back to normal"); }
  { LoopGuard g; size_t step = 0;
    for (uint64_t k = 100; k < 160; ++k) g.after_commit(k, ++step, int(k), 0, "GoToExit");
    const std::vector<uint64_t> moves = {1, 2, 3};
    check(g.triggers() == 0 && !g.forcing() && !g.cautious() && g.reseed(5) == 5 && g.budget(16) == 16 && g.log().empty() &&
              g.root_filter(moves, [](uint64_t m, uint64_t& k) { k = m; return true; }) == moves,
          "a walk with no loop is left untouched (no trigger, same seed, same budget, same moves)"); }
  { LoopGuard g; g.set_enabled(false); size_t step = 0;
    for (int i = 0; i < 20; ++i) g.after_commit(i % 2 ? A : B, ++step, 0, 0, "x");
    check(!g.enabled() && g.triggers() == 0 && !g.forcing() && !g.cautious() && g.budget(8) == 8, "--loop-guard off: the guard never acts, even on a bounce"); }
  { LoopGuard g; check(g.enabled(), "the guard is on by default"); }
}

// ---- drivers ----
struct MazeOut { bool ok = false; std::string actions, chain; int triggers = 0, cells = 0, path = 0; std::string err; };
static MazeOut run_b1(const MazeGrid& grid, bool guard, rl::MazeRl* r, int steps, int budget) {
  MazeOut o; HeadlessMaze mz(grid); mz.set_search_budget(budget); if (steps > 0) mz.set_step_budget(steps);
  if (!guard) mz.enable_loop_guard(false);
  if (r) { mz.set_rl(r); r->begin_run(); }
  Sha256HashChain ch; MazeResult out; o.ok = mz.run_unit("", &ch, out); if (r) r->end_run();
  o.actions = sha_actions(out.actions); o.chain = hx(ch.head()); o.triggers = out.loop_triggers; o.cells = unique_cells(grid, out.actions); o.path = out.path_length; o.err = out.err;
  return o;
}

int main() {
  guard_alone();
  // B1 and Stage A, official search, no model: default (guard on) vs --loop-guard off.
  const MazeGrid b1(1, 16, 16), sa = ScaledPitMaze::make();
  const MazeOut b1_on = run_b1(b1, true, nullptr, 0, 16), b1_off = run_b1(b1, false, nullptr, 0, 16);
  const MazeOut sa_on = run_b1(sa, true, nullptr, 0, 16), sa_off = run_b1(sa, false, nullptr, 0, 16);
  std::printf("  maze on  ok=%d path=%d cells=%d triggers=%d actions=%s chain=%s\n  maze off ok=%d path=%d actions=%s chain=%s\n", b1_on.ok, b1_on.path, b1_on.cells, b1_on.triggers, b1_on.actions.c_str(), b1_on.chain.c_str(), b1_off.ok, b1_off.path, b1_off.actions.c_str(), b1_off.chain.c_str());
  std::printf("  StageA on  ok=%d path=%d triggers=%d actions=%s err=%s\n  StageA off ok=%d path=%d actions=%s err=%s\n", sa_on.ok, sa_on.path, sa_on.triggers, sa_on.actions.c_str(), sa_on.err.c_str(), sa_off.ok, sa_off.path, sa_off.actions.c_str(), sa_off.err.c_str());
  check(b1_on.ok && b1_on.triggers == 0 && b1_on.actions == b1_off.actions && b1_on.chain == b1_off.chain, "maze official search: no loop, so the default run equals --loop-guard off");
  check(sa_on.actions == sa_off.actions || sa_on.triggers > 0, "Stage A official search: identical unless the guard triggered");
  // MazeRl decide path (the L1/null layer) with the null value and no model: this search bounces between two cells.
  rl::MazeRl r_on, r_off; for (rl::MazeRl* r : {&r_on, &r_off}) { r->flags.puct_value = rl::ValueSrc::Null; }
  const MazeOut n_off = run_b1(b1, false, &r_off, 40, 8), n_on = run_b1(b1, true, &r_on, 40, 8);
  rl::MazeRl r_on2; r_on2.flags.puct_value = rl::ValueSrc::Null; const MazeOut n_on2 = run_b1(b1, true, &r_on2, 40, 8);
  std::printf("  null decide off: cells=%d triggers=%d actions=%s err=%s\n  null decide on:  cells=%d triggers=%d actions=%s err=%s\n", n_off.cells, n_off.triggers, n_off.actions.c_str(), n_off.err.c_str(), n_on.cells, n_on.triggers, n_on.actions.c_str(), n_on.err.c_str());
  check(n_off.triggers == 0, "decide path, --loop-guard off: no guard activity (old behavior)");
  check(n_off.cells > 2 || (n_on.triggers > 0 && n_on.cells > n_off.cells), "decide path: if the old run bounced, the default run triggers and reaches more cells");
  check(n_on.actions == n_on2.actions && n_on.chain == n_on2.chain, "decide path with the guard on is run-to-run deterministic");
  // Stage B unit driver (correct answers): default vs off.
  auto sb = [](bool guard) { StageBDoorMaze m(StageBDoorFactory::make()); if (!guard) m.enable_loop_guard(false); StageBDoorResult out;
    (void)m.run_unit([](void*, int door, const TraceL1Task&) -> const char* { return door == 0 ? "dividor" : "multiplexor"; }, nullptr, out); return out; };
  const StageBDoorResult sb_on = sb(true), sb_off = sb(false);
  std::printf("  StageB on win=%d path=%d triggers=%d  off win=%d path=%d\n", int(sb_on.win), sb_on.path_length, sb_on.loop_triggers, int(sb_off.win), sb_off.path_length);
  check(sb_on.actions == sb_off.actions && sb_on.h_world_sha == sb_off.h_world_sha && sb_on.loop_triggers == 0, "Stage B unit driver: navigation has no loop, so the default run equals --loop-guard off");
  // Pins. --loop-guard off must give the values the code produced before the guard became the default
  // (taken from the pre-change headers on 2026-11-13). The default-on values are Scratch/provisional and
  // are listed with their causes in docs/notes/loop-guard-default-hash-moves.md.
  const std::string sb_off_actions = sha_actions(sb_off.actions);
  std::printf("  StageB off actions=%s\n", sb_off_actions.c_str());
  check(b1_off.actions == "0c21184625c3c03b8cfb4259213604f731098618cbbb578e0473eaa44333aa4e" &&
        b1_off.chain == "de479b0a2dd6c4ad69dc379c34139a3a64a7a27d6c9074d59e22a6b632e79533", "pin, off: maze 16x16 actions and chain head are the old values");
  check(sa_off.actions == "a3f8f6381ab03d1bc508a2c652af8fb0bd68ba2c6221d88ae32f3ddf3c2de0c8" && sa_off.err == "step budget", "pin, off: Stage A unit run is the old (looping, step budget) run");
  check(n_off.actions == "dc09b5181f0d62f6a2c814e940c9bb53c0e14099d17f39d108efdf40124995d4" && n_off.cells == 2, "pin, off: null decide is the old 2-cell bounce");
  check(sb_off_actions == "a1b8944a16e60bdc2f5c8c9f9a1d9858a0569fe672e4017bff2217ff87d2fad6", "pin, off: Stage B unit actions are the old values");
  check(b1_on.actions == b1_off.actions, "pin, default: maze unchanged");
  check(sa_on.ok && sa_on.path == 478 && sa_on.triggers == 1 && sa_on.actions == "0723de73b63cced9f4e365602298cf97b11eef2cd81eb73275a26e882e655873", "pin, default (provisional): Stage A unit reaches the exit, 1 trigger, 478 moves");
  check(n_on.triggers == 5 && n_on.cells == 3 && n_on.actions == "f21fb000b0fb4b070fd546bbf66964a1e268cb35168f2895101e55abab71a197", "pin, default (provisional): null decide, 5 triggers, 3 cells");
  std::printf("LOOP-GUARD: %d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
