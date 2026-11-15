// L1 model-driven PUCT, L2 query-before-plan, sealed DecisionTuple.
// Also includes L2 engagement/fork tests, sealed-tuple file verifier + byte-flip tamper tests,
//   --seed-mem (between-run L2 store from the official oracle B1 walk), --verify-tuples.
// Flags-on single-layer ablations on the grid mazes, decide only (no in-engine walk, no LithoKraft).
// CPU, real Little Nemo weights only for model modes. Official runners never build or run this.
//
//   pchie_rl_l1l2_test.exe --unit                     no model: fidelity, poison control, L2, sealed-tuple, guards
//   pchie_rl_l1l2_test.exe --bench                    forward timing on Little Nemo (CPU)
//   pchie_rl_l1l2_test.exe --maze b1|stagea|stageb [--rl-...] [--poison] [--steps N] [--budget N] [--label X] [--b1-seed N]
//                          [--root DIR] [--workers N (default: all threads)] [--mem DIR/scratch/rl/mem/F.jsonl] [--loop-guard on|off] [--tesla on|off] [--tesla-seed os|dev-insecure] [--attest on|off]
//                                                     loop guard v2 is on by default; off gives the old default walk
//                                                     one decide-only run; prints an [RLON] token line
//   pchie_rl_l1l2_test.exe --compare-l1 L1.log NULL.log ORACLE.log      planner comparison verdict (exit 0 / 10 stop and report / 2 invalid)
//   pchie_rl_l1l2_test.exe --compare-l2 SEEDED.log EMPTY.log NULL.log   query-before-plan verdict (exit 0 / 2 invalid)
//   Launchers: experiments/next-runs/. Commands and durations: docs/notes/next-runs.md.
#include "pchie/HeadlessMaze.h"
#include "pchie/neml/NemlQuery.h"
#include "pchie/MazeOracle.h"
#include "pchie/ScaledPitMaze.h"
#include "pchie/Dividor.h"
#include "pchie/MirrorSimSideLog.h"
#include "pchie/Multiplexor.h"
#include "pchie/NemotronModel.h"
#include "pchie/JoinJobDispatcher.h"  // --workers N (B4 join dispatcher; thread-count-invariant by design, verified per run)
#include "pchie/Sha256.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/StageBDoorFactory.h"
#include "pchie/StageBDoorMaze.h"
#include "pchie/rl/DecisionTupleVerify.h"
#include "pchie/rl/MazeL1.h"
#include "pchie/rl/PlanMemory.h"
#include "pchie/rl/RlFlags.h"
#include "pchie/rl/RunCompare.h"
#include "pchie/research/MazeAdvisor.h"  // opt-in --research-* layers (maze runs only)
#include "pchie/ModelPath.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace pchie;

static const std::string& model_path() {
  static const std::string p = [] {
    std::string path, err;
    if (!pchie::resolve_model_path(path, err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return std::string();
    }
    return path;
  }();
  return p;
}

static std::string default_rl_root() {
  std::error_code ec;
  const auto p = std::filesystem::temp_directory_path(ec) / "pchie-rl-l1l2";
  if (!ec) std::filesystem::create_directories(p, ec);
  return p.generic_string();
}
static std::string g_root = default_rl_root();
static int g_fail = 0;

static void check(bool ok, const char* what) {
  std::printf("[RLON] %-92s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
}
static std::string hexv(const std::vector<uint8_t>& v) { return rl::to_hex(v); }
static std::string sha_actions(const std::vector<int32_t>& a) {
  std::vector<uint8_t> b;
  for (int32_t x : a) rl::append_i32_le(b, x);
  return hexv(Sha256::hash(b.data(), b.size()));
}
static std::string scratch(const std::string& rel) { return g_root + "/scratch/rl/" + rel; }

struct Tok {
  std::string actions, world, chain, side, tuple_head, qbp_head;
  uint64_t digest = 0;
  int path = 0;
  bool ok = false, exit = false;
  std::string err;
};
static bool same(const Tok& a, const Tok& b) {
  return a.actions == b.actions && a.world == b.world && a.chain == b.chain && a.side == b.side && a.ok == b.ok;
}

// ---------------- B1 unit helpers ----------------
static Tok b1_unit(uint64_t seed, rl::MazeRl* r, MazeResult* keep = nullptr) {
  HeadlessMaze m(seed);
  m.set_search_budget(16);
  if (r) m.set_rl(r);
  Sha256HashChain ch;
  MazeResult out;
  if (r) r->begin_run();
  const bool ok = m.run_unit("", &ch, out);
  if (r) r->end_run();
  Tok t;
  t.actions = sha_actions(out.actions);
  t.world = hexv(out.h_world_sha);
  t.chain = hexv(ch.head());
  t.path = out.path_length;
  t.ok = ok;
  t.exit = out.reached_exit;
  t.err = out.err;
  if (r) {
    t.digest = r->st.tree_digest;
    t.tuple_head = r->tuple_chain.head_hex();
    t.qbp_head = r->qbp_chain.head_hex();
  }
  if (keep) *keep = out;
  return t;
}

static Tok stagea_unit(rl::MazeRl* r) {
  const MazeGrid g = ScaledPitMaze::make();
  MirrorSimSideLog log;
  HeadlessMaze m(g);
  m.enable_mirror_sim(&log);
  if (r) m.set_rl(r);
  MazeResult out;
  if (r) r->begin_run();
  const bool ok = m.run_unit("", nullptr, out);
  if (r) r->end_run();
  Tok t;
  t.actions = sha_actions(out.actions);
  t.world = hexv(out.h_world_sha);
  t.side = hexv(log.sha256());
  t.path = out.path_length;
  t.ok = ok;
  t.exit = out.reached_exit;
  t.err = out.err;
  return t;
}

static const char* propose_right(void*, int door, const TraceL1Task&) {
  return door == 0 ? Dividor::kName : Multiplexor::kName;
}
static Tok stageb_unit(rl::MazeRl* r) {
  const StageBDoorGrid g = StageBDoorFactory::make();
  MirrorSimSideLog log;
  StageBDoorMaze m(g);
  m.enable_mirror_sim(&log);
  if (r) m.set_rl(r);
  StageBDoorResult out;
  if (r) r->begin_run();
  const bool ok = m.run_unit(&propose_right, nullptr, out);
  if (r) r->end_run();
  Tok t;
  t.actions = sha_actions(out.actions);
  t.world = hexv(out.h_world_sha);
  t.side = hexv(out.mirror_log_sha);
  t.path = out.path_length;
  t.ok = ok;
  t.exit = out.reached_exit;
  t.err = out.err;
  return t;
}

static void print_tok(const char* tag, const Tok& t) {
  std::printf("[RLON] %s ok=%d exit=%d path=%d err=%s actions=%s world=%s chain=%s side=%s digest=%016llx\n", tag,
              int(t.ok), int(t.exit), t.path, t.err.c_str(), t.actions.c_str(), t.world.c_str(),
              t.chain.c_str(), t.side.c_str(), (unsigned long long)t.digest);
}

// Captures (situation, decision) pairs per decision for seeding the L2 store.
struct SitCapture {
  std::vector<std::array<uint8_t, 32>> sits;
};

static rl::MemoryRow make_row(const std::array<uint8_t, 32>& sit, int32_t act, rl::TruthState t) {
  rl::MemoryRow r;
  r.situation_hash = sit;
  r.plan_fragment = {act};
  r.provenance_run = rl::sha256_bytes(std::string("rlon-unit-seed-fixture"));
  r.truth = t;
  return r;
}

static void unit_tests() {
  std::printf("[RLON] unit section (no model). Official tree untouched; scratch logs under %s/scratch/rl/\n", g_root.c_str());
  // --- guards ---
  {
    rl::MazeRl r;
    check(!r.engaged(), "default MazeRl (all flags off) is not engaged");
    check(rl::official_runner_clean(r.flags), "default MazeRl flags pass official_runner_clean");
    const Tok off = b1_unit(1, nullptr);
    const Tok idle = b1_unit(1, &r);
    print_tok("maze seed=1 official(unit)", off);
    check(same(off, idle), "flags off: set_rl(idle) == no set_rl (actions, H_world, chain head)");
    check(r.st.decisions == 0 && r.st.tuples_sealed == 0, "flags off: rl loop never ran, no tuple sealed");
    rl::MazeRl need;
    need.flags.puct_prior = rl::PriorSrc::Model;
    need.flags.puct_value = rl::ValueSrc::Model;
    const Tok fc = b1_unit(1, &need);
    check(!fc.ok && fc.err.find("without a model") != std::string::npos, "L1 without a model fails closed");
    for (int k = 0; k < 6; ++k) {
      rl::MazeRl u;
      if (k == 0) u.flags.fork_score = true;
      if (k == 1) u.flags.belief = true;
      if (k == 2) u.flags.vision_only = true;
      if (k == 3) u.flags.options = true;
      if (k == 4) u.flags.symbolic_peer = true;
      if (k == 5) u.flags.verifier = true;
      const Tok x = b1_unit(1, &u);
      if (x.ok || x.err.find("not wired") == std::string::npos) { std::printf("[RLON]   unwired flag %d did not fail closed\n", k); ++g_fail; }
    }
    check(true, "L3/L4/L5/L6 flags on the maze path fail closed (no silent official-path identity)");
  }
  // --- L1 fidelity: rl loop with official sources == official path ---
  {
    bool all = true;
    for (uint64_t s : {1ULL, 2ULL, 3ULL, 7ULL, 42ULL, 99ULL, 1000ULL, 123456789ULL}) {
      rl::MazeRl r;
      r.force_rl_loop = true;
      const Tok a = b1_unit(s, nullptr), b = b1_unit(s, &r);
      if (!same(a, b)) { all = false; std::printf("[RLON]   fidelity mismatch seed=%llu\n", (unsigned long long)s); print_tok("off", a); print_tok("rl", b); }
    }
    check(all, "fidelity: rl loop with oracle prior+value == official maze (8 seeds, actions/H_world/chain)");
    rl::MazeRl r;
    r.force_rl_loop = true;
    const Tok a = stageb_unit(nullptr), b = stageb_unit(&r);
    print_tok("StageB official(unit right-answers)", a);
    check(same(a, b), "fidelity: rl loop == official Stage B unit (actions/H_world/side log)");
    rl::MazeRl ra;
    ra.force_rl_loop = true;
    const Tok c = stagea_unit(nullptr), d = stagea_unit(&ra);
    print_tok("StageA official(unit)", c);
    check(same(c, d), "fidelity: rl loop == official Stage A unit 64x64 (actions/H_world/side log)");
  }
  // --- poison positive control ---
  {
    rl::MazeRl r;
    r.force_rl_loop = true;
    r.poison_oracle = true;
    const Tok a = b1_unit(1, nullptr), b = b1_unit(1, &r);
    print_tok("maze seed=1 rl-loop oracle POISONED", b);
    check(r.st.oracle_calls > 0, "poison control: oracle path consults the oracle (calls > 0)");
    check(!same(a, b), "poison control: poisoning the oracle moves the oracle-valued result");
  }
  // --- L2 query-before-plan (oracle prior+value, L1 off) ---
  const std::string memdir = scratch("mem/");
  std::remove((memdir + "unit_empty.jsonl").c_str());
  std::remove((memdir + "unit_seeded.jsonl").c_str());
  std::remove((memdir + "unit_bad.jsonl").c_str());
  std::remove((memdir + "unit_tamper.jsonl").c_str());
  std::remove((memdir + "unit_div.jsonl").c_str());
  std::remove((memdir + "unit_poison.jsonl").c_str());
  std::remove(scratch("qbp_unit.jsonl").c_str());
  std::remove(scratch("tuples_unit.jsonl").c_str());
  const Tok off = b1_unit(1, nullptr);
  std::vector<std::array<uint8_t, 32>> sits;
  std::vector<int32_t> acts;
  {
    rl::FilePlanMemory mem(memdir + "unit_empty.jsonl", rl::MinTruth::Sim);
    rl::MazeRl r;
    r.flags.qbp = true;
    r.memory = &mem;
    r.qbp_log = scratch("qbp_unit.jsonl");
    r.tuple_log = scratch("tuples_unit.jsonl");
    r.run_label = "unit-L2-empty";
    MazeResult out;
    // capture situation hashes by re-deriving them in the on_decision hook
    r.on_decision = [&](rl::MazeRl& rr) { (void)rr; };
    const Tok t = b1_unit(1, &r, &out);
    check(same(off, t), "L2 empty store: actions, H_world and official chain head identical to flags-off");
    check(r.st.mem_queries == r.st.decisions && r.st.mem_misses == r.st.decisions && r.st.mem_hits == 0,
          "L2 empty store: one miss per decision, zero hits");
    check(r.qbp_chain.length() == r.st.decisions, "L2 empty store: one hit/miss record per decision on the scratch chain");
    // sealed-tuple on this run
    bool rehash = r.tuples.size() == r.st.decisions && !r.tuples.empty();
    for (size_t i = 0; rehash && i < r.tuples.size(); ++i) rehash = rl::tuple_rehashes_to(r.tuples[i], r.seals[i]);
    check(rehash, "Tuples: every sealed DecisionTuple re-hashes to its seal");
    if (!r.tuples.empty()) {
      rl::DecisionTuple bad = r.tuples[0];
      bad.decision ^= 1;
      check(!rl::tuple_rehashes_to(bad, r.seals[0]), "Tuples: a tampered tuple (decision flipped) no longer matches its seal");
      bool match = true;
      for (size_t i = 0; i < r.tuples.size() && i < out.actions.size(); ++i) match = match && r.tuples[i].decision == out.actions[i];
      check(match, "Tuples: tuple decisions equal the committed actions, in order");
    }
    std::printf("[RLON] L2-empty decisions=%llu tuple_side_head=%s qbp_side_head=%s\n",
                (unsigned long long)r.st.decisions, r.tuple_chain.head_hex().c_str(), r.qbp_chain.head_hex().c_str());
    // second identical run: side heads deterministic
    rl::FilePlanMemory mem2(memdir + "unit_empty.jsonl", rl::MinTruth::Sim);
    rl::MazeRl r2;
    r2.flags.qbp = true;
    r2.memory = &mem2;
    r2.run_label = "unit-L2-empty";
    const Tok t2 = b1_unit(1, &r2);
    check(same(t, t2) && r2.tuple_chain.head_hex() == r.tuple_chain.head_hex() && r2.qbp_chain.head_hex() == r.qbp_chain.head_hex(),
          "Tuples/L2: a second identical run gives identical tokens and identical tuple/qbp side-chain heads");
    // derive situation hashes from tuples: situation key is computed inside; recompute here identically
    acts = out.actions;
  }
  // Seeded verified store from the flags-off path. Situation hashes are recomputed by a capture run.
  {
    struct Cap { std::vector<std::array<uint8_t, 32>>* v; };
    rl::FilePlanMemory probe(memdir + "unit_none.jsonl", rl::MinTruth::Sim);
    rl::MazeRl r;
    r.flags.qbp = true;
    r.memory = &probe;
    r.run_label = "unit-capture";
    // read back the qbp records to recover situation hashes
    r.qbp_log = scratch("qbp_capture.jsonl");
    std::remove(r.qbp_log.c_str());
    (void)b1_unit(1, &r);
    std::ifstream in(r.qbp_log);
    std::string line;
    while (std::getline(in, line)) {
      const auto p = line.find("\"situation\":\"");
      std::array<uint8_t, 32> h{};
      if (p != std::string::npos && rl::from_hex(line.substr(p + 13, 64), h)) sits.push_back(h);
    }
    check(sits.size() == acts.size() && !sits.empty(), "L2 capture: one situation hash per committed decision");
  }
  if (!sits.empty() && sits.size() == acts.size()) {
    // correct verified fragments
    {
      rl::FilePlanMemory w(memdir + "unit_seeded.jsonl", rl::MinTruth::Sim);
      bool all = true;
      for (size_t i = 0; i < sits.size(); ++i) all = all && w.append_between_runs(make_row(sits[i], acts[i], rl::TruthState::SimVerified));
      check(all, "L2 between-run append of verified rows succeeds (no run live)");
      rl::FilePlanMemory mem(memdir + "unit_seeded.jsonl", rl::MinTruth::Sim);
      check(mem.load_ok() && !mem.broken_link(), "L2 store reloads with a valid row chain");
      rl::MazeRl r;
      r.flags.qbp = true;
      r.memory = &mem;
      bool refused = false;
      r.on_decision = [&](rl::MazeRl& rr) {
        if (rr.st.decisions == 1) refused = !rr.memory->append_between_runs(make_row(sits[0], acts[0], rl::TruthState::SimVerified));
      };
      const Tok t = b1_unit(1, &r);
      check(refused, "L2 append attempted mid-run is refused");
      check(r.st.mem_hits == r.st.decisions && r.st.mem_rejected == 0, "L2 seeded correct store: every decision is a hit and passes gate + legality");
      check(t.actions == off.actions && t.world == off.world, "L2 seeded correct store: actions and H_world unchanged (hits agree with search)");
      std::printf("[RLON] L2-seeded chain=%s (flags-off %s) digest=%016llx hits=%llu\n", t.chain.c_str(), off.chain.c_str(),
                  (unsigned long long)t.digest, (unsigned long long)r.st.mem_hits);
      check(t.chain != off.chain, "L2 seeded store: hits raise root priors, so the official-format chain head moves (recorded as an L2 move)");
      rl::FilePlanMemory hum(memdir + "unit_seeded.jsonl", rl::MinTruth::Human);
      rl::MazeRl rh;
      rh.flags.qbp = true;
      rh.memory = &hum;
      const Tok th = b1_unit(1, &rh);
      check(rh.st.mem_hits == 0 && same(th, off), "L2 min-truth=human: SimVerified rows are not returned; identical to flags-off");
    }
    // unverified / refuted rows
    {
      rl::FilePlanMemory w(memdir + "unit_bad.jsonl", rl::MinTruth::Sim);
      for (size_t i = 0; i < sits.size(); ++i)
        (void)w.append_between_runs(make_row(sits[i], acts[i], (i % 2) ? rl::TruthState::Refuted : rl::TruthState::Unverified));
      rl::FilePlanMemory mem(memdir + "unit_bad.jsonl", rl::MinTruth::Sim);
      rl::MazeRl r;
      r.flags.qbp = true;
      r.memory = &mem;
      const Tok t = b1_unit(1, &r);
      check(r.st.mem_hits == 0 && same(t, off), "L2 unverified/refuted rows are never returned; identical to flags-off");
    }
    // tampered row
    {
      {
        rl::FilePlanMemory w(memdir + "unit_tamper.jsonl", rl::MinTruth::Sim);
        for (size_t i = 0; i < sits.size(); ++i) (void)w.append_between_runs(make_row(sits[i], acts[i], rl::TruthState::SimVerified));
      }
      std::ifstream in(memdir + "unit_tamper.jsonl");
      std::stringstream ss;
      ss << in.rdbuf();
      in.close();
      std::string s = ss.str();
      const auto p = s.find("\"plan_fragment\":[");
      if (p != std::string::npos) s[p + 17] = (s[p + 17] == '9') ? '8' : '9';
      std::ofstream o(memdir + "unit_tamper.jsonl", std::ios::binary | std::ios::trunc);
      o << s;
      o.close();
      rl::FilePlanMemory mem(memdir + "unit_tamper.jsonl", rl::MinTruth::Sim);
      check(mem.broken_link() && !mem.load_ok(), "L2 tampered row: store reports the broken link");
      rl::MazeRl r;
      r.flags.qbp = true;
      r.memory = &mem;
      const Tok t = b1_unit(1, &r);
      check(r.st.mem_hits == 0 && same(t, off), "L2 tampered store: nothing returned; identical to flags-off");
    }
    // poisoned fragments: illegal move and gate-reject token at decision 0
    {
      MazeGrid g0(1, 16, 16);
      const auto leg = g0.legal_actions();
      int32_t illegal = 0;
      for (int32_t a = 7; a <= 10 && !illegal; ++a) {
        bool is_legal = false;
        for (MazeAction l : leg) is_legal = is_legal || int32_t(l) == a;
        if (!is_legal) illegal = a;
      }
      rl::FilePlanMemory w(memdir + "unit_poison.jsonl", rl::MinTruth::Sim);
      (void)w.append_between_runs(make_row(sits[0], illegal, rl::TruthState::SimVerified));
      (void)w.append_between_runs(make_row(sits[1], 1, rl::TruthState::SimVerified));  // reject-set id
      rl::FilePlanMemory mem(memdir + "unit_poison.jsonl", rl::MinTruth::Sim);
      rl::MazeRl r;
      r.flags.qbp = true;
      r.memory = &mem;
      const Tok t = b1_unit(1, &r);
      check(illegal != 0 && r.st.mem_rejected == 2 && r.st.mem_hits == 0 && same(t, off),
            "L2 poisoned fragments (wall move, gate-reject id) are rejected and planning falls back; identical to flags-off");
    }
    // divergent but legal verified fragment at decision 0: must stay a proposal
    {
      // First decision with at least two legal moves; propose a legal move the search did not pick.
      MazeGrid g0(1, 16, 16);
      int32_t alt = 0;
      size_t di = 0;
      for (size_t i = 0; i < acts.size() && !alt; ++i) {
        for (MazeAction l : g0.legal_actions()) if (int32_t(l) != acts[i] && int32_t(l) <= 10 && !alt) { alt = int32_t(l); di = i; }
        if (!alt) (void)g0.apply(static_cast<MazeAction>(acts[i]));
      }
      std::printf("[RLON] L2-divergent decision_index=%zu official_action=%d proposed_alt=%d\n", di, acts[di], alt);
      rl::FilePlanMemory w(memdir + "unit_div.jsonl", rl::MinTruth::Sim);
      if (alt) (void)w.append_between_runs(make_row(sits[di], alt, rl::TruthState::SimVerified));
      rl::FilePlanMemory mem(memdir + "unit_div.jsonl", rl::MinTruth::Sim);
      rl::MazeRl r;
      r.flags.qbp = true;
      r.memory = &mem;
      MazeResult out;
      const Tok t = b1_unit(1, &r, &out);
      std::printf("[RLON] L2-divergent action_at_index=%d (flags-off %d) actions_changed=%d hits=%llu ok=%d path=%d chain=%s actions=%s\n",
                  di < out.actions.size() ? out.actions[di] : -1, acts[di], int(t.actions != off.actions),
                  (unsigned long long)r.st.mem_hits, int(t.ok), t.path, t.chain.c_str(), t.actions.c_str());
      check(alt != 0 && r.st.mem_hits >= 1, "L2 divergent legal fragment: hit is used as a raised root prior (proposal)");
      check(t.chain != off.chain, "L2 divergent fragment moves the chain head (prior change is hashed); attributed to the hit row");
      check(!(t.ok && !t.exit) && (t.exit || !t.err.empty()), "L2 divergent fragment: no false exit claim (honest stall or solve)");
      // Same row at a smaller boost: does oracle search override a wrong memory proposal?
      for (int32_t boost : {rl::kQ20 / 4, rl::kQ20 / 16}) {
        rl::FilePlanMemory mem2(memdir + "unit_div.jsonl", rl::MinTruth::Sim);
        rl::MazeRl r2;
        r2.flags.qbp = true;
        r2.memory = &mem2;
        r2.mem_boost_q20 = boost;
        MazeResult out2;
        const Tok t2 = b1_unit(1, &r2, &out2);
        std::printf("[RLON] L2-divergent boost_q20=%d hits=%llu ok=%d exit=%d actions_changed=%d err=%s chain=%s\n", boost,
                    (unsigned long long)r2.st.mem_hits, int(t2.ok), int(t2.exit), int(t2.actions != off.actions), t2.err.c_str(),
                    t2.chain.c_str());
      }
    }
  }
  // L1 gate fail-closed on model edges
  {
    rl::MazeRl r;
    PchGate gate(std::vector<int32_t>{9});  // East rejected
    float lg6[6] = {1.0f, 1.0f, 5.0f, 1.0f, 0.0f, 0.0f};
    const auto e = r.model_edges({MazeAction::North, MazeAction::East, MazeAction::West}, lg6, 1.0f, 7, gate, {});
    bool east0 = false;
    for (const auto& x : e) if (x.token == 9) east0 = x.prior_q20 == 0;
    check(east0 && r.st.gate_fail_closed == 1, "L1 gate: a rejected edge keeps prior 0 (fail closed) even with the top logit");
  }
}


// ---------------- L2 engagement + fork, sealed-tuple artifact ----------------
static const std::string kZeroHead(64, '0');
static void closeout_tests() {
  std::printf("[RL-UNIT] unit section (no model)\n");
  const std::string memdir = scratch("mem/");
  const Tok off = b1_unit(1, nullptr);
  // capture situations + committed actions (oracle sources, rl loop)
  std::vector<std::array<uint8_t, 32>> sits;
  std::vector<int32_t> acts;
  {
    std::remove((memdir + "co_probe_empty.jsonl").c_str());
    rl::FilePlanMemory probe(memdir + "co_probe_empty.jsonl", rl::MinTruth::Sim);
    rl::MazeRl r;
    r.flags.qbp = true;
    r.memory = &probe;
    r.run_label = "co-capture";
    r.qbp_log = scratch("qbp_co_capture.jsonl");
    std::remove(r.qbp_log.c_str());
    MazeResult out;
    (void)b1_unit(1, &r, &out);
    acts = out.actions;
    std::ifstream in(r.qbp_log);
    std::string line;
    while (std::getline(in, line)) {
      const auto p = line.find("\"situation\":\"");
      std::array<uint8_t, 32> h{};
      if (p != std::string::npos && rl::from_hex(line.substr(p + 13, 64), h)) sits.push_back(h);
    }
  }
  check(!sits.empty() && sits.size() == acts.size(), "L2 capture: one situation per committed decision");
  if (sits.empty() || sits.size() != acts.size()) return;
  std::remove((memdir + "co_seeded.jsonl").c_str());
  {
    rl::FilePlanMemory w(memdir + "co_seeded.jsonl", rl::MinTruth::Sim);
    for (size_t i = 0; i < sits.size(); ++i) (void)w.append_between_runs(make_row(sits[i], acts[i], rl::TruthState::SimVerified));
  }
  // A. engagement guard: qbp off with a store present -> no L2 activity, zero qbp head
  {
    rl::FilePlanMemory mem(memdir + "co_seeded.jsonl", rl::MinTruth::Sim);
    rl::MazeRl r;
    r.force_rl_loop = true;
    r.memory = &mem;
    const Tok t = b1_unit(1, &r);
    check(r.st.mem_queries == 0 && r.qbp_chain.length() == 0 && r.qbp_chain.head_hex() == kZeroHead && same(t, off),
          "L2 off (store present): zero queries, qbp_side_head all zero, identical to flags-off");
  }
  // B. qbp on without a store fails closed
  {
    rl::MazeRl r;
    r.flags.qbp = true;
    const Tok t = b1_unit(1, &r);
    check(!t.ok && t.err.find("without a plan-memory store") != std::string::npos && r.st.decisions == 0,
          "L2 on without a store fails closed (no silent no-L2 'flags-on' run)");
  }
  // C. qbp on with the seeded store: side head non-zero, one record per decision
  {
    rl::FilePlanMemory mem(memdir + "co_seeded.jsonl", rl::MinTruth::Sim);
    rl::MazeRl r;
    r.flags.qbp = true;
    r.memory = &mem;
    const Tok t = b1_unit(1, &r);
    std::printf("[RL-UNIT] L2-seeded decisions=%llu hits=%llu qbp_side_head=%s\n", (unsigned long long)r.st.decisions,
                (unsigned long long)r.st.mem_hits, r.qbp_chain.head_hex().c_str());
    check(r.qbp_chain.head_hex() != kZeroHead && r.qbp_chain.length() == r.st.decisions && r.st.mem_hits == r.st.decisions,
          "L2 engaged: qbp_side_head non-zero, one record per decision, all hits");
    (void)t;
  }
  // D. partial memory: first k situations known -> reuse; the rest miss and fall back to search (fork a new plan)
  {
    const size_t k = std::min<size_t>(10, sits.size() / 2);
    std::remove((memdir + "co_partial.jsonl").c_str());
    {
      rl::FilePlanMemory w(memdir + "co_partial.jsonl", rl::MinTruth::Sim);
      for (size_t i = 0; i < k; ++i) (void)w.append_between_runs(make_row(sits[i], acts[i], rl::TruthState::SimVerified));
    }
    rl::FilePlanMemory mem(memdir + "co_partial.jsonl", rl::MinTruth::Sim);
    rl::MazeRl r;
    r.flags.qbp = true;
    r.memory = &mem;
    const Tok t = b1_unit(1, &r);
    std::printf("[RL-UNIT] L2-partial k=%zu decisions=%llu hits=%llu misses=%llu rejected=%llu\n", k,
                (unsigned long long)r.st.decisions, (unsigned long long)r.st.mem_hits, (unsigned long long)r.st.mem_misses,
                (unsigned long long)r.st.mem_rejected);
    check(r.st.mem_hits >= k && r.st.mem_misses > 0 && r.st.mem_hits + r.st.mem_misses + r.st.mem_rejected == r.st.decisions,
          "L2 partial store: seen situations hit, unseen ones miss and plan by search");
    check(t.actions == off.actions && t.world == off.world, "L2 partial store: hits agree with search; actions and H_world unchanged");
  }
  // E. situation key binds the weights: rows written under other weights are never returned
  {
    rl::FilePlanMemory mem(memdir + "co_seeded.jsonl", rl::MinTruth::Sim);
    rl::MazeRl r;
    r.flags.qbp = true;
    r.memory = &mem;
    r.weights_hash.fill(0x11);
    const Tok t = b1_unit(1, &r);
    check(r.st.mem_hits == 0 && r.st.mem_misses == r.st.decisions && t.actions == off.actions,
          "L2 rows from other weights never hit (situation key binds weights_hash)");
  }
  // F/G. sealed-tuple artifact: file verifier, official chain untouched, byte-flip tamper
  {
    const std::string tl = scratch("tuples_co_g5.jsonl");
    std::remove(tl.c_str());
    rl::MazeRl r;
    r.force_rl_loop = true;
    r.tuple_log = tl;
    r.run_label = "co-g5";
    MazeResult out;
    const Tok t = b1_unit(1, &r, &out);
    check(same(t, off), "Tuple sealing on (oracle sources): official-format chain head, actions, H_world identical to flags-off");
    const auto v = rl::verify_tuple_log(tl, r.tuple_chain.head_hex());
    std::printf("[RL-UNIT] tuple verify lines=%zu seal_pass=%zu seal_fail=%zu parse_fail=%zu head=%s expect=%s ok=%d\n", v.lines,
                v.seal_pass, v.seal_fail, v.parse_fail, v.head_hex.c_str(), r.tuple_chain.head_hex().c_str(), int(v.ok));
    check(v.ok && v.seal_pass == r.st.decisions && v.lines == r.st.decisions, "Tuple verifier: every tuple line re-seals and the side-chain head re-hashes");
    std::ifstream in(tl, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    in.close();
    const std::string orig = ss.str();
    const size_t nl = orig.find('\n');
    const std::string tmp = scratch("tuples_co_g5_tamper.jsonl");
    auto verify_text = [&](const std::string& txt) {
      std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
      o << txt;
      o.close();
      return rl::verify_tuple_log(tmp, r.tuple_chain.head_hex());
    };
    // every single-byte flip (low bit) of line 0 must fail verification
    size_t flips = 0, caught = 0, by_seal = 0, by_parse = 0, by_head = 0;
    for (size_t i = 0; nl != std::string::npos && i < nl; ++i) {
      std::string x = orig;
      x[i] = char(uint8_t(x[i]) ^ 1u);
      if (x[i] == '\n' || orig[i] == '\n') continue;
      ++flips;
      const auto vv = verify_text(x);
      if (!vv.ok) {
        ++caught;
        if (vv.parse_fail) ++by_parse;
        else if (vv.seal_fail) ++by_seal;
        else if (!vv.head_match) ++by_head;
      }
    }
    std::printf("[RL-UNIT] tuple tamper line0_bytes=%zu caught=%zu (seal=%zu parse=%zu head_only=%zu)\n", flips, caught, by_seal,
                by_parse, by_head);
    check(flips > 0 && caught == flips, "Tuple tamper: every single-byte flip of a tuple line -> verify FAIL");
    // decision-token flip specifically fails the seal (not just the chain)
    {
      std::string x = orig;
      const auto p = x.find("\"decision_token\":");
      if (p != std::string::npos) x[p + 17] = char(uint8_t(x[p + 17]) ^ 1u);
      const auto vv = verify_text(x);
      check(p != std::string::npos && !vv.ok && vv.seal_fail >= 1, "Tuple tamper: flipped decision_token -> seal mismatch (FAIL)");
    }
    // drop the last line / swap two lines -> head mismatch even though every remaining seal passes
    {
      std::string x = orig;
      x.pop_back();
      x = x.substr(0, x.rfind('\n') + 1);
      const auto vv = verify_text(x);
      check(!vv.ok && vv.seal_fail == 0 && !vv.head_match, "Tuple tamper: truncated log (last tuple dropped) -> head mismatch (FAIL)");
      const size_t nl2 = orig.find('\n', nl + 1);
      std::string y = orig.substr(nl + 1, nl2 - nl) + orig.substr(0, nl + 1) + orig.substr(nl2 + 1);
      const auto vy = verify_text(y);
      check(!vy.ok && vy.seal_fail == 0 && !vy.head_match, "Tuple tamper: two tuples reordered -> head mismatch (FAIL)");
    }
    check(verify_text(orig).ok, "Tuple control: untampered copy verifies PASS");
    std::remove(tmp.c_str());
    // canonical bytes: every flip changes the seal
    if (!r.tuples.empty()) {
      const auto c = rl::decision_tuple_canonical(r.tuples[0]);
      size_t moved = 0;
      for (size_t i = 0; i < c.size(); ++i) {
        auto d = c;
        d[i] ^= 1u;
        moved += rl::sha256_bytes(d.data(), d.size()) != r.seals[0] ? 1 : 0;
      }
      std::printf("[RL-UNIT] tuple canonical_bytes=%zu flips_that_move_seal=%zu\n", c.size(), moved);
      check(moved == c.size(), "Tuple canonical form: every byte is covered by the seal");
    }
  }
}

static int seed_mem(int argc, char** argv) {
  std::string out, label = "co_seed";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--seed-mem" && i + 1 < argc) out = argv[++i];
    else if (a == "--label" && i + 1 < argc) label = argv[++i];
  }
  if (out.find("/scratch/rl/mem/") == std::string::npos && out.find("\\scratch\\rl\\mem\\") == std::string::npos) {
    std::printf("[RL-UNIT] seed-mem path must be under scratch/rl/mem/\n");
    return 3;
  }
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) { std::printf("[RL-UNIT] load err %s\n", err.c_str()); return 2; }
  m.init_cache(256, 8);
  const std::string probe = scratch("mem/" + label + "_probe_empty.jsonl");
  std::remove(probe.c_str());
  std::remove(out.c_str());
  rl::FilePlanMemory pm(probe, rl::MinTruth::Sim);
  rl::MazeRl r;
  r.force_rl_loop = true;  // oracle prior + value: the official decide, model only tokenizes + weights hash
  r.flags.qbp = true;
  r.memory = &pm;
  r.model = &m;
  r.run_label = label;
  r.qbp_log = scratch("qbp_" + label + ".jsonl");
  std::remove(r.qbp_log.c_str());
  HeadlessMaze mz(MazeGrid(1, 16, 16));
  mz.set_rl(&r);
  Sha256HashChain ch;
  MazeResult res;
  r.begin_run();
  const bool ok = mz.run(m, "", &ch, res);
  r.end_run();
  std::vector<std::array<uint8_t, 32>> sits;
  std::ifstream in(r.qbp_log);
  std::string line;
  while (std::getline(in, line)) {
    const auto p = line.find("\"situation\":\"");
    std::array<uint8_t, 32> h{};
    if (p != std::string::npos && rl::from_hex(line.substr(p + 13, 64), h)) sits.push_back(h);
  }
  std::printf("[RL-UNIT] seed walk ok=%d exit=%d path=%d decisions=%llu sits=%zu actions=%s chain_head=%s err=%s\n", int(ok),
              int(res.reached_exit), res.path_length, (unsigned long long)r.st.decisions, sits.size(),
              sha_actions(res.actions).c_str(), hexv(ch.head()).c_str(), res.err.c_str());
  if (!ok || !res.reached_exit || sits.size() != res.actions.size()) { std::printf("[RL-UNIT] seed refused (walk not a verified solve)\n"); return 4; }
  rl::FilePlanMemory w(out, rl::MinTruth::Sim);
  size_t n = 0;
  for (size_t i = 0; i < sits.size(); ++i) {
    rl::MemoryRow row;
    row.situation_hash = sits[i];
    row.plan_fragment = {res.actions[i]};
    row.provenance_run = rl::sha256_bytes("closeout-seed: official oracle B1 seed=1 walk, actions " + sha_actions(res.actions));
    row.truth = rl::TruthState::SimVerified;  // verified by the simulator: this walk reached the exit
    n += w.append_between_runs(row) ? 1 : 0;
  }
  rl::FilePlanMemory chk(out, rl::MinTruth::Sim);
  std::printf("[RL-UNIT] seed rows_written=%zu reload_ok=%d broken_link=%d weights=%s\n", n, int(chk.load_ok()), int(chk.broken_link()),
              rl::to_hex(r.weights_hash).c_str());
  return (n == sits.size() && chk.load_ok()) ? 0 : 5;
}

static int verify_tuples_cli(int argc, char** argv) {
  std::string path, expect;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--verify-tuples" && i + 1 < argc) path = argv[++i];
    else if (a == "--expect" && i + 1 < argc) expect = argv[++i];
  }
  const auto v = rl::verify_tuple_log(path, expect);
  std::printf("[RL-UNIT] verify-tuples path=%s lines=%zu seal_pass=%zu seal_fail=%zu parse_fail=%zu head=%s expect=%s head_match=%d %s %s\n",
              path.c_str(), v.lines, v.seal_pass, v.seal_fail, v.parse_fail, v.head_hex.c_str(), expect.c_str(), int(v.head_match),
              v.ok ? "PASS" : "FAIL", v.first_error.c_str());
  return v.ok ? 0 : 1;
}

// ---------------- model modes ----------------
static int bench() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) { std::printf("[RLON] load err %s\n", err.c_str()); return 2; }
  m.init_cache(256, 8);
  const auto ids = m.tokenize(rl::MazeRl::header_text());
  ModelBranch b = m.new_branch();
  const auto t0 = std::chrono::steady_clock::now();
  for (int32_t t : ids) m.forward(b, t);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("[RLON] bench header_tokens=%zu secs=%.3f tok_per_s=%.2f n_past=%u\n", ids.size(), s, double(ids.size()) / s, b.n_past);
  for (const char* w : {" North", " South", " East", " West", " Pickup", " Use", " closer", " farther", " stuck", "\nMoves:", "\nVerdict:"}) {
    const auto v = m.tokenize_no_bos(w);
    std::printf("[RLON] bench tok %-12s n=%zu first=%d\n", w[0] == '\n' ? w + 1 : w, v.size(), v.empty() ? -1 : v[0]);
  }
  // warm pass: fork the primed header and forward 32 more tokens
  ModelBranch c = m.fork(b);
  const auto t1 = std::chrono::steady_clock::now();
  const auto more = m.tokenize_no_bos(" Position x=3 y=5. Target Key at x=10 y=2. Holding key: no. Door: closed.");
  for (int32_t t : more) m.forward(c, t);
  const double s1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  std::printf("[RLON] bench warm tokens=%zu secs=%.3f tok_per_s=%.2f\n", more.size(), s1, double(more.size()) / s1);
  m.release(c);
  m.release(b);
  return 0;
}

static int maze_run(int argc, char** argv) {
  std::string maze = "b1", label = "run";
  bool poison = false;
  int steps = 0, ctx = 256, budget = 16;
  uint64_t b1_seed = 1;  // --b1-seed N: another B1 maze (scratch comparisons; default 1 is the B1 of record)
  std::string mempath;
  bool loop_guard = true;  // loop guard v2, the default since 2026-11-13
  StreamAuthFlags stream_auth;  // TESLA + attestation on by default
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--maze" && i + 1 < argc) maze = argv[++i];
    else if (a == "--poison") poison = true;
    else if (a == "--b1-seed" && i + 1 < argc) b1_seed = std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
    else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
    else if (a == "--budget" && i + 1 < argc) budget = std::atoi(argv[++i]);
    else if (a == "--label" && i + 1 < argc) label = argv[++i];
    else if (a == "--mem" && i + 1 < argc) mempath = argv[++i];
    else if (a == "--loop-guard" && i + 1 < argc) {
      const std::string v = argv[++i];
      if (v != "on" && v != "off") { std::printf("[RLON] --loop-guard takes on or off\n"); return 3; }
      loop_guard = v == "on";
    }
  }
  const auto pr = rl::parse_rl_flags(argc, argv);
  if (!pr.ok) { std::printf("[RLON] flag error: %s\n", pr.err.c_str()); return 3; }
  const auto sa = parse_stream_auth_flags(argc, argv);
  if (!sa.ok) { std::printf("[RLON] %s\n", sa.err.c_str()); return 3; }
  stream_auth = sa.flags;
  // L2 stores live under <root>/scratch/rl/mem/ only (the same rule as --seed-mem), checked before the model loads.
  if (!mempath.empty()) {
    std::string why;
    if (!rl::mem_path_under_scratch(g_root, mempath, why)) { std::printf("[RLON] refused --mem %s: %s\n", mempath.c_str(), why.c_str()); return 3; }
  }
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) { std::printf("[RLON] load err %s\n", err.c_str()); return 2; }
  m.init_cache(uint32_t(ctx), 8);
  // Default: every available thread (capped at the dispatcher limit). --workers N overrides. Results do not depend on N.
  const unsigned hw_ = std::thread::hardware_concurrency();
  int workers_ = hw_ == 0 ? 1 : (hw_ > unsigned(JoinJobDispatcher::kMaxWorkers) ? JoinJobDispatcher::kMaxWorkers : int(hw_));
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--workers" && i + 1 < argc) {
      const int w = std::atoi(argv[i + 1]);
      if (w >= 1 && w <= JoinJobDispatcher::kMaxWorkers) workers_ = w;
    }
  static JoinJobDispatcher disp_(workers_ > 1 ? workers_ : 1);
  if (workers_ > 1) m.set_dispatcher(&disp_);
  std::printf("[RLON] workers=%d (join dispatcher; results do not depend on the thread count)\n", workers_);
  rl::MazeRl r;
  r.flags = pr.flags;
  r.model = &m;
  r.max_ctx = uint32_t(ctx);
  r.poison_oracle = poison;
  r.run_label = label;
  r.tuple_log = scratch("tuples_" + label + ".jsonl");
  r.qbp_log = scratch("qbp_" + label + ".jsonl");
  std::remove(r.tuple_log.c_str());
  std::remove(r.qbp_log.c_str());
  rl::FilePlanMemory* mem = nullptr;
  if (!mempath.empty()) { mem = new rl::FilePlanMemory(mempath, pr.flags.qbp_min_truth); r.memory = mem; }
  // Research layers (--research-*, off by default). Unknown research flags fail closed.
  const auto research_args = research::parse_research_flags(argc, argv);
  if (!research_args.ok) { std::printf("[RLON] research flag error: %s\n", research_args.err.c_str()); return 3; }
  std::unique_ptr<research::MazeResearchAdvisor> advisor;
  if (research_args.flags.any_on()) {
    if (maze == "stageb") { std::printf("[RLON] research layers are wired into b1 and stagea only (fail closed)\n"); return 3; }
    advisor = std::make_unique<research::MazeResearchAdvisor>(research_args.flags);
    std::string rerr;
    if (!advisor->prepare(rerr)) { std::printf("[RLON] %s\n", rerr.c_str()); return 3; }
    advisor->set_jobs(&disp_);  // Tree of Thoughts scoring; same result at every thread count
    r.research = advisor.get();
    std::printf("[RLON] research {%s}\n", research_args.flags.identity_line().c_str());
  }
  const bool eng = r.engaged();
  if (b1_seed != 1) std::printf("[RLON] b1_seed=%llu (scratch comparison maze)\n", (unsigned long long)b1_seed);
  std::printf("[RLON] start maze=%s label=%s engaged=%d flags={%s} poison=%d steps=%d budget=%d ctx=%d loop_guard=%s tesla=%s attest=%s\n", maze.c_str(), label.c_str(),
              int(eng), pr.flags.identity_line().c_str(), int(poison), steps, budget, ctx, loop_guard ? "on" : "off",
              stream_auth.tesla ? "on" : "off", stream_auth.attest ? "on" : "off");
  std::fflush(stdout);
  const auto t0 = std::chrono::steady_clock::now();
  Tok t;
  std::string extra;
  if (maze == "b1" || maze == "stagea") {
    const bool a = maze == "stagea";
    const MazeGrid grid = a ? ScaledPitMaze::make() : MazeGrid(b1_seed, 16, 16);
    HeadlessMaze mz(grid);
    mz.set_search_budget(budget);
    mz.enable_loop_guard(loop_guard);
    mz.set_stream_auth(stream_auth);
    MirrorSimSideLog log;
    if (a) mz.enable_mirror_sim(&log);
    if (steps > 0) mz.set_step_budget(steps);
    if (eng) mz.set_rl(&r);
    // Episodic memory prior (on by default for these scratch runs; --neml-reason off disables; empty ledger = unused).
    if (!neml::apply_cli(argc, argv)) { std::printf("[RLON] --neml takes on, off or a folder; --neml-reason takes on or off\n"); return 3; }
    const neml::Snapshot snap = neml::process_config().memory_prior ? neml::Snapshot::take(neml::process_config()) : neml::Snapshot::empty();
    if (!snap.runs.empty()) mz.set_memory(&snap);
    std::printf("[RLON] memory prior %s (snapshot runs=%zu sha256=%s)\n", snap.runs.empty() ? "unused" : "ON", snap.runs.size(),
                hexv(snap.sha256).c_str());
    Sha256HashChain ch;
    MazeResult out;
    r.begin_run();
    const bool ok = mz.run(m, "", a ? nullptr : &ch, out);
    r.end_run();
    t.actions = sha_actions(out.actions);
    t.world = hexv(out.h_world_sha);
    t.chain = a ? "" : hexv(ch.head());
    t.side = a ? hexv(log.sha256()) : "";
    t.path = out.path_length;
    t.ok = ok;
    t.exit = out.reached_exit;
    t.err = out.err;
    // Post-hoc evaluation only (never fed to the tree): replay and measure remaining BFS distance.
    MazeGrid rp = grid;
    for (int32_t x : out.actions) (void)rp.apply(static_cast<MazeAction>(x));
    const int rem = a ? MazeOracle::shortest_length_avoiding_pits(rp) : MazeOracle::shortest_length(rp);
    const int start = a ? MazeOracle::shortest_length_avoiding_pits(grid) : MazeOracle::shortest_length(grid);
    int uniq = 0;
    {
      std::vector<uint8_t> seen(size_t(grid.width() * grid.height()), 0);
      MazeGrid w = grid;
      seen[size_t(w.y() * grid.width() + w.x())] = 1;
      uniq = 1;
      for (int32_t x : out.actions) {
        (void)w.apply(static_cast<MazeAction>(x));
        const size_t k = size_t(w.y() * grid.width() + w.x());
        if (!seen[k]) { seen[k] = 1; ++uniq; }
      }
    }
    std::ostringstream o;
    o << "bfs_start=" << start << " bfs_remaining_posthoc=" << rem << " have_key=" << int(rp.have_key()) << " door_open="
      << int(rp.door_open()) << " in_pit=" << int(rp.in_pit()) << " unique_cells=" << uniq
      << " mirror_refused=" << out.mirror_refused << " loop_triggers=" << out.loop_triggers;
    extra = o.str();
  } else if (maze == "stageb") {
    const StageBDoorGrid g = StageBDoorFactory::make();
    MirrorSimSideLog log;
    StageBDoorMaze mz(g);
    mz.set_search_budget(budget);
    mz.enable_loop_guard(loop_guard);
    mz.set_stream_auth(stream_auth);
    mz.enable_mirror_sim(&log);
    if (steps > 0) mz.set_step_budget(steps);
    if (eng) mz.set_rl(&r);
    StageBDoorResult out;
    r.begin_run();
    const bool ok = mz.run(m, out);
    r.end_run();
    t.actions = sha_actions(out.actions);
    t.world = hexv(out.h_world_sha);
    t.side = hexv(out.mirror_log_sha);
    t.path = out.path_length;
    t.ok = ok;
    t.exit = out.reached_exit;
    t.err = out.err;
    std::ostringstream o;
    o << "door0=" << int(out.door_open[0]) << " door1=" << int(out.door_open[1]) << " attempts=" << out.attempts_per_door[0]
      << "/" << out.attempts_per_door[1] << " stalled_door=" << out.stalled_door << " steps_used=" << out.steps_used
      << " cap=" << out.cap << " bfs=" << out.bfs_length << " loop_triggers=" << out.loop_triggers;
    extra = o.str();
  } else {
    std::printf("[RLON] unknown maze\n");
    return 3;
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("[RLON] result maze=%s label=%s ok=%d exit=%d path=%d err=%s %s\n", maze.c_str(), label.c_str(), int(t.ok),
              int(t.exit), t.path, t.err.c_str(), extra.c_str());
  std::printf("[RLON] tokens label=%s actions_sha256=%s h_world_sha256=%s chain_head=%s side_log_sha256=%s tree_digest=%016llx\n",
              label.c_str(), t.actions.c_str(), t.world.c_str(), t.chain.c_str(), t.side.c_str(),
              (unsigned long long)r.st.tree_digest);
  std::printf("[RLON] stats label=%s decisions=%llu oracle_calls=%llu model_forwards=%llu leaf_evals=%llu cache_hits=%llu gate_fail_closed=%llu mem_q=%llu mem_hit=%llu mem_miss=%llu mem_rej=%llu tuples=%llu tuple_side_head=%s qbp_side_head=%s max_ctx_used=%u wall_s=%.1f\n",
              label.c_str(), (unsigned long long)r.st.decisions, (unsigned long long)r.st.oracle_calls,
              (unsigned long long)r.st.model_forwards, (unsigned long long)r.st.model_leaf_evals,
              (unsigned long long)r.st.model_cache_hits, (unsigned long long)r.st.gate_fail_closed,
              (unsigned long long)r.st.mem_queries, (unsigned long long)r.st.mem_hits, (unsigned long long)r.st.mem_misses,
              (unsigned long long)r.st.mem_rejected, (unsigned long long)r.st.tuples_sealed, r.tuple_chain.head_hex().c_str(),
              r.qbp_chain.head_hex().c_str(), r.st.max_ctx_used, secs);
  bool rehash = r.tuples.size() == r.seals.size();
  for (size_t i = 0; rehash && i < r.tuples.size(); ++i) rehash = rl::tuple_rehashes_to(r.tuples[i], r.seals[i]);
  if (advisor)
    std::printf("[RLON] research label=%s decisions=%llu proposals=%llu applied=%llu rejected=%llu capped=%llu go_explore=%llu lookahead=%llu lookahead_unknown=%llu transitions=%llu lessons=%llu side_head=%s digest=%s\n",
                label.c_str(), (unsigned long long)advisor->stats().decisions, (unsigned long long)r.st.research_proposals,
                (unsigned long long)r.st.research_applied, (unsigned long long)r.st.research_rejected,
                (unsigned long long)r.st.research_capped, (unsigned long long)advisor->stats().go_explore_proposals,
                (unsigned long long)advisor->stats().lookahead_proposals,
                (unsigned long long)advisor->stats().lookahead_unknown_state, (unsigned long long)advisor->stats().transitions,
                (unsigned long long)advisor->stats().lessons, advisor->log().head().c_str(), advisor->digest().c_str());
  if (advisor)
    std::printf("[RLON] research label=%s tot_runs=%llu tot=%llu smt_questions=%llu smt_unreachable=%llu smt=%llu z3_checks=%llu z3_agree=%llu\n",
                label.c_str(), (unsigned long long)advisor->stats().tot_runs, (unsigned long long)advisor->stats().tot_proposals,
                (unsigned long long)advisor->stats().smt_questions, (unsigned long long)advisor->stats().smt_unreachable,
                (unsigned long long)advisor->stats().smt_proposals, (unsigned long long)advisor->stats().z3_checks,
                (unsigned long long)advisor->stats().z3_agree);
  std::printf("[RLON] g5 label=%s tuples_rehash=%s live_pages_end=%u\n", label.c_str(), rehash ? "PASS" : "FAIL",
              m.pool_swa().live_pages() + m.pool_glob().live_pages());
  delete mem;
  return 0;
}

// ---------------- run comparers (read finished logs; no model) ----------------
static int compare_cli(const std::string& which, int argc, char** argv, int at) {
  if (argc < at + 4) { std::printf("[RLON] usage: %s A.log B.log C.log\n", which.c_str()); return 2; }
  rl::RlonRun runs[3];
  for (int k = 0; k < 3; ++k) {
    std::ifstream in(argv[at + 1 + k], std::ios::binary);
    if (!in) { std::printf("[RLON] cannot read %s\n", argv[at + 1 + k]); return 2; }
    std::stringstream ss; ss << in.rdbuf(); runs[k] = rl::parse_rlon_log(ss.str());
  }
  const rl::Verdict v = which == "--compare-l1" ? rl::judge_l1_separation(runs[0], runs[1], runs[2]) : rl::judge_l2_qbp(runs[0], runs[1], runs[2]);
  for (const auto& l : v.lines) std::printf("%s\n", l.c_str());
  return v.exit_code;
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--compare-l1" || a == "--compare-l2") return compare_cli(a, argc, argv, i);
  }
  for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--root" && i + 1 < argc) g_root = argv[i + 1];
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--bench") return bench();
    if (a == "--seed-mem") return seed_mem(argc, argv);
    if (a == "--verify-tuples") return verify_tuples_cli(argc, argv);
    if (a == "--maze") return maze_run(argc, argv);
  }
  unit_tests();
  closeout_tests();
  std::printf("[RLON] unit %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
