// Fast checks (no model) for the Planner comparison and Query-before-plan run comparers and the L2 store path guard.
// Fixtures are the real [RLON] lines from the 2026-11-12 runs (provisional), plus edited copies for the planner-comparison cases.
#include <cstdio>
#include <string>
#include "pchie/rl/RunCompare.h"

using namespace pchie::rl;
static int passes = 0, fails = 0;
static void check(bool ok, const std::string& what) { std::printf("[CMP] %-96s %s\n", what.c_str(), ok ? "PASS" : "FAIL"); ok ? ++passes : ++fails; }
static std::string rep(std::string s, const std::string& a, const std::string& b) { const auto p = s.find(a); if (p != std::string::npos) s.replace(p, a.size(), b); return s; }
static std::string relabel(std::string s, const std::string& from, const std::string& to) { size_t p = 0; while ((p = s.find(from, p)) != std::string::npos) { s.replace(p, from.size(), to); p += to.size(); } return s; }

static const char* kFlags0 = "fork_score=0 fork_horizon=2 fork_budget=0 belief=0 vision_only=0 options=0 options_set= symbolic_peer=0 symbolic_weight_q20=0 verifier=0 verifier_mode=record";
static std::string start(const std::string& label, int engaged, const std::string& f, int steps, int budget) {
  return "[RLON] start maze=b1 label=" + label + " engaged=" + std::to_string(engaged) + " flags={" + f + " " + kFlags0 + "} poison=0 steps=" + std::to_string(steps) + " budget=" + std::to_string(budget) + " ctx=256\n"; }

// ---- Query-before-plan, real logs (b1, 25 steps, budget 8, 2 workers) ----
static const std::string kL1Empty = "[RLON] workers=2 (multithreaded build)\n" +
  start("b1_L1_qbp_empty_s25b8", 1, "puct_model=1 puct_prior=model puct_value=model qbp=1 qbp_write=0 qbp_min_truth=sim", 25, 8) +
  "[RLON] result maze=b1 label=b1_L1_qbp_empty_s25b8 ok=0 exit=0 path=0 err=step budget bfs_start=50 bfs_remaining_posthoc=49 have_key=0 door_open=0 in_pit=0 unique_cells=2 mirror_refused=0\n"
  "[RLON] tokens label=b1_L1_qbp_empty_s25b8 actions_sha256=199f76a0a06c224c8ab9204539b0218dd12618013763f34ed1328b32b0a7fb9f h_world_sha256= chain_head=2d4695244c08023b1704c6a966d5a7b2d06667a6652f3df0a52848fa75770ff3 side_log_sha256= tree_digest=950e4288aa4da13d\n"
  "[RLON] stats label=b1_L1_qbp_empty_s25b8 decisions=25 oracle_calls=0 model_forwards=3020 leaf_evals=200 cache_hits=0 gate_fail_closed=0 mem_q=25 mem_hit=0 mem_miss=25 mem_rej=0 tuples=25 tuple_side_head=735f68dc09c50aa779f249d8be067f4aeb07f696aac7aef17dde9a2252424615 qbp_side_head=536ed7f23d44e4afb3cecf9707b1a972942631c21abcd662f3a26f87eb08632e max_ctx_used=182 wall_s=1542.5\n"
  "[RLON] g5 label=b1_L1_qbp_empty_s25b8 tuples_rehash=PASS live_pages_end=0\n";
static const std::string kL1Seeded = "[RLON] workers=2 (multithreaded build)\n" +
  start("b1_L1_qbp_seeded_s25b8", 1, "puct_model=1 puct_prior=model puct_value=model qbp=1 qbp_write=0 qbp_min_truth=sim", 25, 8) +
  "[RLON] result maze=b1 label=b1_L1_qbp_seeded_s25b8 ok=0 exit=0 path=0 err=step budget bfs_start=50 bfs_remaining_posthoc=49 have_key=0 door_open=0 in_pit=0 unique_cells=2 mirror_refused=0\n"
  "[RLON] tokens label=b1_L1_qbp_seeded_s25b8 actions_sha256=199f76a0a06c224c8ab9204539b0218dd12618013763f34ed1328b32b0a7fb9f h_world_sha256= chain_head=edb81f7faa4f0559e512e828ecf034e6a89a9b8959ab6e5c4c59f691e65f6fcb side_log_sha256= tree_digest=0353b5bff294b474\n"
  "[RLON] stats label=b1_L1_qbp_seeded_s25b8 decisions=25 oracle_calls=0 model_forwards=2888 leaf_evals=200 cache_hits=0 gate_fail_closed=0 mem_q=25 mem_hit=25 mem_miss=0 mem_rej=0 tuples=25 tuple_side_head=542606ebf81eb44510d74d139aef69038abae190ccf7d6d01e60be37f9dac779 qbp_side_head=8212e308d7244709056fa3bfd0f8bab2ad830b255de8edf7a881332d4fcd4514 max_ctx_used=179 wall_s=1481.4\n"
  "[RLON] g5 label=b1_L1_qbp_seeded_s25b8 tuples_rehash=PASS live_pages_end=0\n";
static const std::string kNullSeeded = "[RLON] workers=2 (multithreaded build)\n" +
  start("b1_null_qbp_seeded_s25b8", 1, "puct_model=0 puct_prior=model puct_value=null qbp=1 qbp_write=0 qbp_min_truth=sim", 25, 8) +
  "[RLON] result maze=b1 label=b1_null_qbp_seeded_s25b8 ok=0 exit=0 path=0 err=step budget bfs_start=50 bfs_remaining_posthoc=49 have_key=0 door_open=0 in_pit=0 unique_cells=2 mirror_refused=0\n"
  "[RLON] tokens label=b1_null_qbp_seeded_s25b8 actions_sha256=199f76a0a06c224c8ab9204539b0218dd12618013763f34ed1328b32b0a7fb9f h_world_sha256= chain_head=608f7ff4b3194d0d2ccf011aaf982bf31f0453de4f99ff0f0d4277b2c14e2d51 side_log_sha256= tree_digest=021fe0ad7757f92c\n"
  "[RLON] stats label=b1_null_qbp_seeded_s25b8 decisions=25 oracle_calls=0 model_forwards=2288 leaf_evals=200 cache_hits=0 gate_fail_closed=0 mem_q=25 mem_hit=25 mem_miss=0 mem_rej=0 tuples=25 tuple_side_head=8c5fae9f38ea87eba007b2430a2161f487e8606516aefabe01be5685a9912f50 qbp_side_head=8988c695e016a6dfe9d74a2b43b8ed392f0cb716152172aeba4155c952096566 max_ctx_used=176 wall_s=1157.5\n"
  "[RLON] g5 label=b1_null_qbp_seeded_s25b8 tuples_rehash=PASS live_pages_end=0\n";

// ---- Planner comparison: the real oracle s60b16 log, and the cancelled L1 partial (start line only) ----
static const std::string kOracle = start("b1_oracle_s60b16", 0, "puct_model=0 puct_prior=current puct_value=oracle qbp=0 qbp_write=0 qbp_min_truth=sim", 60, 16) +
  "[RLON] result maze=b1 label=b1_oracle_s60b16 ok=1 exit=1 path=50 err= bfs_start=50 bfs_remaining_posthoc=0 have_key=1 door_open=1 in_pit=0 unique_cells=49 mirror_refused=0\n"
  "[RLON] tokens label=b1_oracle_s60b16 actions_sha256=0c21184625c3c03b8cfb4259213604f731098618cbbb578e0473eaa44333aa4e h_world_sha256=457668e71e112ab1fee54d08220ad5dc445245f53a1a91390c95d8c324be8ec5 chain_head=05f42ccc5b7ae53d06e99c3c3eb42585ee4016873fa86fb345c2553fbe8b9ae0 side_log_sha256= tree_digest=0000000000000000\n"
  "[RLON] stats label=b1_oracle_s60b16 decisions=0 oracle_calls=0 model_forwards=0 leaf_evals=0 cache_hits=0 gate_fail_closed=0 mem_q=0 mem_hit=0 mem_miss=0 mem_rej=0 tuples=0 tuple_side_head=0000000000000000000000000000000000000000000000000000000000000000 qbp_side_head=0000000000000000000000000000000000000000000000000000000000000000 max_ctx_used=0 wall_s=13.1\n"
  "[RLON] g5 label=b1_oracle_s60b16 tuples_rehash=PASS live_pages_end=0\n";
static const std::string kL1Partial = start("b1_L1_s60b16", 1, "puct_model=1 puct_prior=model puct_value=model qbp=0 qbp_write=0 qbp_min_truth=sim", 60, 16);
// Edited copies of the L1 empty-store run turned into s60b16 runs without L2 (qbp=0), for the Planner comparison cases.
static std::string as13(std::string s, const std::string& label) {
  s = relabel(s, "b1_L1_qbp_empty_s25b8", label); s = rep(s, "steps=25 budget=8", "steps=60 budget=16"); s = rep(s, "qbp=1 qbp_write=0", "qbp=0 qbp_write=0");
  s = rep(s, "mem_q=25 mem_hit=0 mem_miss=25", "mem_q=0 mem_hit=0 mem_miss=0"); return rep(s, "qbp_side_head=536ed7f23d44e4afb3cecf9707b1a972942631c21abcd662f3a26f87eb08632e", "qbp_side_head=0000000000000000000000000000000000000000000000000000000000000000"); }
static std::string as_null(std::string s) { return rep(s, "puct_model=1 puct_prior=model puct_value=model", "puct_model=0 puct_prior=model puct_value=null"); }

int main() {
  // parser
  const RlonRun e = parse_rlon_log(kL1Empty);
  check(e.complete() && e.label() == "b1_L1_qbp_empty_s25b8" && e.get(e.flags, "qbp") == "1" && e.get(e.start, "budget") == "8", "parser: start line, flags block and settings");
  check(e.get(e.result, "err") == "step budget" && e.num(e.result, "bfs_start") == 50 && e.progress() == 1, "parser: a value with a space (err=step budget) does not swallow the next key");
  check(e.num(e.stats, "model_forwards") == 3020 && e.get(e.g5, "tuples_rehash") == "PASS", "parser: stats and g5 lines");
  check(!parse_rlon_log(kL1Partial).complete(), "parser: a cancelled run (start line only) is incomplete");

  // Query-before-plan on the real logs
  const Verdict v15 = judge_l2_qbp(parse_rlon_log(kL1Seeded), e, parse_rlon_log(kNullSeeded));
  for (auto& l : v15.lines) std::printf("    %s\n", l.c_str());
  check(v15.valid && v15.verdict == "L2_ENGAGED_WALK_UNCHANGED" && v15.exit_code == 0, "1.5 real 2026-11-12 logs: L2 engaged on the side chain, hits did not change the walk");
  bool has_pct = false; for (auto& l : v15.lines) if (l.find("(-4.4%)") != std::string::npos) has_pct = true;
  check(has_pct, "1.5 report: seeded store cut model forwards by 4.4% against the empty store");
  check(!judge_l2_qbp(e, e, parse_rlon_log(kNullSeeded)).valid, "1.5: an empty store passed off as the seeded run is INVALID (no hits)");
  check(!judge_l2_qbp(parse_rlon_log(rep(kL1Seeded, "qbp_write=0", "qbp_write=1")), e, parse_rlon_log(kNullSeeded)).valid, "1.5: mid-run store writes (qbp_write=1) are INVALID");
  check(!judge_l2_qbp(parse_rlon_log(rep(kL1Seeded, "qbp_side_head=8212e308d7244709056fa3bfd0f8bab2ad830b255de8edf7a881332d4fcd4514", "qbp_side_head=0000000000000000000000000000000000000000000000000000000000000000")), e, parse_rlon_log(kNullSeeded)).valid,
        "query-before-plan: no L2 record on the scratch side chain is INVALID");
  const Verdict v15c = judge_l2_qbp(parse_rlon_log(rep(rep(kL1Seeded, "actions_sha256=199f", "actions_sha256=aaaa"), "bfs_remaining_posthoc=49", "bfs_remaining_posthoc=40")), e, parse_rlon_log(kNullSeeded));
  check(v15c.valid && v15c.verdict == "L2_ENGAGED_WALK_CHANGED", "1.5: a seeded walk that differs is reported as changed");

  // Planner comparison cases
  const RlonRun orc = parse_rlon_log(kOracle);
  const std::string l1 = as13(kL1Empty, "b1_L1_s60b16"), nul = as_null(as13(kL1Empty, "b1_null_s60b16"));
  const Verdict same = judge_l1_separation(parse_rlon_log(l1), parse_rlon_log(nul), orc);
  for (auto& l : same.lines) std::printf("    %s\n", l.c_str());
  check(same.valid && same.verdict == "NO_SEPARATION_SAME_WALK" && same.stop_and_report && same.exit_code == 10, "1.3: L1 and null commit the same walk -> no separation, STOP AND REPORT (exit 10)");
  const std::string better = rep(rep(l1, "actions_sha256=199f", "actions_sha256=bbbb"), "bfs_remaining_posthoc=49", "bfs_remaining_posthoc=30");
  const Verdict sep = judge_l1_separation(parse_rlon_log(better), parse_rlon_log(nul), orc);
  check(sep.valid && sep.verdict == "SEPARATED" && !sep.stop_and_report && sep.exit_code == 0, "1.3: L1 gets further than null -> SEPARATED (exit 0)");
  const std::string worse = rep(l1, "actions_sha256=199f", "actions_sha256=cccc");
  const Verdict nb = judge_l1_separation(parse_rlon_log(worse), parse_rlon_log(nul), orc);
  check(nb.valid && nb.verdict == "NO_SEPARATION_NOT_BETTER" && nb.stop_and_report, "1.3: a different walk with no extra progress is not separation -> STOP AND REPORT");
  check(!judge_l1_separation(parse_rlon_log(kL1Partial), parse_rlon_log(nul), orc).valid, "1.3: a cancelled L1 run is INVALID, never read as a result");
  check(!judge_l1_separation(parse_rlon_log(rep(better, "oracle_calls=0", "oracle_calls=7")), parse_rlon_log(nul), orc).valid, "1.3: L1 that called the oracle is INVALID");
  check(!judge_l1_separation(parse_rlon_log(rep(better, "budget=16", "budget=8")), parse_rlon_log(nul), orc).valid, "1.3: mismatched budget between L1 and null is INVALID");
  check(!judge_l1_separation(parse_rlon_log(better), parse_rlon_log(better), orc).valid, "1.3: the null slot must hold a null-value run");
  check(!judge_l1_separation(parse_rlon_log(rep(better, "tuples_rehash=PASS", "tuples_rehash=FAIL")), parse_rlon_log(nul), orc).valid, "1.3: a run whose sealed tuples do not re-hash is INVALID");

  // L2 store path guard
  std::string why;
  check(mem_path_under_scratch("X:/pchie-runs/rl", "X:/pchie-runs/rl/scratch/rl/mem/b1_seed_official.jsonl", why), "store guard: a store under <root>/scratch/rl/mem/ is allowed");
  check(mem_path_under_scratch("X:\\pchie-runs\\rl", "x:/pchie-runs/rl/scratch/rl/mem/x.jsonl", why), "store guard: slashes and drive-letter case are normalized");
  check(!mem_path_under_scratch("X:/pchie-runs/rl", "X:/pchie-runs/rl/scratch/rl/mem/../../official.jsonl", why), "store guard: .. is refused");
  check(!mem_path_under_scratch("X:/pchie-runs/rl", "Y:/other/records/plan_memory.jsonl", why), "store guard: a store outside the scratch tree is refused");
  check(!mem_path_under_scratch("X:/pchie-runs/rl", "X:/pchie-runs/rl/scratch/rl/mem/", why), "store guard: the folder itself is not a store");
  std::printf("RUN-COMPARE: %d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
