// Bolt PCHIE B7 - budget and timeout tests.
// EXPECTED OUTPUTS (defined before implementation; not a golden lock):
// Unit-test fixture only (never feeds a golden or baseline):
//   - B1 maze seed=1, 16x16, node budgets 1, 10, 100 and 1000, 3 runs each.
//     Never creates more PUCT nodes than the cap (max_nodes <= budget).
//     Each capped search appends at least one BudgetExhausted record and
//     still returns a best-so-far action list of only legal maze actions.
//     3/3 H_bolt identical within a budget (run-to-run, not a locked golden).
//     Different caps are not required to share one H_bolt (weaker search may
//     change the path). A budget of 1 need not reach the exit.
//   - An open/unlimited run (no node cap, default B1 search_budget=16) matches
//     the live B1 run_unit H_bolt and chain length, and has no BudgetExhausted
//     records (default behavior is unchanged).
//   - A tick budget (deterministic, not wall-clock) is honored: ticks used
//     never exceed the cap, a BudgetExhausted record is present, 3/3 H_bolt.
//   - A token (committed-action) budget is honored: path_length <= cap, a
//     BudgetExhausted record is present, 3/3 H_bolt, actions stay legal.
//   - node budget <= 0 in the capped list fails cleanly.
//   - Watch / budget-slider UI, a real wall-clock timeout, and CUDA fail
//     cleanly with an explicit gap message, not a fake pass.
// Real Little Nemo (load once):
//   - Node budgets 10 and 100, 2 runs each: never exceed the cap, exhaust
//     record present, 2/2 same H_bolt per budget, legal actions, no KV leak.
//     FNV chain head != first 8 of SHA-256; FNV of SHA bytes != first 8.
// Hashes stay provisional / unlocked. No B8 / B9.
#include "pchie/BudgetHarness.h"
#include "pchie/HeadlessMaze.h"
#include "pchie/GauntletChainTrace.h"
#include "pchie/InjectionChainAudit.h"
#include "pchie/ChainRecord.h"
#include "pchie/NemotronModel.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/Sha256.h"
#include "pchie/MazeGrid.h"
#include "pchie/ModelPath.h"

#include <cstdio>
#include <cstring>
#include <string>
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


static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[BUDGET] %-76s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static uint64_t fnv_bytes(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  uint64_t h = FNV_OFFSET;
  constexpr uint64_t P = 1099511628211ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= uint64_t(b[i]);
    h *= P;
  }
  return h;
}

static bool collide8(const std::vector<uint8_t>& sha, uint64_t fnv) {
  return sha.size() >= 8 && std::memcmp(&fnv, sha.data(), 8) == 0;
}

static bool legal_action(int32_t a) {
  return a == int32_t(MazeAction::North) || a == int32_t(MazeAction::South) ||
         a == int32_t(MazeAction::East) || a == int32_t(MazeAction::West) ||
         a == int32_t(MazeAction::Pickup) || a == int32_t(MazeAction::Use);
}

static void test_unit_node_budgets() {
  HeadlessMaze ref(1, 16, 16);
  GauntletChainTrace ref_ch;
  ref_ch.reset();
  MazeResult rr;
  const bool ref_ok = ref.run_unit("", &ref_ch, rr);
  check(ref_ok && rr.ok && rr.reached_exit,
        "unit: reference maze seed=1 open run succeeds (within 1.2x BFS)");

  BudgetHarness g;
  g.set_seed(1);
  g.set_node_budgets({1, 10, 100, 1000});
  g.set_runs(3);
  BudgetResult out;
  const bool ok = g.run_unit(out);
  check(ok && out.pass && out.rows.size() == 4, "unit: 3-run gauntlet over node budgets 1,10,100,1000 passes");

  bool rows_ok = out.rows.size() == 4;
  bool open_ok = out.open_head_sha == ref_ch.head() && out.open_chain_length == ref_ch.length() &&
                 out.open_head_sha.size() == 32;
  check(open_ok, "unit: unlimited/open run matches the live maze H_bolt and chain length");

  const int want[] = {1, 10, 100, 1000};
  for (size_t i = 0; i < out.rows.size(); ++i) {
    const BudgetRow& row = out.rows[i];
    if (!(row.node_budget == want[i] && row.runs == 3 && row.unique_hashes == 1 && row.pass &&
          row.max_nodes > 0 && row.max_nodes <= row.node_budget && row.exhaust_records >= 1 &&
          row.off_legal == 0 && row.head_sha.size() == 32 && row.first_diff.index < 0 &&
          row.chain_length > 0))
      rows_ok = false;
    std::printf("[BUDGET] unit nodes=%d max_nodes=%d exhaust=%d exit=%d chain_len=%zu sha=%s\n",
                row.node_budget, row.max_nodes, row.exhaust_records, int(row.reached_exit),
                row.chain_length,
                row.head_sha.empty() ? "" : GgufWeightsHash::to_hex(row.head_sha).c_str());
  }
  check(rows_ok, "unit: each node cap is honored, BudgetExhausted is recorded, 3/3 H_bolt, legal actions");

  if (!out.open_head_sha.empty()) {
    const uint64_t hfnv = fnv_bytes(out.open_head_sha.data(), out.open_head_sha.size());
    check(!collide8(out.open_head_sha, hfnv), "unit: open H_bolt FNV-of-SHA != first 8 of SHA-256");
    std::printf("[BUDGET] unit open H_bolt sha256=%s chain_len=%zu (provisional, unlocked)\n",
                GgufWeightsHash::to_hex(out.open_head_sha).c_str(), out.open_chain_length);
  }

  BudgetHarness bad;
  bad.set_node_budgets({1, 0, 10});
  bad.set_runs(3);
  BudgetResult bo;
  check(bad.run_unit(bo) == false && !bo.pass && bo.err.find("budget") != std::string::npos,
        "unit: a non-positive node budget in the list fails cleanly");
}

static void test_unit_tick_and_token() {
  BudgetHarness g;
  g.set_seed(1);
  BudgetResult ticks;
  const bool tok = g.run_tick_budget_unit(/*ticks*/8, /*runs*/3, ticks);
  check(tok && ticks.pass && ticks.tick_limit_used > 0 && ticks.tick_limit_used <= 8 &&
            ticks.tick_exhaust_records >= 1,
        "unit: tick budget 8 is honored (deterministic, not wall-clock) with BudgetExhausted");
  BudgetResult tokens;
  const bool tnk = g.run_token_budget_unit(/*tokens*/5, /*runs*/3, tokens);
  check(tnk && tokens.pass && tokens.token_limit_used > 0 && tokens.token_limit_used <= 5 &&
            tokens.token_exhaust_records == 1,
        "unit: token budget 5 is honored (path_length <= 5) with one BudgetExhausted");
  check(tok && ticks.tick_exhaust_records == 1,
        "unit: tick budget 8 records BudgetExhausted once (no double-append after best-so-far)");

  BudgetResult t1;
  const bool t1ok = g.run_tick_budget_unit(/*ticks*/1, /*runs*/3, t1);
  check(t1ok && t1.pass && t1.tick_limit_used == 1 && t1.tick_exhaust_records == 1,
        "unit: tick budget 1 uses exactly one tick and records exhaustion once");
  BudgetResult k1;
  const bool k1ok = g.run_token_budget_unit(/*tokens*/1, /*runs*/3, k1);
  check(k1ok && k1.pass && k1.token_limit_used == 1 && k1.token_exhaust_records == 1,
        "unit: token budget 1 commits exactly one legal action and records exhaustion once");

  BudgetResult z;
  check(g.run_tick_budget_unit(/*ticks*/0, /*runs*/3, z) == false && !z.pass &&
            z.err.find("budget") != std::string::npos,
        "unit: a non-positive tick budget fails cleanly");

  BudgetResult gap;
  check(g.run_watch_unsupported(gap) == false && !gap.pass &&
            (gap.err.find("watch") != std::string::npos || gap.err.find("slider") != std::string::npos),
        "unit: watch / budget-slider UI fails cleanly (not faked)");
  check(g.run_wallclock_unsupported(gap) == false && !gap.pass &&
            (gap.err.find("wall") != std::string::npos || gap.err.find("clock") != std::string::npos),
        "unit: real wall-clock timeout fails cleanly (tick budget is the honest path)");
  check(g.run_cuda_unsupported(gap) == false && !gap.pass && gap.err.find("CUDA") != std::string::npos,
        "unit: CUDA configuration fails cleanly (not faked)");
}

static uint32_t u32le(const std::vector<uint8_t>& b, size_t off) {
  if (off + 4 > b.size()) return 0;
  return uint32_t(b[off]) | (uint32_t(b[off + 1]) << 8) | (uint32_t(b[off + 2]) << 16) |
         (uint32_t(b[off + 3]) << 24);
}

static void test_unit_adversarial() {
  // Direct driver checks (unit-test fixture only). Confirms empty-commit is not
  // fatal at node_budget=1 and that BudgetExhausted payload is reason/limit/used.
  HeadlessMaze maze(1, 16, 16);
  maze.set_node_budget(1);
  maze.set_search_budget(16);
  InjectionChainAudit ch;
  MazeResult res;
  (void)maze.run_unit("", &ch, res);
  bool legal = !res.actions.empty();
  for (int32_t a : res.actions)
    if (!legal_action(a)) legal = false;
  check(res.err != "empty commit" && !res.actions.empty() && legal && res.max_nodes == 1 &&
            res.budget_exhausted && res.exhaust_records >= 1,
        "unit: node_budget=1 returns a legal best-so-far (no empty-commit failure)");

  int be = 0;
  bool payload_ok = true;
  const auto& kinds = ch.trace().kinds();
  const auto& pays = ch.payloads();
  if (kinds.size() != pays.size()) payload_ok = false;
  for (size_t i = 0; i < kinds.size(); ++i) {
    if (kinds[i] != ChainRecordKind::BudgetExhausted) continue;
    ++be;
    if (pays[i].size() != 12 || u32le(pays[i], 0) != 1u /* nodes */ || u32le(pays[i], 4) != 1u ||
        u32le(pays[i], 8) != 1u)
      payload_ok = false;
  }
  check(be == res.exhaust_records && be == int(res.actions.size()) && payload_ok,
        "unit: node_budget=1 appends one BudgetExhausted(reason=nodes,limit=1,used=1) per action");

  // Tick exhaustion must stop further primitives after the best-so-far action.
  HeadlessMaze tm(1, 16, 16);
  tm.set_tick_budget(8);
  InjectionChainAudit tch;
  MazeResult tr;
  (void)tm.run_unit("", &tch, tr);
  int te = 0;
  bool tick_payload = true;
  const auto& tk = tch.trace().kinds();
  const auto& tp = tch.payloads();
  for (size_t i = 0; i < tk.size(); ++i) {
    if (tk[i] != ChainRecordKind::BudgetExhausted) continue;
    ++te;
    if (tp[i].size() != 12 || u32le(tp[i], 0) != 2u /* ticks */ || u32le(tp[i], 4) != 8u ||
        u32le(tp[i], 8) > 8u)
      tick_payload = false;
  }
  check(tr.ticks_used == 8 && te == 1 && tick_payload && !tr.actions.empty(),
        "unit: tick budget 8 stops after one best-so-far and one ticks payload");
}

static void test_real_little_nemo() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight budget: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[BUDGET] load err: %s\n", err.c_str());
    return;
  }
  m.init_cache(/*max_ctx*/32, /*branches*/8);
  const uint32_t live0 = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  BudgetHarness g;
  g.set_seed(1);
  g.set_node_budgets({10, 100});
  g.set_runs(2);
  BudgetResult out;
  const bool ok = g.run(m, out);
  check(ok && out.pass && out.rows.size() == 2, "real Little Nemo: 2-run gauntlet over node budgets 10,100 passes");
  bool rows_ok = out.rows.size() == 2;
  for (const BudgetRow& row : out.rows) {
    if (!(row.runs == 2 && row.unique_hashes == 1 && row.pass && row.max_nodes > 0 &&
          row.max_nodes <= row.node_budget && row.exhaust_records >= 1 && row.off_legal == 0 &&
          row.first_diff.index < 0 && row.head_sha.size() == 32))
      rows_ok = false;
    std::printf("[BUDGET] real nodes=%d max_nodes=%d exhaust=%d sha=%s\n", row.node_budget, row.max_nodes,
                row.exhaust_records,
                row.head_sha.empty() ? "" : GgufWeightsHash::to_hex(row.head_sha).c_str());
  }
  check(rows_ok, "real Little Nemo: each cap is honored, BudgetExhausted recorded, 2/2 H_bolt, legal actions");
  if (!out.rows.empty() && out.rows[0].head_sha.size() >= 8) {
    const uint64_t hfnv = fnv_bytes(out.rows[0].head_sha.data(), out.rows[0].head_sha.size());
    check(!collide8(out.rows[0].head_sha, hfnv), "real Little Nemo: H_bolt FNV-of-SHA != first 8 of SHA-256");
    uint64_t first8 = 0;
    std::memcpy(&first8, out.rows[0].head_sha.data(), 8);
    const bool agree = out.rows[0].head_fnv == first8;
    check(!agree, "real Little Nemo: FNV-1a64 chain head != first 8 bytes of SHA-256 H_bolt");
    if (agree)
      std::printf("[BUDGET] ALERT: FNV chain head agrees with first 8 of SHA-256 H_bolt -- report it immediately\n");
  }

  const uint32_t live_end = m.pool_swa().live_pages() + m.pool_glob().live_pages();
  check(live_end == live0, "real Little Nemo: no KV page leak after budget runs");
  std::printf("[BUDGET] live_pages start=%u end=%u\n", live0, live_end);
}

int main() {
  std::printf("[BUDGET] budget and timeout tests (headless maze; watch/wall-clock/CUDA deferred)\n");
  test_unit_node_budgets();
  test_unit_tick_and_token();
  test_unit_adversarial();
  test_real_little_nemo();
  std::printf("[BUDGET] budget %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
