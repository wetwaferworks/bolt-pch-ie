// Bolt PCHIE B5 - headless freeze and recovery tests.
// EXPECTED OUTPUTS (defined before implementation; not a golden lock):
// Unit-test fixture only (never feeds a golden or baseline):
//   - FreezeWatchdog: allocate_slot issues unique ids; has_duplicate_slot() is false
//     after a normal allocate/begin/end sequence.
//   - force_freeze on an in-flight slot latches frozen(), records last_freeze,
//     and try_commit of that slot fails (the stalled rollout is not committed).
//   - resetAfterFreeze clears the latch and bumps generation. A try_commit with
//     the pre-reset generation is rejected (stale worker).
//   - A slot marked reset_stuck stays reserved: allocate_slot never returns it
//     (no duplicate slot) until finish_reset. After finish_reset the id may be
//     reused only once the reservation is gone.
//   - force_freeze on a slot that is not in flight fails cleanly.
//   - allocate after freeze (before resetAfterFreeze) still does not reuse the
//     frozen slot id.
//   - FreezeRecovery over B1 seed=1, 16x16, 3 trials, freeze at
//     (GoToKeyA, step 0, iter 0): every trial succeeds, post-recovery H_bolt,
//     H_world, action list, and chain length equal the clean B1 run_unit (and
//     the live HeadlessMaze reference). The stalled slot is not in
//     committed_slots. has_duplicate_slot is false.
//   - A second freeze point (step 1, iter 2) also matches the clean run (3 trials).
//   - trials < 3 fails cleanly.
//   - Watch mode and the mutex/CV LithoLattice handshake authority fail cleanly
//     with an explicit gap message, not a fake pass.
// Real Little Nemo (load once):
//   - 3 freeze trials at (GoToKeyA, step 0, iter 0) match the clean real B1 run
//     (H_bolt, H_world, chain length). NCM was produced on the clean path; no
//     KV leak. FNV chain head != first 8 of SHA-256; FNV of SHA bytes != first 8.
// Hashes stay provisional / unlocked. No B6.
#include "pchie/FreezeRecovery.h"
#include "pchie/FreezeWatchdog.h"
#include "pchie/HeadlessMaze.h"
#include "pchie/GauntletChainTrace.h"
#include "pchie/DeterminismGauntlet.h"
#include "pchie/NemotronModel.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/Sha256.h"
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
  std::printf("[FREEZE] %-76s %s\n", what, ok ? "PASS" : "FAIL");
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

static bool contains_slot(const std::vector<uint32_t>& v, uint32_t s) {
  for (uint32_t x : v)
    if (x == s) return true;
  return false;
}

static void test_watchdog_slots() {
  FreezeWatchdog w;
  w.reset();
  const uint32_t a = w.allocate_slot();
  const uint32_t b = w.allocate_slot();
  const uint32_t c = w.allocate_slot();
  check(a != b && b != c && a != c, "unit: allocate_slot issues unique ids");
  check(w.begin_rollout(a) && w.begin_rollout(b) && w.begin_rollout(c),
        "unit: begin_rollout accepts freshly allocated slots");
  w.end_rollout(a);
  w.end_rollout(b);
  w.end_rollout(c);
  check(!w.has_duplicate_slot() && w.live_in_flight() == 0,
        "unit: no duplicate slot after a normal allocate/begin/end sequence");

  const uint32_t s = w.allocate_slot();
  check(w.begin_rollout(s), "unit: begin in-flight slot for freeze");
  const uint32_t gen0 = w.generation();
  check(w.force_freeze(s, /*task*/5, /*step*/0, /*iter*/0) && w.frozen() && w.last_freeze().valid &&
            w.last_freeze().slot == s && w.last_freeze().generation == gen0,
        "unit: force_freeze latches frozen() and records last_freeze");
  check(!w.try_commit(s, gen0) && w.slot_discarded(s),
        "unit: stalled slot cannot be committed (rollout discarded)");
  const uint32_t s2 = w.allocate_slot();
  check(s2 != s && w.slot_reserved(s),
        "unit: allocate after freeze does not reuse the frozen slot id");

  w.resetAfterFreeze();
  check(!w.frozen() && w.generation() != gen0,
        "unit: resetAfterFreeze clears the latch and bumps generation");
  check(!w.try_commit(s, gen0), "unit: try_commit with the pre-reset generation is rejected");

  // Stuck reset: slot stays reserved; no duplicate; stale commit still rejected.
  FreezeWatchdog w2;
  w2.reset();
  const uint32_t t = w2.allocate_slot();
  check(w2.begin_rollout(t) && w2.force_freeze(t, 5, 0, 0), "unit: freeze a slot for stuck-reset");
  w2.mark_reset_stuck(t);
  w2.resetAfterFreeze();
  const uint32_t t2 = w2.allocate_slot();
  check(t2 != t && w2.slot_reserved(t) && !w2.has_duplicate_slot(),
        "unit: reset_stuck slot stays reserved (no duplicate slot) after resetAfterFreeze");
  check(!w2.try_commit(t, w2.generation()) && !w2.try_commit(t, w2.last_freeze().generation),
        "unit: stuck reset cannot commit in the new or old generation");
  w2.finish_reset(t);
  check(!w2.slot_reserved(t), "unit: finish_reset releases the reservation");
  const uint32_t t3 = w2.allocate_slot();
  (void)t3;
  check(!w2.slot_reserved(t) && !w2.has_duplicate_slot(),
        "unit: after finish_reset the table has no reserved or duplicate slot");

  FreezeWatchdog w3;
  w3.reset();
  check(!w3.force_freeze(99, 5, 0, 0) && !w3.frozen(),
        "unit: force_freeze on a slot that is not in flight fails cleanly");
  check(!w3.begin_rollout(99), "unit: begin_rollout of an unallocated slot fails cleanly");
}

static void test_unit_recovery() {
  HeadlessMaze ref(1, 16, 16);
  GauntletChainTrace ref_ch;
  ref_ch.reset();
  MazeResult rr;
  const bool ref_ok = ref.run_unit("", &ref_ch, rr);
  check(ref_ok && rr.ok && rr.reached_exit && rr.within_oracle,
        "unit: reference maze seed=1 run succeeds (within 1.2x BFS)");

  FreezeRecovery g;
  g.set_seed(1);
  g.set_trials(3);
  g.set_freeze_point(HeadlessMaze::kGoToKeyA, /*step*/0, /*iter*/0);
  FreezeResult out;
  const bool ok = g.run_unit(out);
  check(ok && out.pass && out.trials.size() == 3 && out.matching == 3,
        "unit: 3 freeze trials at (GoToKeyA, step 0, iter 0) pass");
  check(out.clean_head_sha == ref_ch.head() && out.clean_world_sha == rr.h_world_sha &&
            out.clean_chain_length == ref_ch.length() && out.clean_head_sha.size() == 32,
        "unit: clean H_bolt / H_world / chain length match the live maze reference");
  bool rows = out.trials.size() == 3;
  for (const FreezeTrial& tr : out.trials) {
    if (!(tr.ok && tr.head_sha == out.clean_head_sha && tr.world_sha == out.clean_world_sha &&
          tr.actions_sha == out.clean_actions_sha && tr.chain_length == out.clean_chain_length &&
          tr.first_diff.index < 0 && tr.discarded_slot >= 0 &&
          !contains_slot(tr.committed_slots, uint32_t(tr.discarded_slot))))
      rows = false;
  }
  check(rows, "unit: each trial matches clean H_bolt and never commits the stalled slot");
  const uint64_t hfnv = fnv_bytes(out.clean_head_sha.data(), out.clean_head_sha.size());
  check(!out.clean_head_sha.empty() && !collide8(out.clean_head_sha, hfnv),
        "unit: H_bolt FNV-of-SHA != first 8 bytes of SHA-256");
  if (!out.trials.empty() && out.clean_head_sha.size() >= 8) {
    uint64_t first8 = 0;
    std::memcpy(&first8, out.clean_head_sha.data(), 8);
    const bool agree = out.trials[0].head_fnv == first8;
    check(!agree, "unit: FNV-1a64 chain head != first 8 bytes of SHA-256 H_bolt");
    if (agree)
      std::printf("[FREEZE] ALERT: FNV chain head agrees with first 8 of SHA-256 H_bolt -- report it immediately\n");
  }
  std::printf("[FREEZE] unit H_bolt sha256=%s chain_len=%zu (provisional, unlocked)\n",
              out.clean_head_sha.empty() ? "" : GgufWeightsHash::to_hex(out.clean_head_sha).c_str(),
              out.clean_chain_length);

  FreezeRecovery g2;
  g2.set_seed(1);
  g2.set_trials(3);
  g2.set_freeze_point(HeadlessMaze::kGoToKeyA, /*step*/1, /*iter*/2);
  FreezeResult out2;
  const bool ok2 = g2.run_unit(out2);
  check(ok2 && out2.pass && out2.matching == 3 && out2.clean_head_sha == out.clean_head_sha,
        "unit: 3 trials at (GoToKeyA, step 1, iter 2) also match the same clean H_bolt");

  FreezeRecovery bad;
  bad.set_trials(2);
  FreezeResult bo;
  check(bad.run_unit(bo) == false && !bo.pass && bo.err.find("trial") != std::string::npos,
        "unit: trials < 3 fails cleanly");

  FreezeResult wout;
  check(g.run_watch_unsupported(wout) == false && !wout.pass &&
            wout.err.find("watch") != std::string::npos,
        "unit: watch mode fails cleanly (not faked)");
  check(g.run_threaded_handshake_unsupported(wout) == false && !wout.pass &&
            (wout.err.find("handshake") != std::string::npos || wout.err.find("mutex") != std::string::npos),
        "unit: threaded LithoLattice handshake authority fails cleanly (not faked)");
}


static void test_adversarial() {
  // Second force_freeze is rejected; first last_freeze is kept.
  {
    FreezeWatchdog w;
    w.reset();
    const uint32_t a = w.allocate_slot();
    const uint32_t b = w.allocate_slot();
    check(w.begin_rollout(a) && w.begin_rollout(b), "adversarial: two in-flight slots");
    check(w.force_freeze(a, 5, 0, 0) && !w.force_freeze(b, 5, 0, 1) && w.last_freeze().slot == a,
          "adversarial: second force_freeze is rejected and last_freeze is unchanged");
  }

  // A freeze point that never occurs does not claim a recovery pass.
  {
    FreezeRecovery g;
    g.set_seed(1);
    g.set_trials(3);
    g.set_freeze_point(/*unknown task*/999, 0, 0);
    FreezeResult out;
    const bool ok = g.run_unit(out);
    check(ok == false && !out.pass && (out.trials.empty() || out.trials[0].discarded_slot < 0),
          "adversarial: a freeze point that never fires fails cleanly (no false recovery)");
  }

  // Freeze on the last primitive, mid budget, still matches the clean hash.
  {
    FreezeRecovery g;
    g.set_seed(1);
    g.set_trials(3);
    g.set_freeze_point(HeadlessMaze::kGoToExit, /*step*/0, /*iter*/7);
    FreezeResult out;
    const bool ok = g.run_unit(out);
    check(ok && out.pass && out.matching == 3 && out.clean_head_sha.size() == 32,
          "adversarial: 3 trials frozen at (GoToExit, step 0, iter 7) match the clean run");
  }

  // Seed=2, 3 unit trials, first iteration: deterministic, not a golden of seed=1.
  {
    FreezeRecovery g;
    g.set_seed(2);
    g.set_trials(3);
    g.set_freeze_point(HeadlessMaze::kGoToKeyA, 0, 0);
    FreezeResult out;
    const bool ok = g.run_unit(out);
    check(ok && out.pass && out.matching == 3 && out.clean_head_sha.size() == 32,
          "adversarial: seed=2 unit 3/3 freeze recovery is deterministic (not a golden)");
  }

  // Chain length 0 vs a one-record chain: first_diff names the missing index.
  {
    GauntletChainTrace a, b;
    a.append(ChainRecord{ChainRecordKind::Backup, {1}});
    const GauntletDiff d = DeterminismGauntlet::first_diff(a, b);
    check(d.index == 0 && d.kind_a_present && !d.kind_b_present && d.kind_a == ChainRecordKind::Backup,
          "adversarial: first_diff of an empty recovery chain names index 0");
  }
}

static void test_real_little_nemo() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight freeze: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[FREEZE] load err: %s\n", err.c_str());
    return;
  }
  m.init_cache(/*max_ctx*/32, /*branches*/8);
  const uint32_t live0 = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  FreezeRecovery g;
  g.set_seed(1);
  g.set_trials(3);
  g.set_freeze_point(HeadlessMaze::kGoToKeyA, 0, 0);
  FreezeResult out;
  const bool ok = g.run(m, out);
  check(ok && out.pass && out.trials.size() == 3 && out.matching == 3,
        "real Little Nemo: 3 freeze trials match the clean run");
  check(out.clean_head_sha.size() == 32 && out.clean_world_sha.size() == 32 && out.clean_chain_length > 0,
        "real Little Nemo: clean H_bolt and H_world are 32-byte SHA-256");
  bool rows = out.trials.size() == 3;
  for (const FreezeTrial& tr : out.trials) {
    if (!(tr.ok && tr.head_sha == out.clean_head_sha && tr.world_sha == out.clean_world_sha &&
          tr.chain_length == out.clean_chain_length && tr.first_diff.index < 0 &&
          tr.discarded_slot >= 0 && !contains_slot(tr.committed_slots, uint32_t(tr.discarded_slot))))
      rows = false;
  }
  check(rows, "real Little Nemo: each trial matches clean H_bolt and never commits the stalled slot");
  const uint64_t hfnv = fnv_bytes(out.clean_head_sha.data(), out.clean_head_sha.size());
  check(!collide8(out.clean_head_sha, hfnv), "real Little Nemo: H_bolt FNV-of-SHA != first 8 of SHA-256");
  if (!out.trials.empty() && out.clean_head_sha.size() >= 8) {
    uint64_t first8 = 0;
    std::memcpy(&first8, out.clean_head_sha.data(), 8);
    const bool agree = out.trials[0].head_fnv == first8;
    check(!agree, "real Little Nemo: FNV-1a64 chain head != first 8 bytes of SHA-256 H_bolt");
    if (agree)
      std::printf("[FREEZE] ALERT: FNV chain head agrees with first 8 of SHA-256 H_bolt -- report it immediately\n");
  }
  std::printf("[FREEZE] real H_bolt sha256=%s fnv=0x%016llX world=%s (provisional, unlocked)\n",
              out.clean_head_sha.empty() ? "" : GgufWeightsHash::to_hex(out.clean_head_sha).c_str(),
              out.trials.empty() ? 0ULL : (unsigned long long)out.trials[0].head_fnv,
              out.clean_world_sha.empty() ? "" : GgufWeightsHash::to_hex(out.clean_world_sha).c_str());

  const uint32_t live_end = m.pool_swa().live_pages() + m.pool_glob().live_pages();
  check(live_end == live0, "real Little Nemo: no KV page leak after freeze trials");
  std::printf("[FREEZE] live_pages start=%u end=%u\n", live0, live_end);
}

int main() {
  std::printf("[FREEZE] freeze and recovery tests (headless maze; watch/handshake-authority deferred)\n");
  test_watchdog_slots();
  test_unit_recovery();
  test_adversarial();
  test_real_little_nemo();
  std::printf("[FREEZE] freeze %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
