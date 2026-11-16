// Bolt PCHIE B4 - CPU determinism gauntlet tests.
// EXPECTED OUTPUTS (defined before implementation; not a golden lock):
// Unit-test fixture only (never feeds a golden or baseline):
//   - Amendment-2 pin: Ln / Sigmoid / Pow / exp_deterministic bits match the
//     existing pchie_search_rules_test vectors exactly.
//   - JoinJobDispatcher(n) for n in {1,2,4,8,16}: size()==n; parallel_for(count)
//     invokes fn(0)..fn(count-1) exactly once; result is independent of n.
//   - n<=0 or n>16 is rejected (size()==0, parallel_for is a no-op).
//   - GauntletChainTrace: appends kinds in order; SHA and FNV heads are deterministic;
//     first_diff of two identical traces is empty; a one-record difference
//     reports that index and both kinds.
//   - Gauntlet over B1 seed=1, 16x16, 100 runs, workers {1,2,4,8,16}: every row
//     100/100, unique_hashes==1, all five rows share the same H_bolt and the
//     same actions / H_world as a single HeadlessMaze::run_unit of seed=1.
//   - set_perturb(config, run) makes that row fail and names the first differing
//     step (index and ChainRecordKind).
//   - CUDA / B3 / Linux are not claimed; requesting them must fail cleanly, not
//     return a fake pass.
// Real Little Nemo (load once):
//   - Same five worker counts, 2 runs each (100-run real-weight and CUDA deferred).
//   - All ten B1::run results share one H_bolt, one H_world, one action list.
//   - NCM was produced; no KV leak; FNV != first 8 bytes of SHA-256.
// Hashes stay provisional / unlocked. No B5.
#include "pchie/DeterminismGauntlet.h"
#include "pchie/GauntletChainTrace.h"
#include "pchie/JoinJobDispatcher.h"
#include "pchie/HeadlessMaze.h"
#include "pchie/NemotronModel.h"
#include "pchie/DetLog.h"
#include "pchie/pchie_det_math.h"
#include "pchie/Sha256.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/pchie_tree.h"
#include "pchie/ChainRecord.h"
#include "pchie/ModelPath.h"

#include <cstdio>
#include <stdexcept>
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
  std::printf("[GAUNTLET] %-76s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static uint64_t bits_of(double d) {
  uint64_t u;
  std::memcpy(&u, &d, 8);
  return u;
}

static uint64_t fnv_bytes(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  uint64_t h = FNV_OFFSET;
  constexpr uint64_t P = 1099511628211ULL;
  for (size_t i = 0; i < n; ++i) { h ^= uint64_t(b[i]); h *= P; }
  return h;
}

static bool collide8(const std::vector<uint8_t>& sha, uint64_t fnv) {
  return sha.size() >= 8 && std::memcmp(&fnv, sha.data(), 8) == 0;
}

static void test_det_math_pins() {
  // Same vectors as tests/pchie_search_rules_test.cpp (Archie amendment 2).
  const double xs[] = {1e-9, 0.1, 0.5, 1.0, 2.0, 3.14159, 1000.0, 262144.0};
  static const uint64_t want[] = {
      0xC034B927F32BFFB8ULL, 0xC0026BB1BBB55515ULL, 0xBFE62E42FEFA39EFULL,
      0x0000000000000000ULL, 0x3FE62E42FEFA39EFULL, 0x3FF250CF662ABF9AULL,
      0x401BA18A998FFFA0ULL, 0x4028F40B5ED9812DULL, 0x3F35FA3DB7114E2BULL,
      0x3FD136561ABDF00AULL, 0x3FE0000000000000ULL, 0x3FE3EB2FD371BE79ULL,
      0x3FEE7B7CBCF0EBF0ULL, 0x3FF9FDF8C0000000ULL, 0x40051CB440000000ULL,
      0x402A73EC00000000ULL, 0x3E21B48660000000ULL, 0x3FD78B5640000000ULL,
      0x3FF0000000000000ULL, 0x4001CDE860000000ULL, 0x40D5829DC0000000ULL};
  uint64_t got[32];
  int k = 0;
  for (double x : xs) got[k++] = bits_of(DetLog::Ln(x));
  for (double x : {-8.0, -1.0, 0.0, 0.5, 3.0}) got[k++] = bits_of(DetLog::Sigmoid(x));
  for (double x : {2.0, 4.0, 40.0}) got[k++] = bits_of(DetLog::Pow(x, 0.7));
  for (float x : {-20.0f, -1.0f, 0.0f, 0.8f, 10.0f}) got[k++] = bits_of(double(exp_deterministic(x)));
  bool ok = sizeof(want) / sizeof(want[0]) == size_t(k);
  for (int i = 0; ok && i < k; ++i) ok = got[i] == want[i];
  check(ok, "unit: Ln / Sigmoid / Pow / exp_deterministic bits pinned (amendment 2)");
}

static void test_join_dispatcher() {
  const int ns[] = {1, 2, 4, 8, 16};
  int sums[5] = {0, 0, 0, 0, 0};
  int visits = 0;
  for (int i = 0; i < 5; ++i) {
    JoinJobDispatcher d(ns[i]);
    check(d.size() == ns[i], (std::string("unit: JoinJobDispatcher(") + std::to_string(ns[i]) + ").size()").c_str());
    const int count = 32;
    // Per-index slots only: a shared sums[i] += k would data-race at workers>1.
    std::vector<int> hit(size_t(count), 0);
    d.parallel_for(count, [&](int k) {
      if (k >= 0 && k < count) hit[size_t(k)] = k + 1;
    });
    bool once = true;
    int s = 0;
    for (int k = 0; k < count; ++k) {
      if (hit[size_t(k)] != k + 1) once = false;
      s += hit[size_t(k)];
    }
    sums[i] = s;
    check(once, (std::string("unit: n=") + std::to_string(ns[i]) + " visits each index exactly once").c_str());
    visits += once ? 1 : 0;
  }
  check(visits == 5 && sums[0] == sums[1] && sums[1] == sums[2] && sums[2] == sums[3] && sums[3] == sums[4],
        "unit: parallel_for reduction is independent of worker count");
  {
    NemotronModel m;
    JoinJobDispatcher d8(8);
    m.set_dispatcher(&d8);
    const int mid = m.dispatcher_size();
    m.set_dispatcher(nullptr);
    const int back = m.dispatcher_size();
    check(mid == 8 && back == 1, "unit: NemotronModel::set_dispatcher routes size 8 then back to serial 1");
  }
  int refused = 0;
  for (int bad : {0, -3, 17}) {
    try { JoinJobDispatcher d(bad); } catch (const std::invalid_argument&) { ++refused; }
  }
  check(refused == 3, "unit: workers <=0 or >16 are refused when the dispatcher is built");
}

static void test_chain_trace() {
  GauntletChainTrace a, b, c;
  a.reset();
  b.reset();
  a.append(ChainRecord{ChainRecordKind::Custom, {1, 2, 3, 4}});
  a.append(ChainRecord{ChainRecordKind::Expansion, {9}});
  a.append(ChainRecord{ChainRecordKind::Commit, {7, 8, 9, 10}});
  b.append(ChainRecord{ChainRecordKind::Custom, {1, 2, 3, 4}});
  b.append(ChainRecord{ChainRecordKind::Expansion, {9}});
  b.append(ChainRecord{ChainRecordKind::Commit, {7, 8, 9, 10}});
  check(a.length() == 3 && a.kinds().size() == 3 && a.kinds()[0] == ChainRecordKind::Custom &&
            a.kinds()[2] == ChainRecordKind::Commit,
        "unit: GauntletChainTrace records kinds in append order");
  check(a.head() == b.head() && a.head().size() == 32 && a.fnv_head() == b.fnv_head() &&
            a.fnv_head().size() == 8,
        "unit: identical records => identical SHA-256 and FNV heads");
  const GauntletDiff same = DeterminismGauntlet::first_diff(a, b);
  check(same.index < 0, "unit: first_diff of identical traces is empty");
  c.append(ChainRecord{ChainRecordKind::Custom, {1, 2, 3, 4}});
  c.append(ChainRecord{ChainRecordKind::Selection, {9}});  // differs in kind at index 1
  c.append(ChainRecord{ChainRecordKind::Commit, {7, 8, 9, 10}});
  const GauntletDiff d = DeterminismGauntlet::first_diff(a, c);
  check(d.index == 1 && d.kind_a == ChainRecordKind::Expansion && d.kind_b == ChainRecordKind::Selection,
        "unit: first_diff names the first differing step and both record kinds");
  const uint64_t fnv = fnv_bytes(a.head().data(), a.head().size());
  check(!collide8(a.head(), fnv), "unit: trace SHA-256 head FNV != first 8 bytes of SHA-256");
  if (collide8(a.head(), fnv))
    std::printf("[GAUNTLET] ALERT: FNV vs SHA disagreement FAILED on chain head -- report it immediately\n");
}

static const char* kind_name(ChainRecordKind k) {
  switch (k) {
    case ChainRecordKind::Genesis: return "Genesis";
    case ChainRecordKind::Model: return "Model";
    case ChainRecordKind::Config: return "Config";
    case ChainRecordKind::Prompt: return "Prompt";
    case ChainRecordKind::Expansion: return "Expansion";
    case ChainRecordKind::Selection: return "Selection";
    case ChainRecordKind::LeafValue: return "LeafValue";
    case ChainRecordKind::Backup: return "Backup";
    case ChainRecordKind::Commit: return "Commit";
    case ChainRecordKind::Handshake: return "Handshake";
    case ChainRecordKind::GateVerdict: return "GateVerdict";
    case ChainRecordKind::Custom: return "Custom";
    default: return "Other";
  }
}

static void test_unit_gauntlet() {
  // Labeled unit-test fixture: B1 maze, no model, 100 runs x 5 worker counts.
  HeadlessMaze ref(1, 16, 16);
  GauntletChainTrace ref_ch;
  ref_ch.reset();
  MazeResult rr;
  const bool ref_ok = ref.run_unit("", &ref_ch, rr);
  check(ref_ok && rr.ok && rr.reached_exit && rr.within_oracle,
        "unit: reference maze seed=1 run succeeds (within 1.2x BFS)");

  DeterminismGauntlet g;
  g.set_workers({1, 2, 4, 8, 16});
  g.set_runs(100);
  g.set_seed(1);
  GauntletResult out;
  const bool ok = g.run_unit(out);
  check(ok && out.pass && out.rows.size() == 5, "unit: 100-run gauntlet over workers 1,2,4,8,16 passes");
  bool rows_ok = out.rows.size() == 5;
  for (const GauntletRow& row : out.rows) {
    if (!(row.runs == 100 && row.unique_hashes == 1 && row.b1_failures == 0 && row.pass &&
          row.head_sha == out.ref_head_sha && row.first_diff.index < 0))
      rows_ok = false;
  }
  check(rows_ok, "unit: each row is 100/100, unique_hashes=1, same H_bolt, no first_diff");
  check(out.ref_head_sha == ref_ch.head() && out.ref_head_sha.size() == 32 &&
            out.ref_world_sha == rr.h_world_sha,
        "unit: gauntlet H_bolt and H_world match the reference maze run_unit");
  check(out.ref_chain_length == ref_ch.length() && out.ref_chain_length > 0,
        "unit: gauntlet chain length matches the reference maze run");
  const uint64_t hfnv = fnv_bytes(out.ref_head_sha.data(), out.ref_head_sha.size());
  check(!out.ref_head_sha.empty() && !collide8(out.ref_head_sha, hfnv),
        "unit: H_bolt FNV != first 8 bytes of SHA-256");
  if (collide8(out.ref_head_sha, hfnv))
    std::printf("[GAUNTLET] ALERT: FNV vs SHA disagreement FAILED on unit H_bolt -- report it immediately\n");

  if (!out.rows.empty() && out.ref_head_sha.size() >= 8) {
    uint64_t first8 = 0;
    std::memcpy(&first8, out.ref_head_sha.data(), 8);
    const bool agree = out.rows[0].head_fnv == first8;
    check(!agree, "unit: FNV-1a64 chain head != first 8 bytes of SHA-256 H_bolt");
    if (agree)
      std::printf("[GAUNTLET] ALERT: FNV chain head agrees with first 8 of SHA-256 H_bolt -- report it immediately\n");
  }
  std::printf("[GAUNTLET] unit H_bolt sha256=%s chain_len=%zu (provisional, unlocked)\n",
              out.ref_head_sha.empty() ? "" : GgufWeightsHash::to_hex(out.ref_head_sha).c_str(),
              out.ref_chain_length);

  DeterminismGauntlet gp;
  gp.set_workers({1, 2});
  gp.set_runs(8);
  gp.set_seed(1);
  gp.set_perturb(/*config_index*/1, /*run_index*/3);
  GauntletResult bad;
  const bool pok = gp.run_unit(bad);
  const GauntletRow* failed = nullptr;
  for (const GauntletRow& row : bad.rows) if (!row.pass) failed = &row;
  check(pok == false && !bad.pass && failed && failed->workers == 2 && failed->first_diff_run == 3 &&
            failed->first_diff.index >= 0,
        "unit: set_perturb fails the named row and reports first differing step");
  if (failed && failed->first_diff.index >= 0) {
    std::printf("[GAUNTLET] perturb first diff run=%d step=%lld kind_a=%s kind_b=%s\n",
                failed->first_diff_run, (long long)failed->first_diff.index,
                kind_name(failed->first_diff.kind_a), kind_name(failed->first_diff.kind_b));
  }

  DeterminismGauntlet gc;
  check(gc.run_cuda_unsupported(bad) == false && !bad.pass && bad.err.find("CUDA") != std::string::npos,
        "unit: CUDA configuration fails cleanly (not faked)");
  check(gc.run_b3_unsupported(bad) == false && !bad.pass && bad.err.find("Trace fixture") != std::string::npos,
        "unit: Trace fixture fails cleanly (not built, not faked)");
}


static void test_adversarial() {
  // Length mismatch: first_diff reports the first missing index and which side has it.
  GauntletChainTrace longc, shortc;
  longc.append(ChainRecord{ChainRecordKind::Custom, {1}});
  longc.append(ChainRecord{ChainRecordKind::Commit, {2}});
  shortc.append(ChainRecord{ChainRecordKind::Custom, {1}});
  const GauntletDiff dl = DeterminismGauntlet::first_diff(longc, shortc);
  check(dl.index == 1 && dl.kind_a_present && !dl.kind_b_present && dl.kind_a == ChainRecordKind::Commit,
        "adversarial: first_diff of a shorter chain names the missing index and kind");

  // Same kind, different payload: index 0, both kinds present and equal.
  GauntletChainTrace p1, p2;
  p1.append(ChainRecord{ChainRecordKind::Expansion, {10}});
  p2.append(ChainRecord{ChainRecordKind::Expansion, {11}});
  const GauntletDiff dp = DeterminismGauntlet::first_diff(p1, p2);
  check(dp.index == 0 && dp.kind_a == ChainRecordKind::Expansion && dp.kind_b == ChainRecordKind::Expansion,
        "adversarial: first_diff of same-kind different payload is index 0");

  // restore_head truncates the recording prefix; a later append matches a fresh chain.
  GauntletChainTrace a, b;
  a.append(ChainRecord{ChainRecordKind::Custom, {1, 2, 3}});
  a.append(ChainRecord{ChainRecordKind::Expansion, {9}});
  a.append(ChainRecord{ChainRecordKind::Commit, {7}});
  const auto mid = a.heads()[0];
  a.restore_head(mid, 1);
  a.append(ChainRecord{ChainRecordKind::Selection, {4}});
  b.append(ChainRecord{ChainRecordKind::Custom, {1, 2, 3}});
  b.append(ChainRecord{ChainRecordKind::Selection, {4}});
  check(a.length() == 2 && a.head() == b.head() && a.fnv_head() == b.fnv_head() &&
            a.kinds().size() == 2 && a.kinds()[1] == ChainRecordKind::Selection,
        "adversarial: GauntletChainTrace restore_head then append matches a fresh chain");

  // Uneven partition: count=5, workers=2 covers 0..4 exactly once (2+3 split).
  {
    JoinJobDispatcher d(2);
    std::vector<int> hit(5, 0);
    d.parallel_for(5, [&](int k) {
      if (k >= 0 && k < 5) hit[size_t(k)] += 1;
    });
    bool okp = true;
    for (int h : hit) if (h != 1) okp = false;
    check(okp && d.size() == 2, "adversarial: workers=2, count=5 visits each index once");
  }

  // A worker exception is rethrown on the coordinator after join (no silent drop).
  {
    JoinJobDispatcher d(4);
    bool threw = false;
    try {
      d.parallel_for(8, [](int k) {
        if (k == 5) throw std::runtime_error("worker-fail");
      });
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()) == "worker-fail";
    }
    check(threw, "adversarial: worker exception is rethrown after join");
  }

  // Empty workers list fails cleanly (does not claim 100/100).
  {
    DeterminismGauntlet g;
    g.set_workers({});
    g.set_runs(2);
    GauntletResult out;
    check(g.run_unit(out) == false && !out.pass && out.err.find("workers") != std::string::npos,
          "adversarial: empty workers list fails cleanly");
  }

  // Seed=2, 3 unit runs, one worker: 3/3 same H_bolt (not the seed=1 baseline).
  {
    DeterminismGauntlet g;
    g.set_workers({1});
    g.set_runs(3);
    g.set_seed(2);
    GauntletResult out;
    const bool ok = g.run_unit(out);
    check(ok && out.pass && out.rows.size() == 1 && out.rows[0].unique_hashes == 1 &&
              out.rows[0].runs == 3 && !out.ref_head_sha.empty(),
          "adversarial: seed=2 unit 3/3 is deterministic (not a golden)");
  }
}

static void test_real_little_nemo() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight gauntlet: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[GAUNTLET] load err: %s\n", err.c_str());
    return;
  }
  m.init_cache(/*max_ctx*/32, /*branches*/8);
  const uint32_t live0 = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  DeterminismGauntlet g;
  g.set_workers({1, 2, 4, 8, 16});
  g.set_runs(2);  // 100-run real-weight deferred
  g.set_seed(1);
  GauntletResult out;
  const bool ok = g.run(m, out);
  check(ok && out.pass && out.rows.size() == 5, "real Little Nemo: 2-run gauntlet over 1,2,4,8,16 workers passes");
  bool rows_ok = out.rows.size() == 5;
  for (const GauntletRow& row : out.rows) {
    if (!(row.runs == 2 && row.unique_hashes == 1 && row.b1_failures == 0 && row.pass &&
          row.head_sha == out.ref_head_sha && row.first_diff.index < 0))
      rows_ok = false;
  }
  check(rows_ok, "real Little Nemo: each worker row is 2/2, one H_bolt, no first_diff");
  check(out.ref_head_sha.size() == 32 && out.ref_world_sha.size() == 32,
        "real Little Nemo: H_bolt and H_world SHA-256 are 32 bytes");
  const uint64_t hfnv = fnv_bytes(out.ref_head_sha.data(), out.ref_head_sha.size());
  check(!collide8(out.ref_head_sha, hfnv), "real Little Nemo: H_bolt FNV != first 8 bytes of SHA-256");
  if (collide8(out.ref_head_sha, hfnv))
    std::printf("[GAUNTLET] ALERT: FNV vs SHA disagreement FAILED on real H_bolt -- report it immediately\n");

  if (!out.rows.empty() && out.ref_head_sha.size() >= 8) {
    uint64_t first8 = 0;
    std::memcpy(&first8, out.ref_head_sha.data(), 8);
    const bool agree = out.rows[0].head_fnv == first8;
    check(!agree, "real Little Nemo: FNV-1a64 chain head != first 8 bytes of SHA-256 H_bolt");
    if (agree)
      std::printf("[GAUNTLET] ALERT: FNV chain head agrees with first 8 of SHA-256 H_bolt -- report it immediately\n");
  }
  std::printf("[GAUNTLET] real H_bolt sha256=%s fnv=0x%016llX world=%s (provisional, unlocked)\n",
              out.ref_head_sha.empty() ? "" : GgufWeightsHash::to_hex(out.ref_head_sha).c_str(),
              (unsigned long long)hfnv,
              out.ref_world_sha.empty() ? "" : GgufWeightsHash::to_hex(out.ref_world_sha).c_str());

  const uint32_t live_end = m.pool_swa().live_pages() + m.pool_glob().live_pages();
  check(live_end == live0, "real Little Nemo: no KV page leak after the gauntlet");
  std::printf("[GAUNTLET] live_pages start=%u end=%u\n", live0, live_end);
}

int main() {
  std::printf("[GAUNTLET] CPU determinism gauntlet tests (PchStack/maze; CUDA/Trace fixture/UI deferred)\n");
  test_det_math_pins();
  test_join_dispatcher();
  test_chain_trace();
  test_unit_gauntlet();
  test_adversarial();
  test_real_little_nemo();
  std::printf("[GAUNTLET] gauntlet %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
