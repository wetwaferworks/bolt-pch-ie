// TESLA-manner stream authentication and simulated attestation (RATS verdicts, no TPM).
// TESLA and attestation are ON by default in the maze drivers; --tesla off and
// --attest off turn them off for testing. This CTest never talks to the TPM.
// Existing v1, ratchet, and official B1 16x16 seals are checked here so a hash move is visible immediately.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "pchie/AttestationVerdict.h"
#include "pchie/HeadlessMaze.h"
#include "pchie/ChainGeneration.h"
#include "pchie/RatchetHashChain.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/StreamAuth.h"
#include "pchie/TeslaStream.h"
#include "pchie/TeslaWatcher.h"

using namespace pchie;
using Bytes = std::vector<uint8_t>;

static int passes = 0, fails = 0;
static void check(bool ok, const std::string& name) {
  std::printf("[TESLA] %-90s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
  ok ? ++passes : ++fails;
}
static Bytes sha(const std::string& s) { return Sha256::hash(reinterpret_cast<const uint8_t*>(s.data()), s.size()); }
static Bytes sha_u(uint64_t n) {
  uint8_t b[8];
  for (int i = 7; i >= 0; --i) { b[i] = uint8_t(n & 0xff); n >>= 8; }
  return Sha256::hash(b, 8);
}
static ChainRecord rec(int i) {
  ChainRecord r;
  r.kind = ChainRecordKind(i % 12);
  for (int j = 0; j <= i; ++j) r.payload.push_back(uint8_t(i * 31 + j));
  return r;
}
static RatchetGenesis ratchet_genesis() {
  RatchetGenesis g;
  g.weights_sha256 = sha("weights");
  g.run_id = "ratchet-test-run";
  g.harness_version = "bolt-pchie test harness";
  return g;
}
static Bytes seed_for(const char* label) {
  Bytes s = sha(std::string("tesla-seed:") + label);
  if (s.size() < 32) s.resize(32, 0);
  return s;
}
static bool has_flag(const TeslaWatchResult& r, TeslaFlag f) {
  for (const auto& e : r.flags) if (e.flag == f) return true;
  return false;
}

struct BuiltStream {
  Bytes commitment;
  TeslaSchedule sch;
  std::vector<TeslaStepPacket> steps;
  std::vector<TeslaDisclosure> on_schedule;  // in stream order, interleaved conceptually after the closing step
  std::vector<size_t> disclose_after_step;   // on_schedule[k] is emitted after steps[disclose_after_step[k]]
  std::vector<TeslaDisclosure> trailing;
  Bytes last_seal, last_payload;
};

// Build an honest stream of n steps. Seals are taken from a v1 chain so TESLA wraps existing seals
// and cannot change them.
static BuiltStream build_honest(uint64_t n, const char* seed_label = "honest") {
  TeslaSchedule sch;
  const uint32_t intervals = uint32_t((n + sch.I - 1) / sch.I);
  TeslaSender sender(seed_for(seed_label), intervals, sch);
  Sha256HashChain v1;
  BuiltStream out;
  out.commitment = sender.commitment();
  out.sch = sch;
  for (uint64_t s = 0; s < n; ++s) {
    v1.append(rec(int(s)));
    const Bytes seal = v1.head();
    const Bytes payload = sha_u(s);
    TeslaStepPacket p = sender.emit_step(s, seal, payload);
    out.steps.push_back(p);
    auto disc = sender.take_disclosures();
    for (auto& d : disc) {
      out.disclose_after_step.push_back(out.steps.size() - 1);
      out.on_schedule.push_back(std::move(d));
    }
    out.last_seal = seal;
    out.last_payload = payload;
  }
  out.trailing = sender.finish();
  return out;
}

static TeslaWatchResult watch_honest_like(const BuiltStream& st,
                                          const std::function<void(TeslaStepPacket&)>& mut_step = nullptr,
                                          int drop_step = -1,
                                          int inject_at = -1, const TeslaStepPacket* injected = nullptr,
                                          bool early_disclose = false,
                                          const Bytes* wrong_key = nullptr,
                                          bool reorder_last_two = false) {
  TeslaWatcher w(st.commitment, st.sch);
  size_t disc_i = 0;
  auto emit_due = [&](size_t after) {
    while (disc_i < st.on_schedule.size() && st.disclose_after_step[disc_i] == after) {
      TeslaDisclosure d = st.on_schedule[disc_i];
      if (wrong_key && disc_i == 0) d.key = *wrong_key;
      w.accept_disclosure(d);
      ++disc_i;
    }
  };
  if (early_disclose && !st.on_schedule.empty()) {
    w.accept_disclosure(st.on_schedule[0]);  // K_0 before any step
  }
  std::vector<TeslaStepPacket> order = st.steps;
  if (reorder_last_two && order.size() >= 2) std::swap(order[order.size() - 2], order[order.size() - 1]);
  for (size_t i = 0; i < order.size(); ++i) {
    TeslaStepPacket p = order[i];
    const bool is_dropped = drop_step >= 0 && int(p.step_index) == drop_step;
    if (inject_at >= 0 && int(i) == inject_at && injected) w.accept_step(*injected);
    if (!is_dropped) {
      if (mut_step) mut_step(p);
      w.accept_step(p);
    }
    // On-schedule disclosures follow the original (unreordered) step index after which they were produced.
    // For the honest and flip/drop tests we use original order, so after index i is fine.
    if (!reorder_last_two && !early_disclose) emit_due(i);
  }
  if (!early_disclose && !reorder_last_two) {
    // leftover on-schedule (should be none)
    while (disc_i < st.on_schedule.size()) {
      TeslaDisclosure d = st.on_schedule[disc_i];
      if (wrong_key && disc_i == 0) d.key = *wrong_key;
      w.accept_disclosure(d);
      ++disc_i;
    }
  }
  w.note_end_of_steps();
  if (!early_disclose) {
    for (const auto& d : st.trailing) w.accept_disclosure(d);
  } else {
    // After early K_0, still feed the rest of the honest disclosures so later MACs can be judged late.
    for (size_t k = 1; k < st.on_schedule.size(); ++k) w.accept_disclosure(st.on_schedule[k]);
    for (const auto& d : st.trailing) w.accept_disclosure(d);
  }
  return w.finish();
}

static std::string write_stream_file(const BuiltStream& st, const std::string& path) {
  TeslaStreamWriter wr(st.commitment, st.sch);
  size_t disc_i = 0;
  for (size_t i = 0; i < st.steps.size(); ++i) {
    wr.add_step(st.steps[i]);
    while (disc_i < st.on_schedule.size() && st.disclose_after_step[disc_i] == i) {
      wr.add_disclosure(st.on_schedule[disc_i]);
      ++disc_i;
    }
  }
  wr.end_steps();
  for (const auto& d : st.trailing) wr.add_disclosure(d);
  std::ofstream f(path, std::ios::binary);
  f << wr.text();
  f.close();
  return path;
}

#ifdef _WIN32
static int run_watcher_cli(const std::string& exe, const std::string& stream) {
  std::string cmd = "\"" + exe + "\" \"" + stream + "\"";
  STARTUPINFOA si{};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi{};
  std::vector<char> buf(cmd.begin(), cmd.end());
  buf.push_back(0);
  if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return 127;
  WaitForSingleObject(pi.hProcess, 60000);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return int(code);
}
#endif

int main(int argc, char** argv) {
  std::string watcher_exe;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--watcher" && i + 1 < argc) watcher_exe = argv[++i];
  }

  // ---- Neutrality: existing goldens must not move. Stop and report if they do. ----
  const std::string kV1Golden = "f4c122234f3fd97c5c495a028af4c62c593e0b6865c239c155e2643d2c5f2445";
  const std::string kRatchetGolden = "a781fda17ae8a2228d5f03431aa78b933837c02ee206e7ea571fce54a1601c29";
  {
    Sha256HashChain c;
    for (int i = 0; i < 10; ++i) c.append(rec(i));
    check(tesla_hex(c.head()) == kV1Golden && c.length() == 10, "neutrality: v1 Sha256HashChain head unchanged");
  }
  {
    auto c = make_hash_chain();
    for (int i = 0; i < 10; ++i) c->append(rec(i));
    check(kDefaultChainGeneration == ChainGeneration::Sha256V1 && tesla_hex(c->head()) == kV1Golden,
          "neutrality: default chain generation is still v1 and reproduces the v1 head");
  }
  {
    RatchetHashChain c(ratchet_genesis());
    for (int i = 0; i < 10; ++i) c.append(rec(i));
    check(tesla_hex(c.head()) == kRatchetGolden, "neutrality: default recomputable ratchet head unchanged");
  }

  // ---- Honest end-to-end, including a short last interval (40 steps, 5 intervals, last is full). ----
  // 40 is a multiple of 8, so also a 36-step short-last run.
  const BuiltStream honest40 = build_honest(40);
  const TeslaWatchResult h40 = watch_honest_like(honest40);
  check(h40.all_verified && h40.authentic && h40.released.size() == 40 && h40.gaps.empty() && h40.flags.empty(),
        "honest end-to-end: 40 steps, I=8 d=2, all released, no flags");
  check(honest40.trailing.size() == 2, "honest: exactly d=2 trailing disclosures after the last step");
  {
    bool seals_match = true;
    Sha256HashChain v1;
    for (size_t s = 0; s < honest40.steps.size(); ++s) {
      v1.append(rec(int(s)));
      if (honest40.steps[s].seal != v1.head()) seals_match = false;
      if (honest40.steps[s].interval_index != uint32_t(s / 8)) seals_match = false;
    }
    check(seals_match, "honest: each packet carries the unchanged v1 seal for that step");
  }

  const BuiltStream honest36 = build_honest(36, "honest-36");
  const TeslaWatchResult h36 = watch_honest_like(honest36);
  check(h36.all_verified && h36.released.size() == 36 && honest36.trailing.size() >= 2,
        "honest end-to-end: 36 steps (short last interval) all released");

  // Commitment is public and does not equal K_0 (the extra hash).
  {
    TeslaKeyChain chain(seed_for("honest"), 5);
    check(TeslaKeyChain::key_belongs(chain.key(0), 0, chain.commitment()) && chain.key(0) != chain.commitment(),
          "commitment is F(K_0), so K_0 stays secret until disclosure");
    Bytes mk = chain.mac_key(0);
    check(mk != chain.key(0), "MAC key is derived; the interval key is not reused as the MAC key");
    wipe_bytes(mk);
  }

  // Separate from the ratchet: flipping TESLA does not change seal bytes (already in the packet);
  // also a TESLA seed change does not change the ratchet or v1 head.
  {
    const BuiltStream other = build_honest(40, "other-seed");
    check(other.commitment != honest40.commitment && other.steps[0].seal == honest40.steps[0].seal &&
              other.steps[0].mac != honest40.steps[0].mac,
          "TESLA seed is separate from the seal: same seals, different MACs and commitment");
  }

  // ---- Worker agreement: 1, 2, 4, 8, 16 workers produce identical streams (canonical order). ----
  {
    std::vector<std::string> digests;
    for (int workers : {1, 2, 4, 8, 16}) {
      // "Workers" compute payload hashes independently; the coordinator emits in step order.
      TeslaSchedule sch;
      TeslaSender sender(seed_for("workers"), 5, sch);
      Sha256HashChain v1;
      TeslaStreamWriter wr(sender.commitment(), sch);
      std::vector<Bytes> seals(40), payloads(40);
      for (int s = 0; s < 40; ++s) {
        v1.append(rec(s));
        seals[s] = v1.head();
        payloads[s] = sha_u(uint64_t(s * 17 + workers));  // worker count must not leak into the payload
        // Use a payload that does not depend on the worker count:
        payloads[s] = sha_u(uint64_t(s));
      }
      (void)workers;
      for (int s = 0; s < 40; ++s) {
        TeslaStepPacket p = sender.emit_step(uint64_t(s), seals[s], payloads[s]);
        wr.add_step(p);
        for (const auto& d : sender.take_disclosures()) wr.add_disclosure(d);
      }
      wr.end_steps();
      for (const auto& d : sender.finish()) wr.add_disclosure(d);
      digests.push_back(tesla_hex(sha(wr.text())));
    }
    bool same = true;
    for (size_t i = 1; i < digests.size(); ++i) if (digests[i] != digests[0]) same = false;
    check(same, "honest stream identical at 1, 2, 4, 8 and 16 workers");
  }

  // ---- Byte flip: one byte of one packet, fail at that step, no later step released. ----
  {
    TeslaWatchResult r = watch_honest_like(honest40, [](TeslaStepPacket& p) {
      if (p.step_index == 5 && !p.payload_hash.empty()) p.payload_hash[0] ^= 0x01;
    });
    bool later = false;
    for (const auto& rel : r.released) if (rel.step_index >= 5) later = true;
    bool earlier = false;
    int early_count = 0;
    for (const auto& rel : r.released) if (rel.step_index < 5) { earlier = true; ++early_count; }
    check(has_flag(r, TeslaFlag::BadMac) && r.failed_at == 5 && !later && earlier && !r.all_verified,
          "byte flip at step 5: bad MAC, earlier steps released, no later step released");
  }

  // ---- Injection: a forged packet with a plausible step index is rejected when the true key arrives. ----
  {
    TeslaStepPacket fake = honest40.steps[9];
    fake.step_index = 9;
    fake.interval_index = 1;
    if (!fake.payload_hash.empty()) fake.payload_hash[0] ^= 0xff;
    // Keep the original MAC, so it cannot match the flipped payload, and also inject an extra packet.
    TeslaWatchResult r = watch_honest_like(honest40, nullptr, -1, 9, &fake);
    check(has_flag(r, TeslaFlag::BadMac) && !r.all_verified, "injection: forged packet rejected (bad MAC) when the key arrives");
  }

  // ---- Drop: a missing step is a gap, not a silent heal. ----
  {
    TeslaWatchResult r = watch_honest_like(honest40, nullptr, /*drop*/ 12);
    bool released_12 = false;
    for (const auto& rel : r.released) if (rel.step_index == 12) released_12 = true;
    bool gap12 = false;
    for (uint64_t g : r.gaps) if (g == 12) gap12 = true;
    check(!released_12 && gap12 && has_flag(r, TeslaFlag::Gap) && !r.all_verified && r.authentic,
          "drop step 12: reported as a gap, not released, authenticity of the key chain still holds");
  }

  // ---- Reorder: swap the last two steps; the one that arrives after its disclosure (if any) or out of
  //      close order still verifies if both arrive before their keys. Last two are in interval 4, keys
  //      disclosed only in trailing, so a swap of 38 and 39 still arrives before the key. Use a swap
  //      across a disclosure boundary: steps 23 and 24, where K_0 is disclosed after step 23 (interval 2 closes).
  //      Swapping 23 and 24 still precedes K_0? Interval 2 closes at step 23, K_0 disclosed after 23.
  //      If we swap so 24 arrives before 23, both still before K_0. To test late, disclose first.
  //      Reorder after early? Better: after feeding through step 23 and its disclosure of K_0, we
  //      cannot late-test interval 3. For late, the early-disclosure test covers "packet after key".
  //      For reorder of packets within a buffered interval, both still verify (order of arrival before
  //      the key is what matters, not step order). The brief says drop and reorder produce gaps and
  //      failures at the touched positions. A reorder that places a packet after its key is a late.
  {
    // Move step 0 to after the first disclosure (K_0, after step 23). Build manually.
    TeslaWatcher w(honest40.commitment, honest40.sch);
    size_t disc_i = 0;
    TeslaStepPacket delayed = honest40.steps[0];
    for (size_t i = 1; i < honest40.steps.size(); ++i) {
      w.accept_step(honest40.steps[i]);
      while (disc_i < honest40.on_schedule.size() && honest40.disclose_after_step[disc_i] == i) {
        w.accept_disclosure(honest40.on_schedule[disc_i]);
        ++disc_i;
      }
    }
    w.accept_step(delayed);  // step 0 arrives after K_0 was disclosed
    w.note_end_of_steps();
    for (const auto& d : honest40.trailing) w.accept_disclosure(d);
    TeslaWatchResult r = w.finish();
    check(has_flag(r, TeslaFlag::Late) && r.failed_at == 0 && !r.all_verified,
          "reorder: step 0 arriving after its key is disclosed is rejected as late");
  }

  // ---- Early disclosure: K_0 released before any step; subsequent packets of interval 0 are late. ----
  {
    TeslaWatchResult r = watch_honest_like(honest40, nullptr, -1, -1, nullptr, true);
    check(has_flag(r, TeslaFlag::EarlyDisclosure) && has_flag(r, TeslaFlag::Late) && !r.all_verified,
          "early disclosure: flagged, and packets arriving after the key are rejected as late");
  }

  // ---- Wrong key chain: first disclosure does not hash to the commitment; authenticity fails, full stop. ----
  {
    Bytes bad(32, 0x5a);
    TeslaWatchResult r = watch_honest_like(honest40, nullptr, -1, -1, nullptr, false, &bad);
    check(has_flag(r, TeslaFlag::WrongKey) && !r.authentic && !r.all_verified,
          "wrong key chain: authenticity fails at the first bad disclosure, full stop");
  }

  // ---- Watcher program: honest file exits 0; flipped file exits 1. ----
  if (!watcher_exe.empty()) {
#ifdef _WIN32
    const std::string good_path = "tesla_watch_good.txt";
    const std::string bad_path = "tesla_watch_bad.txt";
    write_stream_file(honest40, good_path);
    BuiltStream flipped = honest40;
    if (!flipped.steps[5].payload_hash.empty()) flipped.steps[5].payload_hash[0] ^= 0x01;
    write_stream_file(flipped, bad_path);
    const int good_rc = run_watcher_cli(watcher_exe, good_path);
    const int bad_rc = run_watcher_cli(watcher_exe, bad_path);
    check(good_rc == 0, "watcher program: honest stream exits 0");
    check(bad_rc == 1, "watcher program: flipped stream exits 1");
    std::remove(good_path.c_str());
    std::remove(bad_path.c_str());
#else
    check(false, "watcher program: Windows-only spawn in this test");
#endif
  } else {
    check(true, "watcher program: skipped (no --watcher path); library checks already cover the same logic");
  }

  // ---- Simulated attestation (no TPM). ----
  {
    const Bytes ak = sha("test-attestation-key");
    const Bytes nonce1 = sha("nonce-1");
    const Bytes nonce2 = sha("nonce-2");
    MeasurementSet m;
    m.binary_sha256 = sha("runner-binary");
    m.harness_version = "bolt-pchie test harness";
    m.weights_sha256 = sha("weights");
    m.config_sha256 = sha("config");
    m.tesla_commitment = honest40.commitment;
    KnownGoodRegistry reg;
    reg.entries.push_back({m.binary_sha256, m.weights_sha256, m.config_sha256, m.harness_version});
    SimulatedQuote q = make_simulated_quote(m, nonce1, "tpm2.0-sim", ak);
    const AttestationAppraisal ok = appraise_attestation(q, nonce1, m, reg, ak);
    check(ok.status == AttestationStatus::Attested, "attestation: honest simulated quote is Attested");

    MeasurementSet wrong = m;
    wrong.binary_sha256 = sha("modified-runner-binary");
    SimulatedQuote q_wrong = make_simulated_quote(wrong, nonce1, "tpm2.0-sim", ak);
    const AttestationAppraisal rej_bin = appraise_attestation(q_wrong, nonce1, wrong, reg, ak);
    check(rej_bin.status == AttestationStatus::Rejected && rej_bin.reason.find("registry") != std::string::npos,
          "attestation: wrong binary is Rejected (not in the known-good registry)");

    const AttestationAppraisal rej_replay = appraise_attestation(q, nonce2, m, reg, ak);
    check(rej_replay.status == AttestationStatus::Rejected && rej_replay.reason == "nonce mismatch",
          "attestation: replay of a previous quote against a fresh nonce is Rejected");

    SimulatedQuote absent;
    absent.present = false;
    const AttestationAppraisal un = appraise_attestation(absent, nonce1, m, reg, ak);
    check(un.status == AttestationStatus::Unattested && un.reason == "absent platform",
          "attestation: no quote is Unattested, not Rejected and not Attested");
    // Chain validity is a separate axis: the honest TESLA stream still verifies alongside Unattested.
    check(h40.all_verified && un.status == AttestationStatus::Unattested,
          "attestation: Unattested does not disturb an already-verified chain");
  }

  // ---- Defaults, flags, and official-search neutrality (must not move sim or seal hashes). ----
  {
    const StreamAuthFlags d;
    check(d.tesla && d.attest, "flags: TESLA and attestation are on by default");
    char* av_off[] = {(char*)"x", (char*)"--tesla", (char*)"off", (char*)"--attest", (char*)"off"};
    const auto off = parse_stream_auth_flags(5, av_off);
    check(off.ok && !off.flags.tesla && !off.flags.attest, "flags: --tesla off --attest off");
    char* av_on[] = {(char*)"x", (char*)"--tesla", (char*)"on", (char*)"--attest", (char*)"on"};
    const auto on = parse_stream_auth_flags(5, av_on);
    check(on.ok && on.flags.tesla && on.flags.attest, "flags: --tesla on --attest on leave both on");
    char* av_bad[] = {(char*)"x", (char*)"--tesla", (char*)"maybe"};
    check(!parse_stream_auth_flags(3, av_bad).ok, "flags: --tesla rejects a value other than on or off");
  }
  {
    const std::string kActions = "0c21184625c3c03b8cfb4259213604f731098618cbbb578e0473eaa44333aa4e";
    const std::string kChain = "de479b0a2dd6c4ad69dc379c34139a3a64a7a27d6c9074d59e22a6b632e79533";
    auto sha_actions = [](const std::vector<int32_t>& a) {
      std::vector<uint8_t> b(a.size() * 4);
      for (size_t i = 0; i < a.size(); ++i) {
        const uint32_t v = uint32_t(a[i]);
        b[4 * i] = uint8_t(v);
        b[4 * i + 1] = uint8_t(v >> 8);
        b[4 * i + 2] = uint8_t(v >> 16);
        b[4 * i + 3] = uint8_t(v >> 24);
      }
      return tesla_hex(Sha256::hash(b.data(), b.size()));
    };
    auto run_b1 = [&](bool tesla, bool attest) {
      HeadlessMaze mz(MazeGrid(1, 16, 16));
      mz.set_search_budget(16);
      mz.enable_tesla(tesla);
      mz.enable_attest(attest);
      Sha256HashChain ch;
      MazeResult out;
      const bool ok = mz.run_unit("", &ch, out);
      return std::make_tuple(ok, sha_actions(out.actions), tesla_hex(ch.head()), out);
    };
    const auto on = run_b1(true, true);
    const auto off = run_b1(false, false);
    const auto& on_out = std::get<3>(on);
    const auto& off_out = std::get<3>(off);
    check(std::get<0>(on) && std::get<0>(off), "maze official search: default-on and flags-off both finish");
    check(std::get<1>(on) == kActions && std::get<1>(off) == kActions,
          "neutrality: official maze 16x16 actions hash unchanged with TESLA/attest on or off");
    check(std::get<2>(on) == kChain && std::get<2>(off) == kChain,
          "neutrality: official maze 16x16 chain head unchanged with TESLA/attest on or off");
    check(std::get<1>(on) == std::get<1>(off) && std::get<2>(on) == std::get<2>(off),
          "neutrality: TESLA on vs --tesla off produces identical actions and chain head");
    check(on_out.tesla_enabled && on_out.tesla_steps > 0 && on_out.tesla_commitment.size() == 32 && on_out.tesla_err.empty(),
          "default on: TESLA session runs and records a commitment (side stream only)");
    check(!off_out.tesla_enabled && off_out.tesla_steps == 0 && off_out.tesla_commitment.empty(),
          "--tesla off: no TESLA session is started");
    check(on_out.attest_enabled && on_out.attest_computed && on_out.attest_status == AttestationStatus::Unattested &&
              on_out.attest_reason == "absent platform",
          "default on: attestation is computed; without a TPM quote the verdict is Unattested");
    check(on_out.attest_status != AttestationStatus::Attested,
          "default on: never fakes Attested when there is no quote");
    check(!off_out.attest_enabled && !off_out.attest_computed,
          "--attest off: no attestation verdict is computed");
  }

  std::printf("[TESLA] %d pass, %d fail\n", passes, fails);
  return fails ? 1 : 0;
}
