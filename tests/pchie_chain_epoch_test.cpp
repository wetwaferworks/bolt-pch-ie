// Epoch re-root at the harness's real fork and resume points (brief s3.6;).
// Resume point: the B1 freeze-recovery retry (restore the chain head, then retry the iteration).
// Fork point:   MctsDualPoolStepBack::rollback (branch again from a snapshotted search node).
// A ratcheted chain opens a new epoch at each; v1 chains behave exactly as before and keep their pinned heads.
#include <cstdio>
#include <string>
#include <vector>
#include "pchie/HeadlessMaze.h"
#include "pchie/ChainEpoch.h"
#include "pchie/MctsDualPoolStepBack.h"
#include "pchie/RatchetHashChain.h"
#include "pchie/Sha256HashChain.h"

using namespace pchie;
using Bytes = std::vector<uint8_t>;
static int passes = 0, fails = 0;
static void check(bool ok, const std::string& name) { std::printf("[EPOCH] %-92s %s\n", name.c_str(), ok ? "PASS" : "FAIL"); ok ? ++passes : ++fails; }
static std::string hx(const Bytes& b) { static const char* d = "0123456789abcdef"; std::string s; for (uint8_t c : b) { s += d[c >> 4]; s += d[c & 15]; } return s; }
static ChainRecord rec(int i) { ChainRecord r; r.kind = ChainRecordKind::Custom; r.payload = {uint8_t(i), uint8_t(i * 3), uint8_t(i * 7)}; return r; }
static RatchetGenesis genesis() { RatchetGenesis g; const std::string w = "epoch test weights"; g.weights_sha256 = Sha256::hash(reinterpret_cast<const uint8_t*>(w.data()), w.size()); g.run_id = "epoch-test-run"; g.harness_version = "bolt-pchie epoch test"; return g; }
static bool freeze_once(void* ctx, uint32_t task, int step, int iter) {
  int* n = static_cast<int*>(ctx);
  if (task == HeadlessMaze::kGoToKeyA && step == 0 && iter == 0 && *n == 0) { ++*n; return true; }
  return false;
}
struct MazeRun { bool ok = false; std::string err; Bytes head; size_t length = 0; int freezes = 0; };
static MazeRun run_b1(IHashChain& chain, bool freeze) {
  MazeRun r; HeadlessMaze m(1, 16, 16); if (freeze) m.set_freeze_check(freeze_once, &r.freezes);
  MazeResult out; try { r.ok = m.run_unit("", &chain, out) && out.ok; r.err = out.err; } catch (const std::exception& e) { r.ok = false; r.err = e.what(); }
  r.head = chain.head(); r.length = chain.length(); return r;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  // ---- Resume point: B1 freeze-recovery retry ----
  Sha256HashChain v1c, v1f; v1c.reset(); v1f.reset();
  const MazeRun c = run_b1(v1c, false), f = run_b1(v1f, true);
  std::printf("  v1 clean  head=%s len=%zu\n  v1 freeze head=%s len=%zu freezes=%d\n", hx(c.head).c_str(), c.length, hx(f.head).c_str(), f.length, f.freezes);
  const std::string kV1B1Head = "de479b0a2dd6c4ad69dc379c34139a3a64a7a27d6c9074d59e22a6b632e79533";
  check(c.ok && f.ok && f.freezes == 1 && f.head == c.head && f.length == c.length, "v1 chain: freeze retry restores the head and ends identical to the clean run (unchanged behavior)");
  check(hx(c.head) == kV1B1Head, "v1 chain: maze head pinned to the value from the unwired harness");
  { RatchetHashChain rc(genesis()), rf(genesis()), rf2(genesis());
    const MazeRun a = run_b1(rc, false), b = run_b1(rf, true), b2 = run_b1(rf2, true);
    std::printf("  ratchet clean  head=%s len=%zu epoch=%u\n  ratchet freeze head=%s len=%zu epoch=%u err=%s\n", hx(a.head).c_str(), a.length, rc.epoch(), hx(b.head).c_str(), b.length, rf.epoch(), b.err.c_str());
    check(a.ok && rc.epoch() == 0, "ratcheted chain: clean maze run needs no epoch");
    check(b.ok && b.freezes == 1 && rf.epoch() == 1 && !rf.detached() && b.length == a.length + 1 && b.head != a.head,
          "ratcheted chain: freeze retry opens epoch 1 at the resume point, so the resumed line is not the clean line");
    check(b2.ok && b2.head == b.head && rf2.epoch_commitment() == rf.epoch_commitment(), "ratcheted chain: resume re-root is deterministic"); }

  // ---- Fork point: DualPool step-back rollback to a snapshotted node ----
  { KVPagePool swa(1, 1, 4, 8), glob(1, 1, 4, 8); MctsDualPoolStepBack sb(swa, glob); ModelBranch b, out;
    Sha256HashChain s; s.reset(); for (int i = 0; i < 3; ++i) s.append(rec(i)); sb.snapshot(0, b, s); const Bytes h0 = s.head(); for (int i = 3; i < 5; ++i) s.append(rec(i));
    check(sb.rollback(0, out, s) && s.head() == h0 && s.length() == 3, "v1 chain: rollback restores the snapshot head and length exactly (unchanged behavior)"); }
  auto fork_run = [](std::vector<Bytes>& heads, RatchetHashChain& r) {
    KVPagePool swa(1, 1, 4, 8), glob(1, 1, 4, 8); MctsDualPoolStepBack sb(swa, glob); ModelBranch b, out;
    for (int i = 0; i < 3; ++i) r.append(rec(i)); sb.snapshot(7, b, r);
    for (int i = 3; i < 5; ++i) r.append(rec(i)); heads.push_back(r.head());           // branch one
    bool ok = sb.rollback(7, out, r); r.append(rec(5)); heads.push_back(r.head());       // fork from node 7
    ok = ok && sb.rollback(7, out, r); r.append(rec(5)); heads.push_back(r.head());      // fork again from node 7
    return ok; };
  { RatchetHashChain r1(genesis()), r2(genesis()); std::vector<Bytes> h1, h2; bool ok1 = false, ok2 = false;
    try { ok1 = fork_run(h1, r1); ok2 = fork_run(h2, r2); } catch (const std::exception& e) { std::printf("  fork run threw: %s\n", e.what()); }
    if (h1.size() < 3) h1.resize(3); if (h2.size() < 3) h2.resize(3);
    check(ok1 && r1.epoch() == 2 && !r1.detached() && r1.length() == 3 + 1 + 1, "ratcheted chain: each rollback to a snapshot opens a new epoch at the fork point");
    check(h1[1] != h1[2], "ratcheted chain: two forks from the same node with the same next step get different heads");
    check(ok2 && h1 == h2, "ratcheted chain: fork re-roots are deterministic"); }
  // ---- The helper leaves every non-ratcheted chain alone ----
  { Sha256HashChain s; s.reset(); s.append(rec(1)); const Bytes h = s.head();
    check(!reroot_if_ratcheted(&s, "fork", "test", {1}) && s.head() == h && s.length() == 1, "reroot_if_ratcheted does nothing to a v1 chain"); }
  std::printf("CHAIN-EPOCH: %d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
