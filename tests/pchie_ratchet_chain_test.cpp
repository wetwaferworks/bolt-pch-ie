// Ratcheted seal chain (generation pchie.seal.v2-ratchet): the v1 chain is untouched, the ratchet is off by
// default, derivation is one-way, resume/fork must re-root, and the head depends on canonical order only.
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "pchie/ChainGeneration.h"
#include "pchie/RatchetHashChain.h"
#include "pchie/Sha256HashChain.h"

using namespace pchie;
using Bytes = std::vector<uint8_t>;
static int passes = 0, fails = 0;
static void check(bool ok, const std::string& name) {
  std::printf("[RATCHET] %-86s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
  ok ? ++passes : ++fails;
}
static std::string hx(const Bytes& b) { static const char* d = "0123456789abcdef"; std::string s; for (uint8_t c : b) { s += d[c >> 4]; s += d[c & 15]; } return s; }
static Bytes sha(const std::string& s) { return Sha256::hash(reinterpret_cast<const uint8_t*>(s.data()), s.size()); }
static ChainRecord rec(int i) { ChainRecord r; r.kind = ChainRecordKind(i % 12); for (int j = 0; j <= i; ++j) r.payload.push_back(uint8_t(i * 31 + j)); return r; }
static RatchetGenesis genesis(const std::string& run = "ratchet-test-run") { RatchetGenesis g; g.weights_sha256 = sha("weights"); g.run_id = run; g.harness_version = "bolt-pchie test harness"; return g; }

int main() {
  // 1. v1 is unchanged: the same ten records give the head recorded from the untouched tree (907d171).
  const std::string kV1Golden = "f4c122234f3fd97c5c495a028af4c62c593e0b6865c239c155e2643d2c5f2445";
  { Sha256HashChain c; for (int i = 0; i < 10; ++i) c.append(rec(i)); check(hx(c.head()) == kV1Golden && c.length() == 10, "v1 Sha256HashChain head unchanged from the untouched tree"); }
  // 2. The switch defaults to v1 and gives the identical v1 head.
  { auto c = make_hash_chain(); for (int i = 0; i < 10; ++i) c->append(rec(i));
    check(kDefaultChainGeneration == ChainGeneration::Sha256V1 && hx(c->head()) == kV1Golden, "default chain generation is v1 (ratchet off) and reproduces the v1 head"); }
  { bool threw = false; try { make_hash_chain(ChainGeneration::RatchetV2); } catch (const std::invalid_argument&) { threw = true; } check(threw, "ratchet generation refuses to start without a genesis"); }
  // 3. Deterministic, and bound to every genesis input.
  auto run = [](const RatchetGenesis& g, int n) { RatchetHashChain c(g); for (int i = 0; i < n; ++i) c.append(rec(i)); return c.head(); };
  const Bytes h = run(genesis(), 10);
  check(h == run(genesis(), 10), "same genesis and records give the same ratcheted head");
  { RatchetGenesis g = genesis(); g.weights_sha256 = sha("other weights"); RatchetGenesis r = genesis("other-run"); RatchetGenesis v = genesis(); v.harness_version = "other harness";
    RatchetGenesis gen = genesis(); gen.generation = "pchie.seal.v2-other";
    check(h != run(g, 10) && h != run(r, 10) && h != run(v, 10) && h != run(gen, 10), "head changes with weights, run id, harness version, and generation id"); }
  check(hx(h) != kV1Golden, "ratcheted head differs from the v1 head over the same records (generations apart)");
  // 3b. Default (recomputable) mode is pinned to the head recorded before the secret-salt mode existed.
  check(hx(h) == "a781fda17ae8a2228d5f03431aa78b933837c02ee206e7ea571fce54a1601c29", "default recomputable ratchet head unchanged (pinned before the salt mode was added)");
  // 3c. Optional secret genesis salt: deterministic, apart from the default, bound to the salt, and strict about its inputs.
  { Bytes s1(32, 0x5a), s2(32, 0x5b);
    const Bytes a = run(RatchetGenesis::with_secret_salt(genesis(), s1), 10), b = run(RatchetGenesis::with_secret_salt(genesis(), s1), 10), c = run(RatchetGenesis::with_secret_salt(genesis(), s2), 10);
    check(a == b && a != h && a != c, "secret-salt mode: deterministic, differs from the default chain, and differs per salt");
    bool t1 = false, t2 = false, t3 = false;
    try { RatchetGenesis g = genesis(); g.secret_salt = s1; run(g, 1); } catch (const std::invalid_argument&) { t1 = true; }
    try { RatchetGenesis g = genesis(); g.generation = kRatchetSaltedGenerationId; run(g, 1); } catch (const std::invalid_argument&) { t2 = true; }
    try { run(RatchetGenesis::with_secret_salt(genesis(), Bytes(8, 1)), 1); } catch (const std::invalid_argument&) { t3 = true; }
    check(t1 && t2 && t3, "secret-salt mode refuses a salt without its generation, the generation without a salt, and a short salt"); }
  // 4. Chain object and the pure steps agree.
  std::vector<Bytes> cks, mks, seals; {
    Bytes ck = RatchetHashChain::genesis_state(genesis()), head = RatchetHashChain::genesis_head(ck); cks.push_back(ck);
    for (int i = 0; i < 10; ++i) { Bytes mk = RatchetHashChain::step_key(ck); Bytes s = RatchetHashChain::seal(mk, head, uint64_t(i) + 1, rec(i)); ck = RatchetHashChain::advance(ck, s); head = s; mks.push_back(mk); seals.push_back(s); cks.push_back(ck); }
    check(head == h, "chain object matches the documented step-by-step derivation"); }
  // 5. Backward fabrication: from the late state ck_10 alone, walk forward 256 steps with arbitrary records and
  //    keys; nothing reached equals any earlier state, step key, or seal. Going back needs an HMAC-SHA256 preimage.
  { std::set<std::string> early; for (int i = 0; i < 10; ++i) { early.insert(hx(cks[size_t(i)])); early.insert(hx(mks[size_t(i)])); early.insert(hx(seals[size_t(i)])); }
    Bytes ck = cks.back(), head = seals.back(); bool clash = false;
    for (int i = 0; i < 256; ++i) { Bytes mk = RatchetHashChain::step_key(ck); Bytes s = RatchetHashChain::seal(mk, head, uint64_t(i) + 11, rec(i % 7)); for (const Bytes* b : {&ck, &mk, &s}) clash |= early.count(hx(*b)) != 0; ck = RatchetHashChain::advance(ck, s); head = s; }
    // An attacker holding ck_10 who tries to reseal step 3 with what it has gets a different seal.
    Bytes forged = RatchetHashChain::seal(RatchetHashChain::step_key(cks.back()), seals[1], 3, rec(2));
    check(!clash && forged != seals[2], "backward fabrication: a late state reaches no earlier state, key, or seal"); }
  // 6. Resume needs a re-root: restore_head detaches the chain, append refuses until reroot.
  { RatchetHashChain c(genesis()); for (int i = 0; i < 5; ++i) c.append(rec(i)); const Bytes ph = c.head(); const size_t pl = c.length();
    RatchetHashChain d(genesis()); d.restore_head(ph, pl); bool threw = false; try { d.append(rec(5)); } catch (const std::logic_error&) { threw = true; }
    check(threw && d.detached(), "restored head refuses to append until re-rooted (no silent straight-line resume)");
    d.reroot("resume", "ratchet-test-run", sha("resume at step 5")); d.append(rec(5));
    RatchetHashChain e(genesis()); e.restore_head(ph, pl); e.reroot("resume", "ratchet-test-run", sha("resume at step 5")); e.append(rec(5));
    c.append(rec(5));
    check(d.head() == e.head() && d.epoch() == 1 && d.head() != c.head(), "re-root is deterministic and differs from the unbroken line");
    RatchetHashChain fa(genesis()), fb(genesis()); fa.restore_head(ph, pl); fb.restore_head(ph, pl);
    fa.reroot("fork", "child-a", sha("branch a")); fb.reroot("fork", "child-b", sha("branch b")); fa.append(rec(5)); fb.append(rec(5));
    check(fa.head() != fb.head() && fa.epoch_commitment() != fb.epoch_commitment(), "two forks of one parent get different epoch commitments and heads");
    RatchetHashChain fw(genesis()); fw.restore_head(ph, pl); fw.reroot("fork", "child-a", sha("branch a, changed weights hash")); fw.append(rec(5));
    check(fw.head() != fa.head(), "epoch commitment covers changed weights or context detail"); }
  // 7. Canonical order: the head depends on record order, so the coordinator must append in canonical order.
  { RatchetHashChain a(genesis()), b(genesis()); for (int i = 0; i < 10; ++i) a.append(rec(i)); for (int i = 9; i >= 0; --i) b.append(rec(i));
    check(a.head() != b.head(), "record order changes the head (canonical order law has teeth)"); }
  std::printf("RATCHET-CHAIN: %d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
