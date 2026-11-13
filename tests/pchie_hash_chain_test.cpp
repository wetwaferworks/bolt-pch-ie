// Synthetic KV UNIT-TEST FIXTURE ONLY: NEVER golden or baseline input. Real-weight integration is covered by the Nemotron model tests, not here.
// Bolt PCHIE - failing-first tests for hash chains + fork snapshot/rollback.
// CPU only. No locks. Coordinator-thread append only.
#include "pchie/ChainRecord.h"
#include "pchie/ChainVerifier.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/Sha256.h"
#include "pchie/ForkSnapshotStore.h"
#include "pchie/pchie_kv_pools.h"
#include "pchie/pchie_tree.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[HASH-CHAIN] %-72s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
}

static ChainRecord rec(ChainRecordKind k, const char* s) {
  ChainRecord r;
  r.kind = k;
  if (s) {
    const size_t n = std::strlen(s);
    r.payload.assign(reinterpret_cast<const uint8_t*>(s), reinterpret_cast<const uint8_t*>(s) + n);
  }
  return r;
}

static void flip_one_bit(std::vector<uint8_t>& bytes, size_t bit_index) {
  const size_t byte = bit_index / 8;
  const size_t bit = bit_index % 8;
  if (byte < bytes.size()) bytes[byte] = uint8_t(bytes[byte] ^ (uint8_t(1) << bit));
}

static void test_fnv_chain_append_changes_head() {
  Fnv1a64HashChain c;
  c.reset();
  const uint64_t h0 = c.head_u64();
  check(h0 == FNV_OFFSET, "FNV reset head == FNV_OFFSET");
  c.append(rec(ChainRecordKind::Model, "weights-v1"));
  check(c.length() == 1, "FNV length == 1 after one append");
  check(c.head_u64() != h0, "FNV append changes head");
  check(c.head().size() == 8, "FNV head is 8 bytes");
}

static void test_fnv_deterministic_and_tamper() {
  Fnv1a64HashChain a, b;
  const auto r0 = rec(ChainRecordKind::Config, "cfg");
  const auto r1 = rec(ChainRecordKind::Prompt, "hello");
  const auto r2 = rec(ChainRecordKind::Commit, "node=7");
  a.reset(); a.append(r0); a.append(r1); a.append(r2);
  b.reset(); b.append(r0); b.append(r1); b.append(r2);
  check(a.head() == b.head(), "FNV identical records => identical head (deterministic)");

  // Tamper: flip one bit in a payload and recompute.
  ChainRecord evil = r1;
  flip_one_bit(evil.payload, 0);
  Fnv1a64HashChain t;
  t.reset(); t.append(r0); t.append(evil); t.append(r2);
  check(t.head() != a.head(), "FNV single-bit payload flip breaks chain head");

  // Verifier rejects tampered log against good head.
  std::vector<ChainRecord> good = {r0, r1, r2};
  std::vector<ChainRecord> bad = {r0, evil, r2};
  Fnv1a64HashChain scratch;
  auto ok = ChainVerifier::verify(scratch, good, a.head());
  check(ok.ok, "ChainVerifier accepts honest FNV log");
  auto badv = ChainVerifier::verify(scratch, bad, a.head());
  check(!badv.ok, "ChainVerifier rejects tampered FNV log");
}

static void test_sha256_known_vector_and_chain() {
  // FIPS 180-2 empty string digest.
  const char* empty_hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  auto dig = Sha256::hash(nullptr, 0);
  check(dig.size() == 32, "SHA-256 digest is 32 bytes");
  std::string got;
  static const char* hex = "0123456789abcdef";
  for (uint8_t b : dig) { got.push_back(hex[b >> 4]); got.push_back(hex[b & 15]); }
  check(got == empty_hex, "SHA-256(\"\") matches FIPS empty digest");

  Sha256HashChain a, b;
  const auto r0 = rec(ChainRecordKind::Model, "m");
  const auto r1 = rec(ChainRecordKind::GateVerdict, "G=1");
  a.reset(); a.append(r0); a.append(r1);
  b.reset(); b.append(r0); b.append(r1);
  check(a.head() == b.head() && a.head().size() == 32, "SHA chain deterministic 32-byte head");
  check(a.head() != std::vector<uint8_t>(32, 0), "SHA chain head is not all zeros after append");

  ChainRecord evil = r1;
  flip_one_bit(evil.payload, 3);
  Sha256HashChain t;
  t.reset(); t.append(r0); t.append(evil);
  check(t.head() != a.head(), "SHA single-bit flip breaks chain head");

  Sha256HashChain scratch;
  auto ok = ChainVerifier::verify(scratch, std::vector<ChainRecord>{r0, r1}, a.head());
  check(ok.ok, "ChainVerifier accepts honest SHA log");
  auto bad = ChainVerifier::verify(scratch, std::vector<ChainRecord>{r0, evil}, a.head());
  check(!bad.ok, "ChainVerifier rejects tampered SHA log");
}

static void test_fnv_vs_sha_disagreement_is_expected() {
  // Same records: algorithms MUST disagree (report if they ever collide by chance — extremely unlikely).
  const auto r = rec(ChainRecordKind::Custom, "same-payload");
  Fnv1a64HashChain f; f.reset(); f.append(r);
  Sha256HashChain s; s.reset(); s.append(r);
  // Compare first 8 bytes of SHA to FNV head — must not be equal (report if equal).
  const auto fh = f.head();
  const auto sh = s.head();
  bool collide8 = fh.size() == 8 && sh.size() >= 8 && std::memcmp(fh.data(), sh.data(), 8) == 0;
  check(!collide8, "FNV head != first 8 bytes of SHA head (no accidental agreement)");
  if (collide8) std::printf("[HASH-CHAIN] ALERT: FNV vs SHA disagreement FAILED — possible collision, report it immediately\n");
}

static void test_chain_restore_head() {
  Fnv1a64HashChain c;
  c.reset();
  c.append(rec(ChainRecordKind::Prompt, "a"));
  c.append(rec(ChainRecordKind::Prompt, "b"));
  const auto head = c.head();
  const size_t len = c.length();
  c.append(rec(ChainRecordKind::Prompt, "c"));
  check(c.head() != head, "FNV head moved after third append");
  c.restore_head(head, len);
  check(c.head() == head && c.length() == len, "FNV restore_head returns prior head/length");
}

static void test_fork_snapshot_rollback_roundtrip() {
  // Small pool: 2 layers, 1 head, dim 4, plenty of pages.
  KVPagePool pool(/*layers*/2, /*kv_heads*/1, /*head_dim*/4, /*max_pages*/256);
  ForkSnapshotStore store(pool);
  Fnv1a64HashChain chain;
  chain.reset();
  chain.append(rec(ChainRecordKind::Genesis, "g"));

  KVPagePool::Table active = pool.fork(KVPagePool::Table{});
  // Write token 0 so a page is allocated.
  float* k = pool.k_write(active, 0, 0);
  check(k != nullptr, "k_write allocates a page");
  k[0] = 1.0f; k[1] = 2.0f;
  const uint32_t live0 = pool.live_pages();
  const uint64_t sum0 = pool.page_checksum(pool.page_of(active, 0));

  store.snapshot(/*node*/1, active, /*sim*/42, chain.head(), chain.length());
  check(store.find(1) != nullptr, "snapshot stores node 1");

  // Mutate after snapshot: CoW should allocate a private page on write.
  float* k2 = pool.k_write(active, 0, 0);
  k2[0] = 9.0f;
  chain.append(rec(ChainRecordKind::Commit, "after"));
  check(pool.live_pages() >= live0, "mutation may allocate CoW page");

  KVPagePool::Table restored;
  uint64_t sim = 0;
  std::vector<uint8_t> head;
  size_t len = 0;
  check(store.rollback(1, restored, sim, head, len), "rollback(1) succeeds");
  check(sim == 42, "rollback restores sim_snapshot_id");
  check(head == store.find(1)->chain_head && len == store.find(1)->chain_length,
        "rollback restores chain head/length");
  chain.restore_head(head, len);

  // Release mutated active, take restored.
  pool.release(active);
  active = pool.fork(restored);
  const uint64_t sum1 = pool.page_checksum(pool.page_of(active, 0));
  check(sum1 == sum0, "rollback KV page checksum bit-identical to snapshot");
  pool.release(active);
  pool.release(restored);
}

static void test_fork_10k_no_leak() {
  KVPagePool pool(2, 1, 4, 512);
  ForkSnapshotStore store(pool);
  Fnv1a64HashChain chain;
  chain.reset();
  KVPagePool::Table active = pool.fork(KVPagePool::Table{});
  float* k = pool.k_write(active, 0, 0);
  k[0] = 3.14f;
  const uint32_t live_base = pool.live_pages();
  store.snapshot(0, active, 1, chain.head(), chain.length());

  for (int i = 0; i < 10000; ++i) {
    // Branch write (CoW), then rollback to snapshot 0.
    float* w = pool.k_write(active, 0, 0);
    w[0] = float(i);
    chain.append(rec(ChainRecordKind::Custom, "x"));
    KVPagePool::Table restored;
    uint64_t sim = 0;
    std::vector<uint8_t> head;
    size_t len = 0;
    if (!store.rollback(0, restored, sim, head, len)) { check(false, "rollback in 10k loop"); return; }
    chain.restore_head(head, len);
    pool.release(active);
    active = pool.fork(restored);
    pool.release(restored);
  }
  check(pool.live_pages() == live_base || pool.live_pages() == live_base + 0,
        "10k fork/rollback: live_pages back to baseline (no page leaks)");
  // Stronger: live pages must not grow without bound.
  check(pool.live_pages() <= live_base + 2, "10k fork/rollback: live_pages within +2 of baseline");
  pool.release(active);
}

int main() {
  std::printf("[HASH-CHAIN] hash-chain + fork-snapshot tests (CPU)\n");
  test_fnv_chain_append_changes_head();
  test_fnv_deterministic_and_tamper();
  test_sha256_known_vector_and_chain();
  test_fnv_vs_sha_disagreement_is_expected();
  test_chain_restore_head();
  test_fork_snapshot_rollback_roundtrip();
  test_fork_10k_no_leak();
  std::printf("[HASH-CHAIN] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}