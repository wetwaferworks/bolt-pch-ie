// Join-order invariant test (2026-11-11).
// Shuffle-arrival: the same work results deposited in many arrival orders (and by 1..16 real
// worker threads through JoinJobDispatcher) must fold to bit-identical values and hashes.
// A naive arrival-order float sum is shown to differ, so the test can actually fail.
#include "pchie/DeterministicJoin.h"
#include <thread>
#include "pchie/JoinJobDispatcher.h"
#include "pchie/Sha256.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace pchie;
static int g_fail = 0, g_pass = 0;
static void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  ok ? ++g_pass : ++g_fail;
}
static std::string hexs(const std::vector<uint8_t>& v) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (uint8_t b : v) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}
// Work item i -> a float with wildly different magnitudes so summation order matters.
static float work_value(uint64_t i) {
  const float mag[4] = {1e8f, 1.0f, -1e8f, 3.14159f};
  return mag[i % 4] * (1.0f + float(i % 7) * 0.001f) + float(i) * 1e-3f;
}
static bool fold_sum(const DeterministicJoin<float>& j, float& out, std::string& err) {
  return j.fold(0.0f, [](float acc, uint64_t, const float& v) { return acc + v; }, out, err);
}
static std::string fold_hash(const DeterministicJoin<float>& j) {
  std::vector<uint8_t> buf;
  std::string err;
  std::vector<uint8_t> out;
  j.fold(std::vector<uint8_t>(), [](std::vector<uint8_t> acc, uint64_t id, const float& v) {
    uint8_t b[12];
    std::memcpy(b, &id, 8);
    std::memcpy(b + 8, &v, 4);
    acc.insert(acc.end(), b, b + 12);
    return acc;
  }, out, err);
  return hexs(Sha256::hash(out.data(), out.size()));
}

int main() {
  std::printf("Join-order invariant (deterministic fold by stable work identity)\n");
  const size_t N = 257;
  DeterministicJoin<float> ref(N);
  for (uint64_t i = 0; i < N; ++i) ref.deposit(i, work_value(i));
  float ref_sum = 0;
  std::string err;
  check(fold_sum(ref, ref_sum, err), "reference in-order fold succeeds");
  const std::string ref_hash = fold_hash(ref);
  std::printf("  reference fold sum bits=0x%08X sha256=%s\n", [&] { uint32_t u; std::memcpy(&u, &ref_sum, 4); return u; }(), ref_hash.c_str());

  // 1) shuffle-arrival: 200 random arrival permutations, single thread
  std::mt19937_64 rng(0x5EED1234ULL);
  bool all_same = true, naive_differs = false;
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<uint64_t> order(N);
    for (uint64_t i = 0; i < N; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    DeterministicJoin<float> j(N);
    float naive = 0.0f;
    for (uint64_t id : order) { j.deposit(id, work_value(id)); naive = naive + work_value(id); }
    float s = 0;
    fold_sum(j, s, err);
    if (std::memcmp(&s, &ref_sum, 4) != 0 || fold_hash(j) != ref_hash) all_same = false;
    if (std::memcmp(&naive, &ref_sum, 4) != 0) naive_differs = true;
  }
  check(all_same, "200 shuffled arrival orders fold bit-identical (sum and sha256)");
  check(naive_differs, "control: a naive arrival-order float sum DOES differ for some order (test is sensitive)");

  // 2) real threads: 1..16 workers deposit by stable id through JoinJobDispatcher
  bool threads_same = true;
  for (int w = 1; w <= 16; ++w) {
    DeterministicJoin<float> j(N);
    JoinJobDispatcher d(w);
    d.parallel_for(int(N), [&](int i) { j.deposit(uint64_t(i), work_value(uint64_t(i))); });
    float s = 0;
    if (!fold_sum(j, s, err) || std::memcmp(&s, &ref_sum, 4) != 0 || fold_hash(j) != ref_hash) threads_same = false;
  }
  check(threads_same, "1..16 JoinJobDispatcher workers fold bit-identical to the reference");

  // 3) fail-closed edges
  {
    DeterministicJoin<float> j(4);
    j.deposit(0, 1.f); j.deposit(1, 2.f); j.deposit(3, 4.f);
    float s = 0; std::string e;
    check(!fold_sum(j, s, e) && e.find("missing") != std::string::npos, "missing work id fails closed");
  }
  {
    DeterministicJoin<float> j(2);
    j.deposit(0, 1.f);
    const bool dup = j.deposit(0, 5.f);
    j.deposit(1, 2.f);
    float s = 0; std::string e;
    check(!dup && !fold_sum(j, s, e) && e.find("duplicate") != std::string::npos, "duplicate work id refused and fold fails closed");
  }
  {
    DeterministicJoin<float> j(2);
    check(!j.deposit(2, 1.f), "out-of-range work id refused");
  }
  {
    DeterministicJoin<float> j(4);
    std::thread a([&] { for (int k = 0; k < 2000; ++k) j.deposit(uint64_t(100 + k), 1.f); });
    std::thread b([&] { for (int k = 0; k < 2000; ++k) j.deposit(uint64_t(200 + k), 2.f); });
    a.join(); b.join();
    float s = 0; std::string e;
    check(!j.complete() && !fold_sum(j, s, e) && e == "work id out of range", "racing out-of-range deposits fail closed without a shared error string");
  }
  {
    DeterministicJoin<float> j(1);
    int wins = 0;
    std::thread a([&] { if (j.deposit(0, 1.f)) ++wins; });
    a.join();
    std::thread b([&] { if (j.deposit(0, 2.f)) ++wins; });
    b.join();
    float s = 0; std::string e;
    check(wins == 1 && !fold_sum(j, s, e) && e == "duplicate work id", "duplicate claim is detected by the atomic slot owner");
  }
  std::printf("JOIN-ORDER: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
