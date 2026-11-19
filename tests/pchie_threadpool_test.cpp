// Bolt PCHIE - ThreadPool: an exception thrown by the work, on any thread, comes back to the caller after every
// worker is done, and the pool keeps working afterwards.
#include "pchie/pchie_gemma4.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[POOL] %-72s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
}

int main() {
  ThreadPool pool(4);
  for (int round = 0; round < 200; ++round) {
    bool caught = false;
    std::atomic<int> ran{0};
    try {
      pool.parallel_for(64, [&](int i) {
        ran.fetch_add(1);
        if (i == round % 64) throw std::runtime_error("item failed");
      });
    } catch (const std::runtime_error& e) {
      caught = std::string(e.what()) == "item failed";
    }
    if (!caught) { check(false, "a throw from one item reaches the caller"); break; }
  }
  check(true, "200 rounds: a throw from one item (on any thread) reaches the caller");
  bool all_caught = true;
  for (int round = 0; round < 50; ++round) {
    try {
      pool.parallel_for(32, [&](int) { throw std::logic_error("every item"); });
      all_caught = false;
    } catch (const std::logic_error&) {
    }
  }
  check(all_caught, "every item throwing: one exception comes back, no terminate");
  std::vector<int> out(1000, 0);
  pool.parallel_for(1000, [&](int i) { out[size_t(i)] = i * 2; });
  long sum = 0;
  for (int v : out) sum += v;
  check(sum == 999L * 1000L, "the pool still works after exceptions (all items run once)");
  std::printf("[POOL] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
