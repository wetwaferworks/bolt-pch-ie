// Harness checks (no model, about a second). Everything tested here is scratch research, off the
// official path: the official chain, goldens and the --rl-* / --research-* defaults are untouched.
//
//   pchie_harness_test --unit [--dir PATH]   fast checks (CTest); side-chain files go under PATH (default .)
#include "HarnessCheck.h"

#include <chrono>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  std::string dir = ".";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--dir" && i + 1 < argc) dir = argv[++i];
  }
  const auto t0 = std::chrono::steady_clock::now();
  harness_test::test_task_eval(dir.c_str());
  harness_test::test_plugs(dir.c_str());
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("harness: %d passed, %d failed (%.0f ms)\n", harness_test::passes(), harness_test::failures(), ms);
  return harness_test::failures() ? 1 : 0;
}
