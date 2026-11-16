// Tiny check counter shared by the harness test files.
#pragma once

#include <cstdio>

namespace harness_test {

inline int& passes() { static int n = 0; return n; }
inline int& failures() { static int n = 0; return n; }

inline void check(bool ok, const char* what) {
  if (ok) {
    ++passes();
  } else {
    ++failures();
    std::printf("FAIL: %s\n", what);
  }
}

constexpr int kWorkerCounts[] = {1, 2, 4, 8, 12};

void test_task_eval(const char* dir);
void test_plugs(const char* dir);

}  // namespace harness_test
