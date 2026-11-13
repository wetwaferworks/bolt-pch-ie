// Bolt PCHIE - single-thread dispatcher. No mutex, no CV, no worker threads.
// Preferred path for deterministic CPU gates (do not use the legacy ThreadPool).
#pragma once

#include "pchie/IJobDispatcher.h"

namespace pchie {

class SerialJobDispatcher final : public IJobDispatcher {
 public:
  void parallel_for(int count, const std::function<void(int)>& fn) override {
    for (int i = 0; i < count; ++i) fn(i);
  }
  int size() const noexcept override { return 1; }
};

}  // namespace pchie