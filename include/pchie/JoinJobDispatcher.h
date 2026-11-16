// Bolt PCHIE B4 - mutex-free fork/join CPU dispatcher for the worker-count gauntlet.
// Concrete final. No mutex, no condition_variable, no atomic: std::thread::join is
// the phase boundary. Workers never touch the hash chain.
// Spawns threads per parallel_for (persistent pools need a CV/atomic wake).
#pragma once

#include "pchie/IJobDispatcher.h"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

namespace pchie {

class JoinJobDispatcher final : public IJobDispatcher {
 public:
  static constexpr int kMaxWorkers = 16;

  explicit JoinJobDispatcher(int workers) {
    if (workers <= 0 || workers > kMaxWorkers)
      throw std::invalid_argument("JoinJobDispatcher: workers must be in 1..16");
    workers_ = workers;
  }

  void parallel_for(int count, const std::function<void(int)>& fn) override {
    if (workers_ <= 0 || count <= 0) return;
    if (workers_ == 1 || count == 1) {
      for (int i = 0; i < count; ++i) fn(i);
      return;
    }
    const int n = workers_ < count ? workers_ : count;
    std::vector<std::thread> ts;
    ts.reserve(size_t(n));
    std::vector<std::exception_ptr> worker_ex(static_cast<size_t>(n));
    try {
      for (int t = 0; t < n; ++t) {
        ts.emplace_back([&, t]() {
          try {
            const int begin = int((int64_t(count) * t) / n);
            const int end = int((int64_t(count) * (t + 1)) / n);
            for (int i = begin; i < end; ++i) fn(i);
          } catch (...) {
            worker_ex[size_t(t)] = std::current_exception();
          }
        });
      }
    } catch (...) {
      for (auto& th : ts)
        if (th.joinable()) th.join();
      throw;
    }
    for (auto& th : ts)
      if (th.joinable()) th.join();
    for (const auto& e : worker_ex)
      if (e) std::rethrow_exception(e);
  }

  int size() const noexcept override { return workers_; }

 private:
  int workers_ = 0;
};

}  // namespace pchie
