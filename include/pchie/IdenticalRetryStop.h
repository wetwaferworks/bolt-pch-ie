// Bolt PCHIE - IdenticalRetryStop: opt-in early stop on identical retries.
// Concrete final. Serial, deterministic. No mutex, no atomic, no threads.
// Rule: with limit N >= 2, observe(key) returns true once the last N observed keys are
// identical (N in a row). A different key restarts the run at 1; reset() clears it.
// limit 0 = OFF (the default): observe() never returns true and keeps no state.
// Limits 1 and < 0 are rejected (set_limit returns false and changes nothing), because
// "1 in a row" would stop on the first failure, which is not an identical-retry rule.
// The caller defines the key and observes only failed attempts. Opt-in, default
// off; official runners never set it.
#pragma once

#include <string>

namespace pchie {

class IdenticalRetryStop final {
 public:
  bool set_limit(int n) {
    if (n != 0 && n < 2) return false;
    limit_ = n;
    reset();
    return true;
  }
  int limit() const noexcept { return limit_; }
  bool enabled() const noexcept { return limit_ >= 2; }
  void reset() {
    last_.clear();
    run_ = 0;
    have_ = false;
  }
  bool observe(const std::string& key) {
    if (!enabled()) return false;
    if (have_ && key == last_) {
      ++run_;
    } else {
      last_ = key;
      run_ = 1;
      have_ = true;
    }
    return run_ >= limit_;
  }
  int run_length() const noexcept { return run_; }

 private:
  int limit_ = 0;
  std::string last_;
  int run_ = 0;
  bool have_ = false;
};

}  // namespace pchie
