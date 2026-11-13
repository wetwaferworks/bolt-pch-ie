// Bolt PCHIE failed-actuation retry - FailureTrace: canonical, quantized failure
// records as token ids for counterfactual injection. Concrete final. No mutex/atomic/thread.
// Each record = trusted frame ids (string LITERALS only, IModel::tokenize_no_bos: harness text,
// BOS-free continuation) joined with untrusted field ids (UntrustedText ->
// IModel::tokenize_untrusted: never a special id, never a BOS). Ids are joined, never strings,
// so nothing merges across a seam. The ids carry no BOS and must follow a primed trusted frame
// (TraceL1RetryHarness forwards them only on a fork of its primed prompt branch).
// Quantization: integers are clamped to [-9999, 9999]; world_changed is 0/1.
// Record text: "\nAttempt <n> failed: component=<p> result=<r> north=<o> world_changed=<0|1>.
//               Try a different component." (fixed; not tuned).
#pragma once

#include "pchie/IModel.h"
#include "pchie/UntrustedText.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class FailureTrace final {
 public:
  void add_failure(IModel& m, int attempt_no, const UntrustedText& proposed, const UntrustedText& reason,
                   int observed, bool world_changed) {
    std::vector<int32_t> b;
    put(b, frame(m, "\nAttempt "));
    put(b, untrusted(m, UntrustedText(std::to_string(quant(attempt_no)))));
    put(b, frame(m, " failed: component="));
    put(b, untrusted(m, proposed));
    put(b, frame(m, " result="));
    put(b, untrusted(m, reason));
    put(b, frame(m, " north="));
    put(b, untrusted(m, UntrustedText(std::to_string(quant(observed)))));
    put(b, frame(m, " world_changed="));
    put(b, untrusted(m, UntrustedText(world_changed ? "1" : "0")));
    put(b, frame(m, ". Try a different component."));
    blocks_.push_back(std::move(b));
  }

  // Newest whole records that fit in `budget` tokens, in chronological order.
  std::vector<int32_t> ids(size_t budget) const {
    size_t used = 0, first = blocks_.size();
    while (first > 0 && used + blocks_[first - 1].size() <= budget) {
      used += blocks_[first - 1].size();
      --first;
    }
    std::vector<int32_t> out;
    out.reserve(used);
    for (size_t i = first; i < blocks_.size(); ++i) out.insert(out.end(), blocks_[i].begin(), blocks_[i].end());
    return out;
  }
  size_t blocks() const noexcept { return blocks_.size(); }
  // Number of newest whole records ids(budget) returns.
  size_t blocks_fitting(size_t budget) const {
    size_t used = 0, n = 0;
    for (size_t k = blocks_.size(); k > 0 && used + blocks_[k - 1].size() <= budget; --k) {
      used += blocks_[k - 1].size();
      ++n;
    }
    return n;
  }
  void clear() { blocks_.clear(); }

 private:
  // Trusted frame: string literals only (a runtime std::string cannot bind here).
  template <size_t N>
  static std::vector<int32_t> frame(IModel& m, const char (&lit)[N]) {
    return m.tokenize_no_bos(std::string(lit, N - 1));
  }
  static std::vector<int32_t> untrusted(IModel& m, const UntrustedText& u) { return m.tokenize_untrusted(u.str()); }
  static void put(std::vector<int32_t>& b, const std::vector<int32_t>& x) { b.insert(b.end(), x.begin(), x.end()); }
  static int quant(int v) { return v < -9999 ? -9999 : (v > 9999 ? 9999 : v); }

  std::vector<std::vector<int32_t>> blocks_;
};

}  // namespace pchie