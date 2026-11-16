// Bolt PCHIE - TESLA-manner watcher. A separate program (tools/pchie_tesla_watcher.cpp) links this
// header; tests use the same type. The safety test is an ordering test on the stream, never a wall clock.
//
// A step packet is safe only if it arrived before its interval key was disclosed, in canonical stream order.
// Late is forged until proven otherwise, and it cannot be proven. A bad MAC flags the stream at that step
// and no later step is released as verified. Missing steps are gaps, never smoothed over. A disclosure that
// does not hash to the commitment fails authenticity, full stop. This layer is tamper evident, not tamper proof.
//
// Untrusted input is bounded: a step or disclosure whose index is past the limits, or a stream with more
// packets than the limits allow, stops the watcher with a flag (no large allocation, no long hash walk).
// A step index that arrives twice is flagged and rejected from that index on, so all_verified is never true
// for a stream with duplicates. The gap report keeps a count plus the first kMaxGapIndices indices, and a
// stream's step space is capped at kMaxSteps whatever I the header claims, so a three-line stream cannot make
// the watcher allocate gigabytes.
#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "pchie/TeslaKeyChain.h"
#include "pchie/TeslaStream.h"

namespace pchie {

enum class TeslaFlag {
  Late,             // step packet arrived after its interval key was disclosed
  BadMac,           // MAC did not verify
  WrongKey,         // disclosure does not hash to the commitment
  EarlyDisclosure,  // key arrived before its interval was due (ahead of schedule)
  Gap,              // a step index is missing from the released stream
  BadInterval,      // step's interval index does not match floor(step / I)
  Duplicate,        // the same step index arrived more than once
  OutOfRange,       // a step or interval index is past the watcher's limits (stream stopped)
  TooLong,          // more packets than the limits allow (stream stopped)
};

// Limits for an untrusted stream. The defaults match the sender's default chain length (4096 intervals).
struct TeslaWatchLimits {
  uint32_t max_intervals = 4096;
  uint64_t max_packets = 0;  // 0 = steps + intervals + slack, derived from max_intervals and I
};

inline const char* tesla_flag_name(TeslaFlag f) {
  switch (f) {
    case TeslaFlag::Late: return "late";
    case TeslaFlag::BadMac: return "bad-mac";
    case TeslaFlag::WrongKey: return "wrong-key";
    case TeslaFlag::EarlyDisclosure: return "early-disclosure";
    case TeslaFlag::Gap: return "gap";
    case TeslaFlag::BadInterval: return "bad-interval";
    case TeslaFlag::Duplicate: return "duplicate";
    case TeslaFlag::OutOfRange: return "out-of-range";
    case TeslaFlag::TooLong: return "too-long";
  }
  return "unknown";
}

struct TeslaFlagEvent {
  TeslaFlag flag;
  uint64_t step_index = 0;     // for step-related flags; 0 if not applicable
  uint32_t interval_index = 0;
};

struct TeslaReleasedStep {
  uint64_t step_index = 0;
  uint32_t interval_index = 0;
  std::vector<uint8_t> seal, payload_hash;
};

struct TeslaWatchResult {
  bool authentic = true;          // false after a wrong-key disclosure (full stop)
  bool all_verified = false;      // authentic, no flags that reject a step, no gaps, every step 0..last released
  uint64_t failed_at = ~uint64_t(0);  // first step index that must not be released; ~0 if none
  std::vector<TeslaReleasedStep> released;
  std::vector<uint64_t> gaps;     // the first kMaxGapIndices missing indices (the report stays bounded)
  uint64_t gap_count = 0;         // every missing index, including those past the stored list
  std::vector<TeslaFlagEvent> flags;
};

class TeslaWatcher {
 public:
  TeslaWatcher(std::vector<uint8_t> commitment, TeslaSchedule sch = {}, TeslaWatchLimits lim = {})
      : commitment_(std::move(commitment)), sch_(sch), lim_(lim) {
    sch_.check();
    if (commitment_.size() != Sha256::kDigestBytes) throw std::invalid_argument("TESLA watcher: commitment must be 32 bytes");
    if (lim_.max_intervals == 0 || lim_.max_intervals > (1u << 20)) throw std::invalid_argument("TESLA watcher: max_intervals out of range");
    // The step space is capped whatever I and max_intervals say (I comes from the stream header): seen_ is
    // one byte per step, so 2^24 steps is at most 16 MB, still 512 times an honest 4096 x 8 stream.
    max_steps_ = std::min<uint64_t>(uint64_t(lim_.max_intervals) * sch_.I, kMaxSteps);
    if (lim_.max_packets == 0) lim_.max_packets = max_steps_ + 2ull * lim_.max_intervals + 64;
  }

  // Arrival order is the order accept_* is called. That order is the clock.
  void accept_step(const TeslaStepPacket& p) {
    if (stopped_) return;
    if (!count_packet()) return;
    const uint64_t arrival = arrival_++;
    if (p.step_index >= max_steps_ || p.interval_index >= lim_.max_intervals) {
      flag(TeslaFlag::OutOfRange, p.step_index, p.interval_index);
      stop();
      return;
    }
    if (p.interval_index != uint32_t(p.step_index / sch_.I)) {
      flag(TeslaFlag::BadInterval, p.step_index, p.interval_index);
      fail_at(p.step_index);
      return;
    }
    if (seen_.size() <= p.step_index) seen_.resize(size_t(p.step_index) + 1, 0);  // bounded by max_steps_
    if (seen_[size_t(p.step_index)]) {
      // A second packet for a step index is never released: that index and everything after it fail.
      flag(TeslaFlag::Duplicate, p.step_index, p.interval_index);
      fail_at(p.step_index);
      return;
    }
    seen_[size_t(p.step_index)] = 1;
    if (!saw_step_ || p.step_index > highest_step_seen_) highest_step_seen_ = p.step_index;
    saw_step_ = true;
    if (key_known(p.interval_index)) {
      // The key was already public when this packet arrived. Reject it. Late cannot be proven honest.
      flag(TeslaFlag::Late, p.step_index, p.interval_index);
      fail_at(p.step_index);
      return;
    }
    buffer_[p.interval_index].push_back(Buffered{arrival, p});
  }

  void accept_disclosure(const TeslaDisclosure& d) {
    if (stopped_) return;
    if (!count_packet()) return;
    arrival_++;
    if (d.interval_index >= lim_.max_intervals) {
      // Checked before any hashing: key_belongs walks interval_index + 1 hash steps.
      flag(TeslaFlag::OutOfRange, 0, d.interval_index);
      stop();
      return;
    }
    if (!disclosed_key_belongs(d.key, d.interval_index)) {
      flag(TeslaFlag::WrongKey, 0, d.interval_index);
      result_.authentic = false;
      stopped_ = true;  // full stop
      return;
    }
    if (!disclosure_is_due(d.interval_index)) {
      flag(TeslaFlag::EarlyDisclosure, 0, d.interval_index);
      // The key is now public. Packets that arrive after this are late. Already-buffered packets of this
      // interval arrived before the (early) disclosure, so they may still have their MACs checked.
    }
    if (int64_t(d.interval_index) <= top_known_) return;  // already known (a repeat); nothing new to unlock
    // A later key also unlocks every earlier interval (loss tolerance for dropped disclosures). Keys are
    // always known for a whole prefix 0..top_known_, so only the intervals above it are new. One hash step
    // per new interval: every key is derived and every interval checked once over the whole stream.
    std::vector<uint8_t> cur = d.key;
    for (int64_t j = int64_t(d.interval_index); j > top_known_; --j) {
      if (j < int64_t(d.interval_index)) cur = tesla_chain_step(cur);
      install_key(uint32_t(j), cur);
      verify_interval(uint32_t(j));
    }
    wipe_bytes(cur);
    top_known_ = int64_t(d.interval_index);
  }

  // End of stream: report gaps among 0..highest_step_seen, and whether everything verified.
  TeslaWatchResult finish() {
    if (saw_step_ || !result_.released.empty()) {
      const uint64_t last = saw_step_ ? highest_step_seen_ : 0;
      const uint64_t end = saw_step_ ? last + 1 : 0;
      std::vector<uint8_t> have;
      for (const auto& r : result_.released) {
        if (r.step_index >= max_steps_) continue;  // cannot happen: out-of-range steps are never buffered
        if (have.size() <= r.step_index) have.resize(size_t(r.step_index) + 1, 0);
        if (have[size_t(r.step_index)]) {  // two released records for one index: never all_verified
          flag(TeslaFlag::Duplicate, r.step_index, r.interval_index);
          if (r.step_index < result_.failed_at) result_.failed_at = r.step_index;
        }
        have[size_t(r.step_index)] = 1;
      }
      for (uint64_t s = 0; s < end; ++s) {
        const bool released = s < have.size() && have[s];
        if (!released) {
          // An index before failed_at that was never released is a gap. Indices at or after failed_at were
          // already rejected (the stream failed there), so they are not reported again as gaps.
          if (s < result_.failed_at) {
            if (result_.gaps.size() < kMaxGapIndices) result_.gaps.push_back(s);
            ++result_.gap_count;
            flag(TeslaFlag::Gap, s, uint32_t(s / sch_.I));
          }
        }
      }
    }
    result_.all_verified = result_.authentic && result_.gap_count == 0 && result_.gaps.empty() && result_.failed_at == ~uint64_t(0) &&
                           !result_.released.empty();
    // Any flag fails the stream as a whole, early disclosure included: an early key is an attack, even
    // though packets buffered before it may still verify and be released. A stopped watch never passes.
    if (!result_.flags.empty() || stopped_) result_.all_verified = false;
    return result_;
  }

  const TeslaWatchResult& snapshot() const { return result_; }

 private:
  struct Buffered {
    uint64_t arrival = 0;
    TeslaStepPacket packet;
  };

  bool key_known(uint32_t i) const { return i < keys_known_.size() && keys_known_[i]; }

  // Is `key` the chain key for interval i? Checked against the nearest key already authenticated below i
  // (RFC 4082 section 3.5), or against the commitment when none is known yet. Each check costs at most
  // i - top_known_ hash steps, so a long stream of disclosures cannot force quadratic work.
  bool disclosed_key_belongs(const std::vector<uint8_t>& key, uint32_t i) const {
    if (key.size() != Sha256::kDigestBytes) return false;
    if (int64_t(i) <= top_known_) return key == keys_[i];
    if (top_known_ < 0) return TeslaKeyChain::key_belongs(key, i, commitment_);
    std::vector<uint8_t> x = key;
    for (int64_t n = int64_t(i); n > top_known_; --n) x = tesla_chain_step(x);
    const bool ok = x == keys_[size_t(top_known_)];
    wipe_bytes(x);
    return ok;
  }

  bool disclosure_is_due(uint32_t j) const {
    // Mid-stream, K_j is due once interval j+d has closed (the watcher has seen step (j+d+1)*I - 1).
    // After note_end_of_steps, the last d keys of the last interval L are also due (trailing disclosures).
    // No wall clock is consulted.
    if (steps_ended_) {
      if (!saw_step_) return false;
      const uint32_t L = uint32_t(highest_step_seen_ / sch_.I);
      return j <= L;
    }
    const uint64_t due_after = uint64_t(j + sch_.d + 1) * sch_.I - 1;
    return saw_step_ && highest_step_seen_ >= due_after;
  }

  void install_key(uint32_t i, const std::vector<uint8_t>& key) {
    if (keys_.size() <= i) {
      keys_.resize(i + 1);
      keys_known_.resize(i + 1, 0);
    }
    if (!keys_known_[i]) {
      keys_[i] = key;
      keys_known_[i] = 1;
    }
  }

  void verify_interval(uint32_t i) {
    auto it = buffer_.find(i);
    if (it == buffer_.end()) return;
    auto& buf = it->second;
    std::sort(buf.begin(), buf.end(), [](const Buffered& a, const Buffered& b) { return a.packet.step_index < b.packet.step_index; });
    std::vector<uint8_t> mk = tesla_mac_key(keys_[i]);
    for (const auto& item : buf) {
      const TeslaStepPacket& p = item.packet;
      if (p.step_index >= result_.failed_at) continue;  // do not release later steps
      const std::vector<uint8_t> got = tesla_step_mac(mk, p);
      if (got != p.mac) {
        flag(TeslaFlag::BadMac, p.step_index, i);
        fail_at(p.step_index);
        continue;
      }
      TeslaReleasedStep r;
      r.step_index = p.step_index;
      r.interval_index = p.interval_index;
      r.seal = p.seal;
      r.payload_hash = p.payload_hash;
      result_.released.push_back(std::move(r));
    }
    wipe_bytes(mk);
    buffer_.erase(it);
    std::sort(result_.released.begin(), result_.released.end(),
              [](const TeslaReleasedStep& a, const TeslaReleasedStep& b) { return a.step_index < b.step_index; });
  }

  void fail_at(uint64_t step) {
    if (step < result_.failed_at) result_.failed_at = step;
    // Drop any already-released step at or after the failure (should be rare; verification is in order).
    auto& rel = result_.released;
    rel.erase(std::remove_if(rel.begin(), rel.end(), [&](const TeslaReleasedStep& r) { return r.step_index >= result_.failed_at; }),
              rel.end());
  }

  bool count_packet() {
    if (++packets_ <= lim_.max_packets) return true;
    flag(TeslaFlag::TooLong, 0, 0);
    stop();
    return false;
  }
  void stop() {
    stopped_ = true;
    for (auto& kv : buffer_) kv.second.clear();
    buffer_.clear();
  }

  void flag(TeslaFlag f, uint64_t step, uint32_t interval) {
    if (result_.flags.size() >= kMaxFlags) return;  // the stream already failed; keep the report bounded
    TeslaFlagEvent e;
    e.flag = f;
    e.step_index = step;
    e.interval_index = interval;
    result_.flags.push_back(e);
  }

 public:
  // Call after the last step packet, before trailing disclosures, so those keys are not treated as early.
  void note_end_of_steps() { steps_ended_ = true; }

 private:
  static constexpr size_t kMaxFlags = 1024;

 public:
  static constexpr size_t kMaxGapIndices = 1024;
  static constexpr uint64_t kMaxSteps = 1ull << 24;

 private:
  std::vector<uint8_t> commitment_;
  TeslaSchedule sch_;
  TeslaWatchLimits lim_;
  uint64_t max_steps_ = 0;
  uint64_t packets_ = 0;
  std::vector<uint8_t> seen_;
  uint64_t arrival_ = 0;
  uint64_t highest_step_seen_ = 0;
  bool saw_step_ = false;
  bool steps_ended_ = false;
  bool stopped_ = false;
  int64_t top_known_ = -1;  // keys are known for every interval 0..top_known_ (-1: none yet)
  std::vector<std::vector<uint8_t>> keys_;
  std::vector<uint8_t> keys_known_;
  std::map<uint32_t, std::vector<Buffered>> buffer_;
  TeslaWatchResult result_;
};

}  // namespace pchie
