// Bolt PCHIE - TESLA-manner watcher. A separate program (tools/pchie_tesla_watcher.cpp) links this
// header; tests use the same type. The safety test is an ordering test on the stream, never a wall clock.
//
// A step packet is safe only if it arrived before its interval key was disclosed, in canonical stream order.
// Late is forged until proven otherwise, and it cannot be proven. A bad MAC flags the stream at that step
// and no later step is released as verified. Missing steps are gaps, never smoothed over. A disclosure that
// does not hash to the commitment fails authenticity, full stop. This layer is tamper evident, not tamper proof.
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
};

inline const char* tesla_flag_name(TeslaFlag f) {
  switch (f) {
    case TeslaFlag::Late: return "late";
    case TeslaFlag::BadMac: return "bad-mac";
    case TeslaFlag::WrongKey: return "wrong-key";
    case TeslaFlag::EarlyDisclosure: return "early-disclosure";
    case TeslaFlag::Gap: return "gap";
    case TeslaFlag::BadInterval: return "bad-interval";
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
  std::vector<uint64_t> gaps;
  std::vector<TeslaFlagEvent> flags;
};

class TeslaWatcher {
 public:
  TeslaWatcher(std::vector<uint8_t> commitment, TeslaSchedule sch = {}) : commitment_(std::move(commitment)), sch_(sch) {
    sch_.check();
    if (commitment_.size() != Sha256::kDigestBytes) throw std::invalid_argument("TESLA watcher: commitment must be 32 bytes");
  }

  // Arrival order is the order accept_* is called. That order is the clock.
  void accept_step(const TeslaStepPacket& p) {
    if (stopped_) return;
    const uint64_t arrival = arrival_++;
    if (p.interval_index != uint32_t(p.step_index / sch_.I)) {
      flag(TeslaFlag::BadInterval, p.step_index, p.interval_index);
      fail_at(p.step_index);
      return;
    }
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
    arrival_++;
    if (!TeslaKeyChain::key_belongs(d.key, d.interval_index, commitment_)) {
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
    install_key(d.interval_index, d.key);
    // A later key also unlocks every earlier interval (loss tolerance for dropped disclosures).
    for (int64_t j = int64_t(d.interval_index); j >= 0; --j) {
      if (!key_known(uint32_t(j))) {
        std::vector<uint8_t> earlier = TeslaKeyChain::derive_earlier(d.key, d.interval_index, uint32_t(j));
        install_key(uint32_t(j), earlier);
        wipe_bytes(earlier);
      }
      verify_interval(uint32_t(j));
    }
  }

  // End of stream: report gaps among 0..highest_step_seen, and whether everything verified.
  TeslaWatchResult finish() {
    if (saw_step_ || !result_.released.empty()) {
      const uint64_t last = saw_step_ ? highest_step_seen_ : 0;
      const uint64_t end = saw_step_ ? last + 1 : 0;
      std::vector<uint8_t> have;
      uint64_t max_rel = 0;
      bool any_rel = false;
      for (const auto& r : result_.released) {
        if (have.size() <= r.step_index) have.resize(r.step_index + 1, 0);
        have[r.step_index] = 1;
        if (!any_rel || r.step_index > max_rel) max_rel = r.step_index;
        any_rel = true;
      }
      for (uint64_t s = 0; s < end; ++s) {
        const bool released = s < have.size() && have[s];
        if (!released) {
          // A step at or after failed_at is a rejected step, not a gap to be "filled". Still record a gap
          // only for missing indices that were never rejected and never arrived? Brief: missing steps are gaps.
          // Any expected index that was not released is a gap, including flipped packets (they existed but
          // failed). That's a bit harsh for fail-at-and-after. Report gaps for indices that were never released
          // and are before failed_at, plus any missing arrivals before the last seen step if they weren't failed.
          if (s < result_.failed_at) {
            result_.gaps.push_back(s);
            flag(TeslaFlag::Gap, s, uint32_t(s / sch_.I));
          }
        }
      }
    }
    result_.all_verified = result_.authentic && result_.gaps.empty() && result_.failed_at == ~uint64_t(0) &&
                           !result_.released.empty();
    // Early-disclosure is a flag, not a silent heal, but already-buffered honest packets may still verify.
    // all_verified requires no flags at all except we should not demand zero EarlyDisclosure if... actually
    // an early disclosure is an attack, so all_verified should be false if any flags exist.
    if (!result_.flags.empty()) result_.all_verified = false;
    return result_;
  }

  const TeslaWatchResult& snapshot() const { return result_; }

 private:
  struct Buffered {
    uint64_t arrival = 0;
    TeslaStepPacket packet;
  };

  bool key_known(uint32_t i) const { return i < keys_known_.size() && keys_known_[i]; }

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

  void flag(TeslaFlag f, uint64_t step, uint32_t interval) {
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
  std::vector<uint8_t> commitment_;
  TeslaSchedule sch_;
  uint64_t arrival_ = 0;
  uint64_t highest_step_seen_ = 0;
  bool saw_step_ = false;
  bool steps_ended_ = false;
  bool stopped_ = false;
  std::vector<std::vector<uint8_t>> keys_;
  std::vector<uint8_t> keys_known_;
  std::map<uint32_t, std::vector<Buffered>> buffer_;
  TeslaWatchResult result_;
};

}  // namespace pchie
