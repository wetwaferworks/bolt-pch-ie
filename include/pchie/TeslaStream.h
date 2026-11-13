// Bolt PCHIE - TESLA-manner step packets, disclosure schedule, and a text stream format.
// Maze drivers construct a TeslaSender through StreamAuthSession, on by default since 2026-11-13.
#pragma once
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "pchie/HmacSha256.h"
#include "pchie/Sha256.h"
#include "pchie/TeslaKeyChain.h"

namespace pchie {

inline void tesla_put_u32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 3; i >= 0; --i) b.push_back(uint8_t(v >> (8 * i))); }
inline void tesla_put_u64(std::vector<uint8_t>& b, uint64_t v) { for (int i = 7; i >= 0; --i) b.push_back(uint8_t(v >> (8 * i))); }
inline void tesla_put_str(std::vector<uint8_t>& b, const char* s) { while (*s) b.push_back(uint8_t(*s++)); }
inline void tesla_put_bytes(std::vector<uint8_t>& b, const std::vector<uint8_t>& v) { b.insert(b.end(), v.begin(), v.end()); }

inline std::string tesla_hex(const std::vector<uint8_t>& b) {
  static const char* d = "0123456789abcdef";
  std::string s; s.resize(b.size() * 2);
  for (size_t i = 0; i < b.size(); ++i) { s[2 * i] = d[b[i] >> 4]; s[2 * i + 1] = d[b[i] & 15]; }
  return s;
}
inline std::vector<uint8_t> tesla_unhex(const std::string& s) {
  if (s.size() % 2) throw std::invalid_argument("TESLA hex: odd length");
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::invalid_argument("TESLA hex: bad digit");
  };
  std::vector<uint8_t> o(s.size() / 2);
  for (size_t i = 0; i < o.size(); ++i) o[i] = uint8_t((nib(s[2 * i]) << 4) | nib(s[2 * i + 1]));
  return o;
}

struct TeslaStepPacket {
  uint64_t step_index = 0;
  uint32_t interval_index = 0;
  std::vector<uint8_t> seal, payload_hash, mac;
};

struct TeslaDisclosure {
  uint32_t interval_index = 0;
  std::vector<uint8_t> key;
};

// Bytes the MAC is taken over: the whole step packet except the MAC itself.
inline std::vector<uint8_t> tesla_step_mac_input(const TeslaStepPacket& p) {
  std::vector<uint8_t> b;
  tesla_put_str(b, "pchie.tesla.v1.step");
  tesla_put_u64(b, p.step_index);
  tesla_put_u32(b, p.interval_index);
  tesla_put_u32(b, uint32_t(p.seal.size()));
  tesla_put_bytes(b, p.seal);
  tesla_put_u32(b, uint32_t(p.payload_hash.size()));
  tesla_put_bytes(b, p.payload_hash);
  return b;
}

inline std::vector<uint8_t> tesla_step_mac(const std::vector<uint8_t>& mac_key, const TeslaStepPacket& p) {
  return hmac_sha256(mac_key, tesla_step_mac_input(p));
}

// One line each, for the watcher program and for tests. No wall-clock fields.
inline std::string tesla_step_line(const TeslaStepPacket& p) {
  return "step|n=" + std::to_string(p.step_index) + "|i=" + std::to_string(p.interval_index) +
         "|seal=" + tesla_hex(p.seal) + "|payload=" + tesla_hex(p.payload_hash) + "|mac=" + tesla_hex(p.mac);
}
inline std::string tesla_disclose_line(const TeslaDisclosure& d) {
  return "disclose|i=" + std::to_string(d.interval_index) + "|key=" + tesla_hex(d.key);
}

inline TeslaStepPacket tesla_parse_step_line(const std::string& line) {
  if (line.rfind("step|", 0) != 0) throw std::runtime_error("not a TESLA step line");
  TeslaStepPacket p;
  std::stringstream ss(line.substr(5));
  std::string kv;
  while (std::getline(ss, kv, '|')) {
    const auto e = kv.find('=');
    if (e == std::string::npos) throw std::runtime_error("bad TESLA step field");
    const std::string k = kv.substr(0, e), v = kv.substr(e + 1);
    if (k == "n") p.step_index = std::stoull(v);
    else if (k == "i") p.interval_index = uint32_t(std::stoul(v));
    else if (k == "seal") p.seal = tesla_unhex(v);
    else if (k == "payload") p.payload_hash = tesla_unhex(v);
    else if (k == "mac") p.mac = tesla_unhex(v);
    else throw std::runtime_error("unknown TESLA step field " + k);
  }
  if (tesla_step_line(p) != line) throw std::runtime_error("non-canonical TESLA step line");
  return p;
}
inline TeslaDisclosure tesla_parse_disclose_line(const std::string& line) {
  if (line.rfind("disclose|", 0) != 0) throw std::runtime_error("not a TESLA disclose line");
  TeslaDisclosure d;
  std::stringstream ss(line.substr(9));
  std::string kv;
  while (std::getline(ss, kv, '|')) {
    const auto e = kv.find('=');
    if (e == std::string::npos) throw std::runtime_error("bad TESLA disclose field");
    const std::string k = kv.substr(0, e), v = kv.substr(e + 1);
    if (k == "i") d.interval_index = uint32_t(std::stoul(v));
    else if (k == "key") d.key = tesla_unhex(v);
    else throw std::runtime_error("unknown TESLA disclose field " + k);
  }
  if (tesla_disclose_line(d) != line) throw std::runtime_error("non-canonical TESLA disclose line");
  return d;
}

inline std::string tesla_stream_header(const std::vector<uint8_t>& commitment, const TeslaSchedule& sch) {
  sch.check();
  return "pchie.tesla.v1.stream\ncommitment=" + tesla_hex(commitment) +
         "\nI=" + std::to_string(sch.I) + "\nd=" + std::to_string(sch.d) + "\n";
}

// Emits step packets in canonical step order, with interval keys disclosed on lag d, and trailing
// disclosures after the last step so the end of the stream can be verified. No wall clock.
class TeslaSender {
 public:
  TeslaSender(std::vector<uint8_t> seed, uint32_t max_intervals, TeslaSchedule sch = {})
      : chain_(std::move(seed), max_intervals, sch), sch_(sch), next_step_(0), last_interval_(0), closed_through_(~0u), finished_(false) {
    sch_.check();
  }

  const std::vector<uint8_t>& commitment() const { return chain_.commitment(); }
  TeslaSchedule schedule() const { return sch_; }
  uint64_t steps_emitted() const { return next_step_; }

  TeslaStepPacket emit_step(uint64_t step_index, const std::vector<uint8_t>& seal, const std::vector<uint8_t>& payload_hash) {
    if (finished_) throw std::logic_error("TESLA sender: stream already finished");
    if (step_index != next_step_) throw std::logic_error("TESLA sender: steps must be emitted in canonical order");
    const uint32_t i = uint32_t(step_index / sch_.I);
    if (i >= chain_.size()) throw std::out_of_range("TESLA sender: step needs more intervals than the chain was built with");
    TeslaStepPacket p;
    p.step_index = step_index;
    p.interval_index = i;
    p.seal = seal;
    p.payload_hash = payload_hash;
    std::vector<uint8_t> mk = chain_.mac_key(i);
    p.mac = tesla_step_mac(mk, p);
    wipe_bytes(mk);
    last_interval_ = i;
    ++next_step_;
    // An interval closes when its last slot is filled.
    if ((step_index % sch_.I) == (sch_.I - 1)) queue_on_schedule_disclosure(i);
    return p;
  }

  // Disclosures queued by the most recent emit_step or by finish(). Caller takes them in order.
  std::vector<TeslaDisclosure> take_disclosures() { auto out = std::move(pending_); pending_.clear(); return out; }

  // Close a short last interval if needed, then disclose the last d interval keys. Call once, after the last step.
  std::vector<TeslaDisclosure> finish() {
    if (finished_) throw std::logic_error("TESLA sender: finish already called");
    if (next_step_ == 0) throw std::logic_error("TESLA sender: no steps to finish");
    finished_ = true;
    const uint32_t L = last_interval_;
    if (closed_through_ != L) queue_on_schedule_disclosure(L);
    // Trailing: the last d keys, K_{L-d+1} .. K_L, not yet disclosed.
    const uint32_t first_trail = (L + 1 >= sch_.d) ? (L + 1 - sch_.d) : 0;
    for (uint32_t j = first_trail; j <= L; ++j) queue_disclosure(j);
    return take_disclosures();
  }

 private:
  void queue_on_schedule_disclosure(uint32_t closed_interval) {
    closed_through_ = closed_interval;
    if (closed_interval >= sch_.d) queue_disclosure(closed_interval - sch_.d);
  }
  void queue_disclosure(uint32_t j) {
    if (j >= chain_.size()) throw std::out_of_range("TESLA sender: disclosure index out of range");
    if (disclosed_.size() <= j) disclosed_.resize(j + 1, 0);
    if (disclosed_[j]) return;  // already queued (e.g. finish after an exact interval close)
    TeslaDisclosure d;
    d.interval_index = j;
    d.key = chain_.key(j);
    pending_.push_back(std::move(d));
    disclosed_[j] = 1;
  }

  TeslaKeyChain chain_;
  TeslaSchedule sch_;
  uint64_t next_step_;
  uint32_t last_interval_;
  uint32_t closed_through_;
  bool finished_;
  std::vector<TeslaDisclosure> pending_;
  std::vector<uint8_t> disclosed_;
};


// Text stream writer used by the watcher program and the tests. The line "end-steps" sits after
// the last step packet (and any on-schedule disclosures that closed with it) and before trailing
// disclosures, so the watcher does not treat those last keys as early.
class TeslaStreamWriter {
 public:
  TeslaStreamWriter(const std::vector<uint8_t>& commitment, const TeslaSchedule& sch) { text_ = tesla_stream_header(commitment, sch); }
  void add_step(const TeslaStepPacket& p) { text_ += tesla_step_line(p); text_ += '\n'; }
  void add_disclosure(const TeslaDisclosure& d) { text_ += tesla_disclose_line(d); text_ += '\n'; }
  void end_steps() { text_ += "end-steps\n"; }
  const std::string& text() const { return text_; }
 private:
  std::string text_;
};

}  // namespace pchie
