// Bolt PCHIE - TESLA stream authentication and attestation flags and session.
//
// ON by default since 13 November 2026. --tesla off and --attest off turn
// each one off for testing. This layer wraps emission and evidence only: it must not change a
// seal, a sim hash, or any official chain head. Without a TPM quote, attestation records
// Unattested and never fakes Attested.
#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "pchie/AttestationVerdict.h"
#include "pchie/ChainRecord.h"
#include "pchie/IHashChain.h"
#include "pchie/Sha256.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/TeslaStream.h"

namespace pchie {

struct StreamAuthFlags {
  bool tesla = true;    // ON by default
  bool attest = true;   // ON by default; verdict is computed even when no TPM is present
};

inline StreamAuthFlags stream_auth_off() {
  StreamAuthFlags f;
  f.tesla = false;
  f.attest = false;
  return f;
}

struct StreamAuthParse {
  StreamAuthFlags flags;
  bool ok = true;
  std::string err;
};

// Reads --tesla on|off and --attest on|off. Other arguments are ignored.
inline StreamAuthParse parse_stream_auth_flags(int argc, char** argv) {
  StreamAuthParse p;
  for (int i = 1; i < argc; ++i) {
    if (!argv[i]) continue;
    const bool is_tesla = std::strcmp(argv[i], "--tesla") == 0;
    const bool is_attest = std::strcmp(argv[i], "--attest") == 0;
    if (!is_tesla && !is_attest) continue;
    if (i + 1 >= argc) {
      p.ok = false;
      p.err = is_tesla ? "--tesla takes on or off" : "--attest takes on or off";
      return p;
    }
    const char* v = argv[++i];
    const bool on = std::strcmp(v, "on") == 0;
    const bool off = std::strcmp(v, "off") == 0;
    if (!on && !off) {
      p.ok = false;
      p.err = is_tesla ? "--tesla takes on or off" : "--attest takes on or off";
      return p;
    }
    if (is_tesla) p.flags.tesla = on;
    else p.flags.attest = on;
  }
  return p;
}

inline std::vector<uint8_t> tesla_derived_seed(uint64_t maze_seed) {
  std::vector<uint8_t> b;
  tesla_put_str(b, "pchie.tesla.v1.derived-seed");
  tesla_put_u64(b, maze_seed);
  return Sha256::hash(b.data(), b.size());
}

// Side session: MACs sealed steps after the official append has already succeeded, and records
// an attestation verdict. Nothing here is written back into the official chain.
class StreamAuthSession {
 public:
  explicit StreamAuthSession(StreamAuthFlags flags = {}) : flags_(flags) {}

  StreamAuthFlags flags() const { return flags_; }

  void set_secret_seed(std::vector<uint8_t> seed) { secret_seed_ = std::move(seed); }

  void start_tesla(uint64_t maze_seed, uint32_t max_intervals = 4096) {
    if (!flags_.tesla) return;
    tesla_ok_ = false;
    try {
      std::vector<uint8_t> seed = secret_seed_;
      if (seed.size() < 32) seed = tesla_derived_seed(maze_seed);
      sender_ = std::make_unique<TeslaSender>(std::move(seed), max_intervals, TeslaSchedule{});
      tesla_ok_ = true;
    } catch (...) {
      tesla_ok_ = false;
      tesla_err_ = "TESLA sender failed to start";
    }
  }

  void on_append(const std::vector<uint8_t>& seal, const ChainRecord& record) {
    if (!flags_.tesla || !tesla_ok_ || !sender_ || tesla_stopped_) return;
    try {
      const std::vector<uint8_t> rec = record.canonical_bytes();
      const std::vector<uint8_t> payload = Sha256::hash(rec.data(), rec.size());
      const uint64_t n = sender_->steps_emitted();
      TeslaStepPacket p = sender_->emit_step(n, seal, payload);
      packets_ += tesla_step_line(p);
      packets_ += '\n';
      for (const auto& d : sender_->take_disclosures()) {
        packets_ += tesla_disclose_line(d);
        packets_ += '\n';
      }
    } catch (...) {
      tesla_stopped_ = true;
      tesla_err_ = "TESLA emit stopped (does not affect the official chain)";
    }
  }

  // For drivers that have a list of committed actions rather than an official IHashChain.
  void on_action_seals(const std::vector<int32_t>& actions) {
    if (!flags_.tesla || !tesla_ok_ || !sender_ || tesla_stopped_) return;
    try {
      Sha256HashChain side;
      for (size_t i = 0; i < actions.size(); ++i) {
        std::vector<uint8_t> payload(4);
        const uint32_t a = uint32_t(actions[i]);
        payload[0] = uint8_t(a);
        payload[1] = uint8_t(a >> 8);
        payload[2] = uint8_t(a >> 16);
        payload[3] = uint8_t(a >> 24);
        side.append(ChainRecord{ChainRecordKind::Custom, payload});
        on_append(side.head(), ChainRecord{ChainRecordKind::Custom, payload});
      }
    } catch (...) {
      tesla_stopped_ = true;
      tesla_err_ = "TESLA action stream stopped (does not affect the official chain)";
    }
  }

  void on_restore() {
    // Official v1 runs do not restore the official chain. If it happens, stop TESLA rather than
    // invent a stream that pretends the rolled-back records were never sealed.
    tesla_stopped_ = true;
    if (tesla_err_.empty()) tesla_err_ = "TESLA stopped after a chain restore";
  }

  void finish_tesla() {
    if (!flags_.tesla || !tesla_ok_ || !sender_ || tesla_finished_) return;
    tesla_finished_ = true;
    if (sender_->steps_emitted() == 0) return;
    try {
      packets_ += "end-steps\n";
      for (const auto& d : sender_->finish()) {
        packets_ += tesla_disclose_line(d);
        packets_ += '\n';
      }
    } catch (...) {
      tesla_err_ = "TESLA finish failed (does not affect the official chain)";
    }
  }

  // No TPM is consulted here. Unit tests and any run without a quote get Unattested.
  void record_unattested_absent() {
    if (!flags_.attest) return;
    attest_computed_ = true;
    SimulatedQuote q;
    q.present = false;
    MeasurementSet m;
    m.binary_sha256.assign(32, 0);
    m.weights_sha256.assign(32, 0);
    m.config_sha256.assign(32, 0);
    m.tesla_commitment = (sender_ ? sender_->commitment() : std::vector<uint8_t>(32, 0));
    if (m.tesla_commitment.size() != 32) m.tesla_commitment.assign(32, 0);
    KnownGoodRegistry empty;
    const std::vector<uint8_t> ak(32, 0);
    const std::vector<uint8_t> nonce(32, 0);
    appraisal_ = appraise_attestation(q, nonce, m, empty, ak);
  }

  void set_appraisal(AttestationAppraisal a) {
    if (!flags_.attest) return;
    attest_computed_ = true;
    // Never accept a caller-supplied Attested when it was not actually appraised against a quote.
    appraisal_ = std::move(a);
  }

  std::vector<uint8_t> tesla_commitment() const { return sender_ ? sender_->commitment() : std::vector<uint8_t>(); }
  std::vector<uint8_t> tesla_commitment_id_bytes() const {
    const auto c = tesla_commitment();
    if (c.size() != 32) return {};
    return tesla_commitment_id(c);
  }
  uint64_t tesla_steps() const { return sender_ ? sender_->steps_emitted() : 0; }
  const std::string& tesla_stream_text() const { return packets_; }
  const std::string& tesla_err() const { return tesla_err_; }
  bool tesla_ok() const { return tesla_ok_ && tesla_err_.empty(); }
  bool attest_computed() const { return attest_computed_; }
  AttestationAppraisal appraisal() const { return appraisal_; }

  template <typename Result>
  void publish(Result& out) {
    finish_tesla();
    if (flags_.attest && !attest_computed_) record_unattested_absent();
    out.tesla_enabled = flags_.tesla;
    out.attest_enabled = flags_.attest;
    out.tesla_commitment = tesla_commitment();
    out.tesla_commitment_id = tesla_commitment_id_bytes();
    out.tesla_steps = tesla_steps();
    out.tesla_err = tesla_err_;
    out.attest_computed = attest_computed_;
    if (attest_computed_) {
      out.attest_status = appraisal_.status;
      out.attest_reason = appraisal_.reason;
    }
  }

 private:
  StreamAuthFlags flags_;
  std::vector<uint8_t> secret_seed_;
  std::unique_ptr<TeslaSender> sender_;
  std::string packets_;
  std::string tesla_err_;
  bool tesla_ok_ = false;
  bool tesla_stopped_ = false;
  bool tesla_finished_ = false;
  bool attest_computed_ = false;
  AttestationAppraisal appraisal_{};
};

class TeslaWrappingChain final : public IHashChain {
 public:
  void bind(IHashChain* inner, StreamAuthSession* session) {
    inner_ = inner;
    session_ = session;
  }
  void reset() override {
    if (inner_) inner_->reset();
  }
  void append(const ChainRecord& record) override {
    if (!inner_) return;
    inner_->append(record);
    if (session_) session_->on_append(inner_->head(), record);
  }
  std::vector<uint8_t> head() const override { return inner_ ? inner_->head() : std::vector<uint8_t>(); }
  size_t length() const override { return inner_ ? inner_->length() : 0; }
  void restore_head(const std::vector<uint8_t>& head, size_t length) override {
    if (inner_) inner_->restore_head(head, length);
    if (session_) session_->on_restore();
  }
  IHashChain* official_chain() override { return inner_ ? inner_->official_chain() : nullptr; }
  const IHashChain* official_chain() const override { return inner_ ? inner_->official_chain() : nullptr; }
  IHashChain* inner() const { return inner_; }

 private:
  IHashChain* inner_ = nullptr;
  StreamAuthSession* session_ = nullptr;
};

// Result fields shared by B1 and Stage B. Placed on the result by StreamAuthSession::publish.
struct StreamAuthResultFields {
  bool tesla_enabled = true;
  bool attest_enabled = true;
  std::vector<uint8_t> tesla_commitment;
  std::vector<uint8_t> tesla_commitment_id;
  uint64_t tesla_steps = 0;
  std::string tesla_err;
  bool attest_computed = false;
  AttestationStatus attest_status = AttestationStatus::Unattested;
  std::string attest_reason;
};

}  // namespace pchie
