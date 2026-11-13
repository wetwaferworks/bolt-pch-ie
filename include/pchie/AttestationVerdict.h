// Bolt PCHIE - attestation appraisal in the manner of RFC 9334 (RATS), against a simulated quote and
// a known-good registry. No TPM is touched here. The real measure / extend / quote round trip is a separate bench check on PCR 23.
// In-process appraisal never talks to the TPM. Without a quote the verdict is Unattested, never Attested.
//
// Verdicts are exactly three: Attested, Unattested, Rejected. They sit beside a run's Official /
// Scratch / Provisional label and never change that label. Chain validity and attestation are
// separate axes and are never merged into one word.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "pchie/HmacSha256.h"
#include "pchie/Sha256.h"
#include "pchie/TeslaStream.h"

namespace pchie {

enum class AttestationStatus { Attested, Unattested, Rejected };

inline const char* attestation_status_name(AttestationStatus s) {
  switch (s) {
    case AttestationStatus::Attested: return "Attested";
    case AttestationStatus::Unattested: return "Unattested";
    case AttestationStatus::Rejected: return "Rejected";
  }
  return "Unknown";
}

struct MeasurementSet {
  std::vector<uint8_t> binary_sha256;      // 32
  std::string harness_version;
  std::vector<uint8_t> weights_sha256;     // 32
  std::vector<uint8_t> config_sha256;      // 32
  std::vector<uint8_t> tesla_commitment;   // 32
};

inline std::vector<uint8_t> measurement_set_bytes(const MeasurementSet& m) {
  auto put32 = [](std::vector<uint8_t>& b, const std::vector<uint8_t>& v) {
    if (v.size() != 32) throw std::invalid_argument("measurement set: hash field must be 32 bytes");
    b.insert(b.end(), v.begin(), v.end());
  };
  std::vector<uint8_t> b;
  tesla_put_str(b, "pchie.attest.sim.v1.measurements");
  put32(b, m.binary_sha256);
  tesla_put_u32(b, uint32_t(m.harness_version.size()));
  tesla_put_str(b, m.harness_version.c_str());
  put32(b, m.weights_sha256);
  put32(b, m.config_sha256);
  put32(b, m.tesla_commitment);
  return b;
}
inline std::vector<uint8_t> measurement_set_hash(const MeasurementSet& m) {
  const std::vector<uint8_t> b = measurement_set_bytes(m);
  return Sha256::hash(b.data(), b.size());
}

struct KnownGoodEntry {
  std::vector<uint8_t> binary_sha256, weights_sha256, config_sha256;
  std::string harness_version;
};
// Append-only in spirit. Tests push entries; nothing here edits or deletes one.
struct KnownGoodRegistry {
  std::vector<KnownGoodEntry> entries;
  bool contains(const MeasurementSet& m) const {
    for (const auto& e : entries) {
      if (e.binary_sha256 == m.binary_sha256 && e.weights_sha256 == m.weights_sha256 &&
          e.config_sha256 == m.config_sha256 && e.harness_version == m.harness_version)
        return true;
    }
    return false;
  }
};

// A stand-in for a TPM quote, signed with a test attestation key by HMAC. Not a real TPM object.
struct SimulatedQuote {
  bool present = true;  // false means no TPM / quote was available
  std::string platform_id;
  std::vector<uint8_t> nonce;
  std::vector<uint8_t> measurement_set_hash;
  std::vector<uint8_t> signature;
};

inline std::vector<uint8_t> simulated_quote_tbs(const SimulatedQuote& q) {
  std::vector<uint8_t> b;
  tesla_put_str(b, "pchie.attest.sim.v1.quote");
  tesla_put_u32(b, uint32_t(q.platform_id.size()));
  tesla_put_str(b, q.platform_id.c_str());
  tesla_put_u32(b, uint32_t(q.nonce.size()));
  tesla_put_bytes(b, q.nonce);
  tesla_put_u32(b, uint32_t(q.measurement_set_hash.size()));
  tesla_put_bytes(b, q.measurement_set_hash);
  return b;
}
inline std::vector<uint8_t> sign_simulated_quote(SimulatedQuote q, const std::vector<uint8_t>& test_ak) {
  q.signature = hmac_sha256(test_ak, simulated_quote_tbs(q));
  return q.signature;
}
inline SimulatedQuote make_simulated_quote(const MeasurementSet& m, const std::vector<uint8_t>& nonce,
                                           const std::string& platform_id, const std::vector<uint8_t>& test_ak) {
  SimulatedQuote q;
  q.present = true;
  q.platform_id = platform_id;
  q.nonce = nonce;
  q.measurement_set_hash = measurement_set_hash(m);
  q.signature = sign_simulated_quote(q, test_ak);
  return q;
}
inline std::vector<uint8_t> simulated_quote_hash(const SimulatedQuote& q) {
  std::vector<uint8_t> b = simulated_quote_tbs(q);
  tesla_put_bytes(b, q.signature);
  return Sha256::hash(b.data(), b.size());
}

struct AttestationAppraisal {
  AttestationStatus status = AttestationStatus::Unattested;
  std::string reason;
};

// Appraise a (possibly missing) quote against the expected nonce, the claimed measurements, and the registry.
// Does not inspect or change any TPM state.
inline AttestationAppraisal appraise_attestation(const SimulatedQuote& quote, const std::vector<uint8_t>& expected_nonce,
                                                 const MeasurementSet& measurements, const KnownGoodRegistry& registry,
                                                 const std::vector<uint8_t>& test_ak) {
  AttestationAppraisal a;
  if (!quote.present) {
    a.status = AttestationStatus::Unattested;
    a.reason = "absent platform";
    return a;
  }
  if (quote.nonce != expected_nonce) {
    a.status = AttestationStatus::Rejected;
    a.reason = "nonce mismatch";
    return a;
  }
  const std::vector<uint8_t> want_sig = hmac_sha256(test_ak, simulated_quote_tbs(quote));
  if (want_sig != quote.signature) {
    a.status = AttestationStatus::Rejected;
    a.reason = "bad signature";
    return a;
  }
  const std::vector<uint8_t> msh = measurement_set_hash(measurements);
  if (msh != quote.measurement_set_hash) {
    a.status = AttestationStatus::Rejected;
    a.reason = "measurement set does not match the quote";
    return a;
  }
  if (!registry.contains(measurements)) {
    a.status = AttestationStatus::Rejected;
    a.reason = "measurement not in the known-good registry";
    return a;
  }
  a.status = AttestationStatus::Attested;
  a.reason = "quote valid, fresh, and every measurement matches the registry";
  return a;
}

}  // namespace pchie
