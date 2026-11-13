// Bolt PCHIE - ratcheted seal chain, chain generation "pchie.seal.v2-ratchet". Concrete final.
//
// This is a NEW chain generation that sits beside Sha256HashChain. It never re-seals a v1 chain, and
// nothing in an existing run uses it unless a caller asks for it through make_hash_chain()
// (ChainGeneration.h), where the default stays v1.
//
// Derivation, in the manner of the symmetric-key chain in Signal's Double Ratchet specification:
//   genesis   G     = "pchie.ratchet.v2.genesis" | lp(generation) | weights_sha256 | lp(run_id) | lp(harness_version)
//             ck_0  = HKDF-Extract(salt = "pchie.ratchet.v2", ikm = G)
//             head_0= HMAC(ck_0, 0x00 | "genesis")
//   step n    mk_n  = HMAC(ck_{n-1}, 0x01 | "seal")                                   step sealing key
//             seal_n= HMAC(mk_n, "pchie.ratchet.v2.step" | head_{n-1} | u64be(n) | canonical_bytes(record_n))
//             ck_n  = HMAC(ck_{n-1}, 0x02 | seal_n)                                   advance
//             head_n= seal_n; ck_{n-1} and mk_n are wiped
//   re-root   C     = SHA256("pchie.ratchet.v2.epoch" | u32be(epoch) | parent_head | u64be(parent_length)
//                            | lp(event) | lp(epoch_run_id) | detail_sha256)
//             ck    = HKDF-Extract(salt = parent_head, ikm = C);  head = HMAC(ck, 0x03 | C); length += 1
// lp(x) = u32be(len(x)) | x. A later chain state cannot be run backward to an earlier one: every step is
// HMAC, so going back means inverting SHA-256.
//
// Genesis modes: RECOMPUTABLE is the default and preferred mode. The genesis inputs
// are public, so anyone holding them and the step records can recompute the whole chain; stopping a private
// end-to-end recompute is the anchor log's job. An optional SECRET GENESIS SALT mode is supported and off by
// default: generation "pchie.seal.v2-ratchet-salted", ck_0 = HKDF-Extract(salt = secret, ikm = "pchie.ratchet.v2.secret-genesis" | G).
// Only a holder of the salt can re-derive such a chain; a public salt commitment lets them check they hold the right one.
//
// Canonical order law: only the coordinator appends, in canonical step order, never worker completion order.
// Resume and fork: restore_head() detaches the chain; append() then refuses until reroot() opens a new epoch,
// so a resumed or forked line never continues as if it were the same straight line.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "pchie/ChainRecord.h"
#include "pchie/HmacSha256.h"
#include "pchie/IHashChain.h"
#include "pchie/Sha256.h"

namespace pchie {

inline constexpr const char* kRatchetGenerationId = "pchie.seal.v2-ratchet";
inline constexpr const char* kRatchetSaltedGenerationId = "pchie.seal.v2-ratchet-salted";
inline bool is_ratchet_generation(const std::string& g) { return g == kRatchetGenerationId || g == kRatchetSaltedGenerationId; }

namespace ratchet_detail {
inline void put_str(std::vector<uint8_t>& b, const std::string& s) { b.insert(b.end(), s.begin(), s.end()); }
inline void put_u32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 3; i >= 0; --i) b.push_back(uint8_t(v >> (8 * i))); }
inline void put_u64(std::vector<uint8_t>& b, uint64_t v) { for (int i = 7; i >= 0; --i) b.push_back(uint8_t(v >> (8 * i))); }
inline void put_lp(std::vector<uint8_t>& b, const std::string& s) { put_u32(b, uint32_t(s.size())); put_str(b, s); }
inline void put_bytes(std::vector<uint8_t>& b, const std::vector<uint8_t>& v) { b.insert(b.end(), v.begin(), v.end()); }
inline std::vector<uint8_t> tagged(uint8_t tag, const char* label) { std::vector<uint8_t> b{tag}; put_str(b, label); return b; }
}  // namespace ratchet_detail

struct RatchetGenesis {
  std::vector<uint8_t> weights_sha256;  // 32 bytes
  std::string run_id;
  std::string harness_version;
  std::string generation = kRatchetGenerationId;
  std::vector<uint8_t> secret_salt;  // empty = recomputable (default). Never written to a chain file or log.

  // Switch a genesis into the secret-salt mode. The salt must be at least 16 random bytes.
  static RatchetGenesis with_secret_salt(RatchetGenesis g, std::vector<uint8_t> salt) {
    g.secret_salt = std::move(salt); g.generation = kRatchetSaltedGenerationId; return g;
  }
  bool salted() const { return !secret_salt.empty(); }
  // Public commitment to the salt, so a salt holder can check they hold the right one.
  std::vector<uint8_t> salt_commitment() const {
    std::vector<uint8_t> b; ratchet_detail::put_str(b, "pchie.ratchet.v2.salt-commitment"); ratchet_detail::put_bytes(b, secret_salt);
    return Sha256::hash(b.data(), b.size());
  }

  std::vector<uint8_t> bytes() const {
    using namespace ratchet_detail;
    if (weights_sha256.size() != 32) throw std::invalid_argument("ratchet genesis: weights_sha256 must be 32 bytes");
    if (run_id.empty()) throw std::invalid_argument("ratchet genesis: run_id is empty");
    if (salted() != (generation == kRatchetSaltedGenerationId)) throw std::invalid_argument("ratchet genesis: the salted generation and a secret salt go together");
    if (salted() && secret_salt.size() < 16) throw std::invalid_argument("ratchet genesis: secret salt must be at least 16 bytes");
    std::vector<uint8_t> g; put_str(g, "pchie.ratchet.v2.genesis"); put_lp(g, generation); put_bytes(g, weights_sha256);
    put_lp(g, run_id); put_lp(g, harness_version); return g;
  }
};

class RatchetHashChain final : public IHashChain {
 public:
  static constexpr size_t kHeadBytes = Sha256::kDigestBytes;

  // ---- pure derivation steps (the chain and any verifier use exactly these) ----
  static std::vector<uint8_t> genesis_state(const RatchetGenesis& g) {
    if (g.salted()) {  // optional secret-salt mode
      std::vector<uint8_t> ikm; ratchet_detail::put_str(ikm, "pchie.ratchet.v2.secret-genesis"); ratchet_detail::put_bytes(ikm, g.bytes());
      return hkdf_extract_sha256(g.secret_salt, ikm);
    }
    const std::vector<uint8_t> salt{'p','c','h','i','e','.','r','a','t','c','h','e','t','.','v','2'};  // recomputable (default)
    return hkdf_extract_sha256(salt, g.bytes());
  }
  static std::vector<uint8_t> genesis_head(const std::vector<uint8_t>& ck0) { return hmac_sha256(ck0, ratchet_detail::tagged(0x00, "genesis")); }
  static std::vector<uint8_t> step_key(const std::vector<uint8_t>& ck) { return hmac_sha256(ck, ratchet_detail::tagged(0x01, "seal")); }
  static std::vector<uint8_t> seal(const std::vector<uint8_t>& mk, const std::vector<uint8_t>& head_prev, uint64_t n, const ChainRecord& r) {
    using namespace ratchet_detail;
    std::vector<uint8_t> m; put_str(m, "pchie.ratchet.v2.step"); put_bytes(m, head_prev); put_u64(m, n); put_bytes(m, r.canonical_bytes());
    return hmac_sha256(mk, m);
  }
  static std::vector<uint8_t> advance(const std::vector<uint8_t>& ck, const std::vector<uint8_t>& seal_n) {
    std::vector<uint8_t> m{0x02}; ratchet_detail::put_bytes(m, seal_n); return hmac_sha256(ck, m);
  }
  static std::vector<uint8_t> epoch_commitment(uint32_t epoch, const std::vector<uint8_t>& parent_head, uint64_t parent_length,
                                               const std::string& event, const std::string& epoch_run_id, const std::vector<uint8_t>& detail_sha256) {
    using namespace ratchet_detail;
    if (detail_sha256.size() != 32) throw std::invalid_argument("ratchet re-root: detail_sha256 must be 32 bytes");
    std::vector<uint8_t> c; put_str(c, "pchie.ratchet.v2.epoch"); put_u32(c, epoch); put_bytes(c, parent_head); put_u64(c, parent_length);
    put_lp(c, event); put_lp(c, epoch_run_id); put_bytes(c, detail_sha256);
    return Sha256::hash(c.data(), c.size());
  }
  static std::vector<uint8_t> reroot_state(const std::vector<uint8_t>& parent_head, const std::vector<uint8_t>& commitment) { return hkdf_extract_sha256(parent_head, commitment); }
  static std::vector<uint8_t> reroot_head(const std::vector<uint8_t>& ck, const std::vector<uint8_t>& commitment) {
    std::vector<uint8_t> m{0x03}; ratchet_detail::put_bytes(m, commitment); return hmac_sha256(ck, m);
  }

  explicit RatchetHashChain(RatchetGenesis g) : genesis_(std::move(g)) { reset(); }
  ~RatchetHashChain() override { wipe_bytes(ck_); wipe_bytes(genesis_.secret_salt); }
  RatchetHashChain(const RatchetHashChain&) = delete;
  RatchetHashChain& operator=(const RatchetHashChain&) = delete;

  void reset() override {
    wipe_bytes(ck_);
    ck_ = genesis_state(genesis_);
    head_ = genesis_head(ck_);
    length_ = 0;
    epoch_ = 0;
    const std::vector<uint8_t> gb = genesis_.bytes();
    epoch_commitment_ = Sha256::hash(gb.data(), gb.size());
    epoch_run_id_ = genesis_.run_id;
    detached_ = false;
  }

  void append(const ChainRecord& record) override {
    if (detached_) throw std::logic_error("ratchet chain: restored head must be re-rooted (reroot) before the next append");
    std::vector<uint8_t> mk = step_key(ck_);
    std::vector<uint8_t> s = seal(mk, head_, uint64_t(length_) + 1, record);
    std::vector<uint8_t> next = advance(ck_, s);
    wipe_bytes(mk);
    wipe_bytes(ck_);
    ck_ = std::move(next);
    head_ = std::move(s);
    ++length_;
  }

  std::vector<uint8_t> head() const override { return head_; }
  size_t length() const override { return length_; }

  // Snapshot/rollback: the chain state cannot be rebuilt from a head (that is the point of the ratchet),
  // so a restored chain is detached until reroot() opens a new epoch from this head.
  void restore_head(const std::vector<uint8_t>& head, size_t length) override {
    wipe_bytes(ck_);
    head_ = head;
    if (head_.size() != kHeadBytes) head_.resize(kHeadBytes, 0);
    length_ = length;
    detached_ = true;
  }

  // Open a new epoch (fork, resume, context reset) from the current head. epoch_run_id names the run the
  // new epoch belongs to (a fork child's id, or the same id on resume). detail_sha256 commits to what
  // changed: branch identity, resume point, and any changed weights or context hash.
  void reroot(const std::string& event, const std::string& epoch_run_id, const std::vector<uint8_t>& detail_sha256) {
    if (event.empty() || epoch_run_id.empty()) throw std::invalid_argument("ratchet re-root: event and epoch_run_id are required");
    const uint32_t e = epoch_ + 1;
    std::vector<uint8_t> c = epoch_commitment(e, head_, uint64_t(length_), event, epoch_run_id, detail_sha256);
    wipe_bytes(ck_);
    ck_ = reroot_state(head_, c);
    head_ = reroot_head(ck_, c);
    ++length_;
    epoch_ = e;
    epoch_commitment_ = std::move(c);
    epoch_run_id_ = epoch_run_id;
    detached_ = false;
  }

  uint32_t epoch() const { return epoch_; }
  const std::vector<uint8_t>& epoch_commitment() const { return epoch_commitment_; }
  const std::string& epoch_run_id() const { return epoch_run_id_; }
  bool detached() const { return detached_; }
  const RatchetGenesis& genesis() const { return genesis_; }

 private:
  RatchetGenesis genesis_;
  std::vector<uint8_t> ck_;
  std::vector<uint8_t> head_;
  size_t length_ = 0;
  uint32_t epoch_ = 0;
  std::vector<uint8_t> epoch_commitment_;
  std::string epoch_run_id_;
  bool detached_ = false;
};

}  // namespace pchie
