// PCHIE transparency anchoring. Hashes are provisional. Implements Elfie's brief "Transparency Anchoring for the PCHIE
// Hash Chain" (2026-11-12): anchor records (brief s3.1), a Merkle log in the manner of RFC 9162 with signed tree heads,
// inclusion and consistency proofs (s3.2), witness cosignatures after the Sigsum design (s3.3), a safety-number style
// fingerprint (s3.4), and verification of ratcheted chain files (s3.6). Additive side log: observes runs, never feeds them.
// In-house only: built on pchie::Sha256 and pchie/HmacSha256.h. Signatures are a stateful hash-based scheme modeled on
// RFC 8554 (LMS/LM-OTS, SHA-256, n=32, w=4, h=10); NOT claimed interoperable with RFC 8554.
// Labels are never upgraded by anchoring. Nothing here touches any public log or network.
#pragma once
#include "pchie/HmacSha256.h"
#include "pchie/OsRandom.h"
#include "pchie/RatchetHashChain.h"
#include "pchie/Sha256.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pchie { namespace anchor {

// Best-effort wipe of secret bytes (volatile writes the compiler may not drop).
inline void secure_zero(void* p, size_t n) { volatile unsigned char* v = static_cast<volatile unsigned char*>(p); for (size_t i = 0; i < n; ++i) v[i] = 0; }
using Bytes = std::vector<uint8_t>;
using H32 = std::array<uint8_t, 32>;

inline Bytes B(const std::string& s) { return Bytes(s.begin(), s.end()); }
inline H32 sha(const Bytes& b) { auto v = Sha256::hash(b.data(), b.size()); H32 h; std::memcpy(h.data(), v.data(), 32); return h; }
inline H32 sha(const std::string& s) { return sha(Bytes(s.begin(), s.end())); }
inline void put(Bytes& b, const H32& h) { b.insert(b.end(), h.begin(), h.end()); }
inline void put(Bytes& b, const std::string& s) { b.insert(b.end(), s.begin(), s.end()); }
inline void put32(Bytes& b, uint32_t v) { for (int i = 3; i >= 0; --i) b.push_back(uint8_t(v >> (8 * i))); }
inline void put16(Bytes& b, uint16_t v) { b.push_back(uint8_t(v >> 8)); b.push_back(uint8_t(v)); }
inline std::string hex(const uint8_t* p, size_t n) { static const char* d = "0123456789abcdef"; std::string s; for (size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; } return s; }
inline std::string hex(const H32& h) { return hex(h.data(), 32); }
inline std::string hex(const Bytes& b) { return hex(b.data(), b.size()); }
inline Bytes unhex(const std::string& s) { Bytes b; if (s.size() % 2) throw std::runtime_error("bad hex"); for (size_t i = 0; i < s.size(); i += 2) { auto v = [](char c) { if (c >= '0' && c <= '9') return c - '0'; if (c >= 'a' && c <= 'f') return c - 'a' + 10; if (c >= 'A' && c <= 'F') return c - 'A' + 10; throw std::runtime_error("bad hex"); }; b.push_back(uint8_t(v(s[i]) * 16 + v(s[i + 1]))); } return b; }
inline H32 unhex32(const std::string& s) { Bytes b = unhex(s); if (b.size() != 32) throw std::runtime_error("bad hash length"); H32 h; std::memcpy(h.data(), b.data(), 32); return h; }

// ---------------- HMAC-SHA256 (RFC 2104) and HKDF (RFC 5869): thin wrappers over pchie/HmacSha256.h ----------------
inline H32 to_h32(const Bytes& v) { if (v.size() != 32) throw std::runtime_error("expected 32 bytes"); H32 h; std::memcpy(h.data(), v.data(), 32); return h; }
inline H32 hmac(const Bytes& key, const Bytes& msg) { return to_h32(pchie::hmac_sha256(key, msg)); }
inline H32 hkdf_extract(const Bytes& salt, const Bytes& ikm) { return to_h32(pchie::hkdf_extract_sha256(salt, ikm)); }
inline Bytes hkdf_expand(const H32& prk, const Bytes& info, size_t len) { return pchie::hkdf_expand_sha256(Bytes(prk.begin(), prk.end()), info, len); }

// ---------------- Merkle tree, RFC 9162 s2.1 ----------------
inline H32 leaf_hash(const Bytes& d) { Bytes b{0x00}; b.insert(b.end(), d.begin(), d.end()); return sha(b); }
inline H32 node_hash(const H32& l, const H32& r) { Bytes b{0x01}; put(b, l); put(b, r); return sha(b); }
inline size_t split_k(size_t n) { size_t k = 1; while (k * 2 < n) k *= 2; return k; }  // largest power of two < n
inline H32 mth(const std::vector<H32>& L, size_t a, size_t b) {  // over leaf hashes L[a:b]
  const size_t n = b - a; if (n == 0) return sha(Bytes{}); if (n == 1) return L[a];
  const size_t k = split_k(n); return node_hash(mth(L, a, a + k), mth(L, a + k, b));
}
inline void path(const std::vector<H32>& L, size_t m, size_t a, size_t b, std::vector<H32>& out) {
  const size_t n = b - a; if (n <= 1) return; const size_t k = split_k(n);
  if (m < k) { path(L, m, a, a + k, out); out.push_back(mth(L, a + k, b)); } else { path(L, m - k, a + k, b, out); out.push_back(mth(L, a, a + k)); }
}
inline std::vector<H32> inclusion_proof(const std::vector<H32>& L, size_t m, size_t n) { std::vector<H32> o; path(L, m, 0, n, o); return o; }
inline void subproof(const std::vector<H32>& L, size_t m, size_t a, size_t b, bool known, std::vector<H32>& out) {
  const size_t n = b - a;
  if (m == n) { if (!known) out.push_back(mth(L, a, b)); return; }
  const size_t k = split_k(n);
  if (m <= k) { subproof(L, m, a, a + k, known, out); out.push_back(mth(L, a + k, b)); }
  else { subproof(L, m - k, a + k, b, false, out); out.push_back(mth(L, a, a + k)); }
}
inline std::vector<H32> consistency_proof(const std::vector<H32>& L, size_t m, size_t n) { std::vector<H32> o; if (m > 0 && m < n) subproof(L, m, 0, n, true, o); return o; }
// RFC 9162 s2.1.3.2
inline bool verify_inclusion(const H32& leaf, uint64_t index, uint64_t size, const std::vector<H32>& p, const H32& root) {
  if (index >= size) return false; uint64_t fn = index, sn = size - 1; H32 r = leaf;
  for (const auto& x : p) {
    if (sn == 0) return false;
    if ((fn & 1) || fn == sn) { r = node_hash(x, r); if (!(fn & 1)) while (!(fn & 1) && fn != 0) { fn >>= 1; sn >>= 1; } }
    else r = node_hash(r, x);
    fn >>= 1; sn >>= 1;
  }
  return sn == 0 && r == root;
}
// RFC 9162 s2.1.4.2
inline bool verify_consistency(uint64_t first, const H32& first_hash, uint64_t second, const H32& second_hash, std::vector<H32> p) {
  if (first == second) return p.empty() && first_hash == second_hash;
  if (first == 0 || first > second || p.empty()) return false;
  if ((first & (first - 1)) == 0) p.insert(p.begin(), first_hash);
  uint64_t fn = first - 1, sn = second - 1;
  while (fn & 1) { fn >>= 1; sn >>= 1; }
  H32 fr = p[0], sr = p[0];
  for (size_t i = 1; i < p.size(); ++i) {
    const H32& c = p[i]; if (sn == 0) return false;
    if ((fn & 1) || fn == sn) { fr = node_hash(c, fr); sr = node_hash(c, sr); if (!(fn & 1)) while (!(fn & 1) && fn != 0) { fn >>= 1; sn >>= 1; } }
    else sr = node_hash(sr, c);
    fn >>= 1; sn >>= 1;
  }
  return fr == first_hash && sr == second_hash && sn == 0;
}

// ---------------- Hash-based signature (modeled on RFC 8554 LMS/LM-OTS; SHA-256, n=32, w=4, p=67, ls=4, h=10) ----------------
struct Hbs {
  static constexpr int kH = 10, kP = 67, kW = 15; static constexpr uint32_t kLeaves = 1u << kH;
  static uint8_t coef(const Bytes& S, int i) { return uint8_t((S[i / 2] >> (4 * (1 - (i % 2)))) & 0xF); }
  static Bytes qsum(const H32& Q) { Bytes S(Q.begin(), Q.end()); uint32_t sum = 0; for (int i = 0; i < 64; ++i) sum += 15 - coef(S, i); sum <<= 4; S.push_back(uint8_t(sum >> 8)); S.push_back(uint8_t(sum)); return S; }
  static H32 chain(const Bytes& I, uint32_t q, uint16_t i, H32 x, int from, int to) {
    for (int j = from; j < to; ++j) { Bytes b(I); put32(b, q); put16(b, i); b.push_back(uint8_t(j)); put(b, x); x = sha(b); } return x;
  }
  static H32 msg_hash(const Bytes& I, uint32_t q, const H32& C, const Bytes& msg) { Bytes b(I); put32(b, q); put16(b, 0x8181); put(b, C); b.insert(b.end(), msg.begin(), msg.end()); return sha(b); }
  static H32 ots_pub_from(const Bytes& I, uint32_t q, const std::vector<H32>& z) { Bytes b(I); put32(b, q); put16(b, 0x8080); for (auto& h : z) put(b, h); return sha(b); }
  static H32 leaf(const Bytes& I, uint32_t q, const H32& K) { Bytes b(I); put32(b, kLeaves + q); put16(b, 0x8282); put(b, K); return sha(b); }
  static H32 inner(const Bytes& I, uint32_t r, const H32& l, const H32& rr) { Bytes b(I); put32(b, r); put16(b, 0x8383); put(b, l); put(b, rr); return sha(b); }
};
struct HbsKey {
  Bytes I; H32 seed{}; uint32_t next_q = 0; std::vector<H32> T;  // T[1..2^(h+1)-1] tree nodes (public)
  H32 x(uint32_t q, uint16_t i) const { Bytes b(I); put32(b, q); put16(b, i); b.push_back(0xFF); put(b, seed); const H32 r = sha(b); secure_zero(b.data(), b.size()); return r; }
  H32 ots_pub(uint32_t q) const { std::vector<H32> z(Hbs::kP); for (int i = 0; i < Hbs::kP; ++i) z[i] = Hbs::chain(I, q, uint16_t(i), x(q, uint16_t(i)), 0, Hbs::kW); return Hbs::ots_pub_from(I, q, z); }
  void build() { T.assign(2 * Hbs::kLeaves, H32{}); for (uint32_t q = 0; q < Hbs::kLeaves; ++q) T[Hbs::kLeaves + q] = Hbs::leaf(I, q, ots_pub(q)); for (uint32_t r = Hbs::kLeaves - 1; r >= 1; --r) T[r] = Hbs::inner(I, r, T[2 * r], T[2 * r + 1]); }
  Bytes pub() const { Bytes p(I); put(p, T[1]); return p; }  // 16 + 32 bytes
  // Key material comes from the OS random generator; key generation refuses if it is unavailable.
  static HbsKey generate(const std::string& label) {
    HbsKey k; Bytes ikm(64, 0); if (!pchie::os_random_bytes(ikm.data(), ikm.size())) throw std::runtime_error("no OS random generator for key generation");
    H32 prk = hkdf_extract(Bytes(), ikm); Bytes info(label.begin(), label.end());
    Bytes o = hkdf_expand(prk, info, 48); k.I.assign(o.begin(), o.begin() + 16); std::memcpy(k.seed.data(), o.data() + 16, 32); k.build();
    secure_zero(ikm.data(), ikm.size()); secure_zero(prk.data(), prk.size()); secure_zero(o.data(), o.size()); return k;
  }
  // Zero the secret seed once a key copy is no longer needed.
  void wipe() { secure_zero(seed.data(), seed.size()); }
  static HbsKey from_seed(const Bytes& I, const H32& seed, uint32_t next_q) { HbsKey k; k.I = I; k.seed = seed; k.next_q = next_q; k.build(); return k; }
  // stateful: every signature consumes one leaf; the caller must persist next_q BEFORE releasing the signature
  Bytes sign(const Bytes& msg) {
    if (next_q >= Hbs::kLeaves) throw std::runtime_error("HBS key exhausted (1024 signatures)");
    const uint32_t q = next_q++;
    Bytes cb(seed.begin(), seed.end()); put32(cb, q); put(cb, std::string("C")); const H32 C = sha(cb); secure_zero(cb.data(), cb.size());
    const Bytes S = Hbs::qsum(Hbs::msg_hash(I, q, C, msg));
    Bytes sig; put32(sig, q); put(sig, C);
    for (int i = 0; i < Hbs::kP; ++i) put(sig, Hbs::chain(I, q, uint16_t(i), x(q, uint16_t(i)), 0, Hbs::coef(S, i)));
    for (uint32_t r = Hbs::kLeaves + q; r > 1; r >>= 1) put(sig, T[r ^ 1]);
    return sig;
  }
};
inline bool hbs_verify(const Bytes& pub, const Bytes& msg, const Bytes& sig) {
  if (pub.size() != 48 || sig.size() != 4 + 32 + 32 * (Hbs::kP + Hbs::kH)) return false;
  const Bytes I(pub.begin(), pub.begin() + 16); H32 root; std::memcpy(root.data(), pub.data() + 16, 32);
  const uint32_t q = (uint32_t(sig[0]) << 24) | (uint32_t(sig[1]) << 16) | (uint32_t(sig[2]) << 8) | sig[3]; if (q >= Hbs::kLeaves) return false;
  H32 C; std::memcpy(C.data(), sig.data() + 4, 32);
  const Bytes S = Hbs::qsum(Hbs::msg_hash(I, q, C, msg));
  std::vector<H32> z(Hbs::kP);
  for (int i = 0; i < Hbs::kP; ++i) { H32 y; std::memcpy(y.data(), sig.data() + 36 + 32 * i, 32); z[i] = Hbs::chain(I, q, uint16_t(i), y, Hbs::coef(S, i), Hbs::kW); }
  H32 node = Hbs::leaf(I, q, Hbs::ots_pub_from(I, q, z)); uint32_t r = Hbs::kLeaves + q;
  for (int l = 0; l < Hbs::kH; ++l, r >>= 1) { H32 sib; std::memcpy(sib.data(), sig.data() + 36 + 32 * (Hbs::kP + l), 32); node = (r & 1) ? Hbs::inner(I, r >> 1, sib, node) : Hbs::inner(I, r >> 1, node, sib); }
  return node == root;
}

// ---------------- Anchor record (brief s3.1; s3.6 adds chain_generation and epoch_id) ----------------
// v1 records (no chain generation) keep their exact original text, so every existing anchor and log root is unchanged.
// Ratcheted runs use the v2 tag and also carry chain_generation and epoch_id.
struct AnchorRecord {
  std::string run_id, label, weights_sha256, chain_head_sha256, prev_anchor_sha256, emitted_utc; uint64_t chain_length = 0;
  std::string chain_generation, epoch_id;  // empty for v1
  std::string canonical() const {
    if (chain_generation.empty())
      return "pchie.anchor.v1|run_id=" + run_id + "|label=" + label + "|weights_sha256=" + weights_sha256 + "|chain_head_sha256=" + chain_head_sha256 +
             "|chain_length=" + std::to_string(chain_length) + "|prev_anchor_sha256=" + prev_anchor_sha256 + "|emitted_utc=" + emitted_utc;
    return "pchie.anchor.v2|run_id=" + run_id + "|label=" + label + "|weights_sha256=" + weights_sha256 + "|chain_head_sha256=" + chain_head_sha256 +
           "|chain_length=" + std::to_string(chain_length) + "|chain_generation=" + chain_generation + "|epoch_id=" + epoch_id +
           "|prev_anchor_sha256=" + prev_anchor_sha256 + "|emitted_utc=" + emitted_utc;
  }
  H32 record_hash() const { return sha(canonical()); }
  static AnchorRecord parse(const std::string& line) {
    AnchorRecord a; const bool v2 = line.rfind("pchie.anchor.v2|", 0) == 0;
    if (!v2 && line.rfind("pchie.anchor.v1|", 0) != 0) throw std::runtime_error("not an anchor record");
    std::stringstream ss(line.substr(16)); std::string kv;
    while (std::getline(ss, kv, '|')) { const auto e = kv.find('='); if (e == std::string::npos) throw std::runtime_error("bad field"); const std::string k = kv.substr(0, e), v = kv.substr(e + 1);
      if (k == "run_id") a.run_id = v; else if (k == "label") a.label = v; else if (k == "weights_sha256") a.weights_sha256 = v; else if (k == "chain_head_sha256") a.chain_head_sha256 = v;
      else if (k == "chain_length") a.chain_length = std::stoull(v); else if (k == "prev_anchor_sha256") a.prev_anchor_sha256 = v; else if (k == "emitted_utc") a.emitted_utc = v;
      else if (v2 && k == "chain_generation") a.chain_generation = v; else if (v2 && k == "epoch_id") a.epoch_id = v; else throw std::runtime_error("unknown field " + k); }
    if (v2 && (a.chain_generation.empty() || a.epoch_id.empty())) throw std::runtime_error("v2 anchor record needs chain_generation and epoch_id");
    if (a.canonical() != line) throw std::runtime_error("non-canonical anchor record"); return a;
  }
  static bool label_ok(const std::string& l) { return l == "Official" || l == "Scratch" || l == "Provisional"; }
};

// ---------------- Signed tree head and cosignature ----------------
struct Sth { std::string log_id; uint64_t tree_size = 0; H32 root{}; std::string timestamp_utc; Bytes sig;
  std::string signed_text() const { return "pchie.sth.v1|log_id=" + log_id + "|tree_size=" + std::to_string(tree_size) + "|root=" + hex(root) + "|timestamp_utc=" + timestamp_utc; }
  std::string line() const { return signed_text() + "|sig=" + hex(sig); }
  static Sth parse(const std::string& l) { const auto p = l.rfind("|sig="); if (p == std::string::npos) throw std::runtime_error("bad sth"); Sth s; s.sig = unhex(l.substr(p + 5)); std::stringstream ss(l.substr(0, p)); std::string kv; std::getline(ss, kv, '|');
    if (kv != "pchie.sth.v1") throw std::runtime_error("bad sth tag");
    while (std::getline(ss, kv, '|')) { const auto e = kv.find('='); const std::string k = kv.substr(0, e), v = kv.substr(e + 1); if (k == "log_id") s.log_id = v; else if (k == "tree_size") s.tree_size = std::stoull(v); else if (k == "root") s.root = unhex32(v); else if (k == "timestamp_utc") s.timestamp_utc = v; }
    if (s.signed_text() + "|sig=" + hex(s.sig) != l) throw std::runtime_error("non-canonical sth"); return s; }
};
inline std::string cosig_text(const std::string& witness, const Sth& s) { return "pchie.cosig.v1|witness=" + witness + "|log_id=" + s.log_id + "|tree_size=" + std::to_string(s.tree_size) + "|root=" + hex(s.root); }

// ---------------- Fingerprint ceremony (brief s3.4) ----------------
// 60 digits in 12 groups of 5, in the manner of Signal safety numbers: two 5200-round iterated SHA-256 halves.
// v1 rule (records with no chain generation): over (version 0x0001, chain_head, run_id). Unchanged, so every
//   fingerprint already read aloud or pinned stays valid.
// v2 rule (ratcheted records; the fingerprint must cover chain_generation): over (version 0x0002,
//   chain_head, lp(chain_generation), run_id). The same head and run id give different numbers across generations.
inline std::string fingerprint_impl(const std::string& chain_head_hex, const Bytes& prefix_tail, const std::string& run_id, uint8_t version, int iterations) {
  const Bytes head = unhex(chain_head_hex); Bytes out;
  for (uint8_t part = 1; part <= 2; ++part) {
    Bytes b{0x00, version, part}; b.insert(b.end(), head.begin(), head.end()); b.insert(b.end(), prefix_tail.begin(), prefix_tail.end()); put(b, run_id); H32 h = sha(b);
    for (int i = 0; i < iterations; ++i) { Bytes c(h.begin(), h.end()); c.insert(c.end(), head.begin(), head.end()); h = sha(c); }
    out.insert(out.end(), h.begin(), h.end());
  }
  std::string s; char buf[8];
  for (int g = 0; g < 12; ++g) { uint64_t v = 0; for (int j = 0; j < 5; ++j) v = (v << 8) | out[g * 5 + j]; std::snprintf(buf, sizeof buf, "%05llu", (unsigned long long)(v % 100000)); if (g) s += ' '; s += buf; }
  return s;
}
inline std::string fingerprint(const std::string& chain_head_hex, const std::string& run_id, int iterations = 5200) {
  return fingerprint_impl(chain_head_hex, Bytes{}, run_id, 0x01, iterations);
}
// Picks the rule from the anchor: empty generation = v1 rule, otherwise the v2 rule covering the generation.
inline std::string fingerprint_for(const std::string& chain_head_hex, const std::string& run_id, const std::string& chain_generation, int iterations = 5200) {
  if (chain_generation.empty()) return fingerprint(chain_head_hex, run_id, iterations);
  Bytes g; put32(g, uint32_t(chain_generation.size())); put(g, chain_generation);
  return fingerprint_impl(chain_head_hex, g, run_id, 0x02, iterations);
}

// ---------------- Run chain re-derivation, v1 visnav side log (generation pchie.visnav.v1-sha256): h0 = 0^32; h = SHA256(h || SHA256(line)) ----------------
inline bool rederive_visnav_chain(const std::string& path, H32& head, uint64_t& len) {
  std::ifstream f(path, std::ios::binary); if (!f) return false; head.fill(0); len = 0; std::string line;
  while (std::getline(f, line)) { if (!line.empty() && line.back() == '\r') line.pop_back(); if (line.empty()) continue; Bytes b(head.begin(), head.end()); put(b, sha(line)); head = sha(b); ++len; }
  return true;
}
// ---------------- Ratcheted chain files (brief s3.6, generation pchie.seal.v2-ratchet) ----------------
// Text format, one entry per line. Seals are stored so a verifier can name the first entry that disagrees.
// The live chain state and step keys are never written: only records, seals, heads and epoch commitments.
//   pchie.ratchet.v2.chainfile
//   genesis|generation=G|run_id=R|weights_sha256=W|harness=H[|salt_commitment=HEX]   (salt_commitment only in secret-salt mode)
//   step|n=N|kind=K|payload=HEX|seal=HEX
//   epoch|n=N|epoch=E|parent_length=L|parent_head=HEX|event=EV|run_id=R|detail_sha256=HEX|head=HEX
inline constexpr const char* kRatchetFileTag = "pchie.ratchet.v2.chainfile";
inline constexpr const char* kVisnavGenerationId = "pchie.visnav.v1-sha256";

inline std::string ratchet_epoch_id(uint32_t epoch, const std::vector<uint8_t>& commitment) { return std::to_string(epoch) + ":" + hex(commitment); }

// Writes a chain file while it drives a RatchetHashChain, so the file and the live chain cannot drift apart.
class RatchetChainFileWriter {
 public:
  RatchetChainFileWriter(const std::string& path, pchie::RatchetHashChain& chain) : f_(path, std::ios::binary), c_(chain) {
    if (!f_) throw std::runtime_error("cannot write " + path);
    const auto& g = c_.genesis();
    f_ << kRatchetFileTag << "\n" << "genesis|generation=" << g.generation << "|run_id=" << g.run_id << "|weights_sha256=" << hex(g.weights_sha256) << "|harness=" << g.harness_version
       << (g.salted() ? "|salt_commitment=" + hex(g.salt_commitment()) : std::string()) << "\n";
  }
  // Continue a branch file from a parent file's lines (used for forks and resumes).
  void copy_entries_from(const std::string& parent_path) {
    std::ifstream in(parent_path, std::ios::binary); std::string l; int i = 0;
    while (std::getline(in, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); if (i++ >= 2 && !l.empty()) f_ << l << "\n"; }
  }
  void append(const pchie::ChainRecord& r) {
    c_.append(r); const auto h = c_.head();
    f_ << "step|n=" << c_.length() << "|kind=" << uint32_t(r.kind) << "|payload=" << hex(r.payload) << "|seal=" << hex(h) << "\n";
  }
  void reroot(const std::string& event, const std::string& run_id, const std::vector<uint8_t>& detail_sha256) {
    const auto ph = c_.head(); const auto pl = c_.length();
    c_.reroot(event, run_id, detail_sha256);
    f_ << "epoch|n=" << c_.length() << "|epoch=" << c_.epoch() << "|parent_length=" << pl << "|parent_head=" << hex(ph) << "|event=" << event << "|run_id=" << run_id
       << "|detail_sha256=" << hex(detail_sha256) << "|head=" << hex(c_.head()) << "\n";
  }
  void close() { f_.close(); }
 private:
  std::ofstream f_; pchie::RatchetHashChain& c_;
};

struct RatchetFileResult {
  bool ok = false; std::string error; H32 head{}; uint64_t length = 0;
  std::string generation, genesis_run_id, epoch_run_id, weights_sha256, harness, epoch_id;
};
inline std::map<std::string, std::string> split_fields(const std::string& line, std::string& tag) {
  std::map<std::string, std::string> m; std::stringstream ss(line); std::string kv; std::getline(ss, tag, '|');
  while (std::getline(ss, kv, '|')) { const auto e = kv.find('='); if (e == std::string::npos) throw std::runtime_error("bad field: " + kv); m[kv.substr(0, e)] = kv.substr(e + 1); }
  return m;
}
// Generation named on a ratcheted chain file's genesis line ("" when the file is not a ratcheted chain file).
inline std::string ratchet_file_generation(const std::string& path) {
  std::ifstream f(path, std::ios::binary); std::string l;
  if (!std::getline(f, l)) return ""; if (!l.empty() && l.back() == '\r') l.pop_back(); if (l != kRatchetFileTag) return "";
  if (!std::getline(f, l)) return ""; if (!l.empty() && l.back() == '\r') l.pop_back();
  const auto p = l.find("|generation="); if (p == std::string::npos) return ""; const auto e = l.find('|', p + 1); return l.substr(p + 12, e == std::string::npos ? std::string::npos : e - p - 12);
}
inline bool is_ratchet_chain_file(const std::string& path) { std::ifstream f(path, std::ios::binary); std::string l; if (!std::getline(f, l)) return false; if (!l.empty() && l.back() == '\r') l.pop_back(); return l == kRatchetFileTag; }

// Re-derive a ratcheted chain from its genesis, step records and epoch events, checking every stored seal and head.
// secret_salt is needed only for the optional secret-salt mode; recomputable chains (the default) need nothing.
inline RatchetFileResult rederive_ratchet_chain(const std::string& path, const Bytes* secret_salt = nullptr) {
  RatchetFileResult R; std::ifstream f(path, std::ios::binary);
  if (!f) { R.error = "cannot read chain file"; return R; }
  std::vector<std::string> L; std::string l; while (std::getline(f, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); if (!l.empty()) L.push_back(l); }
  if (L.empty() || L[0] != kRatchetFileTag) { R.error = "not a ratcheted chain file (no pchie.ratchet.v2.chainfile header)"; return R; }
  try {
    if (L.size() < 2) { R.error = "missing genesis line"; return R; }
    std::string tag; auto g = split_fields(L[1], tag);
    if (tag != "genesis") { R.error = "line 2 is not genesis"; return R; }
    if (!pchie::is_ratchet_generation(g["generation"])) { R.error = "unknown chain generation " + g["generation"]; return R; }
    pchie::RatchetGenesis gen; gen.generation = g["generation"]; gen.run_id = g.at("run_id"); gen.harness_version = g.at("harness"); gen.weights_sha256 = unhex(g.at("weights_sha256"));
    if (gen.generation == pchie::kRatchetSaltedGenerationId) {
      if (!secret_salt || secret_salt->empty()) { R.error = "secret-salt genesis: the salt is required to re-derive this chain (recomputable chains need none)"; return R; }
      gen.secret_salt = *secret_salt;
      if (!g.count("salt_commitment") || hex(gen.salt_commitment()) != g["salt_commitment"]) { R.error = "secret-salt genesis: the salt does not match the chain's salt commitment"; return R; }
    } else if (g.count("salt_commitment")) { R.error = "salt commitment on a recomputable chain"; return R; }
    R.generation = gen.generation; R.genesis_run_id = gen.run_id; R.epoch_run_id = gen.run_id; R.weights_sha256 = hex(gen.weights_sha256); R.harness = gen.harness_version;
    std::vector<uint8_t> ck = pchie::RatchetHashChain::genesis_state(gen), head = pchie::RatchetHashChain::genesis_head(ck);
    const std::vector<uint8_t> gb = gen.bytes(); std::vector<uint8_t> commit = pchie::Sha256::hash(gb.data(), gb.size());
    uint64_t n = 0; uint32_t epoch = 0;
    for (size_t i = 2; i < L.size(); ++i) {
      auto m = split_fields(L[i], tag); const std::string where = "line " + std::to_string(i + 1);
      if (std::stoull(m.at("n")) != n + 1) { R.error = where + ": entry number " + m["n"] + " where " + std::to_string(n + 1) + " was expected"; return R; }
      if (tag == "step") {
        pchie::ChainRecord r; r.kind = pchie::ChainRecordKind(uint32_t(std::stoul(m.at("kind")))); r.payload = unhex(m.at("payload"));
        std::vector<uint8_t> mk = pchie::RatchetHashChain::step_key(ck);
        std::vector<uint8_t> s = pchie::RatchetHashChain::seal(mk, head, n + 1, r);
        if (hex(s) != m.at("seal")) { R.error = where + ": seal of entry " + std::to_string(n + 1) + " does not re-derive"; return R; }
        std::vector<uint8_t> next = pchie::RatchetHashChain::advance(ck, s); pchie::wipe_bytes(mk); pchie::wipe_bytes(ck); ck = std::move(next); head = std::move(s);
      } else if (tag == "epoch") {
        const uint32_t e = uint32_t(std::stoul(m.at("epoch")));
        if (std::stoull(m.at("parent_length")) != n || m.at("parent_head") != hex(head)) { R.error = where + ": epoch parent is not the current head (not one straight line)"; return R; }
        if (e != epoch + 1) { R.error = where + ": epoch " + m["epoch"] + " where " + std::to_string(epoch + 1) + " was expected"; return R; }
        const std::vector<uint8_t> detail = unhex(m.at("detail_sha256"));
        std::vector<uint8_t> c = pchie::RatchetHashChain::epoch_commitment(e, head, n, m.at("event"), m.at("run_id"), detail);
        pchie::wipe_bytes(ck); ck = pchie::RatchetHashChain::reroot_state(head, c); head = pchie::RatchetHashChain::reroot_head(ck, c);
        if (hex(head) != m.at("head")) { R.error = where + ": epoch head does not re-derive"; return R; }
        epoch = e; commit = c; R.epoch_run_id = m.at("run_id");
      } else { R.error = where + ": unknown entry type " + tag; return R; }
      ++n;
    }
    pchie::wipe_bytes(ck); pchie::wipe_bytes(gen.secret_salt);
    std::memcpy(R.head.data(), head.data(), 32); R.length = n; R.epoch_id = ratchet_epoch_id(epoch, commit); R.ok = true;
  } catch (const std::exception& e) { R.error = std::string("malformed chain file: ") + e.what(); R.ok = false; }
  return R;
}
}}  // namespace pchie::anchor
