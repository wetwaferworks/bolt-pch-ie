// SCRATCH unit tests for pchie_anchor.h
#include "pchie_anchor.h"
#include <chrono>
#include <iostream>
using namespace pchie::anchor;
static int fails = 0, passes = 0;
#define CHECK(c, msg) do { if (c) ++passes; else { ++fails; std::cout << "FAIL " << msg << "\n"; } } while (0)
int main() {
  CHECK(hex(sha(std::string("abc"))) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
  // RFC 4231 TC1 HMAC; RFC 5869 TC1 HKDF
  CHECK(hex(hmac(Bytes(20, 0x0b), B("Hi There"))) == "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7", "hmac rfc4231 tc1");
  { Bytes ikm(22, 0x0b), salt = unhex("000102030405060708090a0b0c"), info = unhex("f0f1f2f3f4f5f6f7f8f9");
    H32 prk = hkdf_extract(salt, ikm); CHECK(hex(prk) == "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5", "hkdf prk tc1");
    CHECK(hex(hkdf_expand(prk, info, 42)) == "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865", "hkdf okm tc1"); }
  // Merkle: all inclusion/consistency proofs for n<=70, plus negative cases and RFC 9162 s2.1.5 proof sizes
  std::vector<H32> L; for (int i = 0; i < 70; ++i) L.push_back(leaf_hash(B("leaf" + std::to_string(i))));
  int inc_ok = 0, inc_bad = 0, con_ok = 0, con_bad = 0, tot_i = 0, tot_c = 0;
  for (size_t n = 1; n <= 70; ++n) { H32 root = mth(L, 0, n);
    // RFC 9162 s2.1.2 stack algorithm agrees with MTH
    { std::vector<H32> st; for (size_t i = 0; i < n; ++i) { st.push_back(L[i]); size_t mc = 0; while ((i >> mc) & 1) ++mc; for (size_t k = 0; k < mc; ++k) { H32 r = st.back(); st.pop_back(); H32 l = st.back(); st.pop_back(); st.push_back(node_hash(l, r)); } }
      while (st.size() > 1) { H32 r = st.back(); st.pop_back(); H32 l = st.back(); st.pop_back(); st.push_back(node_hash(l, r)); } CHECK(st[0] == root, "stack root n=" + std::to_string(n)); }
    for (size_t m = 0; m < n; ++m) { auto p = inclusion_proof(L, m, n); ++tot_i; inc_ok += verify_inclusion(L[m], m, n, p, root); H32 bad = L[m]; bad[0] ^= 1; inc_bad += !verify_inclusion(bad, m, n, p, root); }
    for (size_t m = 1; m < n; ++m) { auto p = consistency_proof(L, m, n); ++tot_c; con_ok += verify_consistency(m, mth(L, 0, m), n, root, p);
      H32 badr = root; badr[5] ^= 1; bool rej1 = !verify_consistency(m, mth(L, 0, m), n, badr, p); auto q = p; q.pop_back(); bool rej2 = !verify_consistency(m, mth(L, 0, m), n, root, q); con_bad += rej1 && rej2; } }
  CHECK(inc_ok == tot_i && inc_bad == tot_i, "inclusion all/neg " + std::to_string(inc_ok) + "/" + std::to_string(tot_i));
  CHECK(con_ok == tot_c && con_bad == tot_c, "consistency all/neg " + std::to_string(con_ok) + "/" + std::to_string(tot_c));
  CHECK(consistency_proof(L, 3, 7).size() == 4 && consistency_proof(L, 4, 7).size() == 1 && consistency_proof(L, 6, 7).size() == 3, "rfc9162 example proof sizes");
  CHECK(inclusion_proof(L, 0, 7).size() == 3 && inclusion_proof(L, 6, 7).size() == 2, "rfc9162 example inclusion sizes");
  // HBS signatures
  auto t0 = std::chrono::steady_clock::now(); HbsKey k = HbsKey::generate("unit"); double kg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  Bytes pub = k.pub(); Bytes s1 = k.sign(B("tree head 1")), s2 = k.sign(B("tree head 2"));
  CHECK(hbs_verify(pub, B("tree head 1"), s1) && hbs_verify(pub, B("tree head 2"), s2), "hbs verify");
  CHECK(!hbs_verify(pub, B("tree head 1x"), s1), "hbs wrong msg rejected");
  { Bytes t = s1; t[100] ^= 1; CHECK(!hbs_verify(pub, B("tree head 1"), t), "hbs tampered sig rejected"); }
  { HbsKey k2 = HbsKey::generate("other"); CHECK(!hbs_verify(k2.pub(), B("tree head 1"), s1), "hbs wrong key rejected"); }
  CHECK(k.next_q == 2, "hbs state advances");
  { HbsKey r = HbsKey::from_seed(k.I, k.seed, 2); CHECK(r.pub() == pub, "hbs deterministic rebuild from seed"); }
  // fingerprint
  const std::string h = "c21d6b8a6de19054000000000000000000000000000000000000000000000000";
  std::string f1 = fingerprint(h, "runA"), f2 = fingerprint(h, "runA"), f3 = fingerprint(h, "runB"), f4 = fingerprint("d" + h.substr(1), "runA");
  CHECK(f1 == f2 && f1 != f3 && f1 != f4 && f1.size() == 71, "fingerprint deterministic/sensitive/60 digits in 12 groups");
  // anchor record canonical round-trip
  AnchorRecord a; a.run_id = "r1"; a.label = "Scratch"; a.weights_sha256 = std::string(64, 'a'); a.chain_head_sha256 = std::string(64, 'b'); a.chain_length = 3; a.prev_anchor_sha256 = std::string(64, '0'); a.emitted_utc = "2026-11-13T00:00:00Z";
  CHECK(AnchorRecord::parse(a.canonical()).canonical() == a.canonical(), "anchor round-trip");
  bool threw = false; try { AnchorRecord::parse(a.canonical() + "|x=1"); } catch (...) { threw = true; } CHECK(threw, "non-canonical anchor rejected");
  std::cout << "keygen_s=" << kg << " sig_bytes=" << s1.size() << " fp_sample=" << f1 << "\n";
  std::cout << (fails ? "UNIT FAIL " : "UNIT PASS ") << passes << " passed, " << fails << " failed\n"; return fails ? 1 : 0;
}
