// sealed DecisionTuple verifier. Reads a tuple JSONL written by
// MazeRl::seal() (scratch/rl/tuples_*.jsonl), rebuilds every DecisionTuple from its fields, checks
// (1) each tuple re-hashes to its recorded seal, and (2) the scratch side-chain head recomputed from
// the raw JSONL lines (h0 = 32 zero bytes, h_n = SHA256(h_{n-1} || line)) equals the expected head.
// Read-only. Never touches IHashChain or any official chain. Schema: docs/G5_DecisionTuple_Schema.md
#pragma once

#include "pchie/rl/ForkScore.h"
#include "pchie/rl/RlCommon.h"
#include "pchie/rl/RlSideChain.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

struct TupleVerifyResult {
  bool ok = false;
  size_t lines = 0;
  size_t seal_pass = 0;
  size_t seal_fail = 0;
  size_t parse_fail = 0;
  std::string head_hex;   // recomputed side-chain head over the raw lines
  bool head_match = false;
  std::string first_error;
};

inline std::string tv_grab(const std::string& line, const char* key) {
  const std::string k = std::string("\"") + key + "\":";
  auto p = line.find(k);
  if (p == std::string::npos) return {};
  p += k.size();
  if (p < line.size() && line[p] == '"') {
    auto e = line.find('"', p + 1);
    if (e == std::string::npos) return {};
    return line.substr(p + 1, e - (p + 1));
  }
  if (p < line.size() && line[p] == '[') {
    auto e = line.find(']', p);
    if (e == std::string::npos) return {};
    return line.substr(p + 1, e - p - 1);
  }
  auto e = line.find_first_of(",}", p);
  if (e == std::string::npos) return {};
  return line.substr(p, e - p);
}

inline bool tv_int(const std::string& s, int64_t& v) {
  if (s.empty()) return false;
  char* end = nullptr;
  v = std::strtoll(s.c_str(), &end, 10);
  return end && *end == '\0';
}

inline bool tv_list(const std::string& inside, std::vector<int32_t>& out) {
  out.clear();
  if (inside.empty()) return true;
  std::stringstream ss(inside);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    int64_t v = 0;
    if (!tv_int(tok, v)) return false;
    out.push_back(int32_t(v));
  }
  return true;
}

// Strict parse: every canonical field must be present and well formed.
inline bool parse_tuple_line(const std::string& line, DecisionTuple& t, std::array<uint8_t, 32>& seal) {
  int64_t v = 0;
  if (!from_hex(tv_grab(line, "state_hash"), t.state_hash)) return false;
  if (line.find("\"options\":[") == std::string::npos) return false;
  if (!tv_list(tv_grab(line, "options"), t.options)) return false;
  if (line.find("\"distribution_q20\":[") == std::string::npos) return false;
  if (!tv_list(tv_grab(line, "distribution_q20"), t.distribution_q20)) return false;
  if (!tv_int(tv_grab(line, "decision_token"), v)) return false;
  t.decision = int32_t(v);
  if (!tv_int(tv_grab(line, "confidence_q20"), v)) return false;
  t.confidence_q20 = int32_t(v);
  if (!from_hex(tv_grab(line, "weights_hash"), t.weights_hash)) return false;
  if (!tv_int(tv_grab(line, "blank_control"), v)) return false;
  t.blank_control_decision = int32_t(v);
  if (!tv_int(tv_grab(line, "verifier"), v) || v < 0 || v > 255) return false;
  t.verifier_verdict = uint8_t(v);
  if (!from_hex(tv_grab(line, "seal"), seal)) return false;
  return true;
}

inline TupleVerifyResult verify_tuple_log(const std::string& path, const std::string& expected_head_hex) {
  TupleVerifyResult r;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    r.first_error = "cannot open " + path;
    return r;
  }
  RlSideChain chain;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    ++r.lines;
    chain.append_string(line);
    DecisionTuple t;
    std::array<uint8_t, 32> seal{};
    if (!parse_tuple_line(line, t, seal)) {
      ++r.parse_fail;
      if (r.first_error.empty()) r.first_error = "parse fail at line " + std::to_string(r.lines);
      continue;
    }
    if (tuple_rehashes_to(t, seal)) {
      ++r.seal_pass;
    } else {
      ++r.seal_fail;
      if (r.first_error.empty()) r.first_error = "seal mismatch at line " + std::to_string(r.lines);
    }
  }
  r.head_hex = chain.head_hex();
  r.head_match = expected_head_hex.empty() || expected_head_hex == r.head_hex;
  if (!r.head_match && r.first_error.empty()) r.first_error = "side-chain head mismatch";
  r.ok = r.lines > 0 && r.parse_fail == 0 && r.seal_fail == 0 && r.head_match;
  return r;
}

}  // namespace rl
}  // namespace pchie
