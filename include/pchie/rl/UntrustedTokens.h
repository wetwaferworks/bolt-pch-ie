// Policy wrapper for untrusted observations (vision symbol grids, world text,
// actuation results, failure notes, recalled memory text).
// Real tokenize_untrusted lives in the later tokenizer; this contract records
// and tests the rules: parse_special=false, no BOS, special-token strings stay
// plain. It does not implement a vocabulary.
#pragma once

#include "pchie/rl/RlCommon.h"

#include <string>
#include <vector>

namespace pchie {
namespace rl {

struct UntrustedTokenizeRequest {
  std::string text;
  bool parse_special = false;  // must stay false
  bool add_bos = false;        // must stay false
};

struct UntrustedTokenizeResult {
  bool ok = false;
  std::string err;
  // Contract-level "tokens": UTF-8 code units grouped as opaque bytes hashed
  // for determinism tests. Not a real vocab encoding.
  std::vector<int32_t> tokens;
};

inline UntrustedTokenizeResult tokenize_untrusted(const UntrustedTokenizeRequest& req) {
  UntrustedTokenizeResult r;
  if (req.parse_special) {
    r.err = "tokenize_untrusted requires parse_special=false";
    return r;
  }
  if (req.add_bos) {
    r.err = "tokenize_untrusted forbids BOS";
    return r;
  }
  r.ok = true;
  r.tokens.reserve(req.text.size());
  for (unsigned char c : req.text) r.tokens.push_back(int32_t(c));
  return r;
}

// Helpers used by tests: these literals must remain ordinary token bytes.
inline constexpr const char* kImStartLiteral = "<|im_start|>";
inline constexpr const char* kBosLiteral = "<s>";

}  // namespace rl
}  // namespace pchie
