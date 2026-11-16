// Bolt PCHIE - replays a record log and confirms every link (scrub mode).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "pchie/ChainRecord.h"
#include "pchie/IHashChain.h"

namespace pchie {

struct ChainVerifyResult {
  bool ok = false;
  size_t first_bad_index = size_t(-1);  // unknown unless honest prefix heads are supplied
  bool first_bad_index_unknown = true;
  const char* message = "";
};

class ChainVerifier final {
 public:
  static ChainVerifyResult verify(IHashChain& scratch,
                                  const std::vector<ChainRecord>& records,
                                  const std::vector<uint8_t>& expected_head) {
    scratch.reset();
    for (const ChainRecord& r : records) scratch.append(r);
    ChainVerifyResult out;
    if (scratch.head() == expected_head) {
      out.ok = true;
      out.message = "ok";
      return out;
    }
    out.ok = false;
    out.message = "final head mismatch";
    // A final head alone cannot locate a divergent record. Leave the index unknown rather than inventing 0.
    out.first_bad_index = size_t(-1);
    return out;
  }
};

}  // namespace pchie