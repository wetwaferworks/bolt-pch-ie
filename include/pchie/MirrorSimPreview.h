// Bolt PCHIE Mirror-Sim gate - player-facing preview.
// Deferred to the in-engine pilot. Concrete final. Fails cleanly, not faked.
#pragma once

#include <string>

namespace pchie {

class MirrorSimPreview final {
 public:
  // Headless: there is no preview UI. Always fails with a clear error.
  static bool request(std::string& err) {
    err = "preview UI not implemented (deferred to the in-engine pilot)";
    return false;
  }
};

}  // namespace pchie
