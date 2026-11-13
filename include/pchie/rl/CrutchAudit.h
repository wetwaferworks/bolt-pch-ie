// L4 crutch audit. Under --rl-vision-only, any world-read API call in the
// planning process must abort the run with CRUTCH_VIOLATION.
// This is a hook the later PchStack/vision wiring will call. Official runners
// never enable it.
#pragma once

#include <string>

namespace pchie {
namespace rl {

class CrutchAudit {
 public:
  void enable(bool on) { on_ = on; }
  bool enabled() const { return on_; }
  bool violated() const { return violated_; }
  const std::string& message() const { return msg_; }

  // Called at a world-read site (get_block, BFS, cMcpWorldReader, world-map).
  // Returns false if the run must abort.
  bool note_world_read(const char* api) {
    if (!on_) return true;
    violated_ = true;
    msg_ = std::string("CRUTCH_VIOLATION: ") + (api ? api : "world-read");
    return false;
  }

  void reset() {
    violated_ = false;
    msg_.clear();
  }

 private:
  bool on_ = false;
  bool violated_ = false;
  std::string msg_;
};

}  // namespace rl
}  // namespace pchie
