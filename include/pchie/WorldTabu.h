// Bolt PCHIE failed-actuation retry - WorldTabu: proposals that already failed on a
// canonical world signature. Concrete final. Serial, deterministic, no mutex/atomic/thread.
// Keyed by (world signature, door/site, proposal name): when the world advances the signature
// changes, so an earlier failure never blocks a try that may be valid in the new state.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class WorldTabu final {
 public:
  bool contains(uint64_t world, int site, const char* name) const {
    if (!name) return false;
    for (const auto& e : e_)
      if (e.world == world && e.site == site && e.name == name) return true;
    return false;
  }
  // Returns false (no change) if the entry is already present.
  bool add(uint64_t world, int site, const char* name) {
    if (!name || contains(world, site, name)) return false;
    e_.push_back(Entry{world, site, std::string(name)});
    return true;
  }
  void clear_for(uint64_t world, int site) {
    std::vector<Entry> keep;
    for (auto& e : e_)
      if (!(e.world == world && e.site == site)) keep.push_back(e);
    e_.swap(keep);
  }
  void clear() { e_.clear(); }
  size_t size() const noexcept { return e_.size(); }

 private:
  struct Entry {
    uint64_t world;
    int site;
    std::string name;
  };
  std::vector<Entry> e_;
};

}  // namespace pchie