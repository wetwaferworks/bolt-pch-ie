// Bolt PCHIE Stage B - door task table and labeled answer key (Q58/Q59).
// Unit/labeled only, never a golden. Separate from the draft Trace key, which
// is not edited. Opaque ids: they do not name or abbreviate the component.
// Order is fixed: door 0 = Dividor (clearer, 5/6 on Trace L1), door 1 =
// Multiplexor. Both tasks are distinguishing. No function-local statics.
#pragma once

#include "pchie/Dividor.h"
#include "pchie/Multiplexor.h"
#include "pchie/Sha256.h"
#include "pchie/TraceFacing.h"
#include "pchie/TraceL1Task.h"
#include "pchie/pchie_tree.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace pchie {

struct StageBDoorAnswer {
  const char* task_id = "";
  const char* component = "";
  TraceFacing facing = TraceFacing::North;
  int expected_output = 0;
};

// Door k uses kStageBDoorTasks[k] / kStageBDoorAnswers[k].
inline const TraceL1Task kStageBDoorTasks[] = {
    {"SB-3F9A", /*s*/12, /*w*/0, /*e*/0, /*goal*/6, TraceFacing::North},
    {"SB-7E2B", /*s*/9, /*w*/20, /*e*/3, /*goal*/20, TraceFacing::North},
};

inline const StageBDoorAnswer kStageBDoorAnswers[] = {
    {"SB-3F9A", Dividor::kName, TraceFacing::North, 6},
    {"SB-7E2B", Multiplexor::kName, TraceFacing::North, 20},
};

class StageBDoorKey final {
 public:
  static int door_count() { return int(sizeof(kStageBDoorTasks) / sizeof(kStageBDoorTasks[0])); }

  static bool task_for_door(int door, TraceL1Task& out) {
    if (door < 0 || door >= door_count()) return false;
    out = kStageBDoorTasks[door];
    return true;
  }

  static const StageBDoorAnswer* for_task(const char* id) {
    if (!id) return nullptr;
    for (const auto& e : kStageBDoorAnswers)
      if (std::strcmp(e.task_id, id) == 0) return &e;
    return nullptr;
  }

  static std::vector<uint8_t> canonical_bytes() {
    std::vector<uint8_t> b;
    auto put_u32 = [&](uint32_t v) {
      for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i)));
    };
    auto put_str = [&](const char* s) {
      const uint32_t n = s ? uint32_t(std::strlen(s)) : 0;
      put_u32(n);
      for (uint32_t i = 0; i < n; ++i) b.push_back(uint8_t(s[i]));
    };
    put_u32(uint32_t(door_count()));
    for (int k = 0; k < door_count(); ++k) {
      const TraceL1Task& t = kStageBDoorTasks[k];
      const StageBDoorAnswer& a = kStageBDoorAnswers[k];
      put_str(t.id);
      put_u32(uint32_t(t.south));
      put_u32(uint32_t(t.west));
      put_u32(uint32_t(t.east));
      put_u32(uint32_t(t.goal));
      put_u32(uint32_t(t.facing));
      put_str(a.component);
      put_u32(uint32_t(a.expected_output));
    }
    return b;
  }

  static std::vector<uint8_t> sha256() {
    const auto b = canonical_bytes();
    return Sha256::hash(b.data(), b.size());
  }

  static uint64_t fnv() {
    const auto b = canonical_bytes();
    uint64_t h = FNV_OFFSET;
    constexpr uint64_t P = 1099511628211ULL;
    for (size_t i = 0; i < b.size(); ++i) {
      h ^= uint64_t(b[i]);
      h *= P;
    }
    return h;
  }
};

}  // namespace pchie