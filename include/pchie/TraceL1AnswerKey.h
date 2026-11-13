// Bolt PCHIE - draft L1 answer key.
// Provisional, unit-test only, never a golden, pending review.
// Concrete final. Deterministic. No official chain records. No mutex.
#pragma once

#include "pchie/Dividor.h"
#include "pchie/Multiplexor.h"
#include "pchie/Sha256.h"
#include "pchie/TraceFacing.h"
#include "pchie/TraceL1Curriculum.h"
#include "pchie/pchie_tree.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace pchie {

struct TraceL1Answer {
  const char* task_id = "";
  const char* component = "";
  TraceFacing facing = TraceFacing::North;
  int expected_output = 0;
};

// File-scope table: constant initialization, no function-local static (no atomics).
inline const TraceL1Answer kTraceL1Answers[] = {
    {"L1-D0", Dividor::kName, TraceFacing::North, 0},
    {"L1-D1", Dividor::kName, TraceFacing::North, 0},
    {"L1-D2", Dividor::kName, TraceFacing::North, 1},
    {"L1-D3", Dividor::kName, TraceFacing::North, 1},
    {"L1-D5", Dividor::kName, TraceFacing::North, 2},
    {"L1-D64", Dividor::kName, TraceFacing::North, 32},
    {"L1-M-U", Multiplexor::kName, TraceFacing::North, 7},
    {"L1-M-P", Multiplexor::kName, TraceFacing::North, 11},
    {"L1-M-U0", Multiplexor::kName, TraceFacing::North, 2},
    {"L1-M-P64", Multiplexor::kName, TraceFacing::North, 64},
};

class TraceL1AnswerKey final {
 public:
  static std::vector<TraceL1Answer> entries() {
    std::vector<TraceL1Answer> v;
    for (const auto& e : kTraceL1Answers) v.push_back(e);
    return v;
  }

  static const TraceL1Answer* for_task(const char* id) {
    if (!id) return nullptr;
    for (const auto& e : kTraceL1Answers) {
      if (std::strcmp(e.task_id, id) == 0) return &e;
    }
    return nullptr;
  }

  static bool lookup(const char* id, TraceL1Answer& out) {
    const auto* p = for_task(id);
    if (!p) return false;
    out = *p;
    return true;
  }

  static std::string readable_text() {
    std::string s;
    s += "# Trace L1 draft answer key\n";
    s += "# Provisional, unit-test only, never a golden, pending review.\n";
    s += "# Picked by Cortana on the maintainer's behalf.\n";
    s += "# L1 = Multiplexor + Dividor. Facing north.\n";
    s += "# Dividor: BACK in, FRONT out (facing north => in=south, out=north).\n";
    s += "# Multiplexor: south=in1, west=in2, east=control (any source, strength\n";
    s += "#   > 0 selects west), north=out.\n";
    s += "# Strengths 0..64; out-of-range rejected.\n";
    s += "#\n";
    const auto tasks = TraceL1Curriculum::tasks();
    for (const auto& t : tasks) {
      TraceL1Answer a;
      if (!lookup(t.id, a)) continue;
      s += "TASK ";
      s += t.id;
      s += "\n";
      s += "  south=";
      s += std::to_string(t.south);
      s += " west=";
      s += std::to_string(t.west);
      s += " east=";
      s += std::to_string(t.east);
      s += "\n";
      s += "  facing=";
      s += trace_facing_name(t.facing);
      s += "\n";
      s += "  goal_output=";
      s += std::to_string(t.goal);
      s += "\n";
      s += "  expected_component=";
      s += a.component;
      s += "\n";
      s += "  expected_output=";
      s += std::to_string(a.expected_output);
      s += "\n";
    }
    return s;
  }

  static bool write_readable(const std::string& path, std::string& err) {
    err.clear();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
      err = "open failed";
      return false;
    }
    const std::string s = readable_text();
    f.write(s.data(), std::streamsize(s.size()));
    if (!f) {
      err = "write failed";
      return false;
    }
    return true;
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
    const auto ents = entries();
    put_u32(uint32_t(ents.size()));
    for (const auto& e : ents) {
      put_str(e.task_id);
      put_str(e.component);
      put_u32(uint32_t(e.facing));
      put_u32(uint32_t(e.expected_output));
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
