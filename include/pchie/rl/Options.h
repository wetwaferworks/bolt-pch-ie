// L5. Macro-actions / options. Short skills over the L4 belief.
// Every primitive step still goes through the real PchGate leaf path (stand-in
// gate in contract tests). A rejected mid-step terminates the option and
// fails closed: no partial action is committed.
// Reserved id range is enforced at registration (also addresses the HTN and NCM extensions
// finding 4).
#pragma once

#include "pchie/rl/BeliefState.h"
#include "pchie/rl/LeafValueSource.h"

#include <string>
#include <vector>

namespace pchie {
namespace rl {

// Reserved option-id range (inclusive). Primitive move tokens stay outside it.
constexpr uint32_t kOptionIdMin = 0x524C0001u;  // "RL" + 1
constexpr uint32_t kOptionIdMax = 0x524C00FFu;

constexpr uint32_t kOptFollowCorridor = 0x524C0001u;
constexpr uint32_t kOptTurnAndReobserve = 0x524C0002u;
constexpr uint32_t kOptBackOutOfDeadEnd = 0x524C0003u;
constexpr uint32_t kOptProbeUnknown = 0x524C0004u;

struct OptionSpec {
  uint32_t id = 0;
  const char* name = "";
  bool (*can_start)(const BeliefState&) = nullptr;
  int32_t (*next_primitive)(const BeliefState&, uint32_t step) = nullptr;  // 0=N 1=E 2=S 3=W, or -1
  bool (*should_stop)(const BeliefState&, uint32_t step) = nullptr;
  uint32_t max_steps = 8;
};

inline bool option_id_reserved(uint32_t id) { return id >= kOptionIdMin && id <= kOptionIdMax; }

struct OptionRegistry {
  std::vector<OptionSpec> specs;
  std::string err;

  bool add_option(const OptionSpec& s) {
    if (!option_id_reserved(s.id)) {
      err = "option id outside reserved range";
      return false;
    }
    if (!s.name || !s.can_start || !s.next_primitive || !s.should_stop) {
      err = "option spec incomplete";
      return false;
    }
    if (s.max_steps == 0 || s.max_steps > 16) {
      err = "option max_steps out of range";
      return false;
    }
    for (const auto& e : specs) {
      if (e.id == s.id) {
        err = "duplicate option id";
        return false;
      }
    }
    specs.push_back(s);
    err.clear();
    return true;
  }
};

enum class OptionStop : uint8_t { Running = 0, Cap = 1, Belief = 2, GateReject = 3, NoPrimitive = 4 };

struct OptionRunResult {
  bool ok = false;
  OptionStop stop = OptionStop::Running;
  std::vector<int32_t> committed;  // primitives accepted by the gate
  Pose start{}, end{};
  uint32_t steps = 0;
};

// Gate is a reject-id list (stand-in). A reject on a mid-step commits nothing
// from this option (fail closed).
inline OptionRunResult run_option(const OptionSpec& spec, BeliefState& belief,
                                  const std::vector<int32_t>& reject_ids) {
  OptionRunResult r;
  r.start = belief.pose();
  if (!spec.can_start || !spec.can_start(belief)) {
    r.stop = OptionStop::Belief;
    return r;
  }
  std::vector<int32_t> acc;
  for (uint32_t step = 0; step < spec.max_steps; ++step) {
    if (spec.should_stop && spec.should_stop(belief, step) && step > 0) {
      r.stop = OptionStop::Belief;
      break;
    }
    const int32_t prim = spec.next_primitive ? spec.next_primitive(belief, step) : -1;
    if (prim < 0) {
      r.stop = OptionStop::NoPrimitive;
      break;
    }
    bool rejected = false;
    for (int32_t id : reject_ids)
      if (id == prim) rejected = true;
    if (rejected) {
      r.stop = OptionStop::GateReject;
      r.ok = false;
      r.end = r.start;
      r.steps = 0;
      r.committed.clear();
      return r;  // no partial commit
    }
    acc.push_back(prim);
    // Contract tests apply move_result themselves or pass Moved for legal floor.
    // The runner does not invent world outcomes; the caller should have updated
    // belief. Here we only record the accepted primitive.
    r.steps = uint32_t(acc.size());
    if (step + 1 == spec.max_steps) r.stop = OptionStop::Cap;
  }
  r.committed = std::move(acc);
  r.end = belief.pose();
  r.ok = r.stop != OptionStop::GateReject;
  return r;
}

// ---- Built-in option predicates (over L4 belief) ----

inline int32_t heading_to_action(Heading h) { return int32_t(h); }

inline Cell ahead(const BeliefState& b) {
  const Pose& p = b.pose();
  return b.cell(p.x + dx_of(p.heading), p.y + dy_of(p.heading));
}

inline int open_dirs(const BeliefState& b, Heading* out, size_t cap) {
  const ActionMask m = b.mask();
  int n = 0;
  const Heading hs[4] = {Heading::N, Heading::E, Heading::S, Heading::W};
  for (Heading h : hs) {
    if (!m.allowed(h)) continue;
    if (out && size_t(n) < cap) out[n] = h;
    ++n;
  }
  return n;
}

inline bool corridor_ahead(const BeliefState& b) {
  // Corridor: current heading is open, left and right are observed walls/bumps.
  const Pose& p = b.pose();
  const Heading left = Heading((int(p.heading) + 3) % 4);
  const Heading right = Heading((int(p.heading) + 1) % 4);
  const Cell L = b.cell(p.x + dx_of(left), p.y + dy_of(left));
  const Cell R = b.cell(p.x + dx_of(right), p.y + dy_of(right));
  const bool side_wall = (L == Cell::Wall || L == Cell::Bumped) && (R == Cell::Wall || R == Cell::Bumped);
  const Cell A = ahead(b);
  return side_wall && A != Cell::Wall && A != Cell::Bumped && A != Cell::Pit;
}

inline bool can_start_follow_corridor(const BeliefState& b) { return corridor_ahead(b); }
inline int32_t next_follow_corridor(const BeliefState& b, uint32_t) {
  return corridor_ahead(b) ? heading_to_action(b.pose().heading) : -1;
}
inline bool stop_follow_corridor(const BeliefState& b, uint32_t) { return !corridor_ahead(b); }

inline bool can_start_turn(const BeliefState& b) {
  Heading tmp[4];
  return open_dirs(b, tmp, 4) >= 1;
}
inline int32_t next_turn(const BeliefState& b, uint32_t) {
  // Turn toward the first open heading that is not straight ahead if possible.
  Heading tmp[4];
  const int n = open_dirs(b, tmp, 4);
  if (n <= 0) return -1;
  for (int i = 0; i < n; ++i)
    if (tmp[i] != b.pose().heading) return heading_to_action(tmp[i]);
  return heading_to_action(tmp[0]);
}
inline bool stop_turn(const BeliefState&, uint32_t step) { return step >= 1; }

inline bool can_start_backout(const BeliefState& b) {
  Heading tmp[4];
  return open_dirs(b, tmp, 4) <= 1;
}
inline int32_t next_backout(const BeliefState& b, uint32_t) {
  const Heading around = Heading((int(b.pose().heading) + 2) % 4);
  if (b.mask().allowed(around)) return heading_to_action(around);
  Heading tmp[4];
  const int n = open_dirs(b, tmp, 4);
  return n > 0 ? heading_to_action(tmp[0]) : -1;
}
inline bool stop_backout(const BeliefState&, uint32_t step) { return step >= 1; }

inline bool can_start_probe(const BeliefState& b) {
  const Pose& p = b.pose();
  const Heading hs[4] = {Heading::N, Heading::E, Heading::S, Heading::W};
  for (Heading h : hs) {
    if (b.mask().allowed(h) && b.cell(p.x + dx_of(h), p.y + dy_of(h)) == Cell::Unknown) return true;
  }
  return false;
}
inline int32_t next_probe(const BeliefState& b, uint32_t) {
  const Pose& p = b.pose();
  const Heading hs[4] = {Heading::N, Heading::E, Heading::S, Heading::W};
  for (Heading h : hs) {
    if (b.mask().allowed(h) && b.cell(p.x + dx_of(h), p.y + dy_of(h)) == Cell::Unknown)
      return heading_to_action(h);
  }
  return -1;
}
inline bool stop_probe(const BeliefState&, uint32_t step) { return step >= 1; }

inline OptionSpec spec_follow_corridor() {
  return OptionSpec{kOptFollowCorridor, "FollowCorridor", can_start_follow_corridor, next_follow_corridor,
                    stop_follow_corridor, 8};
}
inline OptionSpec spec_turn() {
  return OptionSpec{kOptTurnAndReobserve, "TurnAndReobserve", can_start_turn, next_turn, stop_turn, 1};
}
inline OptionSpec spec_backout() {
  return OptionSpec{kOptBackOutOfDeadEnd, "BackOutOfDeadEnd", can_start_backout, next_backout, stop_backout,
                    2};
}
inline OptionSpec spec_probe() {
  return OptionSpec{kOptProbeUnknown, "ProbeUnknown", can_start_probe, next_probe, stop_probe, 1};
}

}  // namespace rl
}  // namespace pchie
