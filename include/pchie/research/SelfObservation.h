// What Little Nemo knows about himself at one step, and nothing more: his dead-reckoned position
// (odometry from his own successful moves), the way he is facing, whether he is holding a key,
// what he can see right where he stands, and a bucket of his current view.
// No research layer reads the world grid; tests build these from his own experience only.
#pragma once

#include <cstdint>

namespace pchie {
namespace research {

/// Moves use the compass order North, East, South, West everywhere in the research layers.
enum Move : uint8_t { kNorth = 0, kEast = 1, kSouth = 2, kWest = 3, kMoves = 4 };

/// Extra actions beside the four moves.
enum Act : uint8_t { kPickup = 4, kUse = 5, kActs = 6 };

inline int move_dx(int m) { return m == kEast ? 1 : (m == kWest ? -1 : 0); }
inline int move_dy(int m) { return m == kSouth ? 1 : (m == kNorth ? -1 : 0); }
inline int reverse_move(int m) { return (m + 2) & 3; }

struct SelfObservation {
  int32_t x = 0;              // odometry, start = (0, 0)
  int32_t y = 0;
  uint8_t heading = kNorth;   // last move he made
  bool have_key = false;
  bool key_here = false;      // he sees a key at his feet
  bool door_near = false;     // he sees a shut door next to him
  bool at_exit = false;
  uint8_t danger_mask = 0;    // bit m set: he can see a pit one step in move m
  uint16_t view_bucket = 0;   // coarse hash of his ASCII view (0 when vision is not wired)
};

/// One packed archive cell: position, heading, key state and view bucket.
inline uint64_t cell_key(const SelfObservation& o, bool use_heading = false) {
  const uint64_t ux = uint64_t(uint32_t(o.x + 32768) & 0xFFFF);
  const uint64_t uy = uint64_t(uint32_t(o.y + 32768) & 0xFFFF);
  const uint64_t h = use_heading ? uint64_t(o.heading & 3) : 0;
  return (ux << 48) | (uy << 32) | (h << 30) | (uint64_t(o.have_key) << 29) | uint64_t(o.view_bucket);
}

/// Position-only key (used for the movement graph).
inline uint64_t place_key(int32_t x, int32_t y) {
  return (uint64_t(uint32_t(x + 32768) & 0xFFFF) << 16) | uint64_t(uint32_t(y + 32768) & 0xFFFF);
}

}  // namespace research
}  // namespace pchie
