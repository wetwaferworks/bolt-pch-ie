// Bolt PCHIE - cardinal facing for a placed Trace component.
#pragma once

namespace pchie {

enum class TraceFacing : int {
  North = 0,
  South = 1,
  East = 2,
  West = 3
};

inline TraceFacing trace_opposite(TraceFacing f) {
  switch (f) {
    case TraceFacing::North: return TraceFacing::South;
    case TraceFacing::South: return TraceFacing::North;
    case TraceFacing::East: return TraceFacing::West;
    case TraceFacing::West: return TraceFacing::East;
  }
  return TraceFacing::North;
}

inline const char* trace_facing_name(TraceFacing f) {
  switch (f) {
    case TraceFacing::North: return "north";
    case TraceFacing::South: return "south";
    case TraceFacing::East: return "east";
    case TraceFacing::West: return "west";
  }
  return "north";
}

}  // namespace pchie
