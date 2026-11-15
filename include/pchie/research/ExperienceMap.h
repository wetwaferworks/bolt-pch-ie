// Little Nemo's own map: only places he has stood on and edges he has tried.
// Each edge is Unknown until he tries it, then Open (he moved) or Blocked (he bumped).
// Iteration order is fixed (std::map), so everything built on it is deterministic.
#pragma once

#include "pchie/research/SelfObservation.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pchie {
namespace research {

enum class EdgeState : uint8_t { Unknown = 0, Open = 1, Blocked = 2 };

struct Place {
  int32_t x = 0;
  int32_t y = 0;
  uint32_t visits = 0;
  EdgeState edge[kMoves] = {EdgeState::Unknown, EdgeState::Unknown, EdgeState::Unknown, EdgeState::Unknown};
  uint32_t tries[kMoves] = {0, 0, 0, 0};
  bool saw_key = false;
  bool saw_door = false;
  bool saw_exit = false;
};

class ExperienceMap {
 public:
  /// Record that he is standing here (and what he sees at his feet).
  void stand(const SelfObservation& o);

  /// Record the outcome of a move tried from (x, y).
  void tried(int32_t x, int32_t y, int move, bool moved);

  /// A door next to him opened: edges he bumped from here become untried again.
  void forget_blocks(int32_t x, int32_t y);

  /// A door opened: every place where he saw a shut door forgets its bumped edges.
  void forget_door_blocks();

  const Place* find(int32_t x, int32_t y) const;
  EdgeState edge(int32_t x, int32_t y, int move) const;
  size_t places() const { return places_.size(); }
  const std::map<uint64_t, Place>& all() const { return places_; }

  /// Shortest path over Open edges only (moves in compass order on ties). Empty when there is none
  /// or when start == goal. ok is false when the goal cannot be reached over known edges.
  std::vector<uint8_t> path(int32_t fx, int32_t fy, int32_t tx, int32_t ty, bool& ok) const;

  /// Nearest known place that still has an Unknown edge (frontier), by path length then key order.
  bool nearest_frontier(int32_t fx, int32_t fy, int32_t& tx, int32_t& ty) const;

  std::string canonical() const;

 private:
  Place& at(int32_t x, int32_t y);
  std::map<uint64_t, Place> places_;
};

}  // namespace research
}  // namespace pchie
