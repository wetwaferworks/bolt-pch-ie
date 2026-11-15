// Plans built only from what Little Nemo has seen himself: his ExperienceMap (places he stood on, ways he
// saw open or bumped) plus the things he saw one step away (pits, the key, a shut door, the exit).
// Two planners use it inside the maze decide loop:
//   - the constraint peer (L6): a door/key graph of his own map, solved exactly (GatedGraphSolver) and,
//     when pinned, checked by Z3 (reachable at the plan length, unreachable one action shorter);
//   - Tree of Thoughts: short move plans over his own map, scored on many threads, ranked in a fixed order.
// Positions are his own (relative to where he first stood). Nothing here reads a maze grid.
#pragma once

#include "pchie/IJobDispatcher.h"
#include "pchie/research/ConstraintSolver.h"
#include "pchie/research/ExperienceMap.h"
#include "pchie/research/SelfObservation.h"
#include "pchie/research/TreeOfThoughts.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace pchie {
namespace research {

/// Things he saw next to him, by place.
struct OwnSight {
  std::set<uint64_t> pits, keys, doors, exits;
  /// Record one view from (x, y). Masks use the compass bits; key_here/at_exit are at his feet.
  void see(int32_t x, int32_t y, uint8_t danger_mask, uint8_t key_mask, uint8_t door_mask, uint8_t exit_mask,
           bool key_here, bool at_exit, bool have_key, bool door_opened);
  std::string canonical() const;
};

/// A door/key question on his own map. coords[i] is node i's place; the goal is the exit if he saw it,
/// otherwise the shut door. One key is assumed to open every door he saw (one door id).
struct OwnGraph {
  GatedGraph g;
  std::vector<std::pair<int32_t, int32_t>> coords;
  bool asked = false;        // there is a goal to plan for
  bool key_held = false;     // his key is modelled as a key lying at the start node
  std::string goal_kind;     // "exit", "door" or "-"
  std::string canonical() const;
};

OwnGraph build_own_graph(const ExperienceMap& map, const OwnSight& sight, int32_t x, int32_t y, bool have_key,
                         bool door_opened);

/// The first real action of a plan as a research action id (kNorth..kWest, kPickup, kUse); -1 if none.
int first_plan_act(const OwnGraph& og, const GatedPlan& plan, int32_t x, int32_t y);

/// Tree of Thoughts over move plans on his own map. Proposals are serial and in compass order; scoring is
/// pure and may run on jobs; the result is the same at every thread count.
TotResult own_map_tree(const ExperienceMap& map, const OwnSight& sight, const SelfObservation& o, bool door_opened,
                       const TotConfig& cfg, IJobDispatcher* jobs);

}  // namespace research
}  // namespace pchie
