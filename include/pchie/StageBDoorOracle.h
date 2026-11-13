// Bolt PCHIE Stage B - BFS oracle over StageBDoorGrid.
// Concrete final. No mutex. State is the agent position under the current
// door flags (doors are opened only by StageBLogicDoor, never by a move).
#pragma once

#include "pchie/StageBDoorGrid.h"

#include <cstdint>
#include <vector>

namespace pchie {

class StageBDoorOracle final {
 public:
  // Stage B HTN primitive ids (distinct from the B1 ids 1..12).
  static constexpr uint32_t kGoToDoor0 = 21;
  static constexpr uint32_t kAnswerDoor0 = 22;
  static constexpr uint32_t kGoToDoor1 = 23;
  static constexpr uint32_t kAnswerDoor1 = 24;
  static constexpr uint32_t kGoToExit = 25;

  static bool goal_reached(const StageBDoorGrid& g, uint32_t task_id) {
    switch (task_id) {
      case kGoToDoor0: return g.adjacent_to_door(0) || g.door_open(0);
      case kAnswerDoor0: return g.door_open(0);
      case kGoToDoor1: return g.adjacent_to_door(1) || g.door_open(1);
      case kAnswerDoor1: return g.door_open(1);
      case kGoToExit: return g.at_exit();
      default: return g.at_exit();
    }
  }

  // Move distance to a navigation subgoal under the current doors, or -1.
  // Answer subgoals are not reachable by moving (returns -1 unless done).
  static int dist_to_subgoal(const StageBDoorGrid& g, uint32_t task_id) {
    if (goal_reached(g, task_id)) return 0;
    std::vector<MazeAction> path;
    return bfs(g, target_of(task_id), /*doors_as_open*/false, path);
  }

  static std::vector<MazeAction> path_to_subgoal(const StageBDoorGrid& g, uint32_t task_id) {
    std::vector<MazeAction> path;
    if (goal_reached(g, task_id)) return path;
    (void)bfs(g, target_of(task_id), false, path);
    return path;
  }

  static int shortest_length_to_exit(const StageBDoorGrid& g) {
    std::vector<MazeAction> path;
    return bfs(g, kTargetExit, false, path);
  }

  static int shortest_length_all_open(const StageBDoorGrid& g) {
    std::vector<MazeAction> path;
    return bfs(g, kTargetExit, true, path);
  }

 private:
  static constexpr int kTargetExit = -1;
  static constexpr int kTargetNone = -2;

  static int target_of(uint32_t task_id) {
    switch (task_id) {
      case kGoToDoor0: return 0;
      case kGoToDoor1: return 1;
      case kGoToExit: return kTargetExit;
      default: return kTargetNone;
    }
  }

  static bool hit(const StageBDoorGrid& g, int x, int y, int target) {
    if (target == kTargetExit) return g.cell(x, y) == StageBCell::Exit;
    if (target >= 0) return g.adjacent_to_door_at(x, y, target);
    return false;
  }

  static int bfs(const StageBDoorGrid& g, int target, bool doors_as_open,
                 std::vector<MazeAction>& path) {
    path.clear();
    const int w = g.width(), h = g.height();
    if (w <= 0 || h <= 0 || target == kTargetNone) return -1;
    if (hit(g, g.x(), g.y(), target)) return 0;
    const int n = w * h;
    std::vector<int> parent(size_t(n), -1);
    std::vector<MazeAction> how(size_t(n), MazeAction::North);
    std::vector<int> q;
    const int s = g.y() * w + g.x();
    parent[size_t(s)] = s;
    q.push_back(s);
    const MazeAction acts[4] = {MazeAction::North, MazeAction::South, MazeAction::East, MazeAction::West};
    const int dxs[4] = {0, 0, 1, -1};
    const int dys[4] = {-1, 1, 0, 0};
    int found = -1;
    for (size_t qi = 0; qi < q.size() && found < 0; ++qi) {
      const int cx = q[qi] % w, cy = q[qi] / w;
      for (int k = 0; k < 4; ++k) {
        const int nx = cx + dxs[k], ny = cy + dys[k];
        if (!g.walkable_with(nx, ny, doors_as_open)) continue;
        const int ni = ny * w + nx;
        if (parent[size_t(ni)] >= 0) continue;
        parent[size_t(ni)] = q[qi];
        how[size_t(ni)] = acts[k];
        if (hit(g, nx, ny, target)) {
          found = ni;
          break;
        }
        q.push_back(ni);
      }
    }
    if (found < 0) return -1;
    std::vector<MazeAction> rev;
    for (int i = found; i != s; i = parent[size_t(i)]) rev.push_back(how[size_t(i)]);
    path.assign(rev.rbegin(), rev.rend());
    return int(path.size());
  }
};

}  // namespace pchie