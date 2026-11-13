// Bolt PCHIE B1 - BFS oracle over MazeGrid.
// Concrete final. No mutex. State is (x, y, have_key, door_open).
#pragma once

#include "pchie/MazeGrid.h"

#include <cstdint>
#include <queue>
#include <vector>

namespace pchie {

class MazeOracle final {
 public:
  static constexpr uint32_t kGoToExit = 4;
  static constexpr uint32_t kGoToKeyA = 5;
  static constexpr uint32_t kPickUpKeyA = 6;
  static constexpr uint32_t kGoToDoorA = 7;
  static constexpr uint32_t kUseKeyA = 8;

  static int shortest_length(const MazeGrid& g) {
    std::vector<MazeAction> path;
    return search(g, /*to_exit*/true, 0, path, /*avoid_pits*/false);
  }

  static std::vector<MazeAction> shortest_path(const MazeGrid& g) {
    std::vector<MazeAction> path;
    (void)search(g, /*to_exit*/true, 0, path, /*avoid_pits*/false);
    return path;
  }

  // BFS to the exit with every pit cell treated as a wall. Returns -1 if
  // unsolvable (including a start already in a pit).
  static int shortest_length_avoiding_pits(const MazeGrid& g) {
    std::vector<MazeAction> path;
    return search(g, /*to_exit*/true, 0, path, /*avoid_pits*/true);
  }

  static std::vector<MazeAction> shortest_path_avoiding_pits(const MazeGrid& g) {
    std::vector<MazeAction> path;
    (void)search(g, /*to_exit*/true, 0, path, /*avoid_pits*/true);
    return path;
  }

  static int dist_to_subgoal(const MazeGrid& g, uint32_t task_id) {
    if (goal_reached(g, task_id)) return 0;
    std::vector<MazeAction> path;
    return search(g, /*to_exit*/false, task_id, path, /*avoid_pits*/false);
  }

 private:
  static int pack(const MazeGrid& g, int x, int y, bool key, bool door) {
    const int layer = (door ? 2 : 0) + (key ? 1 : 0);
    return layer * (g.width() * g.height()) + y * g.width() + x;
  }

  static bool goal_reached(const MazeGrid& g, uint32_t task_id) {
    switch (task_id) {
      case kGoToKeyA: return g.on_key() || g.have_key();
      case kPickUpKeyA: return g.have_key();
      case kGoToDoorA: return g.adjacent_to_door() || g.door_open();
      case kUseKeyA: return g.door_open();
      case kGoToExit: return g.at_exit();
      default: return g.at_exit();
    }
  }

  // Returns path length, or -1. If to_exit, ignores task_id and walks to the exit.
  static int search(const MazeGrid& start, bool to_exit, uint32_t task_id,
                    std::vector<MazeAction>& path, bool avoid_pits) {
    path.clear();
    if (start.width() <= 0 || start.height() <= 0) return -1;
    if (avoid_pits && start.in_pit()) return -1;
    if (to_exit) {
      if (start.at_exit()) return 0;
    } else if (goal_reached(start, task_id)) {
      return 0;
    }

    const int w = start.width();
    const int h = start.height();
    if (w > 256 || h > 256) return -1;
    const int nstate = 4 * w * h;
    std::vector<int> dist(size_t(nstate), -1);
    std::vector<int> parent(size_t(nstate), -1);
    std::vector<MazeAction> how(size_t(nstate), MazeAction::North);
    std::queue<MazeGrid> q;

    const int sidx = pack(start, start.x(), start.y(), start.have_key(), start.door_open());
    if (sidx < 0 || sidx >= nstate) return -1;
    dist[size_t(sidx)] = 0;
    q.push(start);

    int found = -1;
    MazeGrid found_g;
    while (!q.empty()) {
      const MazeGrid cur = q.front();
      q.pop();
      const int cidx = pack(cur, cur.x(), cur.y(), cur.have_key(), cur.door_open());
      const auto acts = cur.legal_actions();
      for (MazeAction a : acts) {
        MazeGrid nxt = cur;
        if (!nxt.apply(a)) continue;
        if (avoid_pits && nxt.in_pit()) continue;
        const int nidx = pack(nxt, nxt.x(), nxt.y(), nxt.have_key(), nxt.door_open());
        if (nidx < 0 || nidx >= nstate) continue;
        if (dist[size_t(nidx)] >= 0) continue;
        dist[size_t(nidx)] = dist[size_t(cidx)] + 1;
        parent[size_t(nidx)] = cidx;
        how[size_t(nidx)] = a;
        const bool hit = to_exit ? nxt.at_exit() : goal_reached(nxt, task_id);
        if (hit) {
          found = nidx;
          found_g = nxt;
          q = {};
          break;
        }
        q.push(std::move(nxt));
      }
    }
    if (found < 0) return -1;
    // Reconstruct.
    std::vector<MazeAction> rev;
    for (int i = found; i != sidx; i = parent[size_t(i)]) {
      if (i < 0 || parent[size_t(i)] < 0) {
        path.clear();
        return -1;
      }
      rev.push_back(how[size_t(i)]);
    }
    path.assign(rev.rbegin(), rev.rend());
    return int(path.size());
  }
};

}  // namespace pchie
