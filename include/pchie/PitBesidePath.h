// Bolt PCHIE Mirror-Sim gate - larger labeled pit-maze fixture.
// Unit-test / B9 scenario only. Never a golden and never the official seed=1
// B1 maze (cells differ by pits placed beside the BFS path). Each pit has a
// bypass: the original BFS path is left intact, so pits-as-walls BFS still
// reaches the exit.
#pragma once

#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"

#include <vector>

namespace pchie {

class PitBesidePath final {
 public:
  static int pit_count(const MazeGrid& g) {
    int n = 0;
    for (int y = 0; y < g.height(); ++y) {
      for (int x = 0; x < g.width(); ++x) {
        if (g.cell(x, y) == MazeCell::Pit) ++n;
      }
    }
    return n;
  }

  static MazeGrid make() {
    MazeGrid g(1, 16, 16);
    const auto path = MazeOracle::shortest_path(g);
    const int w = g.width();
    const int h = g.height();
    std::vector<uint8_t> onpath(size_t(w) * size_t(h), 0);
    auto mark = [&](int x, int y) {
      if (x >= 0 && y >= 0 && x < w && y < h) onpath[size_t(y) * size_t(w) + size_t(x)] = 1;
    };
    auto is_path = [&](int x, int y) -> bool {
      if (x < 0 || y < 0 || x >= w || y >= h) return true;
      return onpath[size_t(y) * size_t(w) + size_t(x)] != 0;
    };
    mark(g.x(), g.y());
    MazeGrid walk = g;
    for (MazeAction a : path) {
      if (!walk.apply(a)) break;
      mark(walk.x(), walk.y());
    }
    auto reserved = [](MazeCell c) {
      return c == MazeCell::Wall || c == MazeCell::Start || c == MazeCell::KeyA ||
             c == MazeCell::DoorA || c == MazeCell::Exit || c == MazeCell::Pit;
    };
    const int dxs[4] = {0, 0, 1, -1};
    const int dys[4] = {-1, 1, 0, 0};
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (!is_path(x, y)) continue;
        for (int k = 0; k < 4; ++k) {
          const int nx = x + dxs[k], ny = y + dys[k];
          if (!g.in_bounds(nx, ny) || is_path(nx, ny)) continue;
          const MazeCell c = g.cell(nx, ny);
          if (reserved(c)) continue;
          if (g.walkable(nx, ny)) g.cells_[g.index(nx, ny)] = MazeCell::Pit;
        }
      }
    }
    return g;
  }
};

}  // namespace pchie
