// Bolt PCHIE Stage A - labeled 64x64 2x2-wide branching pit maze.
// Concrete final. Unit/labeled only, never a golden, never the official
// seed=1 16x16 B1 maze. This is a full branching
// maze with 2x2-wide corridors, not the carved start-key-door-exit
// 1-wide corridor style. Pits are 1x1 and sit along one wall of a
// corridor, leaving a safe 1-cell lane on the other side. Every pit
// has that bypass; pits-as-walls BFS still reaches the exit.
// Deterministic Splitmix64. No mutex.
#pragma once

#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/NodeSeed.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace pchie {

class ScaledPitMaze final {
 public:
  static constexpr uint64_t kLabeledSeed = 64;
  static constexpr int kWidth = 64;
  static constexpr int kHeight = 64;
  // 2x2 floor rooms separated by 1-wide walls. Separate stream
  // from MazeGrid::generate (0xB100...).
  static constexpr uint64_t kMazeStream = 0xB164A11EULL;
  static constexpr int kStep = 3;

  static int pit_count(const MazeGrid& g) {
    int n = 0;
    for (int y = 0; y < g.height(); ++y) {
      for (int x = 0; x < g.width(); ++x) {
        if (g.cell(x, y) == MazeCell::Pit) ++n;
      }
    }
    return n;
  }

  static int rooms_x(int width) { return (width > 0) ? (width - 1) / kStep : 0; }
  static int rooms_y(int height) { return (height > 0) ? (height - 1) / kStep : 0; }
  static int room_origin_x(int i) { return 1 + kStep * i; }
  static int room_origin_y(int j) { return 1 + kStep * j; }

  static MazeGrid make(uint64_t seed = kLabeledSeed, int width = kWidth, int height = kHeight) {
    // Construct then fully overwrite so no 1-wide carved generate() geometry remains.
    MazeGrid g(seed, width, height);
    g.seed_ = seed;
    g.w_ = width;
    g.h_ = height;
    if (g.w_ < 0) g.w_ = 0;
    if (g.h_ < 0) g.h_ = 0;
    g.cells_.assign(size_t(g.w_) * size_t(g.h_), MazeCell::Wall);
    g.have_key_ = false;
    g.door_open_ = false;
    g.x_ = 0;
    g.y_ = 0;

    const int rx = rooms_x(g.w_);
    const int ry = rooms_y(g.h_);
    if (rx < 2 || ry < 2) return g;

    auto setc = [&](int x, int y, MazeCell c) {
      if (g.in_bounds(x, y)) g.cells_[g.index(x, y)] = c;
    };

    for (int j = 0; j < ry; ++j) {
      for (int i = 0; i < rx; ++i) {
        const int x = room_origin_x(i);
        const int y = room_origin_y(j);
        setc(x, y, MazeCell::Floor);
        setc(x + 1, y, MazeCell::Floor);
        setc(x, y + 1, MazeCell::Floor);
        setc(x + 1, y + 1, MazeCell::Floor);
      }
    }

    // east[j*rx+i] connects (i,j) to (i+1,j); south[j*rx+i] connects (i,j) to (i,j+1).
    std::vector<uint8_t> east(size_t(rx) * size_t(ry), 0);
    std::vector<uint8_t> south(size_t(rx) * size_t(ry), 0);
    auto rid = [&](int i, int j) { return j * rx + i; };

    auto carve_east = [&](int i, int j) {
      east[size_t(rid(i, j))] = 1;
      const int xw = room_origin_x(i) + 2;
      const int y0 = room_origin_y(j);
      setc(xw, y0, MazeCell::Floor);
      setc(xw, y0 + 1, MazeCell::Floor);
    };
    auto carve_south = [&](int i, int j) {
      south[size_t(rid(i, j))] = 1;
      const int yw = room_origin_y(j) + 2;
      const int x0 = room_origin_x(i);
      setc(x0, yw, MazeCell::Floor);
      setc(x0 + 1, yw, MazeCell::Floor);
    };

    uint64_t rng = NodeSeed::Splitmix64(seed ^ kMazeStream);
    auto rnd = [&]() -> uint64_t {
      rng = NodeSeed::Splitmix64(rng);
      return rng;
    };

    // Recursive-backtracker perfect maze on the room grid (branching, many
    // dead ends, not a single carved start-key-door-exit corridor).
    std::vector<uint8_t> vis(size_t(rx) * size_t(ry), 0);
    std::vector<int> st;
    vis[0] = 1;
    st.push_back(0);
    const int di[4] = {1, -1, 0, 0};
    const int dj[4] = {0, 0, 1, -1};
    while (!st.empty()) {
      const int id = st.back();
      const int i = id % rx;
      const int j = id / rx;
      int opts[4];
      int nopt = 0;
      for (int k = 0; k < 4; ++k) {
        const int ni = i + di[k];
        const int nj = j + dj[k];
        if (ni < 0 || nj < 0 || ni >= rx || nj >= ry) continue;
        if (vis[size_t(rid(ni, nj))]) continue;
        opts[nopt++] = k;
      }
      if (nopt == 0) {
        st.pop_back();
        continue;
      }
      const int k = opts[int(rnd() % uint64_t(nopt))];
      const int ni = i + di[k];
      const int nj = j + dj[k];
      if (ni == i + 1) carve_east(i, j);
      else if (ni == i - 1) carve_east(ni, j);
      else if (nj == j + 1) carve_south(i, j);
      else carve_south(i, nj);
      vis[size_t(rid(ni, nj))] = 1;
      st.push_back(rid(ni, nj));
    }

    const int sx = room_origin_x(0);
    const int sy = room_origin_y(0);
    setc(sx, sy, MazeCell::Start);
    g.x_ = sx;
    g.y_ = sy;

    const int exi = rx - 1;
    const int eyi = ry - 1;
    setc(room_origin_x(exi) + 1, room_origin_y(eyi) + 1, MazeCell::Exit);

    auto connected = [&](int i, int j, int ni, int nj) -> bool {
      if (ni == i + 1 && nj == j) return east[size_t(rid(i, j))] != 0;
      if (ni == i - 1 && nj == j) return east[size_t(rid(ni, j))] != 0;
      if (nj == j + 1 && ni == i) return south[size_t(rid(i, j))] != 0;
      if (nj == j - 1 && ni == i) return south[size_t(rid(i, nj))] != 0;
      return false;
    };

    // Room BFS from start to exit for the door edge.
    std::vector<int> prev(size_t(rx) * size_t(ry), -1);
    std::vector<int> q;
    q.push_back(rid(0, 0));
    prev[size_t(rid(0, 0))] = rid(0, 0);
    size_t qi = 0;
    while (qi < q.size()) {
      const int id = q[qi++];
      const int i = id % rx, j = id / rx;
      for (int k = 0; k < 4; ++k) {
        const int ni = i + di[k], nj = j + dj[k];
        if (ni < 0 || nj < 0 || ni >= rx || nj >= ry) continue;
        if (!connected(i, j, ni, nj)) continue;
        const int nid = rid(ni, nj);
        if (prev[size_t(nid)] >= 0) continue;
        prev[size_t(nid)] = id;
        q.push_back(nid);
      }
    }
    const int exit_id = rid(exi, eyi);
    if (prev[size_t(exit_id)] < 0) return g;

    std::vector<int> path;
    for (int id = exit_id;;) {
      path.push_back(id);
      if (id == rid(0, 0)) break;
      id = prev[size_t(id)];
      if (id < 0) {
        path.clear();
        break;
      }
    }
    if (path.size() < 2) return g;
    // path is exit ... start; reverse to start ... exit.
    for (size_t a = 0, b = path.size() - 1; a < b; ++a, --b) std::swap(path[a], path[b]);

    // Door spans the 2-wide opening on an edge past the first hop so the
    // start room (and any earlier branches) stay on the key side.
    const int door_at = (int(path.size()) >= 4) ? int(path.size()) / 2 : 1;
    const int da = path[size_t(door_at - 1)];
    const int db = path[size_t(door_at)];
    const int dai = da % rx, daj = da / rx;
    const int dbi = db % rx, dbj = db / rx;
    int door_x0 = 0, door_y0 = 0, door_x1 = 0, door_y1 = 0;
    if (dbi == dai + 1 && dbj == daj) {
      door_x0 = room_origin_x(dai) + 2;
      door_y0 = room_origin_y(daj);
      door_x1 = door_x0;
      door_y1 = door_y0 + 1;
    } else if (dbi == dai - 1 && dbj == daj) {
      door_x0 = room_origin_x(dbi) + 2;
      door_y0 = room_origin_y(daj);
      door_x1 = door_x0;
      door_y1 = door_y0 + 1;
    } else if (dbj == daj + 1 && dbi == dai) {
      door_x0 = room_origin_x(dai);
      door_y0 = room_origin_y(daj) + 2;
      door_x1 = door_x0 + 1;
      door_y1 = door_y0;
    } else {
      door_x0 = room_origin_x(dai);
      door_y0 = room_origin_y(dbj) + 2;
      door_x1 = door_x0 + 1;
      door_y1 = door_y0;
    }
    setc(door_x0, door_y0, MazeCell::DoorA);
    setc(door_x1, door_y1, MazeCell::DoorA);

    // Key: reachable from start with the door edge blocked; prefer a dead end.
    auto blocked = [&](int i, int j, int ni, int nj) {
      return (i == dai && j == daj && ni == dbi && nj == dbj) ||
             (i == dbi && j == dbj && ni == dai && nj == daj);
    };
    auto degree = [&](int i, int j) {
      int d = 0;
      for (int k = 0; k < 4; ++k) {
        const int ni = i + di[k], nj = j + dj[k];
        if (ni < 0 || nj < 0 || ni >= rx || nj >= ry) continue;
        if (connected(i, j, ni, nj) && !blocked(i, j, ni, nj)) ++d;
      }
      return d;
    };
    std::vector<uint8_t> reach(size_t(rx) * size_t(ry), 0);
    std::vector<int> fq;
    fq.push_back(rid(0, 0));
    reach[0] = 1;
    size_t fi = 0;
    while (fi < fq.size()) {
      const int id = fq[fi++];
      const int i = id % rx, j = id / rx;
      for (int k = 0; k < 4; ++k) {
        const int ni = i + di[k], nj = j + dj[k];
        if (ni < 0 || nj < 0 || ni >= rx || nj >= ry) continue;
        if (!connected(i, j, ni, nj) || blocked(i, j, ni, nj)) continue;
        const int nid = rid(ni, nj);
        if (reach[size_t(nid)]) continue;
        reach[size_t(nid)] = 1;
        fq.push_back(nid);
      }
    }
    int key_i = 0, key_j = 0;
    bool found_key_room = false;
    int best_score = -1;
    for (int j = 0; j < ry; ++j) {
      for (int i = 0; i < rx; ++i) {
        if (!reach[size_t(rid(i, j))]) continue;
        if (i == 0 && j == 0) continue;
        const int deg = degree(i, j);
        // Prefer dead ends, then rooms farther from the origin.
        const int score = ((deg == 1) ? 10000 : 0) + i + j;
        if (!found_key_room || score > best_score) {
          found_key_room = true;
          best_score = score;
          key_i = i;
          key_j = j;
        }
      }
    }
    if (!found_key_room) {
      // Fallback: another cell in the start room.
      setc(sx + 1, sy + 1, MazeCell::KeyA);
    } else {
      int kx = room_origin_x(key_i);
      int ky = room_origin_y(key_j);
      if (kx == sx && ky == sy) {
        kx = sx + 1;
        ky = sy + 1;
      }
      setc(kx, ky, MazeCell::KeyA);
    }

    // Pit candidates: 1x1 floor cells along a wall, with a safe opposite lane.
    auto reserved = [](MazeCell c) {
      return c == MazeCell::Wall || c == MazeCell::Start || c == MazeCell::KeyA ||
             c == MazeCell::DoorA || c == MazeCell::Exit || c == MazeCell::Pit;
    };
    auto open_lane = [&](int x, int y) {
      if (!g.in_bounds(x, y)) return false;
      const MazeCell c = g.cell(x, y);
      return !reserved(c);
    };
    // cand stores packed x,y and the chosen safe-lane packed cell.
    std::vector<int> cand;
    std::vector<int> lane_of;
    const int dxs[4] = {0, 0, 1, -1};
    const int dys[4] = {-1, 1, 0, 0};
    for (int y = 0; y < g.h_; ++y) {
      for (int x = 0; x < g.w_; ++x) {
        if (g.cell(x, y) != MazeCell::Floor) continue;
        for (int k = 0; k < 4; ++k) {
          const int wx = x + dxs[k], wy = y + dys[k];
          const int lx = x - dxs[k], ly = y - dys[k];
          if (!g.in_bounds(wx, wy) || g.cell(wx, wy) != MazeCell::Wall) continue;
          if (!open_lane(lx, ly)) continue;
          cand.push_back(y * g.w_ + x);
          lane_of.push_back(ly * g.w_ + lx);
          break;  // first valid orientation (N,S,E,W) is deterministic
        }
      }
    }
    if (cand.empty()) return g;

    // Pre-pit solvability: the 1-key branching layout must already reach
    // the exit. If it does not, leave the grid as-is (tests will fail).
    if (MazeOracle::shortest_length_avoiding_pits(g) < 0) return g;

    const int pit_n = 1 + int(rnd() % uint64_t(cand.size()));
    for (int i = int(cand.size()) - 1; i > 0; --i) {
      const int j = int(rnd() % uint64_t(i + 1));
      std::swap(cand[size_t(i)], cand[size_t(j)]);
      std::swap(lane_of[size_t(i)], lane_of[size_t(j)]);
    }
    std::vector<uint8_t> taken(size_t(g.w_) * size_t(g.h_), 0);
    int placed = 0;
    for (size_t i = 0; i < cand.size() && placed < pit_n; ++i) {
      const int p = cand[i];
      const int l = lane_of[i];
      if (taken[size_t(p)] || taken[size_t(l)]) continue;
      const int x = p % g.w_;
      const int y = p / g.w_;
      if (g.cell(x, y) != MazeCell::Floor) continue;
      g.cells_[g.index(x, y)] = MazeCell::Pit;
      // Keep the 1-cell opposite lane, and also reject any pit that would
      // make pits-as-walls BFS unable to reach the exit (e.g. trapping the
      // start or the key behind both lanes of a 2x2 room).
      if (MazeOracle::shortest_length_avoiding_pits(g) < 0) {
        g.cells_[g.index(x, y)] = MazeCell::Floor;
        continue;
      }
      taken[size_t(p)] = 1;
      taken[size_t(l)] = 1;
      ++placed;
    }
    return g;
  }
};

}  // namespace pchie
