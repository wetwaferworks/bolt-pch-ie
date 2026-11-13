// Bolt PCHIE B1 - headless flat-world maze grid.
// Concrete final. No mutex. Seeded, deterministic. Headless only (no voxels, no watch mode).
// Default size is 16x16, 1-key (the spec's 32x32 / 3-key / pressure-plate tiers are deferred;
// (an open review question). Always-solvable corridor plus seeded dead ends.
#pragma once

#include "pchie/NodeSeed.h"

#include <cstdint>
#include <vector>

namespace pchie {

class PitBesidePath;
class ScaledPitMaze;

enum class MazeCell : uint8_t {
  Floor = 0,
  Wall = 1,
  Start = 2,
  KeyA = 3,
  DoorA = 4,
  Exit = 5,
  Pit = 6
};

// Action ids avoid the default PchGate reject set and EOS / <turn|>.
enum class MazeAction : int32_t {
  North = 7,
  South = 8,
  East = 9,
  West = 10,
  Pickup = 11,
  Use = 12
};

class MazeGrid final {
  friend class PitBesidePath;
  friend class ScaledPitMaze;

 public:
  static constexpr int kDefaultWidth = 16;
  static constexpr int kDefaultHeight = 16;

  MazeGrid() = default;

  MazeGrid(uint64_t seed, int width, int height) : seed_(seed), w_(width), h_(height) {
    generate();
  }

  // Labeled unit-test fixture only: unsolvable (all walls, no exit). Never a golden.
  static MazeGrid closed_box(int width, int height) {
    MazeGrid g;
    g.seed_ = 0;
    g.w_ = width;
    g.h_ = height;
    if (g.w_ < 0) g.w_ = 0;
    if (g.h_ < 0) g.h_ = 0;
    g.cells_.assign(size_t(g.w_) * size_t(g.h_), MazeCell::Wall);
    g.x_ = 0;
    g.y_ = 0;
    g.have_key_ = false;
    g.door_open_ = false;
    return g;
  }

  // Labeled unit-test fixture only: one pit on the apparent shortcut, with a
  // safe route around it to the exit. Never a golden or baseline.
  // Layout (7x6, # wall, S start, P pit, E exit):
  //   #######
  //   #S# #E#
  //   # P   #
  //   # ### #
  //   #     #
  //   #######
  // Pit-avoiding BFS length is 10.
  static MazeGrid pit_safe_bypass() {
    MazeGrid g;
    g.seed_ = 0;
    g.w_ = 7;
    g.h_ = 6;
    g.cells_.assign(size_t(g.w_) * size_t(g.h_), MazeCell::Wall);
    auto setc = [&](int x, int y, MazeCell c) { g.cells_[g.index(x, y)] = c; };
    setc(1, 1, MazeCell::Start);
    setc(3, 1, MazeCell::Floor);
    setc(5, 1, MazeCell::Exit);
    setc(1, 2, MazeCell::Floor);
    setc(2, 2, MazeCell::Pit);
    setc(3, 2, MazeCell::Floor);
    setc(4, 2, MazeCell::Floor);
    setc(5, 2, MazeCell::Floor);
    setc(1, 3, MazeCell::Floor);
    setc(5, 3, MazeCell::Floor);
    setc(1, 4, MazeCell::Floor);
    setc(2, 4, MazeCell::Floor);
    setc(3, 4, MazeCell::Floor);
    setc(4, 4, MazeCell::Floor);
    setc(5, 4, MazeCell::Floor);
    g.x_ = 1;
    g.y_ = 1;
    g.have_key_ = false;
    g.door_open_ = false;
    return g;
  }

  uint64_t seed() const noexcept { return seed_; }
  int width() const noexcept { return w_; }
  int height() const noexcept { return h_; }
  int x() const noexcept { return x_; }
  int y() const noexcept { return y_; }
  bool have_key() const noexcept { return have_key_; }
  bool door_open() const noexcept { return door_open_; }

  MazeCell cell(int x, int y) const {
    if (!in_bounds(x, y)) return MazeCell::Wall;
    return cells_[index(x, y)];
  }

  bool in_bounds(int x, int y) const noexcept { return x >= 0 && y >= 0 && x < w_ && y < h_; }

  bool is_pit(int x, int y) const {
    return in_bounds(x, y) && cells_[index(x, y)] == MazeCell::Pit;
  }

  bool in_pit() const noexcept { return is_pit(x_, y_); }

  bool walkable(int x, int y) const {
    if (!in_bounds(x, y)) return false;
    const MazeCell c = cells_[index(x, y)];
    if (c == MazeCell::Wall) return false;
    if (c == MazeCell::DoorA && !door_open_) return false;
    return true;  // Pit is enterable (fall in); not a wall.
  }

  // For the pit-avoiding BFS: pits are treated as walls.
  bool walkable_safe(int x, int y) const { return walkable(x, y) && !is_pit(x, y); }

  bool on_key() const noexcept { return in_bounds(x_, y_) && cells_[index(x_, y_)] == MazeCell::KeyA && !have_key_; }

  bool at_exit() const noexcept { return in_bounds(x_, y_) && cells_[index(x_, y_)] == MazeCell::Exit; }

  bool adjacent_to_door() const {
    int dx = 0, dy = 0;
    if (!find(MazeCell::DoorA, dx, dy)) return false;
    const int ax = x_ > dx ? x_ - dx : dx - x_;
    const int ay = y_ > dy ? y_ - dy : dy - y_;
    return (ax + ay) == 1;
  }

  bool find(MazeCell c, int& ox, int& oy) const {
    for (int y = 0; y < h_; ++y) {
      for (int x = 0; x < w_; ++x) {
        if (cells_[index(x, y)] == c) {
          ox = x;
          oy = y;
          return true;
        }
      }
    }
    ox = oy = -1;
    return false;
  }

  std::vector<MazeAction> legal_actions() const {
    std::vector<MazeAction> out;
    if (in_pit()) return out;  // fail/stuck: no further moves, never a win
    if (walkable(x_, y_ - 1)) out.push_back(MazeAction::North);
    if (walkable(x_, y_ + 1)) out.push_back(MazeAction::South);
    if (walkable(x_ + 1, y_)) out.push_back(MazeAction::East);
    if (walkable(x_ - 1, y_)) out.push_back(MazeAction::West);
    if (on_key()) out.push_back(MazeAction::Pickup);
    if (have_key_ && !door_open_ && adjacent_to_door()) out.push_back(MazeAction::Use);
    return out;
  }

  bool apply(MazeAction a) {
    if (in_pit()) return false;  // stuck: cannot leave or act, never a win
    switch (a) {
      case MazeAction::North: return step(0, -1);
      case MazeAction::South: return step(0, 1);
      case MazeAction::East: return step(1, 0);
      case MazeAction::West: return step(-1, 0);
      case MazeAction::Pickup:
        if (!on_key()) return false;
        have_key_ = true;
        cells_[index(x_, y_)] = MazeCell::Floor;
        return true;
      case MazeAction::Use:
        if (!(have_key_ && !door_open_ && adjacent_to_door())) return false;
        door_open_ = true;
        return true;
      default:
        return false;
    }
  }

  std::vector<uint8_t> canonical_bytes() const {
    std::vector<uint8_t> b;
    auto put_u64 = [&](uint64_t v) {
      for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i)));
    };
    auto put_u32 = [&](uint32_t v) {
      for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i)));
    };
    put_u64(seed_);
    put_u32(uint32_t(w_));
    put_u32(uint32_t(h_));
    put_u32(uint32_t(x_));
    put_u32(uint32_t(y_));
    b.push_back(have_key_ ? 1 : 0);
    b.push_back(door_open_ ? 1 : 0);
    b.insert(b.end(), reinterpret_cast<const uint8_t*>(cells_.data()),
             reinterpret_cast<const uint8_t*>(cells_.data()) + cells_.size() * sizeof(MazeCell));
    return b;
  }

  bool operator==(const MazeGrid& o) const {
    return seed_ == o.seed_ && w_ == o.w_ && h_ == o.h_ && x_ == o.x_ && y_ == o.y_ &&
           have_key_ == o.have_key_ && door_open_ == o.door_open_ && cells_ == o.cells_;
  }
  bool operator!=(const MazeGrid& o) const { return !(*this == o); }

 private:
  size_t index(int x, int y) const { return size_t(y) * size_t(w_) + size_t(x); }

  bool step(int dx, int dy) {
    const int nx = x_ + dx, ny = y_ + dy;
    if (!walkable(nx, ny)) return false;
    x_ = nx;
    y_ = ny;
    return true;
  }

  void set_floor_interior(int x, int y) {
    if (x <= 0 || y <= 0 || x >= w_ - 1 || y >= h_ - 1) return;
    cells_[index(x, y)] = MazeCell::Floor;
  }

  void carve(int x0, int y0, int x1, int y1, uint64_t& rng) {
    const bool horiz_first = (NodeSeed::Splitmix64(rng) & 1ull) != 0;
    rng = NodeSeed::Splitmix64(rng);
    auto run = [&](int sx, int sy, int tx, int ty, bool horiz) {
      int x = sx, y = sy;
      if (horiz) {
        while (x != tx) {
          x += (tx > x) ? 1 : -1;
          set_floor_interior(x, y);
        }
        while (y != ty) {
          y += (ty > y) ? 1 : -1;
          set_floor_interior(x, y);
        }
      } else {
        while (y != ty) {
          y += (ty > y) ? 1 : -1;
          set_floor_interior(x, y);
        }
        while (x != tx) {
          x += (tx > x) ? 1 : -1;
          set_floor_interior(x, y);
        }
      }
    };
    run(x0, y0, x1, y1, horiz_first);
  }


  bool reachable_door_closed(MazeCell target) const {
    if (w_ <= 0 || h_ <= 0) return false;
    int kx = -1, ky = -1;
    if (!find(target, kx, ky)) return false;
    const int n = w_ * h_;
    std::vector<uint8_t> seen(size_t(n), 0);
    std::vector<int> st;
    auto push = [&](int x, int y) {
      if (!in_bounds(x, y)) return;
      const MazeCell c = cells_[index(x, y)];
      if (c == MazeCell::Wall || c == MazeCell::DoorA) return;
      const int i = y * w_ + x;
      if (seen[size_t(i)]) return;
      seen[size_t(i)] = 1;
      st.push_back(i);
    };
    push(x_, y_);
    while (!st.empty()) {
      const int i = st.back();
      st.pop_back();
      const int x = i % w_, y = i / w_;
      if (x == kx && y == ky) return true;
      push(x, y - 1);
      push(x, y + 1);
      push(x + 1, y);
      push(x - 1, y);
    }
    return false;
  }

  void layout_fallback() {
    cells_.assign(size_t(w_) * size_t(h_), MazeCell::Wall);
    const int sx = 1, sy = 1;
    const int kx = w_ - 3, ky = 1;
    const int dx = w_ - 3, dy = h_ - 3;
    const int ex = 1, ey = h_ - 2;
    uint64_t rng = NodeSeed::Splitmix64(seed_ ^ 0xB1FBULL);
    set_floor_interior(sx, sy);
    set_floor_interior(kx, ky);
    set_floor_interior(dx, dy);
    set_floor_interior(ex, ey);
    carve(sx, sy, kx, ky, rng);
    carve(kx, ky, dx, dy, rng);
    carve(dx, dy, ex, ey, rng);
    cells_[index(sx, sy)] = MazeCell::Start;
    cells_[index(kx, ky)] = MazeCell::KeyA;
    cells_[index(dx, dy)] = MazeCell::DoorA;
    cells_[index(ex, ey)] = MazeCell::Exit;
    x_ = sx;
    y_ = sy;
    have_key_ = false;
    door_open_ = false;
  }

  void generate() {
    cells_.clear();
    have_key_ = false;
    door_open_ = false;
    x_ = 0;
    y_ = 0;
    if (w_ < 5 || h_ < 5) {
      if (w_ < 0) w_ = 0;
      if (h_ < 0) h_ = 0;
      cells_.assign(size_t(w_) * size_t(h_), MazeCell::Wall);
      return;
    }
    cells_.assign(size_t(w_) * size_t(h_), MazeCell::Wall);
    uint64_t rng = NodeSeed::Splitmix64(seed_ ^ 0xB100000000000001ULL);

    auto rint = [&](int lo, int hi) -> int {
      if (hi <= lo) return lo;
      rng = NodeSeed::Splitmix64(rng);
      return lo + int(rng % uint64_t(hi - lo + 1));
    };

    int sx = 1, sy = 1;
    int kx = w_ - 3, ky = 1;
    int dx = w_ - 3, dy = h_ - 3;
    int ex = 1, ey = h_ - 2;

    auto occupied = [&](int x, int y, int n, const int* xs, const int* ys) {
      for (int i = 0; i < n; ++i) if (xs[i] == x && ys[i] == y) return true;
      return false;
    };

    // Seeded interior placements; fall back to the corridor corners if they collide.
    for (int t = 0; t < 64; ++t) {
      const int x = rint(1, w_ - 2);
      const int y = rint(1, h_ - 2);
      int xs[1] = {sx}, ys[1] = {sy};
      if (!occupied(x, y, 1, xs, ys)) { kx = x; ky = y; break; }
    }
    for (int t = 0; t < 64; ++t) {
      const int x = rint(1, w_ - 2);
      const int y = rint(1, h_ - 2);
      int xs[2] = {sx, kx}, ys[2] = {sy, ky};
      if (!occupied(x, y, 2, xs, ys)) { dx = x; dy = y; break; }
    }
    for (int t = 0; t < 64; ++t) {
      const int x = rint(1, w_ - 2);
      const int y = rint(1, h_ - 2);
      int xs[3] = {sx, kx, dx}, ys[3] = {sy, ky, dy};
      if (!occupied(x, y, 3, xs, ys)) { ex = x; ey = y; break; }
    }
    // Final collision fallback (always distinct for w,h >= 5).
    if ((kx == sx && ky == sy) || (dx == sx && dy == sy) || (dx == kx && dy == ky) ||
        (ex == sx && ey == sy) || (ex == kx && ey == ky) || (ex == dx && ey == dy)) {
      sx = 1; sy = 1;
      kx = w_ - 3; ky = 1;
      dx = w_ - 3; dy = h_ - 3;
      ex = 1; ey = h_ - 2;
    }

    set_floor_interior(sx, sy);
    set_floor_interior(kx, ky);
    set_floor_interior(dx, dy);
    set_floor_interior(ex, ey);
    carve(sx, sy, kx, ky, rng);
    carve(kx, ky, dx, dy, rng);
    carve(dx, dy, ex, ey, rng);

    const int extras = w_ + h_;
    for (int i = 0; i < extras; ++i) {
      const int x = rint(1, w_ - 2);
      const int y = rint(1, h_ - 2);
      if (cell(x, y) == MazeCell::Wall) continue;
      const int dir = rint(0, 3);
      const int dxs[4] = {0, 0, 1, -1};
      const int dys[4] = {-1, 1, 0, 0};
      int cx = x, cy = y;
      const int steps = rint(1, 3);
      for (int s = 0; s < steps; ++s) {
        cx += dxs[dir];
        cy += dys[dir];
        set_floor_interior(cx, cy);
      }
    }

    cells_[index(sx, sy)] = MazeCell::Start;
    cells_[index(kx, ky)] = MazeCell::KeyA;
    cells_[index(dx, dy)] = MazeCell::DoorA;
    cells_[index(ex, ey)] = MazeCell::Exit;
    x_ = sx;
    y_ = sy;
    // The key must be reachable and the exit must need the door (1-key B1 tier).
    if (!reachable_door_closed(MazeCell::KeyA) || reachable_door_closed(MazeCell::Exit)) layout_fallback();
  }

  uint64_t seed_ = 0;
  int w_ = 0, h_ = 0, x_ = 0, y_ = 0;
  bool have_key_ = false, door_open_ = false;
  std::vector<MazeCell> cells_;
};

}  // namespace pchie
