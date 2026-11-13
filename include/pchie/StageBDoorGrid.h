// Bolt PCHIE Stage B - labeled sectional grid with two L1 logic-gate doors.
// Concrete final. Unit/labeled only, never a golden, never the official B1
// grid. MazeGrid and its 1-key DoorA are not used or edited; only
// the MazeAction move ids (North/South/East/West) are reused. A door opens
// only through StageBLogicDoor (gate + SimTraceGrader pass). No pits in
// this first fixture. Deterministic. No mutex, no atomic.
#pragma once

#include "pchie/MazeGrid.h"

#include <cstdint>
#include <vector>

namespace pchie {

class StageBDoorFactory;
class StageBLogicDoor;

enum class StageBCell : uint8_t {
  Floor = 0,
  Wall = 1,
  Start = 2,
  Door = 3,
  Exit = 4
};

class StageBDoorGrid final {
  friend class StageBDoorFactory;
  friend class StageBLogicDoor;

 public:
  static constexpr int kDoors = 2;

  StageBDoorGrid() = default;

  uint64_t seed() const noexcept { return seed_; }
  int width() const noexcept { return w_; }
  int height() const noexcept { return h_; }
  int x() const noexcept { return x_; }
  int y() const noexcept { return y_; }

  bool in_bounds(int x, int y) const noexcept { return x >= 0 && y >= 0 && x < w_ && y < h_; }

  StageBCell cell(int x, int y) const {
    if (!in_bounds(x, y)) return StageBCell::Wall;
    return cells_[index(x, y)];
  }

  // Door id at (x, y), or -1 when the cell is not a door.
  int door_at(int x, int y) const {
    if (!in_bounds(x, y)) return -1;
    return int(door_ids_[index(x, y)]);
  }

  bool door_open(int k) const noexcept { return k >= 0 && k < kDoors && door_open_[k]; }

  bool walkable(int x, int y) const { return walkable_with(x, y, false); }

  // doors_as_open is an oracle-only view; it never changes the world.
  bool walkable_with(int x, int y, bool doors_as_open) const {
    if (!in_bounds(x, y)) return false;
    const StageBCell c = cells_[index(x, y)];
    if (c == StageBCell::Wall) return false;
    if (c == StageBCell::Door && !doors_as_open && !door_open(door_at(x, y))) return false;
    return true;
  }

  bool adjacent_to_door(int k) const { return adjacent_to_door_at(x_, y_, k); }

  bool adjacent_to_door_at(int px, int py, int k) const {
    if (k < 0 || k >= kDoors) return false;
    const int dxs[4] = {0, 0, 1, -1};
    const int dys[4] = {-1, 1, 0, 0};
    for (int i = 0; i < 4; ++i) {
      const int nx = px + dxs[i], ny = py + dys[i];
      if (cell(nx, ny) == StageBCell::Door && door_at(nx, ny) == k) return true;
    }
    return false;
  }

  bool at_exit() const { return cell(x_, y_) == StageBCell::Exit; }

  // Moves only. Answering a door is not a grid action; see StageBLogicDoor.
  std::vector<MazeAction> legal_actions() const {
    std::vector<MazeAction> out;
    if (walkable(x_, y_ - 1)) out.push_back(MazeAction::North);
    if (walkable(x_, y_ + 1)) out.push_back(MazeAction::South);
    if (walkable(x_ + 1, y_)) out.push_back(MazeAction::East);
    if (walkable(x_ - 1, y_)) out.push_back(MazeAction::West);
    return out;
  }

  bool apply(MazeAction a) {
    switch (a) {
      case MazeAction::North: return step(0, -1);
      case MazeAction::South: return step(0, 1);
      case MazeAction::East: return step(1, 0);
      case MazeAction::West: return step(-1, 0);
      default: return false;
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
    // Domain tag so a Stage B world can never hash like a MazeGrid world.
    b.push_back('S');
    b.push_back('B');
    b.push_back('D');
    b.push_back('G');
    put_u64(seed_);
    put_u32(uint32_t(w_));
    put_u32(uint32_t(h_));
    put_u32(uint32_t(x_));
    put_u32(uint32_t(y_));
    for (int k = 0; k < kDoors; ++k) b.push_back(door_open_[k] ? 1 : 0);
    for (StageBCell c : cells_) b.push_back(uint8_t(c));
    for (int8_t d : door_ids_) b.push_back(uint8_t(d));
    return b;
  }

  bool operator==(const StageBDoorGrid& o) const {
    if (seed_ != o.seed_ || w_ != o.w_ || h_ != o.h_ || x_ != o.x_ || y_ != o.y_) return false;
    for (int k = 0; k < kDoors; ++k)
      if (door_open_[k] != o.door_open_[k]) return false;
    return cells_ == o.cells_ && door_ids_ == o.door_ids_;
  }
  bool operator!=(const StageBDoorGrid& o) const { return !(*this == o); }

 private:
  size_t index(int x, int y) const { return size_t(y) * size_t(w_) + size_t(x); }

  bool step(int dx, int dy) {
    const int nx = x_ + dx, ny = y_ + dy;
    if (!walkable(nx, ny)) return false;
    x_ = nx;
    y_ = ny;
    return true;
  }

  // Friend-only (StageBLogicDoor). Opens exactly one door id.
  bool open_door(int k) {
    if (k < 0 || k >= kDoors || door_open_[k]) return false;
    door_open_[k] = true;
    return true;
  }

  uint64_t seed_ = 0;
  int w_ = 0, h_ = 0, x_ = 0, y_ = 0;
  bool door_open_[kDoors] = {false, false};
  std::vector<StageBCell> cells_;
  std::vector<int8_t> door_ids_;
};

}  // namespace pchie