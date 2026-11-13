// Bolt PCHIE - where a B1 grid lives in LithoKraft voxels, and the read-back check.
// Concrete final. Pure, deterministic, no mutex. The game builds the maze from LkMazeText
// (cPchieMazeWorld); before the model plays, every cell is read back over MCP
// (world_get_block) and must match the B1 grid the model decides on, so the maze the
// model sees and the maze the player walks cannot drift apart (B1 spec, watch mode).
#pragma once

#include "pchie/MazeGrid.h"

#include <cmath>
#include <cstdint>
#include <string>

namespace pchie {

// Block ids the game logs as "pchie maze: legend ..." (runtime registry ids).
struct LkLegend {
  uint16_t air = 0;
  uint16_t floor = 0, wall = 0, start = 0, exit = 0, key = 0, door = 0;
};

class LkMazeVoxelMap final {
 public:
  static constexpr int kLayers = 3;  // floorY, floorY + 1, floorY + 2

  LkMazeVoxelMap() = default;
  LkMazeVoxelMap(int origin_x, int floor_y, int origin_z) : ox_(origin_x), fy_(floor_y), oz_(origin_z) {}

  int origin_x() const noexcept { return ox_; }
  int floor_y() const noexcept { return fy_; }
  int origin_z() const noexcept { return oz_; }
  int block_x(int cx) const noexcept { return ox_ + cx; }
  int block_z(int cy) const noexcept { return oz_ + cy; }
  double centre_x(int cx) const noexcept { return double(ox_ + cx) + 0.5; }
  double centre_z(int cy) const noexcept { return double(oz_ + cy) + 0.5; }

  bool world_to_cell(double x, double z, int& cx, int& cy) const noexcept {
    cx = int(std::floor(x)) - ox_;
    cy = int(std::floor(z)) - oz_;
    return true;
  }

  // Expected ids for (floorY, floorY+1, floorY+2) of one cell.
  static void expected(MazeCell c, const LkLegend& L, uint16_t out[kLayers]) {
    out[0] = L.floor;
    out[1] = L.air;
    out[2] = L.air;
    switch (c) {
      case MazeCell::Wall: out[1] = L.wall; out[2] = L.wall; break;
      case MazeCell::Start: out[0] = L.start; break;
      case MazeCell::Exit: out[0] = L.exit; break;
      case MazeCell::KeyA: out[1] = L.key; break;
      case MazeCell::DoorA: out[1] = L.door; out[2] = L.door; break;
      case MazeCell::Pit: out[0] = L.air; break;
      case MazeCell::Floor: break;
    }
  }

  // reader(x, y, z, id&) -> bool. Returns the number of mismatched cells (-1 on a read error).
  template <class Reader>
  int verify(const MazeGrid& g, const LkLegend& L, Reader&& reader, std::string& first) const {
    int bad = 0;
    first.clear();
    for (int y = 0; y < g.height(); ++y) {
      for (int x = 0; x < g.width(); ++x) {
        uint16_t want[kLayers];
        expected(g.cell(x, y), L, want);
        bool cell_bad = false;
        for (int k = 0; k < kLayers; ++k) {
          uint16_t got = 0;
          if (!reader(block_x(x), fy_ + k, block_z(y), got)) {
            first = "read failed at cell (" + std::to_string(x) + "," + std::to_string(y) + ")";
            return -1;
          }
          if (got != want[k]) {
            if (first.empty())
              first = "cell (" + std::to_string(x) + "," + std::to_string(y) + ") layer " +
                      std::to_string(k) + ": want " + std::to_string(want[k]) + " got " +
                      std::to_string(got);
            cell_bad = true;
          }
        }
        if (cell_bad) ++bad;
      }
    }
    return bad;
  }

 private:
  int ox_ = 0, fy_ = 0, oz_ = 0;
};

}  // namespace pchie
