// Bolt PCHIE Stage B - labeled sectional 2x2-corridor factory, two doors.
// Concrete final. Unit/labeled only, never a golden. Not the Stage A 64x64
// grid and not the official B1 grid. Three sections side by side,
// each a 3x3 grid of 2x2 rooms with 1-wide walls, each its own
// recursive-backtracker room tree. Sections are linked only by a 2-cell
// door throat: door 0 (section 0 -> 1) and door 1 (section 1 -> 2).
// No pits. Deterministic Splitmix64. No mutex.
#pragma once

#include "pchie/NodeSeed.h"
#include "pchie/StageBDoorGrid.h"

#include <cstdint>
#include <vector>

namespace pchie {

class StageBDoorFactory final {
 public:
  static constexpr uint64_t kLabeledSeed = 7;
  static constexpr uint64_t kStream = 0x5B0D0025ULL;  // separate from B1 / Stage A streams
  static constexpr int kStep = 3;
  static constexpr int kSections = 3;
  static constexpr int kSectionRoomsX = 3;
  static constexpr int kRoomsY = 3;
  static constexpr int kRoomsX = kSections * kSectionRoomsX;
  static constexpr int kWidth = 1 + kStep * kRoomsX;   // 28
  static constexpr int kHeight = 1 + kStep * kRoomsY;  // 10

  static int room_origin_x(int i) { return 1 + kStep * i; }
  static int room_origin_y(int j) { return 1 + kStep * j; }

  static StageBDoorGrid make(uint64_t seed = kLabeledSeed) {
    StageBDoorGrid g;
    g.seed_ = seed;
    g.w_ = kWidth;
    g.h_ = kHeight;
    g.cells_.assign(size_t(kWidth) * size_t(kHeight), StageBCell::Wall);
    g.door_ids_.assign(size_t(kWidth) * size_t(kHeight), int8_t(-1));
    for (int k = 0; k < StageBDoorGrid::kDoors; ++k) g.door_open_[k] = false;

    auto setc = [&](int x, int y, StageBCell c) {
      if (g.in_bounds(x, y)) g.cells_[g.index(x, y)] = c;
    };
    for (int j = 0; j < kRoomsY; ++j) {
      for (int i = 0; i < kRoomsX; ++i) {
        const int x = room_origin_x(i), y = room_origin_y(j);
        setc(x, y, StageBCell::Floor);
        setc(x + 1, y, StageBCell::Floor);
        setc(x, y + 1, StageBCell::Floor);
        setc(x + 1, y + 1, StageBCell::Floor);
      }
    }

    uint64_t rng = NodeSeed::Splitmix64(seed ^ kStream);
    auto rnd = [&]() -> uint64_t {
      rng = NodeSeed::Splitmix64(rng);
      return rng;
    };

    // Door rows first so each section's tree starts at its entry room.
    int door_row[kSections - 1];
    for (int s = 0; s < kSections - 1; ++s) door_row[s] = int(rnd() % uint64_t(kRoomsY));

    const int di[4] = {1, -1, 0, 0};
    const int dj[4] = {0, 0, 1, -1};
    for (int s = 0; s < kSections; ++s) {
      const int i0 = s * kSectionRoomsX;
      const int entry_j = (s == 0) ? 0 : door_row[s - 1];
      std::vector<uint8_t> vis(size_t(kSectionRoomsX) * size_t(kRoomsY), 0);
      auto lid = [&](int i, int j) { return j * kSectionRoomsX + (i - i0); };
      std::vector<int> st;
      vis[size_t(lid(i0, entry_j))] = 1;
      st.push_back(lid(i0, entry_j));
      while (!st.empty()) {
        const int id = st.back();
        const int i = i0 + id % kSectionRoomsX;
        const int j = id / kSectionRoomsX;
        int opts[4];
        int nopt = 0;
        for (int k = 0; k < 4; ++k) {
          const int ni = i + di[k], nj = j + dj[k];
          if (ni < i0 || ni >= i0 + kSectionRoomsX || nj < 0 || nj >= kRoomsY) continue;
          if (vis[size_t(lid(ni, nj))]) continue;
          opts[nopt++] = k;
        }
        if (nopt == 0) {
          st.pop_back();
          continue;
        }
        const int k = opts[int(rnd() % uint64_t(nopt))];
        const int ni = i + di[k], nj = j + dj[k];
        if (ni != i) {
          const int xw = room_origin_x(ni > i ? i : ni) + 2;
          const int y0 = room_origin_y(j);
          setc(xw, y0, StageBCell::Floor);
          setc(xw, y0 + 1, StageBCell::Floor);
        } else {
          const int yw = room_origin_y(nj > j ? j : nj) + 2;
          const int x0 = room_origin_x(i);
          setc(x0, yw, StageBCell::Floor);
          setc(x0 + 1, yw, StageBCell::Floor);
        }
        vis[size_t(lid(ni, nj))] = 1;
        st.push_back(lid(ni, nj));
      }
    }

    // Door throats: the wall column between the last room of section s and
    // the first room of section s+1, both cells of the 2-wide opening.
    for (int s = 0; s < kSections - 1; ++s) {
      const int x = room_origin_x((s + 1) * kSectionRoomsX - 1) + 2;
      const int y = room_origin_y(door_row[s]);
      for (int dy = 0; dy < 2; ++dy) {
        setc(x, y + dy, StageBCell::Door);
        g.door_ids_[g.index(x, y + dy)] = int8_t(s);
      }
    }

    const int sx = room_origin_x(0), sy = room_origin_y(0);
    setc(sx, sy, StageBCell::Start);
    g.x_ = sx;
    g.y_ = sy;
    setc(room_origin_x(kRoomsX - 1) + 1, room_origin_y(kRoomsY - 1) + 1, StageBCell::Exit);
    return g;
  }
};

}  // namespace pchie