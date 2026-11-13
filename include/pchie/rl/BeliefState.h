// L4. Belief-state layer (partial observability).
// Planning may use only this belief plus untrusted self-observation of move
// outcomes (still an open design question; this code assumes proprioception is allowed and
// is treated as untrusted). No get_block, BFS, cMcpWorldReader, or world-map
// read may feed planning, wall masks, or values under --rl-vision-only.
#pragma once

#include "pchie/rl/RlCommon.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

enum class Cell : uint8_t { Unknown = 0, Floor = 1, Wall = 2, Pit = 3, Visited = 4, Bumped = 5 };

enum class Heading : uint8_t { N = 0, E = 1, S = 2, W = 3 };

struct Pose {
  int32_t x = 0, y = 0;
  Heading heading = Heading::N;
};

enum class MoveOutcome : uint8_t { Moved = 0, Blocked = 1, Fell = 2 };  // Fell = terminal

struct ActionMask {
  bool n = true, e = true, s = true, w = true;
  bool allowed(Heading h) const {
    switch (h) {
      case Heading::N: return n;
      case Heading::E: return e;
      case Heading::S: return s;
      case Heading::W: return w;
    }
    return false;
  }
};

// Untrusted vision symbol grid: (2*radius+1) rows, each a string of
// U F W P V B. Row 0 is north. Cells are untrusted on ingest.
struct UntrustedSymbolGrid {
  int32_t radius = 0;
  std::vector<std::string> cells;  // row-major, north to south
};

struct PredictedObs {
  UntrustedSymbolGrid grid;
  bool predicted = true;  // L3 look-ahead nodes over belief are tagged predicted
};

inline int32_t dx_of(Heading h) { return h == Heading::E ? 1 : h == Heading::W ? -1 : 0; }
inline int32_t dy_of(Heading h) { return h == Heading::S ? 1 : h == Heading::N ? -1 : 0; }

inline char cell_char(Cell c) {
  switch (c) {
    case Cell::Floor: return 'F';
    case Cell::Wall: return 'W';
    case Cell::Pit: return 'P';
    case Cell::Visited: return 'V';
    case Cell::Bumped: return 'B';
    default: return 'U';
  }
}

inline Cell cell_from_char(char ch) {
  switch (ch) {
    case 'F': return Cell::Floor;
    case 'W': return Cell::Wall;
    case 'P': return Cell::Pit;
    case 'V': return Cell::Visited;
    case 'B': return Cell::Bumped;
    default: return Cell::Unknown;
  }
}

class BeliefState {
 public:
  void observe(const UntrustedSymbolGrid& g, const Pose& at) {
    pose_ = at;
    if (g.radius < 0) return;
    const int32_t dim = 2 * g.radius + 1;
    if (int32_t(g.cells.size()) != dim) return;
    for (int32_t r = 0; r < dim; ++r) {
      if (int32_t(g.cells[size_t(r)].size()) != dim) return;
    }
    // Row 0 is north => smaller y.
    for (int32_t r = 0; r < dim; ++r) {
      for (int32_t c = 0; c < dim; ++c) {
        const int32_t x = at.x + (c - g.radius);
        const int32_t y = at.y + (r - g.radius);
        const Cell seen = cell_from_char(g.cells[size_t(r)][size_t(c)]);
        if (seen == Cell::Unknown) continue;  // leave existing belief
        if (x == at.x && y == at.y) {
          set_cell(x, y, Cell::Visited);
          bump_revisit(x, y);
        } else {
          // Do not demote a Bumped cell back to Floor/Wall on a later glance
          // unless the new observation is also Bumped or Wall/Pit.
          const Cell cur = cell(x, y);
          if (cur == Cell::Bumped && (seen == Cell::Floor || seen == Cell::Visited)) continue;
          set_cell(x, y, seen == Cell::Visited ? Cell::Floor : seen);
        }
      }
    }
  }

  // action: 0=N 1=E 2=S 3=W (primitive). Fall is terminal.
  void move_result(int32_t action, MoveOutcome outcome) {
    if (terminal_) return;
    Heading h;
    switch (action) {
      case 0: h = Heading::N; break;
      case 1: h = Heading::E; break;
      case 2: h = Heading::S; break;
      case 3: h = Heading::W; break;
      default: return;
    }
    pose_.heading = h;
    const int32_t nx = pose_.x + dx_of(h);
    const int32_t ny = pose_.y + dy_of(h);
    if (outcome == MoveOutcome::Blocked) {
      set_cell(nx, ny, Cell::Bumped);
      return;
    }
    if (outcome == MoveOutcome::Fell) {
      set_cell(nx, ny, Cell::Pit);
      terminal_ = true;
      return;
    }
    pose_.x = nx;
    pose_.y = ny;
    set_cell(nx, ny, Cell::Visited);
    bump_revisit(nx, ny);
  }

  ActionMask mask() const {
    ActionMask m;
    auto blocked = [&](Heading h) {
      const Cell c = cell(pose_.x + dx_of(h), pose_.y + dy_of(h));
      return c == Cell::Wall || c == Cell::Pit || c == Cell::Bumped;
    };
    m.n = !blocked(Heading::N);
    m.e = !blocked(Heading::E);
    m.s = !blocked(Heading::S);
    m.w = !blocked(Heading::W);
    return m;
  }

  PredictedObs predict(const Pose& p) const {
    PredictedObs o;
    o.predicted = true;
    o.grid.radius = 1;
    o.grid.cells = {"UUU", "UUU", "UUU"};
    for (int32_t r = 0; r < 3; ++r) {
      for (int32_t c = 0; c < 3; ++c) {
        const int32_t x = p.x + (c - 1);
        const int32_t y = p.y + (r - 1);
        o.grid.cells[size_t(r)][size_t(c)] = cell_char(cell(x, y));
      }
    }
    if (o.grid.cells[1].size() >= 3) o.grid.cells[1][1] = 'V';
    return o;
  }

  std::array<uint8_t, 32> digest() const {
    std::vector<uint8_t> b;
    append_i32_le(b, pose_.x);
    append_i32_le(b, pose_.y);
    b.push_back(uint8_t(pose_.heading));
    b.push_back(terminal_ ? 1 : 0);
    // Stable order: sort keys.
    std::vector<std::pair<uint64_t, uint32_t>> items;
    items.reserve(cells_.size());
    for (const auto& kv : cells_) items.push_back(kv);
    std::sort(items.begin(), items.end());
    append_u32_le(b, uint32_t(items.size()));
    for (const auto& kv : items) {
      for (int i = 0; i < 8; ++i) b.push_back(uint8_t(kv.first >> (8 * i)));
      append_u32_le(b, kv.second);
    }
    std::vector<std::pair<uint64_t, uint32_t>> revs;
    revs.reserve(revisits_.size());
    for (const auto& kv : revisits_) revs.push_back(kv);
    std::sort(revs.begin(), revs.end());
    append_u32_le(b, uint32_t(revs.size()));
    for (const auto& kv : revs) {
      for (int i = 0; i < 8; ++i) b.push_back(uint8_t(kv.first >> (8 * i)));
      append_u32_le(b, kv.second);
    }
    return sha256_bytes(b.data(), b.size());
  }

  Cell cell(int32_t x, int32_t y) const {
    auto it = cells_.find(key(x, y));
    return it == cells_.end() ? Cell::Unknown : Cell(it->second);
  }
  uint32_t revisit_count(int32_t x, int32_t y) const {
    auto it = revisits_.find(key(x, y));
    return it == revisits_.end() ? 0u : it->second;
  }
  const Pose& pose() const { return pose_; }
  bool terminal() const { return terminal_; }

 private:
  static uint64_t key(int32_t x, int32_t y) {
    return (uint64_t(uint32_t(x)) << 32) | uint64_t(uint32_t(y));
  }
  void set_cell(int32_t x, int32_t y, Cell c) { cells_[key(x, y)] = uint32_t(c); }
  void bump_revisit(int32_t x, int32_t y) { revisits_[key(x, y)] += 1; }

  Pose pose_{};
  bool terminal_ = false;
  std::map<uint64_t, uint32_t> cells_;
  std::map<uint64_t, uint32_t> revisits_;
};

}  // namespace rl
}  // namespace pchie
