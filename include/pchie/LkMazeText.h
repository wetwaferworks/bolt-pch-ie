// Bolt PCHIE - B1 grid <-> LithoKraft --pchie-maze text (one char per cell).
// Concrete final. Pure, deterministic, no mutex. Labeled view-only example input, never a golden.
// Format (must match LithoKraft cPchieMazeWorld.h):
//   PCHIE-B1 <width> <height> <seed>
//   <height rows of width chars>   # wall  . floor  S start  K key  D door  E exit  P pit
// Row y = 0 is the north row (B1 North = y - 1 = world -z).
#pragma once

#include "pchie/MazeGrid.h"

#include <string>

namespace pchie {

class LkMazeText final {
 public:
  static char glyph(MazeCell c) {
    switch (c) {
      case MazeCell::Wall: return '#';
      case MazeCell::Floor: return '.';
      case MazeCell::Start: return 'S';
      case MazeCell::KeyA: return 'K';
      case MazeCell::DoorA: return 'D';
      case MazeCell::Exit: return 'E';
      case MazeCell::Pit: return 'P';
    }
    return '?';
  }

  static std::string encode(const MazeGrid& g) {
    std::string s = "PCHIE-B1 " + std::to_string(g.width()) + " " + std::to_string(g.height()) + " " +
                    std::to_string(g.seed()) + "\n";
    for (int y = 0; y < g.height(); ++y) {
      for (int x = 0; x < g.width(); ++x) s += glyph(g.cell(x, y));
      s += "\n";
    }
    return s;
  }
};

}  // namespace pchie
