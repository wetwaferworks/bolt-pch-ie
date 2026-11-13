// Bolt PCHIE - turns the reasoning model's committed B1 actions into in-engine steps.
// Concrete final. Pure, deterministic, no mutex. The decisions are the model's; this only
// says how a keyboard player carries each one out in LithoKraft:
//   North/South/East/West -> Walk to the neighbour cell centre (a KeyA cell holds a Lamp
//                            block, so BreakKey first: the player breaks it from the cell before)
//   Pickup                -> PickUp (no input; the key block was broken on the way in)
//   Use                   -> BreakDoor for every DoorA cell (both Frosted Glass blocks of each).
//                            B1 has one door cell, so this is one BreakDoor exactly as in the in-engine pilot;
//                            the Stage A 64x64 fixture has a 2-wide door (two DoorA cells), and
//                            the grid's Use opens the whole door, so the engine must too.
//                            (2026-11-10 LithoKraft testbeds; backup *.pre-lkbed.bak)
// It replays the actions on a copy of the grid; an action that is illegal on the grid fails
// the whole plan (fail closed), so a plan can never ask the player to walk through a wall.
#pragma once

#include "pchie/MazeGrid.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

struct LkStep {
  enum class Kind : uint8_t { Walk, BreakKey, PickUp, BreakDoor };
  Kind kind = Kind::Walk;
  int from_x = 0, from_y = 0;  // player cell before the step
  int to_x = 0, to_y = 0;      // Walk: destination; BreakKey/BreakDoor: the cell whose block breaks
  int32_t action = 0;          // the MazeAction this step serves
  int action_index = 0;        // index into the model's action list
};

class LkActuationPlan final {
 public:
  static bool build(const MazeGrid& start, const std::vector<int32_t>& actions, std::vector<LkStep>& out,
                    std::string& err) {
    out.clear();
    err.clear();
    MazeGrid g = start;
    bool key_block_present = true;
    for (size_t i = 0; i < actions.size(); ++i) {
      const auto a = static_cast<MazeAction>(actions[i]);
      const int x = g.x(), y = g.y();
      LkStep s;
      s.from_x = x;
      s.from_y = y;
      s.action = actions[i];
      s.action_index = int(i);
      int dx = 0, dy = 0;
      std::vector<LkStep> door_steps;
      switch (a) {
        case MazeAction::North: dy = -1; break;
        case MazeAction::South: dy = 1; break;
        case MazeAction::East: dx = 1; break;
        case MazeAction::West: dx = -1; break;
        case MazeAction::Pickup: s.kind = LkStep::Kind::PickUp; s.to_x = x; s.to_y = y; break;
        case MazeAction::Use: {
          for (int cy = 0; cy < g.height(); ++cy)
            for (int cx = 0; cx < g.width(); ++cx)
              if (g.cell(cx, cy) == MazeCell::DoorA) {
                LkStep d = s;
                d.kind = LkStep::Kind::BreakDoor;
                d.to_x = cx;
                d.to_y = cy;
                door_steps.push_back(d);
              }
          if (door_steps.empty()) {
            err = "Use without a door";
            out.clear();
            return false;
          }
          s = door_steps.front();
          break;
        }
        default:
          err = "unknown action id " + std::to_string(actions[i]);
          out.clear();
          return false;
      }
      if (dx || dy) {
        s.kind = LkStep::Kind::Walk;
        s.to_x = x + dx;
        s.to_y = y + dy;
        if (g.cell(s.to_x, s.to_y) == MazeCell::KeyA && key_block_present) {
          LkStep b = s;
          b.kind = LkStep::Kind::BreakKey;
          out.push_back(b);
          key_block_present = false;
        }
      }
      if (!g.apply(a)) {
        err = "action " + std::to_string(i) + " (id " + std::to_string(actions[i]) + ") is illegal on the grid";
        out.clear();
        return false;
      }
      if (!door_steps.empty()) {
        for (const LkStep& d : door_steps) out.push_back(d);
      } else {
        out.push_back(s);
      }
    }
    return true;
  }
};

}  // namespace pchie
