// Bolt PCHIE - the grid-maze world for the hemispheres, with its answer key.
//
// States are MazeGrid positions; the six moves are North, South, East, West, Pickup
// and Use. The answer key is MazeOracle's breadth-first search (it counts the key and
// the door): a move is cleared when it shortens the way to the exit, or reaches it.
//
// Episodes are guided: whatever a hemisphere proposes is graded, then the key's own
// move is played. Every model is therefore graded on the same states, so scores are
// comparable and the set of states can be worked out once.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#pragma once
#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/hemispheres/SampleSet.h"
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace hemispheres {

class GridMazeWorld final {
public:
    static constexpr uint32_t kActions = 6;
    static constexpr uint32_t kFeatures = 8;
    static MazeAction action(uint32_t i) { return MazeAction(int32_t(MazeAction::North) + int32_t(i)); }
    static const char* action_name(uint32_t i) {
        static const char* names[kActions] = {"North", "South", "East", "West", "Pickup", "Use"};
        return i < kActions ? names[i] : "?";
    }
    static uint32_t legal_mask(const MazeGrid& g) {
        uint32_t m = 0;
        for (MazeAction a : g.legal_actions()) {
            const int32_t i = int32_t(a) - int32_t(MazeAction::North);
            if (i >= 0 && i < int32_t(kActions)) m |= 1u << uint32_t(i);
        }
        return m;
    }
    // Where the agent should head next: the key until it is held, then the door until it
    // is open, then the exit.
    static bool target(const MazeGrid& g, int& tx, int& ty) {
        if (!g.have_key() && g.find(MazeCell::KeyA, tx, ty)) return true;
        if (!g.door_open() && g.find(MazeCell::DoorA, tx, ty)) return true;
        return g.find(MazeCell::Exit, tx, ty);
    }
    static void features(const MazeGrid& g, int8_t out[kFeatures]) {
        int tx = g.x(), ty = g.y();
        target(g, tx, ty);
        const int dx = tx - g.x(), dy = ty - g.y();
        out[0] = int8_t(dx > 0 ? 64 : dx < 0 ? -64 : 0);
        out[1] = int8_t(dy > 0 ? 64 : dy < 0 ? -64 : 0);
        out[2] = int8_t(g.have_key() ? 64 : 0);
        out[3] = int8_t(g.door_open() ? 64 : 0);
        out[4] = int8_t(g.walkable(g.x(), g.y() - 1) ? 0 : 64);
        out[5] = int8_t(g.walkable(g.x(), g.y() + 1) ? 0 : 64);
        out[6] = int8_t(g.walkable(g.x() + 1, g.y()) ? 0 : 64);
        out[7] = int8_t(g.walkable(g.x() - 1, g.y()) ? 0 : 64);
    }
    // A short text the model reads (north is up, y grows southward).
    static std::string describe(const MazeGrid& g) {
        int tx = g.x(), ty = g.y();
        const bool any = target(g, tx, ty);
        const char* what = !g.have_key() ? "the key" : !g.door_open() ? "the door" : "the exit";
        std::string s = "Goal: ";
        s += what;
        if (any) {
            const int dx = tx - g.x(), dy = ty - g.y();
            s += ", " + std::to_string(dx < 0 ? -dx : dx) + (dx < 0 ? " west" : " east");
            s += " and " + std::to_string(dy < 0 ? -dy : dy) + (dy < 0 ? " north" : " south");
        }
        s += ". Open: ";
        bool first = true;
        const int nx[4] = {0, 0, 1, -1}, ny[4] = {-1, 1, 0, 0};
        for (uint32_t i = 0; i < 4; ++i)
            if (g.walkable(g.x() + nx[i], g.y() + ny[i])) { s += first ? "" : ", "; s += action_name(i); first = false; }
        if (first) s += "none";
        s += g.on_key() ? ". You stand on the key." : ".";
        if (g.adjacent_to_door() && !g.door_open()) s += g.have_key() ? " The door is next to you." : " The door is next to you, locked.";
        return s;
    }
    // The answer key.
    static Grade grade(const MazeGrid& g, uint32_t move) {
        if (move >= kActions) return Grade::NotYet;
        MazeGrid after = g;
        if (!after.apply(action(move))) return Grade::NotYet;
        if (after.at_exit()) return Grade::Cleared;
        const int before_len = MazeOracle::shortest_length(g), after_len = MazeOracle::shortest_length(after);
        return (after_len >= 0 && before_len >= 0 && after_len < before_len) ? Grade::Cleared : Grade::NotYet;
    }
    // The states of a guided episode: every position on the key's shortest path, start
    // included, exit excluded. Empty if the maze has no way out.
    static std::vector<MazeGrid> guided_states(const MazeGrid& start) {
        std::vector<MazeGrid> out;
        const std::vector<MazeAction> path = MazeOracle::shortest_path(start);
        MazeGrid g = start;
        for (MazeAction a : path) {
            out.push_back(g);
            if (!g.apply(a)) { out.clear(); return out; }
        }
        if (!g.at_exit()) out.clear();
        return out;
    }
};

} // namespace hemispheres
} // namespace pchie
