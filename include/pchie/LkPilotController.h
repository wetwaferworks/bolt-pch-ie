// Bolt PCHIE - closed-loop pilot that carries out LkActuationPlan steps in LithoKraft.
// Concrete final. Serial, no mutex, no atomic. Uses only the player's normal input path
// (ILkPilotHost: held move, mouse look, one-frame attack) and reads the player's real
// position back after every step, so the walk is the engine's, not a replay of the grid.
// The steps themselves come from the reasoning model's committed actions; this class never
// chooses a route. Every observed cell is appended to an engine trace (side log only, never
// the official chain, never a golden).
#pragma once

#include "pchie/ILkPilotHost.h"
#include "pchie/LkActuationPlan.h"
#include "pchie/LkMazeVoxelMap.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace pchie {

class LkPilotController final {
 public:
  static constexpr double kPi = 3.14159265358979323846;
  static constexpr double kEyeHeight = 1.62;   // feet -> eye (first-person camera)
  static constexpr double kWalkPitch = -0.30;  // look slightly down while walking (watchable)
  static constexpr double kArrive = 0.14;      // blocks from the cell centre
  static constexpr double kTurnStop = 0.35;    // radians: stop and turn in place above this
  static constexpr int kStepTimeoutMs = 12000;

  LkPilotController(ILkPilotHost& host, const LkMazeVoxelMap& map) : host_(host), map_(map) {}

  static double wrap(double a) {
    while (a > kPi) a -= 2 * kPi;
    while (a < -kPi) a += 2 * kPi;
    return a;
  }

  double yaw_gain() const noexcept { return gain_yaw_; }
  double pitch_gain() const noexcept { return gain_pitch_; }
  const std::string& trace() const noexcept { return trace_; }
  int walks() const noexcept { return walks_; }
  int breaks() const noexcept { return breaks_; }

  // Measures radians per look pixel (yaw and pitch). Fails closed when look input is ignored
  // (window unfocused / cursor not captured / pilot not controllable).
  bool calibrate(std::string& err) {
    LkPlayerState a, b;
    if (!host_.player_state(a, err)) return false;
    if (!host_.look(100.0, 0.0, err)) return false;
    host_.sleep_ms(150);
    if (!host_.player_state(b, err)) return false;
    const double dy = wrap(b.yaw - a.yaw);
    if (std::fabs(dy) < 1e-4) {
      err = "look input had no effect (game window not focused, cursor not captured, or menu open)";
      return false;
    }
    gain_yaw_ = dy / 100.0;
    if (!host_.look(-100.0, 0.0, err)) return false;
    host_.sleep_ms(150);
    if (!host_.player_state(a, err)) return false;
    if (!host_.look(0.0, 40.0, err)) return false;
    host_.sleep_ms(150);
    if (!host_.player_state(b, err)) return false;
    const double dp = b.pitch - a.pitch;  // pitch -= look.y * gain
    gain_pitch_ = std::fabs(dp) > 1e-4 ? -dp / 40.0 : gain_yaw_;
    return true;
  }

  bool face(double yaw, double pitch, std::string& err) {
    for (int it = 0; it < 40; ++it) {
      LkPlayerState s;
      if (!host_.player_state(s, err)) return false;
      const double ey = wrap(yaw - s.yaw), ep = pitch - s.pitch;
      if (std::fabs(ey) < 0.03 && std::fabs(ep) < 0.04) return true;
      double px = ey / gain_yaw_, py = -ep / gain_pitch_;
      if (px > 2048) px = 2048;
      if (px < -2048) px = -2048;
      if (py > 2048) py = 2048;
      if (py < -2048) py = -2048;
      if (!host_.look(px, py, err)) return false;
      host_.sleep_ms(40);
    }
    err = "face: did not converge";
    return false;
  }

  bool walk_to(int cx, int cy, std::string& err) {
    const double tx = map_.centre_x(cx), tz = map_.centre_z(cy);
    int waited = 0;
    bool moving = false;
    while (waited < kStepTimeoutMs) {
      LkPlayerState s;
      if (!host_.player_state(s, err)) return false;
      const double ddx = tx - s.x, ddz = tz - s.z;
      const double dist = std::sqrt(ddx * ddx + ddz * ddz);
      const double want = std::atan2(ddz, ddx);
      const double ey = wrap(want - s.yaw);
      if (dist < kArrive || (dist < 0.30 && std::fabs(ey) > kPi / 2)) {
        if (!host_.move(0, 0, true, err)) return false;
        host_.sleep_ms(120);
        LkPlayerState e;
        if (!host_.player_state(e, err)) return false;
        int ox = 0, oy = 0;
        map_.world_to_cell(e.x, e.z, ox, oy);
        note_cell(ox, oy);
        if (ox != cx || oy != cy) {
          err = "walk ended in cell (" + std::to_string(ox) + "," + std::to_string(oy) + "), wanted (" +
                std::to_string(cx) + "," + std::to_string(cy) + ")";
          return false;
        }
        ++walks_;
        return true;
      }
      if (std::fabs(ey) > kTurnStop) {
        if (moving && !host_.move(0, 0, true, err)) return false;
        moving = false;
        if (!face(want, kWalkPitch, err)) return false;
        continue;
      }
      if (std::fabs(ey) > 0.04 && !host_.look(ey / gain_yaw_, 0.0, err)) return false;
      if (!host_.move(1.0, 0.0, true, err)) return false;
      moving = true;
      host_.sleep_ms(25);
      waited += 25;
    }
    (void)host_.move(0, 0, true, err);
    err = "walk to (" + std::to_string(cx) + "," + std::to_string(cy) + ") timed out";
    return false;
  }

  // Break the block at layer (floorY + layer) of cell (cx, cy), facing it from where the player stands.
  bool break_block(int cx, int cy, int layer, std::string& err) {
    const int bx = map_.block_x(cx), by = map_.floor_y() + layer, bz = map_.block_z(cy);
    static const double kPitchTry[4] = {0.0, 0.12, -0.12, 0.25};
    for (double off : kPitchTry) {
      LkPlayerState s;
      if (!host_.player_state(s, err)) return false;
      const double ddx = (bx + 0.5) - s.x, ddz = (bz + 0.5) - s.z;
      const double horiz = std::sqrt(ddx * ddx + ddz * ddz);
      const double pitch = std::atan2((by + 0.5) - (s.y + kEyeHeight), horiz) + off;
      if (!face(std::atan2(ddz, ddx), pitch, err)) return false;
      if (!host_.action("attack", err)) return false;
      host_.sleep_ms(200);
      uint16_t id = 1;
      if (!host_.get_block(bx, by, bz, id, err)) return false;
      if (id == 0) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "B(%d,%d,%d);", cx, cy, layer);
        trace_ += buf;
        ++breaks_;
        return true;
      }
    }
    err = "block at cell (" + std::to_string(cx) + "," + std::to_string(cy) + ") layer " + std::to_string(layer) +
          " did not break";
    return false;
  }

  bool run_step(const LkStep& st, std::string& err) {
    switch (st.kind) {
      case LkStep::Kind::Walk: return walk_to(st.to_x, st.to_y, err);
      case LkStep::Kind::BreakKey: return break_block(st.to_x, st.to_y, 1, err);
      case LkStep::Kind::PickUp: trace_ += "K;"; return true;
      case LkStep::Kind::BreakDoor:
        if (!break_block(st.to_x, st.to_y, 2, err)) return false;
        return break_block(st.to_x, st.to_y, 1, err);
    }
    err = "unknown step";
    return false;
  }

 private:
  void note_cell(int x, int y) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "W(%d,%d);", x, y);
    trace_ += buf;
  }

  ILkPilotHost& host_;
  LkMazeVoxelMap map_;
  double gain_yaw_ = 0.0022, gain_pitch_ = 0.0022;
  std::string trace_;
  int walks_ = 0, breaks_ = 0;
};

}  // namespace pchie
