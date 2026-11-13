// Bolt PCHIE - runtime boundary between the pilot controller and a LithoKraft pilot.
// Pure interface (a runtime boundary: real = LithoKraftEngineHost over MCP stdio; test =
// a kinematic fake). Every call is synchronous and serial; no mutex, no atomic.
#pragma once

#include <cstdint>
#include <string>

namespace pchie {

struct LkPlayerState {
  double x = 0, y = 0, z = 0;
  double yaw = 0, pitch = 0;  // radians; forward = (cos yaw, sin yaw) in (x, z)
};

class ILkPilotHost {
 public:
  virtual ~ILkPilotHost() = default;
  virtual bool player_state(LkPlayerState& s, std::string& err) = 0;
  virtual bool get_block(int x, int y, int z, uint16_t& id, std::string& err) = 0;
  // Held movement (sticky until changed), same physics as the keyboard.
  virtual bool move(double forward, double strafe, bool sneak, std::string& err) = 0;
  // Relative look in mouse pixels (one-shot).
  virtual bool look(double dx, double dy, std::string& err) = 0;
  // One-frame button: "attack" breaks the block under the crosshair.
  virtual bool action(const std::string& which, std::string& err) = 0;
  virtual void sleep_ms(int ms) = 0;
};

}  // namespace pchie
