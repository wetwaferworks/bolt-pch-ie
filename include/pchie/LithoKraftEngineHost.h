// Bolt PCHIE - LithoKraft as a PCHIE pilot host, over the engine's own MCP server
// (litholattice::mcp::cMcpServer, deny-by-default; the player grants it in the game console:
// `mcp grant world_read player_control`). Concrete final; implements the ILkPilotHost runtime
// boundary. Serial, no mutex, no atomic. Rate-limited tool results ("rate_limited") are retried
// after a short sleep; any other tool error fails the call (fail closed).
#pragma once

#include "pchie/ILkPilotHost.h"
#include "pchie/McpStdioClient.h"
#include "pchie/MiniJson.h"

#include <cstdio>
#include <string>

namespace pchie {

class LithoKraftEngineHost final : public ILkPilotHost {
 public:
  explicit LithoKraftEngineHost(McpStdioClient& c) : c_(c) {}

  bool initialize(std::string& err) {
    MiniJson r;
    if (!c_.request("initialize",
                    "{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},"
                    "\"clientInfo\":{\"name\":\"pchie_lk_pilot\",\"version\":\"0.7.0\"}}",
                    r, err, 120000))
      return false;
    server_name_ = r.get("serverInfo") ? r.get("serverInfo")->str_or("name", "") : "";
    return c_.notify("notifications/initialized", "{}", err);
  }

  const std::string& server_name() const noexcept { return server_name_; }
  int tool_calls() const noexcept { return calls_; }

  // Returns the tool's text payload; is_error mirrors the result's isError flag.
  bool call_tool(const std::string& name, const std::string& args_json, std::string& text, bool& is_error,
                 std::string& err) {
    for (int attempt = 0; attempt < 50; ++attempt) {
      MiniJson r;
      ++calls_;
      if (!c_.request("tools/call", "{\"name\":" + MiniJson::quote(name) + ",\"arguments\":" + args_json + "}", r,
                      err))
        return false;
      is_error = r.is_true("isError");
      text.clear();
      if (const MiniJson* content = r.get("content"))
        for (const MiniJson& part : content->arr) text += part.str_or("text", "");
      if (is_error && text.find("rate") != std::string::npos && text.find("limit") != std::string::npos) {
        sleep_ms(60);
        continue;
      }
      return true;
    }
    err = name + ": still rate limited";
    return false;
  }

  bool call_ok(const std::string& name, const std::string& args_json, MiniJson* payload, std::string& err) {
    std::string text;
    bool is_error = false;
    if (!call_tool(name, args_json, text, is_error, err)) return false;
    if (is_error) {
      err = name + ": " + text;
      return false;
    }
    if (payload && !MiniJson::parse(text, *payload, err)) {
      err = name + ": bad payload: " + err;
      return false;
    }
    return true;
  }

  // Polls until the player has granted world_read + player_control (deny-by-default until then).
  bool wait_for_grant(int timeout_ms, std::string& err) {
    int waited = 0;
    std::string text, last;
    while (waited < timeout_ms) {
      bool e1 = true, e2 = true;
      if (!call_tool("player_get_state", "{}", text, e1, err)) return false;
      if (!e1) {
        if (!call_tool("player_move", "{\"pilot\":\"player\",\"forward\":0,\"strafe\":0}", text, e2, err)) return false;
        if (!e2) return true;
      }
      last = text;
      sleep_ms(250);
      waited += 250;
    }
    err = "no grant within " + std::to_string(timeout_ms / 1000) + " s (type `mcp grant world_read player_control` in the console); last: " + last;
    return false;
  }

  bool player_state(LkPlayerState& s, std::string& err) override {
    MiniJson p;
    if (!call_ok("player_get_state", "{}", &p, err)) return false;
    s.x = p.num_or("x", 0);
    s.y = p.num_or("y", 0);
    s.z = p.num_or("z", 0);
    s.yaw = p.num_or("yaw", 0);
    s.pitch = p.num_or("pitch", 0);
    return true;
  }

  bool get_block(int x, int y, int z, uint16_t& id, std::string& err) override {
    MiniJson p;
    char a[96];
    std::snprintf(a, sizeof(a), "{\"x\":%d,\"y\":%d,\"z\":%d}", x, y, z);
    if (!call_ok("world_get_block", a, &p, err)) return false;
    id = uint16_t(p.num_or("blockId", 0));
    return true;
  }

  bool move(double forward, double strafe, bool sneak, std::string& err) override {
    char a[160];
    std::snprintf(a, sizeof(a), "{\"pilot\":\"player\",\"forward\":%.3f,\"strafe\":%.3f,\"sneak\":%s}", forward, strafe,
                  sneak ? "true" : "false");
    return call_ok("player_move", a, nullptr, err);
  }

  bool look(double dx, double dy, std::string& err) override {
    char a[128];
    std::snprintf(a, sizeof(a), "{\"pilot\":\"player\",\"dx\":%.3f,\"dy\":%.3f}", dx, dy);
    return call_ok("player_look", a, nullptr, err);
  }

  bool action(const std::string& which, std::string& err) override {
    return call_ok("player_action", "{\"pilot\":\"player\",\"action\":" + MiniJson::quote(which) + "}", nullptr, err);
  }

  bool release(std::string& err) { return call_ok("player_release", "{\"pilot\":\"player\"}", nullptr, err); }

  void sleep_ms(int ms) override { Sleep(DWORD(ms)); }

 private:
  McpStdioClient& c_;
  std::string server_name_;
  int calls_ = 0;
};

}  // namespace pchie
