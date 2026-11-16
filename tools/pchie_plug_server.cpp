// pchie_plug_server: the harness game plugs over stdin/stdout, one JSON object per line, for the thin
// Gymnasium adapter in tools/gym. Off the official path and outside every hashed path: it writes no
// chain or side chain, reads no keys, and opens no network connection.
//
//   {"cmd":"list"}                      -> {"ok":true,"plugs":[...]}
//   {"cmd":"make","plug":"NAME"}        -> {"ok":true,"info":{...}}
//   {"cmd":"reset","seed":N}            -> {"ok":true,"obs":{...},"state":"HEX"}
//   {"cmd":"step","action":K}           -> {"ok":true,"obs":{...},"reward":R,"terminated":B,"action_ok":B,"state":"HEX"}
//   {"cmd":"grade"}                     -> {"ok":true,"success":B,"progress":P,"target":T}
//   {"cmd":"close"}                     -> {"ok":true} and exit
// Errors: {"ok":false,"err":"..."}.
#include "pchie/harness/GamePlug.h"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

using namespace pchie::harness;

namespace {

std::string q(const std::string& s) { return "\"" + json_escape(s) + "\""; }

// Finds "key": and returns the raw value text (string unescaped for simple cases, or a bare token).
bool field(const std::string& line, const std::string& key, std::string& out) {
  const std::string k = "\"" + key + "\"";
  size_t p = line.find(k);
  if (p == std::string::npos) return false;
  p = line.find(':', p + k.size());
  if (p == std::string::npos) return false;
  ++p;
  while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
  if (p >= line.size()) return false;
  out.clear();
  if (line[p] == '"') {
    for (size_t i = p + 1; i < line.size(); ++i) {
      if (line[i] == '\\' && i + 1 < line.size()) { out.push_back(line[++i]); continue; }
      if (line[i] == '"') return true;
      out.push_back(line[i]);
    }
    return false;
  }
  size_t e = p;
  while (e < line.size() && line[e] != ',' && line[e] != '}' && line[e] != ' ') ++e;
  out = line.substr(p, e - p);
  return !out.empty();
}

bool to_u64(const std::string& s, uint64_t& v) {
  if (s.empty() || s.size() > 20) return false;
  v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    v = v * 10 + uint64_t(c - '0');
  }
  return true;
}

std::string obs_json(const PlugObservation& o) {
  return "{\"text\":" + q(o.text) + ",\"glyphs\":" + q(o.glyphs) + ",\"cols\":" + std::to_string(o.glyph_cols) +
         ",\"rows\":" + std::to_string(o.glyph_rows) + "}";
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::unique_ptr<IGamePlug> game;
  int n_actions = 0;
  bool reset_done = false;
  std::string line;
  auto reply = [](const std::string& s) { std::cout << s << "\n" << std::flush; };
  auto fail = [&](const std::string& e) { reply("{\"ok\":false,\"err\":" + q(e) + "}"); };
  while (std::getline(std::cin, line)) {
    if (line.size() > 4096) { fail("line too long"); continue; }
    std::string cmd;
    if (!field(line, "cmd", cmd)) { fail("no cmd"); continue; }
    if (cmd == "list") {
      std::string s = "{\"ok\":true,\"plugs\":[";
      const auto names = plug_names();
      for (size_t i = 0; i < names.size(); ++i) s += (i ? "," : "") + q(names[i]);
      reply(s + "]}");
    } else if (cmd == "make") {
      std::string name, err;
      if (!field(line, "plug", name)) { fail("make needs plug"); continue; }
      game = make_plug(name, err);
      reset_done = false;
      if (!game) { fail(err); continue; }
      const PlugInfo i = game->info();
      n_actions = int(i.actions.size());
      std::string acts;
      for (size_t k = 0; k < i.actions.size(); ++k) acts += (k ? "," : "") + q(i.actions[k]);
      reply("{\"ok\":true,\"info\":{\"name\":" + q(i.name) + ",\"version\":" + q(i.version) + ",\"family\":" + q(i.family) +
            ",\"stub\":" + (i.stub ? "true" : "false") + ",\"actions\":[" + acts + "],\"instructions\":" + q(i.instructions) + "}}");
    } else if (cmd == "reset") {
      if (!game) { fail("make a plug first"); continue; }
      std::string sv;
      uint64_t seed = 0;
      if (field(line, "seed", sv) && sv != "null" && !to_u64(sv, seed)) { fail("seed must be a whole number"); continue; }
      const PlugObservation o = game->reset(seed);
      reset_done = true;
      reply("{\"ok\":true,\"obs\":" + obs_json(o) + ",\"state\":\"" + game->state_digest() + "\"}");
    } else if (cmd == "step") {
      if (!game || !reset_done) { fail("reset first"); continue; }
      std::string av;
      uint64_t a = 0;
      if (!field(line, "action", av) || !to_u64(av, a) || a >= uint64_t(n_actions)) { fail("action out of range"); continue; }
      const PlugStep s = game->step(int(a));
      reply("{\"ok\":true,\"obs\":" + obs_json(s.obs) + ",\"reward\":" + std::to_string(s.reward) + ",\"terminated\":" +
            (s.done ? "true" : "false") + ",\"success\":" + (s.success ? "true" : "false") + ",\"action_ok\":" +
            (s.action_ok ? "true" : "false") + ",\"state\":\"" + game->state_digest() + "\"}");
    } else if (cmd == "grade") {
      if (!game || !reset_done) { fail("reset first"); continue; }
      const PlugGrade g = game->grade();
      reply("{\"ok\":true,\"success\":" + std::string(g.success ? "true" : "false") + ",\"progress\":" + std::to_string(g.progress) +
            ",\"target\":" + std::to_string(g.target) + "}");
    } else if (cmd == "close") {
      reply("{\"ok\":true}");
      return 0;
    } else {
      fail("unknown cmd");
    }
  }
  return 0;
}
