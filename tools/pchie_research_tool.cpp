// Research side tool.
//   pchie_research_tool freeze OUT LOG [LOG ...]   freeze a dynamics model from research side-chain logs
//   pchie_research_tool verify LOG                 re-walk one research side-chain log
// freeze checks every log's chain first, learns from its transition lines only, writes the frozen model to
// OUT and prints its SHA-256 (the pin for --research-dynamics-sha).
#include "pchie/research/FrozenDynamics.h"
#include "pchie/research/SideLog.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace pchie::research;

namespace {
bool field(const std::string& line, const char* key, long long& out) {
  const std::string k = std::string("\"") + key + "\":";
  const size_t p = line.find(k);
  if (p == std::string::npos) return false;
  char* end = nullptr;
  out = std::strtoll(line.c_str() + p + k.size(), &end, 10);
  return end != line.c_str() + p + k.size();
}
bool ufield(const std::string& line, const char* key, unsigned long long& out) {
  const std::string k = std::string("\"") + key + "\":";
  const size_t p = line.find(k);
  if (p == std::string::npos) return false;
  char* end = nullptr;
  out = std::strtoull(line.c_str() + p + k.size(), &end, 10);
  return end != line.c_str() + p + k.size();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc >= 3 && std::string(argv[1]) == "verify") {
    std::string head, err;
    size_t lines = 0;
    if (!SideLog::verify(argv[2], head, lines, err)) {
      std::printf("research log: FAIL (%s)\n", err.c_str());
      return 1;
    }
    std::printf("research log: OK lines=%zu head=%s\n", lines, head.c_str());
    return 0;
  }
  if (argc < 4 || std::string(argv[1]) != "freeze") {
    std::printf("usage: pchie_research_tool freeze OUT LOG [LOG ...] | verify LOG\n");
    return 2;
  }
  DynamicsTrainer trainer;
  size_t used = 0;
  for (int i = 3; i < argc; ++i) {
    std::string head, err;
    size_t lines = 0;
    if (!SideLog::verify(argv[i], head, lines, err)) {
      std::printf("refused %s: %s\n", argv[i], err.c_str());
      return 1;
    }
    std::ifstream in(argv[i], std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
      if (line.find("\"event\":\"transition\"") == std::string::npos) continue;
      unsigned long long st = 0, nx = 0;
      long long a = 0, r = 0;
      if (!ufield(line, "state", st) || !field(line, "action", a) || !ufield(line, "next", nx) ||
          !field(line, "reward_q20", r) || a < 0 || a > 255) {
        std::printf("bad transition line in %s\n", argv[i]);
        return 1;
      }
      trainer.add(Transition{st, uint8_t(a), nx, int32_t(r)});
      ++used;
    }
  }
  const FrozenDynamics model = trainer.freeze();
  std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
  out << model.canonical() << "\n";
  if (!out) {
    std::printf("cannot write %s\n", argv[2]);
    return 1;
  }
  std::printf("frozen: transitions=%zu entries=%zu sha256=%s\n", used, model.size(), model.sha256().c_str());
  return 0;
}
