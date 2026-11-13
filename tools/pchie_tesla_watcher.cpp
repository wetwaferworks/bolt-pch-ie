// TESLA-manner stream watcher: a separate program from the runner. Reads a text stream, writes a
// verdict, never consults a wall clock and never talks to a TPM.
//
// Usage: pchie_tesla_watcher <stream.txt>
// Exit 0 only when the stream is authentic, every step 0..last is released, and there are no flags.
#include <cstdio>
#include <fstream>
#include <string>
#include "pchie/TeslaWatcher.h"

using namespace pchie;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: pchie_tesla_watcher <stream.txt>\n");
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  std::string line;
  if (!std::getline(in, line) || line != "pchie.tesla.v1.stream") {
    std::fprintf(stderr, "not a TESLA stream file\n");
    return 2;
  }
  std::string commitment_hex, i_line, d_line;
  if (!std::getline(in, commitment_hex) || !std::getline(in, i_line) || !std::getline(in, d_line)) {
    std::fprintf(stderr, "missing stream header\n");
    return 2;
  }
  if (commitment_hex.rfind("commitment=", 0) != 0 || i_line.rfind("I=", 0) != 0 || d_line.rfind("d=", 0) != 0) {
    std::fprintf(stderr, "bad stream header\n");
    return 2;
  }
  TeslaSchedule sch;
  sch.I = uint32_t(std::stoul(i_line.substr(2)));
  sch.d = uint32_t(std::stoul(d_line.substr(2)));
  TeslaWatcher w(tesla_unhex(commitment_hex.substr(11)), sch);
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    if (line == "end-steps") w.note_end_of_steps();
    else if (line.rfind("step|", 0) == 0) w.accept_step(tesla_parse_step_line(line));
    else if (line.rfind("disclose|", 0) == 0) w.accept_disclosure(tesla_parse_disclose_line(line));
    else {
      std::fprintf(stderr, "unknown stream line: %s\n", line.c_str());
      return 2;
    }
  }
  const TeslaWatchResult r = w.finish();
  std::printf("TESLA watcher\n");
  std::printf("commitment=%s\n", commitment_hex.substr(11).c_str());
  std::printf("I=%u d=%u\n", sch.I, sch.d);
  std::printf("released=%zu gaps=%zu flags=%zu failed_at=%s\n", r.released.size(), r.gaps.size(), r.flags.size(),
              r.failed_at == ~uint64_t(0) ? "none" : std::to_string(r.failed_at).c_str());
  for (const auto& f : r.flags) {
    std::printf("  flag %s step=%llu interval=%u\n", tesla_flag_name(f.flag),
                (unsigned long long)f.step_index, f.interval_index);
  }
  std::printf("authentic=%s all_verified=%s\n", r.authentic ? "PASS" : "FAIL", r.all_verified ? "PASS" : "FAIL");
  return r.all_verified && r.authentic ? 0 : 1;
}
