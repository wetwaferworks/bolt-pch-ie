// TESLA-manner stream watcher: a separate program from the runner. Reads a text stream, writes a
// verdict, never consults a wall clock and never talks to a TPM.
//
// Usage: pchie_tesla_watcher <stream.txt>
// Exit 0 only when the stream is authentic, every step 0..last is released, and there are no flags.
// Exit 1 for a stream that does not verify, 2 for a file that cannot be read as a stream. Malformed or
// oversized input fails with a message; the watcher's own limits bound memory and work.
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include "pchie/TeslaWatcher.h"

using namespace pchie;

static constexpr uintmax_t kMaxStreamBytes = 64ull << 20;  // far above an honest 4096-interval stream
static constexpr size_t kMaxLineBytes = 4096;

static int run(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: pchie_tesla_watcher <stream.txt>\n");
    return 2;
  }
  std::error_code ec;
  const uintmax_t size = std::filesystem::file_size(argv[1], ec);
  if (ec) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  if (size > kMaxStreamBytes) {
    std::fprintf(stderr, "stream file is too large (%llu bytes)\n", (unsigned long long)size);
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
  auto num = [](const std::string& v, uint32_t& out) {
    if (v.empty() || v.size() > 9 || v.find_first_not_of("0123456789") != std::string::npos) return false;
    out = uint32_t(std::stoul(v));
    return true;
  };
  TeslaSchedule sch;
  if (!num(i_line.substr(2), sch.I) || !num(d_line.substr(2), sch.d)) {
    std::fprintf(stderr, "bad stream header\n");
    return 2;
  }
  TeslaWatcher w(tesla_unhex(commitment_hex.substr(11)), sch);
  while (std::getline(in, line)) {
    if (line.size() > kMaxLineBytes) {
      std::fprintf(stderr, "stream line is too long\n");
      return 2;
    }
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
  std::printf("released=%zu gaps=%llu flags=%zu failed_at=%s\n", r.released.size(), (unsigned long long)r.gap_count, r.flags.size(),
              r.failed_at == ~uint64_t(0) ? "none" : std::to_string(r.failed_at).c_str());
  for (const auto& f : r.flags) {
    std::printf("  flag %s step=%llu interval=%u\n", tesla_flag_name(f.flag),
                (unsigned long long)f.step_index, f.interval_index);
  }
  std::printf("authentic=%s all_verified=%s\n", r.authentic ? "PASS" : "FAIL", r.all_verified ? "PASS" : "FAIL");
  return r.all_verified && r.authentic ? 0 : 1;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "malformed stream: %s\n", e.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "malformed stream\n");
    return 2;
  }
}
