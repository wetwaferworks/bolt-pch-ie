// Small self-contained fuzz driver for the three parsers that read outside input: MiniJson (MCP replies),
// the TESLA stream lines read by the watcher, and the GGUF reader. It mutates built-in seeds with a fixed
// pseudo-random sequence, so a run is reproducible from its seed. Build it with PCHIE_FUZZ=ON, ideally
// together with PCHIE_SANITIZE=address,undefined; any crash or sanitizer report is a finding.
//   pchie_fuzz <json|tesla|gguf|all> [iterations] [seed]
#include "pchie/MiniJson.h"
#include "pchie/TeslaStream.h"
#include "pchie/TeslaWatcher.h"
#include "pchie/pchie_gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace pchie;
using Bytes = std::vector<uint8_t>;

namespace {

struct Rng {
  uint64_t s;
  uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  size_t below(size_t n) { return n ? size_t(next() % n) : 0; }
};

void mutate(Bytes& b, Rng& r, size_t max_len) {
  static const uint8_t interesting[] = {0x00, 0x01, 0x7f, 0x80, 0xff, '"', '\\', '|', '\n', '{', '[', '-', 'e', '0', '9'};
  const int rounds = 1 + int(r.below(6));
  for (int k = 0; k < rounds; ++k) {
    switch (r.below(7)) {
      case 0: if (!b.empty()) b[r.below(b.size())] ^= uint8_t(1u << r.below(8)); break;
      case 1: if (!b.empty()) b[r.below(b.size())] = interesting[r.below(sizeof interesting)]; break;
      case 2: if (b.size() < max_len) b.insert(b.begin() + long(r.below(b.size() + 1)), uint8_t(r.next())); break;
      case 3: if (!b.empty()) b.erase(b.begin() + long(r.below(b.size()))); break;
      case 4: if (!b.empty()) b.resize(r.below(b.size())); break;
      case 5: if (b.size() >= 4) { const size_t at = r.below(b.size() - 3); const uint32_t v[] = {0u, 0xffffffffu, 0x7fffffffu, 0x80000000u, 0x10000u};
                const uint32_t x = v[r.below(5)]; std::memcpy(b.data() + at, &x, 4); } break;
      case 6: if (!b.empty() && b.size() < max_len) { const size_t a = r.below(b.size()), n = 1 + r.below(std::min<size_t>(16, b.size() - a));
                Bytes chunk(b.begin() + long(a), b.begin() + long(a + n)); b.insert(b.begin() + long(r.below(b.size() + 1)), chunk.begin(), chunk.end()); } break;
    }
    if (b.size() > max_len) b.resize(max_len);
  }
}

// ---- MiniJson ----
bool run_json(const Bytes& in) {
  MiniJson j; std::string err;
  const std::string text(in.begin(), in.end());
  const bool loose = MiniJson::parse(text, j, err);
  // Tight limits (the shape PMCP-style callers use) must never accept what the defaults refuse.
  MiniJson::Limits tight;
  tight.max_bytes = 2048; tight.max_depth = 8; tight.max_string_bytes = 64; tight.max_items = 16;
  tight.max_values = 128; tight.reject_duplicate_keys = true;
  MiniJson t; std::string terr;
  const bool strict = MiniJson::parse(text, t, terr, tight);
  if (strict && !loose) { std::fprintf(stderr, "json: tight limits accepted what defaults refused\n"); std::abort(); }
  return loose;
}
std::vector<Bytes> json_seeds() {
  std::vector<Bytes> s;
  for (const char* t : {R"({"jsonrpc":"2.0","id":7,"result":{"content":[{"type":"text","text":"{\"x\":-12.5,\"blockId\":3}"}],"isError":false}})",
                        R"([1,-0.5e+3,true,false,null,"\u00e9\n\ud83d\ude00",{"a":[[]]}])", R"("plain")", "0"})
    s.emplace_back(t, t + std::strlen(t));
  return s;
}

// ---- TESLA stream (same line handling as the watcher tool, from memory) ----
bool run_tesla(const Bytes& in) {
  std::istringstream ss(std::string(in.begin(), in.end()));
  std::string line, c, il, dl;
  if (!std::getline(ss, line) || line != "pchie.tesla.v1.stream") return false;
  if (!std::getline(ss, c) || !std::getline(ss, il) || !std::getline(ss, dl)) return false;
  if (c.rfind("commitment=", 0) != 0 || il.rfind("I=", 0) != 0 || dl.rfind("d=", 0) != 0) return false;
  auto num = [](const std::string& v, uint32_t& out) {
    if (v.empty() || v.size() > 9 || v.find_first_not_of("0123456789") != std::string::npos) return false;
    out = uint32_t(std::stoul(v)); return true; };
  TeslaSchedule sch;
  if (!num(il.substr(2), sch.I) || !num(dl.substr(2), sch.d)) return false;
  try {
    TeslaWatcher w(tesla_unhex(c.substr(11)), sch);
    while (std::getline(ss, line)) {
      if (line.size() > 4096) return false;
      if (line == "end-steps") w.note_end_of_steps();
      else if (line.rfind("step|", 0) == 0) w.accept_step(tesla_parse_step_line(line));
      else if (line.rfind("disclose|", 0) == 0) w.accept_disclosure(tesla_parse_disclose_line(line));
    }
    (void)w.finish();
    return true;
  } catch (const std::exception&) { return false; }  // a refusal by exception is a clean outcome; a crash is not
}
std::vector<Bytes> tesla_seeds() {
  std::vector<Bytes> s;
  const std::string base = "pchie.tesla.v1.stream\ncommitment=" + std::string(64, 'a') + "\nI=4\nd=2\n";
  for (const std::string& t : {base + "step|0|0|" + std::string(64, 'b') + "|" + std::string(64, 'c') + "\ndisclose|0|" + std::string(64, 'd') + "\nend-steps\n",
                               base + "step|5|1|00|00\nend-steps\nstep|6|1|00|00\n", base})
    s.emplace_back(t.begin(), t.end());
  return s;
}

// ---- GGUF reader (through a temp file, as the loader reads files) ----
std::filesystem::path g_gguf_tmp;
bool run_gguf(const Bytes& in) {
  { std::ofstream f(g_gguf_tmp, std::ios::binary | std::ios::trunc); f.write(reinterpret_cast<const char*>(in.data()), std::streamsize(in.size())); }
  GgufFile g; std::string err;
  if (!g.open(g_gguf_tmp.string(), err)) return false;
  (void)g.directory_digest();
  return true;
}
void put_u32(Bytes& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i))); }
void put_u64(Bytes& b, uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i))); }
void put_str(Bytes& b, const std::string& s) { put_u64(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }
std::vector<Bytes> gguf_seeds() {
  Bytes b = {'G', 'G', 'U', 'F'};
  put_u32(b, 3); put_u64(b, 1); put_u64(b, 2);           // version, tensors, kv pairs
  put_str(b, "general.alignment"); put_u32(b, 4); put_u32(b, 32);   // type 4 = u32
  put_str(b, "general.name"); put_u32(b, 8); put_str(b, "fuzz");    // type 8 = string
  put_str(b, "t"); put_u32(b, 1); put_u64(b, 8); put_u32(b, 0); put_u64(b, 0);  // 1-d F32 tensor of 8
  while (b.size() % 32) b.push_back(0);
  for (int i = 0; i < 32; ++i) b.push_back(0);
  return {b};
}

int fuzz(const char* name, bool (*run)(const Bytes&), const std::vector<Bytes>& seeds, long iters, uint64_t seed, size_t max_len) {
  Rng r{seed * 0x9E3779B97F4A7C15ull + 1};
  long seeds_ok = 0, accepted = 0;
  for (const Bytes& s : seeds) seeds_ok += run(s) ? 1 : 0;
  for (long i = 0; i < iters; ++i) {
    Bytes b = seeds[r.below(seeds.size())];
    mutate(b, r, max_len);
    accepted += run(b) ? 1 : 0;
  }
  std::printf("FUZZ %s: %ld inputs (seeds accepted %ld/%zu, mutants accepted %ld), seed %llu, no crash\n", name, iters + long(seeds.size()),
              seeds_ok, seeds.size(), accepted, (unsigned long long)seed);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string which = argc > 1 ? argv[1] : "all";
  const long iters = argc > 2 ? std::atol(argv[2]) : 20000;
  const uint64_t seed = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1;
  if (iters < 0 || iters > 100000000) { std::fprintf(stderr, "iterations must be 0..100000000\n"); return 2; }
  g_gguf_tmp = std::filesystem::temp_directory_path() / ("pchie-fuzz-" + std::to_string(seed) + ".gguf");
  if (which == "json" || which == "all") fuzz("json", run_json, json_seeds(), iters, seed, 4096);
  if (which == "tesla" || which == "all") fuzz("tesla", run_tesla, tesla_seeds(), iters, seed, 8192);
  if (which == "gguf" || which == "all") fuzz("gguf", run_gguf, gguf_seeds(), iters, seed, 4096);
  std::error_code ec; std::filesystem::remove(g_gguf_tmp, ec);
  return 0;
}
