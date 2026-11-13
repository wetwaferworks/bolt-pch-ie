// Scratch NEML tool: unit tests (--test), csv backfill (--backfill runs.csv DIR), digest print (--digest DIR).
// Real tokenizer = Gemma4Model tokenizer of Little Nemo's GGUF (no forward pass). Scratch only.
#include "NemlLedger.h"
#include "pchie/pchie_gemma4.h"
#include "pchie/Sha256.h"
#include "pchie/ModelPath.h"
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace pchie;

static const std::string& model_path() {
  static const std::string p = [] {
    std::string path, err;
    if (!pchie::resolve_model_path(path, err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return std::string();
    }
    return path;
  }();
  return p;
}
static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("[NEML] %-70s %s\n", what, ok ? "PASS" : "FAIL"); if (!ok) ++g_fail; }
static std::vector<std::string> csv_split(const std::string& l) { std::vector<std::string> v; std::string c; bool q = false; for (char ch : l) { if (ch == '"') { q = !q; continue; } if (ch == ',' && !q) { v.push_back(c); c.clear(); } else if (ch != '\r') c += ch; } v.push_back(c); return v; }
static std::string hexs(const std::string& s) { auto h = Sha256::hash(reinterpret_cast<const uint8_t*>(s.data()), s.size()); char b[65]; for (int i = 0; i < 32; ++i) std::snprintf(b + 2 * i, 3, "%02x", h[i]); return b; }

static neml::MazeRunReceipt mk(int i, bool exit, int nev) {
  neml::Recorder r; r.forks = uint16_t(i % 5);
  for (int k = 0; k < nev; ++k) r.event(uint8_t(k % 3), 3 * k + 1, uint8_t(100 + k), k, -k, "unit_event_tag_long_xx");
  if (exit) r.event(neml::kExitFound, 3 * nev + 2, 255, 9, 9, "exit");
  return r.finish(1700000000u + i, 40 + i, exit, "UNIT synthetic receipt number " + std::to_string(i) + " padding padding");
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "--test";
  Gemma4Model m; std::string err;
  if (!m.load(model_path().c_str(), err)) { std::printf("LOAD ERROR: %s\n", err.c_str()); return 1; }
  auto ntok = [&](const std::string& x) { return m.tok().encode(x).size(); };
  if (mode == "--test") {
    check(sizeof(neml::VisualLandmarkEvent) == 32, "struct size: VisualLandmarkEvent == 32 B");
    check(sizeof(neml::MazeRunReceipt) == 256, "struct size: MazeRunReceipt == 256 B");
    neml::Ring ring; ring.load("__none__");
    for (int i = 0; i < 20; ++i) ring.append(mk(i, i % 3 != 0, 7));
    auto ord = ring.ordered();
    check(ring.h.total_written == 20 && ord.size() == 16 && ord.front().run_id == 5 && ord.back().run_id == 20, "ring wrap: 20 appended -> 16 kept, run_id 5..20 oldest->newest");
    std::error_code ec;
    const auto unit_dir = std::filesystem::temp_directory_path() / "pchie_neml_unit";
    std::filesystem::create_directories(unit_dir, ec);
    const std::string p = (unit_dir / "nemo_episodic.bin").string();
    check(ring.save(p), "round-trip: save");
    neml::Ring r2; check(r2.load(p) && std::memcmp(&r2.h, &ring.h, sizeof ring.h) == 0 && std::memcmp(r2.slots.data(), ring.slots.data(), 16 * 256) == 0, "round-trip: load is byte-identical (header + 16 slots)");
    { std::ifstream f(p, std::ios::binary | std::ios::ate); check(size_t(f.tellg()) == 32 + 16 * 256, "round-trip: file size 32 + 16*256 = 4128 B"); }
    int misses = 0; for (auto& r : ord) misses += r.exit_reached ? 0 : 1;
    size_t nt = 0; const std::string d = neml::digest(r2.ordered(), r2.h.total_written, ntok, 384, &nt);
    check(misses > 0 && d.find(" MISS ") != std::string::npos, "honest misses kept: MISS receipts stored and shown in digest");
    auto e = mk(1, true, 9); check(e.key_events[5].event_type == neml::kExitFound && e.key_events[4].event_type <= 2, "key_events: 9 events + exit -> first 5 + ExitFound kept");
    check(nt <= 384, "token cap: digest of 16 dense receipts <= 384 real tokens");
    std::printf("[NEML] token cap detail: %zu tokens; lines kept=%zu of 16\n", nt, size_t(std::count(d.begin(), d.end(), '#')));
    neml::Ring small; small.load("__none__"); for (int i = 0; i < 2; ++i) small.append(mk(i, i == 0, 2));
    size_t ns = 0; const std::string ds = neml::digest(small.ordered(), 2, ntok, 384, &ns);
    check(ds.find("#1 ") != std::string::npos && ds.find("#2 ") != std::string::npos && ns <= 384, "token cap: small ring keeps every run line");
    const std::string d2 = neml::digest(r2.ordered(), r2.h.total_written, ntok, 384);
    neml::Ring r3; r3.load(p); const std::string d3 = neml::digest(r3.ordered(), r3.h.total_written, ntok, 384);
    check(d == d2 && d == d3, "deterministic digest: identical bytes across rebuilds and reloads");
    std::printf("[NEML] unit digest sha256=%s\n", hexs(d).c_str());
    const std::string pr1 = neml::priors_text(neml::priors(r2.ordered())), pr2 = neml::priors_text(neml::priors(r3.ordered()));
    check(pr1 == pr2, "deterministic priors text");
    std::printf("[NEML] unit %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail); return g_fail ? 1 : 0;
  }
  if (mode == "--backfill" && argc > 3) {
    std::ifstream f(argv[2]); std::string line; std::getline(f, line);
    std::vector<std::string> hdr = csv_split(line);
    auto col = [&](const std::string& n) { for (size_t i = 0; i < hdr.size(); ++i) if (hdr[i] == n) return int(i); return -1; };
    const std::string dir = argv[3], rp = dir + "/nemo_episodic.bin";
    neml::Ring ring; if (!ring.load(rp)) { std::printf("bad ring\n"); return 1; }
    int added = 0;
    while (std::getline(f, line)) {
      std::vector<std::string> v = csv_split(line);
      auto g = [&](const char* n) { const int i = col(n); return (i >= 0 && i < int(v.size())) ? v[i] : std::string(); };
      auto uq = [](std::string x) { if (x.size() >= 2 && x.front() == '"') x = x.substr(1, x.size() - 2); return x; };
      const std::string name = uq(g("name")), dec = uq(g("decisions")), rc = uq(g("rc"));
      if (dec.empty() || rc != "0" || name.rfind("VOID", 0) == 0) { std::printf("[NEML] backfill skip %s (no completed result)\n", name.c_str()); continue; }
      const std::string vc = uq(g("view_caused")); const size_t sl = vc.find('/');
      neml::Recorder r; r.forks = uint16_t(sl == std::string::npos ? 0 : std::atoi(vc.c_str() + sl + 1));
      r.dead_ends = uint16_t(std::atoi(uq(g("bumps")).c_str()));
      std::tm tm{}; const std::string et = uq(g("end_et")); std::sscanf(et.c_str(), "%d-%d-%dT%d:%d:%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec);
      tm.tm_year -= 1900; tm.tm_mon -= 1; const uint32_t ts = uint32_t(_mkgmtime(&tm) + 5 * 3600);  // ET (UTC-5) -> UTC
      auto rec = r.finish(ts, std::atoi(dec.c_str()), uq(g("exit")) == "1", "BACKFILL csv " + name);
      const uint32_t id = ring.append(rec); ++added;
      std::printf("[NEML] backfill run_id=%u %s steps=%s forks=%u dead_ends(bumps)=%u exit=%s (csv counters only; no key events)\n", id, name.c_str(), dec.c_str(), rec.mcts_forks_triggered, rec.dead_ends_hit, uq(g("exit")).c_str());
    }
    if (!ring.save(rp)) { std::printf("save failed\n"); return 1; }
    { std::ofstream o(dir + "/nemo_priors.txt", std::ios::binary); o << neml::priors_text(neml::priors(ring.ordered())); }
    std::printf("[NEML] backfill added=%d lifetime=%u\n", added, ring.h.total_written); return 0;
  }
  if (mode == "--digest" && argc > 2) {
    const std::string dir = argv[2]; neml::Ring ring; if (!ring.load(dir + "/nemo_episodic.bin")) return 1;
    size_t nt = 0; const std::string d = neml::digest(ring.ordered(), ring.h.total_written, ntok, 384, &nt);
    { std::ofstream o(dir + "/nemo_digest.txt", std::ios::binary); o << d; }
    std::printf("%s[NEML] digest tokens=%zu (real tokenizer, cap 384) sha256=%s\n", d.c_str(), nt, hexs(d).c_str()); return 0;
  }
  std::printf("usage: neml_tool --test | --backfill runs.csv DIR | --digest DIR\n"); return 2;
}

