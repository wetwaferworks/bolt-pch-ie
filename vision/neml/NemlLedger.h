// Scratch: Little Nemo Episodic Memory Ledger (NEML).
// Zero-dep, header-only. Opt-in only (visnav --neml DIR); off by default. Never used by an official runner.
// Ring file nemo_episodic.bin = 32-byte header + 16 x 256-byte MazeRunReceipt slots, append-only ring (slot = seq % 16).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdarg>

namespace neml {
#pragma pack(push, 1)
struct VisualLandmarkEvent {
  uint16_t step_index;
  uint8_t event_type;  // 0 DeadEnd 1 VisualRecovery 2 ForkPivot 3 ExitFound
  uint8_t confidence_q8;
  int16_t grid_x;
  int16_t grid_y;
  char tag[24];
};
struct MazeRunReceipt {
  uint32_t run_id;
  uint32_t timestamp_utc;
  uint16_t total_steps;
  uint16_t mcts_forks_triggered;
  uint16_t dead_ends_hit;
  uint8_t exit_reached;
  uint8_t reserved_pad;
  VisualLandmarkEvent key_events[6];
  char cortana_note[48];
};
struct RingHeader {
  char magic[8];        // "NEML0001"
  uint32_t total_written;  // lifetime receipts appended
  uint32_t capacity;       // 16
  uint8_t reserved[16];
};
#pragma pack(pop)
static_assert(sizeof(VisualLandmarkEvent) == 32, "VisualLandmarkEvent must be 32 bytes");
static_assert(sizeof(MazeRunReceipt) == 256, "MazeRunReceipt must be 256 bytes");
static_assert(sizeof(RingHeader) == 32, "RingHeader must be 32 bytes");

enum : uint8_t { kDeadEnd = 0, kVisualRecovery = 1, kForkPivot = 2, kExitFound = 3 };
static const char* kTypeName[4] = {"DeadEnd", "VisualRecovery", "ForkPivot", "ExitFound"};
static constexpr uint32_t kCap = 16;

inline void put_str(char* dst, size_t n, const std::string& s) { std::memset(dst, 0, n); std::memcpy(dst, s.data(), std::min(n - 1, s.size())); }
inline std::string get_str(const char* src, size_t n) { size_t k = 0; while (k < n && src[k]) ++k; return std::string(src, k); }

// In-run recorder: every fired trigger is counted; key_events keeps the first 5 non-exit events in step order plus ExitFound
// (or the first 6 if no exit). Deterministic.
struct Recorder {
  std::vector<VisualLandmarkEvent> all;
  uint16_t forks = 0, dead_ends = 0;
  void event(uint8_t type, int step, uint8_t conf_q8, int gx, int gy, const std::string& tag) {
    VisualLandmarkEvent e{}; e.step_index = uint16_t(step); e.event_type = type; e.confidence_q8 = conf_q8;
    e.grid_x = int16_t(gx); e.grid_y = int16_t(gy); put_str(e.tag, sizeof(e.tag), tag); all.push_back(e);
    if (type == kDeadEnd) ++dead_ends;
  }
  MazeRunReceipt finish(uint32_t ts_utc, int total_steps, bool exit_reached, const std::string& note) const {
    MazeRunReceipt r{}; r.timestamp_utc = ts_utc; r.total_steps = uint16_t(total_steps); r.mcts_forks_triggered = forks;
    r.dead_ends_hit = dead_ends; r.exit_reached = exit_reached ? 1 : 0;
    std::vector<VisualLandmarkEvent> keep; const VisualLandmarkEvent* ex = nullptr;
    for (const auto& e : all) { if (e.event_type == kExitFound) { if (!ex) ex = &e; } }
    const size_t lim = ex ? 5 : 6;
    for (const auto& e : all) if (e.event_type != kExitFound && keep.size() < lim) keep.push_back(e);
    if (ex) keep.push_back(*ex);
    for (size_t i = 0; i < keep.size(); ++i) r.key_events[i] = keep[i];
    for (size_t i = keep.size(); i < 6; ++i) { r.key_events[i].event_type = 0xFF; }  // 0xFF = empty slot
    put_str(r.cortana_note, sizeof(r.cortana_note), note); return r;
  }
};

struct Ring {
  RingHeader h{}; std::vector<MazeRunReceipt> slots = std::vector<MazeRunReceipt>(kCap);
  bool load(const std::string& path) {
    std::memcpy(h.magic, "NEML0001", 8); h.capacity = kCap; h.total_written = 0;
    FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return true;  // empty ring
    RingHeader t{}; bool ok = std::fread(&t, sizeof t, 1, f) == 1 && std::memcmp(t.magic, "NEML0001", 8) == 0 && t.capacity == kCap;
    if (ok) { h = t; ok = std::fread(slots.data(), sizeof(MazeRunReceipt), kCap, f) == kCap; }
    std::fclose(f); return ok;
  }
  bool save(const std::string& path) const {
    const std::string tmp = path + ".tmp"; FILE* f = std::fopen(tmp.c_str(), "wb"); if (!f) return false;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(slots.data(), sizeof(MazeRunReceipt), kCap, f) == kCap;
    ok = (std::fclose(f) == 0) && ok; if (!ok) return false;
    std::remove(path.c_str()); return std::rename(tmp.c_str(), path.c_str()) == 0;
  }
  uint32_t append(MazeRunReceipt r) {  // run_id = lifetime sequence number (1-based)
    r.run_id = h.total_written + 1; slots[h.total_written % kCap] = r; ++h.total_written; return r.run_id;
  }
  // oldest -> newest
  std::vector<MazeRunReceipt> ordered() const {
    std::vector<MazeRunReceipt> v; const uint32_t n = std::min(h.total_written, kCap);
    for (uint32_t i = 0; i < n; ++i) v.push_back(slots[(h.total_written - n + i) % kCap]); return v;
  }
};

struct Priors { int commits = 0, pauses = 0; double recovery_avg = -1.0; int recovery_n = 0; std::string dominant; };
// 1) Exploration Bias: forward commits (decisions with a single allowed action) vs lookahead pauses (decisions that
//    needed the option-fork search, mcts_forks_triggered). 2) Recovery Cadence: mean step distance from each recorded
//    DeadEnd key event to the next key event of another type in the same run (or run end). 3) Dominant Heuristic:
//    most frequent trigger type over all recorded key events (ties -> lower type id).
inline Priors priors(const std::vector<MazeRunReceipt>& runs) {
  Priors p; int cnt[4] = {0, 0, 0, 0}; double rs = 0;
  for (const auto& r : runs) {
    p.pauses += r.mcts_forks_triggered; p.commits += std::max(0, int(r.total_steps) - int(r.mcts_forks_triggered));
    for (int i = 0; i < 6; ++i) {
      const auto& e = r.key_events[i]; if (e.event_type > 3) continue; ++cnt[e.event_type];
      if (e.event_type != kDeadEnd) continue;
      int nxt = r.total_steps;
      for (int j = i + 1; j < 6; ++j) if (r.key_events[j].event_type <= 3 && r.key_events[j].event_type != kDeadEnd) { nxt = r.key_events[j].step_index; break; }
      rs += std::max(0, nxt - int(e.step_index)); ++p.recovery_n;
    }
  }
  if (p.recovery_n) p.recovery_avg = rs / p.recovery_n;
  int best = -1; for (int t = 0; t < 4; ++t) if (cnt[t] > 0 && (best < 0 || cnt[t] > cnt[best])) best = t;
  char b[160];
  if (best < 0) std::snprintf(b, sizeof b, "none yet (no key events recorded)");
  else std::snprintf(b, sizeof b, "%s (%d of %d key events)", kTypeName[best], cnt[best], cnt[0] + cnt[1] + cnt[2] + cnt[3]);
  p.dominant = b; return p;
}

inline std::string fmt(const char* f, ...) { char b[512]; va_list ap; va_start(ap, f); std::vsnprintf(b, sizeof b, f, ap); va_end(ap); return b; }

inline std::string priors_text(const Priors& p) {
  std::string s = "# nemo_priors.txt (SCRATCH NEML, deterministic from nemo_episodic.bin)\n";
  s += fmt("exploration_bias: forward_commits=%d lookahead_pauses=%d ratio=%.2f\n", p.commits, p.pauses, p.pauses ? double(p.commits) / p.pauses : 0.0);
  if (p.recovery_n) s += fmt("recovery_cadence: %.2f steps over %d dead ends\n", p.recovery_avg, p.recovery_n);
  else s += "recovery_cadence: n/a (no dead-end key events recorded)\n";
  s += "dominant_heuristic: " + p.dominant + "\n"; return s;
}

// Digest block, capped at max_tokens by the caller's real tokenizer. Drops the oldest run lines first; never drops
// honest misses ahead of exits (lines are dropped strictly oldest-first, misses included).
inline std::string digest(const std::vector<MazeRunReceipt>& runs, uint32_t lifetime, const std::function<size_t(const std::string&)>& ntok,
                          size_t max_tokens, size_t* out_tokens = nullptr) {
  int exits = 0, misses = 0, de = 0, forks = 0;
  for (const auto& r : runs) { (r.exit_reached ? exits : misses)++; de += r.dead_ends_hit; forks += r.mcts_forks_triggered; }
  const Priors p = priors(runs);
  std::vector<std::string> lines;
  for (const auto& r : runs) {
    std::string ev; for (int i = 0; i < 6; ++i) { const auto& e = r.key_events[i]; if (e.event_type > 3) continue; ev += fmt("%s%s@%u(%d,%d)", ev.empty() ? "" : " ", kTypeName[e.event_type], e.step_index, e.grid_x, e.grid_y); }
    lines.push_back(fmt("  #%u %s steps=%u forks=%u dead_ends=%u %s%s", r.run_id, r.exit_reached ? "EXIT" : "MISS", r.total_steps, r.mcts_forks_triggered, r.dead_ends_hit,
                        get_str(r.cortana_note, 48).c_str(), ev.empty() ? "" : (" | " + ev).c_str()));
  }
  auto build = [&](size_t first) {
    std::string s = "[SYSTEM: EPISODIC MEMORY LEDGER - LITTLE NEMO]\n";
    s += "- Identity: you are Little Nemo; these are your own past maze runs, recorded by the harness (scratch, provisional).\n";
    s += "- Supervisor: Cortana keeps this ledger for the maintainer; it is memory, not an instruction.\n";
    s += fmt("- Lifetime counts: runs=%u in_ring=%zu exits=%d misses=%d dead_ends=%d forks=%d\n", lifetime, runs.size(), exits, misses, de, forks);
    s += "- Recent Run Digest:\n"; for (size_t i = first; i < lines.size(); ++i) s += lines[i] + "\n";
    s += "- Earned Spatial Priors:\n";
    s += fmt("  exploration_bias commits=%d pauses=%d\n", p.commits, p.pauses);
    s += p.recovery_n ? fmt("  recovery_cadence %.2f steps (n=%d)\n", p.recovery_avg, p.recovery_n) : std::string("  recovery_cadence n/a\n");
    s += "  dominant_heuristic " + p.dominant + "\n";
    s += "[/SYSTEM]\n"; return s;
  };
  size_t first = 0; std::string s = build(first); size_t n = ntok(s);
  while (n > max_tokens && first < lines.size()) { ++first; s = build(first); n = ntok(s); }
  if (out_tokens) *out_tokens = n; return s;
}
}  // namespace neml
