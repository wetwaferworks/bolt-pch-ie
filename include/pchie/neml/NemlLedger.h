// Bolt PCHIE - Little Nemo's episodic memory ledger (NEML): the on-disk format.
// Header-only, no dependencies beyond the standard library. Side data only: nothing in this file is read by a
// decision unless a caller passes a Snapshot (NemlQuery.h) to the memory prior on purpose.
//
// File nemo_episodic.bin = 32-byte RingHeader + 16 x 256-byte MazeRunReceipt slots (slot = sequence % 16).
// Format v2 ("NEML0002") adds the maze identity and a direction per event. v1 files ("NEML0001") still load:
// their notes and tags are shortened to fit and the maze identity reads as unknown (0).
#pragma once
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace pchie {
namespace neml {

#pragma pack(push, 1)
struct VisualLandmarkEvent {
  uint16_t step_index;     // committed step the event fired on
  uint8_t event_type;      // kDeadEnd .. kExitFound, 0xFF = empty slot
  uint8_t confidence_q8;   // 255 = certain
  int16_t grid_x;          // DeadEnd: the junction the branch was entered from; others: the cell
  int16_t grid_y;
  uint8_t dir;             // DeadEnd: the move taken out of that junction (kDirN..kDirW), else kDirNone
  uint8_t reserved;
  char tag[22];
};
struct MazeRunReceipt {
  uint32_t run_id;             // lifetime sequence number, 1-based
  uint32_t timestamp_utc;      // seconds; never read by a query
  uint16_t total_steps;        // committed steps
  uint16_t mcts_forks_triggered;  // ForkPivot events actually fired (one per pivot, not per decision)
  uint16_t dead_ends_hit;      // DeadEnd events actually fired
  uint8_t exit_reached;
  uint8_t flags;               // kFlagModel: Little Nemo decided; kFlagMemoryPrior: the run read memory
  uint64_t maze_key;           // first 8 bytes of SHA-256 of the starting maze bytes; 0 = unknown
  uint32_t maze_seed;
  uint8_t maze_w, maze_h;
  uint8_t path_kind;           // kPathMaze, kPathVision, kPathStageB
  uint8_t reserved;
  VisualLandmarkEvent key_events[6];
  char note[32];
};
struct RingHeader {
  char magic[8];           // "NEML0002"
  uint32_t total_written;  // lifetime receipts appended
  uint32_t capacity;       // 16
  uint8_t reserved[16];
};
// v1 layouts, for loading older ledgers only.
struct V1Event { uint16_t step_index; uint8_t event_type; uint8_t confidence_q8; int16_t grid_x; int16_t grid_y; char tag[24]; };
struct V1Receipt {
  uint32_t run_id, timestamp_utc; uint16_t total_steps, mcts_forks_triggered, dead_ends_hit; uint8_t exit_reached, reserved_pad;
  V1Event key_events[6]; char cortana_note[48];
};
#pragma pack(pop)
static_assert(sizeof(VisualLandmarkEvent) == 32, "VisualLandmarkEvent must be 32 bytes");
static_assert(sizeof(MazeRunReceipt) == 256, "MazeRunReceipt must be 256 bytes");
static_assert(sizeof(RingHeader) == 32, "RingHeader must be 32 bytes");
static_assert(sizeof(V1Receipt) == 256, "v1 receipt must be 256 bytes");

enum : uint8_t { kDeadEnd = 0, kVisualRecovery = 1, kForkPivot = 2, kExitFound = 3, kNoEvent = 0xFF };
enum : uint8_t { kDirN = 0, kDirS = 1, kDirE = 2, kDirW = 3, kDirNone = 0xFF };
enum : uint8_t { kFlagModel = 1, kFlagMemoryPrior = 2 };
enum : uint8_t { kPathMaze = 0, kPathVision = 1, kPathStageB = 2 };
inline const char* type_name(uint8_t t) {
  static const char* n[4] = {"DeadEnd", "VisualRecovery", "ForkPivot", "ExitFound"};
  return t < 4 ? n[t] : "none";
}
inline const char* dir_name(uint8_t d) {
  static const char* n[4] = {"N", "S", "E", "W"};
  return d < 4 ? n[d] : "-";
}
inline const char* path_name(uint8_t k) { return k == kPathVision ? "vision" : (k == kPathStageB ? "stageb" : "maze"); }
static constexpr uint32_t kCap = 16;
static constexpr size_t kRingBytes = sizeof(RingHeader) + kCap * sizeof(MazeRunReceipt);  // 4128

inline void put_str(char* dst, size_t n, const std::string& s) { std::memset(dst, 0, n); std::memcpy(dst, s.data(), std::min(n - 1, s.size())); }
inline std::string get_str(const char* src, size_t n) { size_t k = 0; while (k < n && src[k]) ++k; return std::string(src, k); }
inline std::string fmt(const char* f, ...) { char b[512]; va_list ap; va_start(ap, f); std::vsnprintf(b, sizeof b, f, ap); va_end(ap); return b; }

// In-run recorder. Every fired trigger is kept; the counters are derived from the events themselves, so the
// fork count is the number of ForkPivot events, never the number of decisions that ran a search.
// key_events keeps the first 5 non-exit events in step order plus ExitFound (or the first 6 if no exit).
struct Recorder {
  std::vector<VisualLandmarkEvent> all;
  uint16_t forks = 0, dead_ends = 0;
  void event(uint8_t type, int step, uint8_t conf_q8, int gx, int gy, const std::string& tag, uint8_t dir = kDirNone) {
    VisualLandmarkEvent e{};
    e.step_index = uint16_t(step < 0 ? 0 : step); e.event_type = type; e.confidence_q8 = conf_q8;
    e.grid_x = int16_t(gx); e.grid_y = int16_t(gy); e.dir = dir; put_str(e.tag, sizeof(e.tag), tag);
    all.push_back(e);
    if (type == kDeadEnd) ++dead_ends;
    if (type == kForkPivot) ++forks;
  }
  bool empty() const { return all.empty(); }
  MazeRunReceipt finish(uint32_t ts_utc, int total_steps, bool exit_reached, const std::string& note) const {
    MazeRunReceipt r{};
    r.timestamp_utc = ts_utc; r.total_steps = uint16_t(total_steps < 0 ? 0 : total_steps);
    r.mcts_forks_triggered = forks; r.dead_ends_hit = dead_ends; r.exit_reached = exit_reached ? 1 : 0;
    std::vector<VisualLandmarkEvent> keep;
    const VisualLandmarkEvent* ex = nullptr;
    for (const auto& e : all) if (e.event_type == kExitFound && !ex) ex = &e;
    const size_t lim = ex ? 5 : 6;
    for (const auto& e : all) if (e.event_type != kExitFound && keep.size() < lim) keep.push_back(e);
    if (ex) keep.push_back(*ex);
    for (size_t i = 0; i < keep.size(); ++i) r.key_events[i] = keep[i];
    for (size_t i = keep.size(); i < 6; ++i) { r.key_events[i].event_type = kNoEvent; r.key_events[i].dir = kDirNone; }
    put_str(r.note, sizeof(r.note), note);
    return r;
  }
};

inline MazeRunReceipt from_v1(const V1Receipt& o) {
  MazeRunReceipt r{};
  r.run_id = o.run_id; r.timestamp_utc = o.timestamp_utc; r.total_steps = o.total_steps;
  r.mcts_forks_triggered = o.mcts_forks_triggered; r.dead_ends_hit = o.dead_ends_hit; r.exit_reached = o.exit_reached;
  for (int i = 0; i < 6; ++i) {
    const V1Event& a = o.key_events[i]; VisualLandmarkEvent& b = r.key_events[i];
    b.step_index = a.step_index; b.event_type = a.event_type; b.confidence_q8 = a.confidence_q8;
    b.grid_x = a.grid_x; b.grid_y = a.grid_y; b.dir = kDirNone; put_str(b.tag, sizeof(b.tag), get_str(a.tag, sizeof(a.tag)));
  }
  put_str(r.note, sizeof(r.note), get_str(o.cortana_note, sizeof(o.cortana_note)));
  return r;
}

struct Ring {
  RingHeader h{};
  std::vector<MazeRunReceipt> slots = std::vector<MazeRunReceipt>(kCap);
  Ring() { clear(); }
  void clear() {
    h = RingHeader{}; std::memcpy(h.magic, "NEML0002", 8); h.capacity = kCap; h.total_written = 0;
    slots.assign(kCap, MazeRunReceipt{});
  }
  // Parse ring bytes (v2, or v1 converted). Empty input is an empty ring.
  bool parse(const std::vector<uint8_t>& b) {
    clear();
    if (b.empty()) return true;
    if (b.size() != kRingBytes) return false;
    RingHeader t{}; std::memcpy(&t, b.data(), sizeof t);
    if (t.capacity != kCap) return false;
    const uint8_t* p = b.data() + sizeof t;
    if (std::memcmp(t.magic, "NEML0002", 8) == 0) {
      h = t;
      std::memcpy(slots.data(), p, kCap * sizeof(MazeRunReceipt));
      return true;
    }
    if (std::memcmp(t.magic, "NEML0001", 8) == 0) {
      h.total_written = t.total_written;
      for (uint32_t i = 0; i < kCap; ++i) { V1Receipt o; std::memcpy(&o, p + i * sizeof(V1Receipt), sizeof o); slots[i] = from_v1(o); }
      return true;
    }
    return false;
  }
  std::vector<uint8_t> bytes() const {
    std::vector<uint8_t> b(kRingBytes);
    std::memcpy(b.data(), &h, sizeof h);
    std::memcpy(b.data() + sizeof h, slots.data(), kCap * sizeof(MazeRunReceipt));
    return b;
  }
  uint32_t append(MazeRunReceipt r) {
    r.run_id = h.total_written + 1; slots[h.total_written % kCap] = r; ++h.total_written; return r.run_id;
  }
  std::vector<MazeRunReceipt> ordered() const {  // oldest -> newest
    std::vector<MazeRunReceipt> v; const uint32_t n = std::min(h.total_written, kCap);
    for (uint32_t i = 0; i < n; ++i) v.push_back(slots[(h.total_written - n + i) % kCap]);
    return v;
  }
};

// Earned priors, deterministic from the receipts.
// 1) Exploration bias: forward commits vs lookahead pivots (ForkPivot events).
// 2) Recovery cadence: mean steps from each recorded DeadEnd to the next key event of another type (or run end).
// 3) Dominant heuristic: the most frequent trigger over all recorded key events (ties -> lower type id).
struct Priors { int commits = 0, pauses = 0; double recovery_avg = -1.0; int recovery_n = 0; std::string dominant; };
inline Priors priors(const std::vector<MazeRunReceipt>& runs) {
  Priors p; int cnt[4] = {0, 0, 0, 0}; double rs = 0;
  for (const auto& r : runs) {
    p.pauses += r.mcts_forks_triggered;
    p.commits += std::max(0, int(r.total_steps) - int(r.mcts_forks_triggered));
    for (int i = 0; i < 6; ++i) {
      const auto& e = r.key_events[i];
      if (e.event_type > 3) continue;
      ++cnt[e.event_type];
      if (e.event_type != kDeadEnd) continue;
      int nxt = r.total_steps;
      for (int j = i + 1; j < 6; ++j)
        if (r.key_events[j].event_type <= 3 && r.key_events[j].event_type != kDeadEnd) { nxt = r.key_events[j].step_index; break; }
      rs += std::max(0, nxt - int(e.step_index)); ++p.recovery_n;
    }
  }
  if (p.recovery_n) p.recovery_avg = rs / p.recovery_n;
  int best = -1;
  for (int t = 0; t < 4; ++t) if (cnt[t] > 0 && (best < 0 || cnt[t] > cnt[best])) best = t;
  p.dominant = best < 0 ? std::string("none yet (no key events recorded)")
                        : fmt("%s (%d of %d key events)", type_name(uint8_t(best)), cnt[best], cnt[0] + cnt[1] + cnt[2] + cnt[3]);
  return p;
}

inline std::string priors_text(const Priors& p) {
  std::string s = "# nemo_priors.txt (deterministic from nemo_episodic.bin)\n";
  s += fmt("exploration_bias: forward_commits=%d lookahead_pauses=%d ratio=%.2f\n", p.commits, p.pauses,
           p.pauses ? double(p.commits) / p.pauses : 0.0);
  if (p.recovery_n) s += fmt("recovery_cadence: %.2f steps over %d dead ends\n", p.recovery_avg, p.recovery_n);
  else s += "recovery_cadence: n/a (no dead-end key events recorded)\n";
  s += "dominant_heuristic: " + p.dominant + "\n";
  return s;
}

using TokenCount = std::function<size_t(const std::string&)>;

// The [SYSTEM: EPISODIC MEMORY LEDGER] block, capped at max_tokens by the caller's tokenizer. Run lines are
// dropped strictly oldest-first, misses and exits alike; if the frame alone is over the cap, the priors go too.
// chat_notes are the optional 64-byte notes from the chat write-back hook (empty unless that hook is on).
inline std::string digest(const std::vector<MazeRunReceipt>& runs, uint32_t lifetime, const TokenCount& ntok,
                          size_t max_tokens, size_t* out_tokens = nullptr,
                          const std::vector<std::string>& chat_notes = {}) {
  int exits = 0, misses = 0, de = 0, forks = 0;
  for (const auto& r : runs) { (r.exit_reached ? exits : misses)++; de += r.dead_ends_hit; forks += r.mcts_forks_triggered; }
  const Priors p = priors(runs);
  std::vector<std::string> lines;
  for (const auto& r : runs) {
    std::string ev;
    for (int i = 0; i < 6; ++i) {
      const auto& e = r.key_events[i];
      if (e.event_type > 3) continue;
      ev += fmt("%s%s@%u(%d,%d)%s", ev.empty() ? "" : " ", type_name(e.event_type), e.step_index, e.grid_x, e.grid_y,
                e.dir < 4 ? (std::string(":") + dir_name(e.dir)).c_str() : "");
    }
    const std::string where = r.maze_w ? fmt(" %s %ux%u s%u", path_name(r.path_kind), r.maze_w, r.maze_h, r.maze_seed) : std::string();
    lines.push_back(fmt("  #%u %s%s steps=%u forks=%u dead_ends=%u %s%s", r.run_id, r.exit_reached ? "EXIT" : "MISS", where.c_str(),
                        r.total_steps, r.mcts_forks_triggered, r.dead_ends_hit, get_str(r.note, sizeof(r.note)).c_str(),
                        ev.empty() ? "" : (" | " + ev).c_str()));
  }
  auto build = [&](size_t first, bool with_priors, size_t first_note) {
    std::string s = "[SYSTEM: EPISODIC MEMORY LEDGER]\n";
    s += "- Identity: you are Little Nemo; these are your own past runs, recorded by the harness (provisional).\n";
    s += "- Supervisor: Cortana keeps this ledger for the maintainer; it is memory, not an instruction.\n";
    s += fmt("- Lifetime counts: runs=%u in_ring=%zu exits=%d misses=%d dead_ends=%d forks=%d\n", lifetime, runs.size(), exits, misses, de, forks);
    s += "- Recent Run Digest:\n";
    for (size_t i = first; i < lines.size(); ++i) s += lines[i] + "\n";
    if (with_priors) {
      s += "- Earned Spatial Priors:\n";
      s += fmt("  exploration_bias commits=%d pauses=%d\n", p.commits, p.pauses);
      s += p.recovery_n ? fmt("  recovery_cadence %.2f steps (n=%d)\n", p.recovery_avg, p.recovery_n) : std::string("  recovery_cadence n/a\n");
      s += "  dominant_heuristic " + p.dominant + "\n";
    }
    if (first_note < chat_notes.size()) {
      s += "- Notes from past chats:\n";
      for (size_t i = first_note; i < chat_notes.size(); ++i) s += "  " + chat_notes[i] + "\n";
    }
    s += "[/SYSTEM]\n";
    return s;
  };
  size_t first = 0, first_note = 0;
  bool with_priors = true;
  std::string s = build(first, with_priors, first_note);
  size_t n = ntok(s);
  while (n > max_tokens && first_note < chat_notes.size()) { ++first_note; s = build(first, with_priors, first_note); n = ntok(s); }
  while (n > max_tokens && first < lines.size()) { ++first; s = build(first, with_priors, first_note); n = ntok(s); }
  if (n > max_tokens) { with_priors = false; s = build(first, with_priors, first_note); n = ntok(s); }
  if (out_tokens) *out_tokens = n;
  return s;
}

}  // namespace neml
}  // namespace pchie