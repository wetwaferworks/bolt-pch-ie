// Bolt PCHIE - NEML read-only queries and the memory prior.
//
// A Snapshot is the ledger as it was when a run started: the raw ring bytes, their SHA-256, and the parsed
// receipts. Every query reads only the snapshot, so its answers depend only on those bytes (never on the clock,
// the file system after start, or receipts written during the run). A run that reads memory hashes the snapshot
// into its own provenance (see HeadlessMaze::set_memory), so its decisions can be reproduced from the bytes.
#pragma once
#include "pchie/Sha256.h"
#include "pchie/neml/NemlLedger.h"
#include "pchie/neml/NemlStore.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace neml {

struct Snapshot {
  std::vector<uint8_t> bytes;     // ring bytes as checked against the journal (empty = empty ledger)
  std::vector<uint8_t> sha256;    // SHA-256 of bytes (of the empty string for an empty ledger)
  std::vector<MazeRunReceipt> runs;  // oldest -> newest
  uint64_t lifetime = 0;
  bool ok = true;                 // false: the ledger was refused (see why); the snapshot holds no memory
  bool busy = false;              // true: another writer held the ledger lock, so nothing was read
  std::string why;                // why the ledger was refused (empty when ok)

  static Snapshot from_bytes(const std::vector<uint8_t>& b) {
    Snapshot s;
    Ring r;
    s.ok = r.parse(b);
    if (s.ok) { s.bytes = b; s.runs = r.ordered(); s.lifetime = r.total(); }
    else s.why = "unreadable ledger bytes";
    s.sha256 = Sha256::hash(s.bytes.data(), s.bytes.size());
    return s;
  }
  static Snapshot empty() { return from_bytes({}); }
  // Takes the snapshot from the ledger (under the save lock, checked against its journal). A ledger that fails
  // its checks gives ok=false and why; callers must not treat that as an empty memory.
  static Snapshot take(const Config& c) {
    const std::string dir = resolved_dir(c);
    std::error_code ec;
    if (!std::filesystem::exists(ring_path(c), ec) && !std::filesystem::exists(ring_journal_path(c), ec)) return empty();
    SaveLock lock(dir);
    if (!lock.held()) { Snapshot s = empty(); s.ok = false; s.busy = true; s.why = busy_text(); return s; }
    Ring r;
    Loaded ld;
    std::string why;
    if (!load_ring(c, r, &why, &ld)) { Snapshot s = empty(); s.ok = false; s.why = why; return s; }
    return from_bytes(ld.state);
  }
};

// Maze identity: first 8 bytes (little-endian) of SHA-256 of the starting maze's canonical bytes.
inline uint64_t maze_key_of(const std::vector<uint8_t>& canonical) {
  const auto h = Sha256::hash(canonical.data(), canonical.size());
  uint64_t k = 0;
  for (int i = 7; i >= 0; --i) k = (k << 8) | h[size_t(i)];
  return k;
}

struct EventHit {
  uint32_t run_id;
  bool run_exit;
  VisualLandmarkEvent e;
};

struct Query {
  const Snapshot& s;
  explicit Query(const Snapshot& snap) : s(snap) {}

  // Receipts for one maze (maze_key, plus the seed when seed >= 0), oldest -> newest.
  std::vector<MazeRunReceipt> runs_for(uint64_t maze_key, int64_t seed = -1) const {
    std::vector<MazeRunReceipt> v;
    for (const auto& r : s.runs)
      if (r.maze_key == maze_key && (seed < 0 || r.maze_seed == uint32_t(seed))) v.push_back(r);
    return v;
  }
  // Recorded key events, filtered by maze (0 = any), type (kNoEvent = any) and cell (when have_cell).
  std::vector<EventHit> events(uint64_t maze_key, uint8_t type = kNoEvent, bool have_cell = false, int x = 0, int y = 0) const {
    std::vector<EventHit> v;
    for (const auto& r : s.runs) {
      if (maze_key && r.maze_key != maze_key) continue;
      for (const auto& e : r.key_events) {
        if (e.event_type > 3) continue;
        if (type != kNoEvent && e.event_type != type) continue;
        if (have_cell && (e.grid_x != x || e.grid_y != y)) continue;
        v.push_back(EventHit{r.run_id, r.exit_reached != 0, e});
      }
    }
    return v;
  }
  // What happened at one junction in past runs of this maze, per direction out of it (N, S, E, W).
  struct Junction {
    std::array<int, 4> dead_ends{};   // DeadEnd events whose branch left this junction that way
    std::array<int, 4> pivots{};      // ForkPivot events that left this junction that way
    std::array<int, 4> exits{};       // runs that reached the exit after a pivot that way
    int total() const { int t = 0; for (int i = 0; i < 4; ++i) t += dead_ends[i] + pivots[i]; return t; }
  };
  Junction junction(uint64_t maze_key, int x, int y) const {
    Junction j;
    for (const auto& h : events(maze_key, kNoEvent, true, x, y)) {
      if (h.e.dir > 3) continue;
      if (h.e.event_type == kDeadEnd) ++j.dead_ends[h.e.dir];
      if (h.e.event_type == kForkPivot) { ++j.pivots[h.e.dir]; if (h.run_exit) ++j.exits[h.e.dir]; }
    }
    return j;
  }
  // Last n outcomes on this maze (true = exit), oldest -> newest.
  std::vector<bool> recent_outcomes(uint64_t maze_key, size_t n) const {
    const auto v = runs_for(maze_key);
    std::vector<bool> o;
    for (size_t i = v.size() > n ? v.size() - n : 0; i < v.size(); ++i) o.push_back(v[i].exit_reached != 0);
    return o;
  }
};

// Memory prior for root expansion: an edge whose direction led into a recorded dead end from this junction keeps
// 1/(1+count) of its prior; priors are then renormalised to the original total in integer Q20 (remainder to the
// lowest token), so the result is a pure function of the snapshot bytes and the edges. Returns true if any prior
// changed. dir_of maps an edge token to kDirN..kDirW (kDirNone for non-moves, which are left alone).
template <class Edge, class DirOf>
bool apply_memory_prior(std::vector<Edge>& edges, const Query::Junction& j, DirOf dir_of) {
  uint64_t before = 0;
  for (const auto& e : edges) before += e.prior_q20;
  bool changed = false;
  for (auto& e : edges) {
    const uint8_t d = dir_of(e.token);
    if (d > 3 || j.dead_ends[d] <= 0) continue;
    e.prior_q20 = uint32_t(uint64_t(e.prior_q20) / uint64_t(1 + j.dead_ends[d]));
    changed = true;
  }
  if (!changed) return false;
  uint64_t after = 0;
  for (const auto& e : edges) after += e.prior_q20;
  if (after == 0 || before == 0) return true;
  uint64_t sum = 0;
  size_t lo = 0;
  for (size_t i = 0; i < edges.size(); ++i) {
    edges[i].prior_q20 = uint32_t(uint64_t(edges[i].prior_q20) * before / after);
    sum += edges[i].prior_q20;
    if (edges[i].token < edges[lo].token) lo = i;
  }
  if (sum < before) edges[lo].prior_q20 += uint32_t(before - sum);
  return true;
}

}  // namespace neml
}  // namespace pchie