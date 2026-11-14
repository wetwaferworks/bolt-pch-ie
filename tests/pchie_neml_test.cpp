// Bolt PCHIE - episodic memory ledger (NEML) checks. No model forward pass; the token cap is measured with Little
// Nemo's real tokenizer when the model file is found (otherwise with a stricter stand-in, and it says so).
// Every ledger here lives in a throwaway folder under the system temp folder; the user's ledger is never touched.
#include "pchie/HeadlessMaze.h"
#include "pchie/MazeGrid.h"
#include "pchie/ModelPath.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/neml/NemlLedger.h"
#include "pchie/neml/NemlQuery.h"
#include "pchie/neml/NemlStore.h"
#include "pchie/neml/NemlTrack.h"
#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace pchie;
namespace fs = std::filesystem;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[NEML] %-78s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
}
static void set_env(const char* k, const char* v) {
#ifdef _WIN32
  _putenv_s(k, v ? v : "");
#else
  if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}
static neml::Config cfg_in(const std::string& name) {
  neml::Config c;
  std::error_code ec;
  const fs::path d = fs::temp_directory_path() / "pchie-neml-test" / name;
  fs::remove_all(d, ec);
  c.dir = d.string();
  return c;
}
static neml::MazeRunReceipt mk(int i, bool exit, int nev) {
  neml::Recorder r;
  for (int k = 0; k < nev; ++k) r.event(uint8_t(k % 3), 3 * k + 1, uint8_t(100 + k), k, -k, "unit_event_tag_long_xx", uint8_t(k % 4));
  if (exit) r.event(neml::kExitFound, 3 * nev + 2, 255, 9, 9, "exit");
  auto rc = r.finish(1700000000u + uint32_t(i), 40 + i, exit, "unit receipt " + std::to_string(i) + " padding");
  rc.maze_key = 0x1234; rc.maze_seed = uint32_t(i); rc.maze_w = rc.maze_h = 16;
  return rc;
}
static std::string hexv(const std::vector<uint8_t>& v) {
  std::string s; char b[3];
  for (uint8_t x : v) { std::snprintf(b, sizeof b, "%02x", x); s += b; }
  return s;
}

struct Run { bool ok; std::vector<int32_t> actions; std::string world, head; MazeResult res; };
static Run run_maze(uint64_t seed, const neml::Config* cfg, bool record_unit, const neml::Snapshot* mem) {
  HeadlessMaze m(seed);
  m.set_search_budget(16);
  if (cfg) m.set_neml_config(cfg);
  m.set_neml_record_unit(record_unit);
  if (mem) m.set_memory(mem);
  Sha256HashChain ch;
  ch.reset();
  Run r;
  r.ok = m.run_unit("", &ch, r.res);
  r.actions = r.res.actions;
  r.world = hexv(r.res.h_world_sha);
  r.head = hexv(ch.head());
  return r;
}

int main() {
  // Real tokenizer for the cap (tokenizer only, no weights read).
  GgufFile gf;
  Gemma4Tokenizer tk;
  bool real_tok = false;
  {
    std::string path, err;
    if (resolve_model_path(path, err) && gf.open(path, err) && tk.load(gf, err)) real_tok = true;
  }
  const neml::TokenCount ntok = [&](const std::string& x) -> size_t {
    if (real_tok) return tk.encode(x).size();
    return x.size() / 2 + 1;  // stand-in: stricter than any real BPE on this ASCII text
  };
  std::printf("[NEML] token counter: %s\n", real_tok ? "Little Nemo's real tokenizer" : "stand-in (model file not found)");

  check(sizeof(neml::VisualLandmarkEvent) == 32 && sizeof(neml::MazeRunReceipt) == 256 && sizeof(neml::RingHeader) == 32,
        "layout: event 32 B, receipt 256 B, header 32 B");

  // ---- default on, and the off switches ----
  const char* keep[] = {"PCHIE_NEML", "PCHIE_NEML_REASON", "PCHIE_NEML_CHAT_WRITEBACK"};
  std::string saved[3];
  for (int i = 0; i < 3; ++i) { const char* v = std::getenv(keep[i]); saved[i] = v ? v : ""; set_env(keep[i], nullptr); }
  {
    const neml::Config d = neml::config_from_env();
    check(d.record && d.memory_prior && !d.chat_writeback, "default: recording ON, memory prior ON, chat write-back OFF");
    set_env("PCHIE_NEML", "0");
    set_env("PCHIE_NEML_REASON", "off");
    const neml::Config o = neml::config_from_env();
    check(!o.record && !o.memory_prior, "disable: PCHIE_NEML=0 and PCHIE_NEML_REASON=off turn both off");
    set_env("PCHIE_NEML", nullptr);
    set_env("PCHIE_NEML_REASON", nullptr);
    neml::Config f;
    char a0[] = "x", a1[] = "--neml", a2[] = "off", a3[] = "--neml-reason", a4[] = "off", a5[] = "--neml-chat-writeback", a6[] = "maybe";
    char* av[] = {a0, a1, a2, a3, a4, a5, a6};
    const int n1 = neml::parse_flag(7, av, 1, f), n2 = neml::parse_flag(7, av, 3, f), n3 = neml::parse_flag(7, av, 5, f);
    check(n1 == 2 && n2 == 2 && n3 == -1 && !f.record && !f.memory_prior, "disable: --neml off / --neml-reason off; a bad value is refused");
    const std::string dd = neml::default_dir();
    check(!dd.empty() && dd.find("bolt-pch-ie") == std::string::npos, "ledger folder: user data folder, outside the repository");
  }
  for (int i = 0; i < 3; ++i) set_env(keep[i], saved[i].empty() ? nullptr : saved[i].c_str());

  // ---- recording writes a receipt and changes nothing ----
  const neml::Config rec_cfg = cfg_in("record");
  neml::Config off_cfg = cfg_in("off");
  off_cfg.record = false;
  const Run base = run_maze(1, &off_cfg, true, nullptr);
  const Run rec = run_maze(1, &rec_cfg, true, nullptr);
  check(base.actions == rec.actions && base.world == rec.world && base.head == rec.head,
        "recording only: actions, H_world and chain head identical with recording on and off");
  check(rec.res.neml_receipt_id == 1 && fs::file_size(neml::ring_path(rec_cfg)) == neml::kRingBytes,
        "default on: one receipt written at teardown (4128-byte ring)");
  check(!fs::exists(neml::ring_path(off_cfg)) && base.res.neml_receipt_id == 0, "disable: recording off writes no file");
  {
    neml::Ring r;
    std::string e;
    const bool ld = neml::load_ring(rec_cfg, r, &e);
    const auto runs = r.ordered();
    const bool shape = ld && runs.size() == 1 && runs[0].total_steps == rec.res.actions.size() &&
                       runs[0].maze_key == HeadlessMaze(1).maze_key() && runs[0].maze_w == 16;
    check(shape, "receipt: steps, maze identity and size match the run");
    int pivots = 0, exits = 0;
    for (const auto& x : runs[0].key_events) { pivots += x.event_type == neml::kForkPivot; exits += x.event_type == neml::kExitFound; }
    check(rec.res.reached_exit == (exits == 1) && runs[0].exit_reached == uint8_t(rec.res.reached_exit), "ExitFound fires exactly when the exit is reached");
    check(runs[0].mcts_forks_triggered == rec.res.neml_forks && rec.res.neml_forks < int(rec.res.actions.size()),
          "forks: counts ForkPivot events only, not every decision");
    std::printf("[NEML] seed 1 unit run: steps=%zu events=%zu forks=%d dead_ends=%d exit=%d\n", rec.res.actions.size(),
                rec.res.neml_events, rec.res.neml_forks, rec.res.neml_dead_ends, int(rec.res.reached_exit));
  }
  {  // Recorder: the fork counter is the ForkPivot events and nothing else
    neml::Recorder r;
    r.event(neml::kForkPivot, 1, 128, 0, 0, "p", neml::kDirE);
    r.event(neml::kDeadEnd, 2, 255, 0, 0, "d", neml::kDirN);
    r.event(neml::kVisualRecovery, 3, 255, 1, 0, "v");
    check(r.forks == 1 && r.dead_ends == 1, "forks: one pivot among three events counts as one fork");
  }

  // ---- ring wrap and the v1 format ----
  {
    neml::Ring ring;
    for (int i = 0; i < 20; ++i) ring.append(mk(i, i % 3 != 0, 7));
    const auto ord = ring.ordered();
    check(ring.h.total_written == 20 && ord.size() == 16 && ord.front().run_id == 5 && ord.back().run_id == 20,
          "ring wrap: 20 appended, newest 16 kept (run ids 5..20, oldest first)");
    neml::Ring back;
    check(back.parse(ring.bytes()) && back.bytes() == ring.bytes(), "round trip: parse(bytes) is byte-identical");
    std::vector<uint8_t> v1(neml::kRingBytes, 0);
    std::memcpy(v1.data(), "NEML0001", 8);
    const uint32_t tw = 2, cap = 16;
    std::memcpy(v1.data() + 8, &tw, 4);
    std::memcpy(v1.data() + 12, &cap, 4);
    neml::V1Receipt o{};
    o.run_id = 1; o.total_steps = 30; o.exit_reached = 0;
    for (auto& e : o.key_events) e.event_type = neml::kNoEvent;
    std::snprintf(o.cortana_note, sizeof o.cortana_note, "old note");
    std::memcpy(v1.data() + 32, &o, sizeof o);
    neml::Ring old;
    check(old.parse(v1) && old.ordered().size() == 2 && old.ordered()[0].total_steps == 30 &&
              neml::get_str(old.ordered()[0].note, 32) == "old note",
          "older ledgers: a v1 file still loads");
  }

  // ---- token cap and honest misses ----
  {
    neml::Ring ring;
    for (int i = 0; i < 16; ++i) ring.append(mk(i, i % 3 != 0, 7));
    size_t nt = 0;
    const std::vector<std::string> notes(16, std::string(63, 'n'));
    const std::string d = neml::digest(ring.ordered(), ring.h.total_written, ntok, 384, &nt, notes);
    check(nt <= 384 && ntok(d) == nt, "token cap: a full ring plus 16 chat notes stays <= 384 tokens");
    std::printf("[NEML] full block: %zu tokens\n", nt);
    check(d.rfind("[SYSTEM: EPISODIC MEMORY LEDGER]", 0) == 0 && d.find("[/SYSTEM]") != std::string::npos, "block: header and footer");
    check(d.find("#16 ") != std::string::npos && d.find("#1 ") == std::string::npos, "token cap: oldest lines go first");
    neml::Ring small;
    small.append(mk(0, false, 2));
    small.append(mk(1, true, 2));
    size_t ns = 0;
    const std::string ds = neml::digest(small.ordered(), 2, ntok, 384, &ns);
    check(ds.find("#1 MISS") != std::string::npos && ds.find("#2 EXIT") != std::string::npos,
          "honest misses: a MISS is kept and shown like an exit");
    check(ds.find("misses=1") != std::string::npos, "honest misses: counted in the lifetime line");
    size_t tiny = 0;
    (void)neml::digest(ring.ordered(), 16, ntok, 200, &tiny);
    check(tiny <= 200, "token cap: a tight cap (200) drops lines and still fits");
  }

  // ---- memory block API and chat write-back (off) ----
  {
    neml::Config c = cfg_in("chat");
    std::string err;
    check(!neml::chat_writeback(c, "hello", err) && !fs::exists(neml::chat_path(c)), "chat write-back: OFF by default, nothing written");
    c.chat_writeback = true;
    check(neml::chat_writeback(c, std::string(100, 'x'), err), "chat write-back: when enabled, writes one note");
    const auto notes = neml::load_chat_notes(c);
    check(notes.size() == 1 && notes[0].size() == 63, "chat write-back: a note is at most 64 bytes");
    size_t nt = 0;
    const std::string blk = neml::memory_block(c, ntok, 384, &nt);
    check(blk.find("Notes from past chats") != std::string::npos && nt <= 384, "memory_block(): reads the ledger and notes under the cap");
  }

  // ---- queries: deterministic, read only from the snapshot ----
  {
    neml::Ring ring;
    for (int i = 0; i < 5; ++i) ring.append(mk(i, i % 2 == 0, 4));
    const auto bytes = ring.bytes();
    const neml::Snapshot a = neml::Snapshot::from_bytes(bytes), b = neml::Snapshot::from_bytes(bytes);
    const neml::Query qa(a), qb(b);
    const auto ja = qa.junction(0x1234, 0, 0), jb = qb.junction(0x1234, 0, 0);
    check(a.sha256 == b.sha256 && ja.dead_ends == jb.dead_ends && ja.pivots == jb.pivots &&
              qa.events(0x1234).size() == qb.events(0x1234).size() && qa.recent_outcomes(0x1234, 3) == qb.recent_outcomes(0x1234, 3),
          "query determinism: same bytes, same answers");
    check(qa.events(0x9999).empty() && qa.runs_for(0x1234, 2).size() == 1, "query: filters by maze and seed");
    check(ja.dead_ends[neml::kDirN] == 5, "query: dead ends per direction at a junction");
    const neml::Config c = cfg_in("snap");
    std::error_code ec;
    fs::create_directories(c.dir, ec);
    (void)neml::write_file_atomic(neml::ring_path(c), bytes);
    const neml::Snapshot s1 = neml::Snapshot::take(c);
    neml::Recorder r;
    (void)neml::write_receipt(c, r, 3, false, "later", neml::ReceiptInfo{});
    const neml::Snapshot s2 = neml::Snapshot::take(c);
    check(s1.sha256 == a.sha256 && s1.runs.size() == 5 && s2.runs.size() == 6 && s2.sha256 != s1.sha256,
          "snapshot: fixed at run start; later writes do not change it");
  }

  // ---- the memory prior: a remembered dead end changes the choice ----
  {
    std::vector<PuctEdge> e(2);
    e[0].token = int32_t(MazeAction::North); e[0].prior_q20 = 1u << 19;
    e[1].token = int32_t(MazeAction::East);  e[1].prior_q20 = 1u << 19;
    const int32_t before = neml::top_prior_token(e);
    neml::Query::Junction j;
    j.dead_ends[neml::kDirN] = 1;
    auto e2 = e;
    const bool changed = neml::apply_memory_prior(e2, j, neml::dir_of_action);
    const int32_t after = neml::top_prior_token(e2);
    check(before == int32_t(MazeAction::North) && changed && after == int32_t(MazeAction::East) &&
              e2[0].prior_q20 + e2[1].prior_q20 == e[0].prior_q20 + e[1].prior_q20,
          "memory prior: a dead end north at this junction makes east the top choice (total kept)");
    neml::Query::Junction none;
    auto e3 = e;
    check(!neml::apply_memory_prior(e3, none, neml::dir_of_action) && e3[0].prior_q20 == e[0].prior_q20, "memory prior: nothing remembered, nothing changes");
  }
  {  // whole maze: remember a dead end at the first junction the seed 1 run passes, then run it again
    neml::Config quiet = cfg_in("prior");
    quiet.record = false;
    const Run plain = run_maze(1, &quiet, false, nullptr);
    MazeGrid w(1, 16, 16);
    int jx = -1, jy = -1, jstep = -1;
    uint8_t jd = neml::kDirNone;
    for (size_t i = 0; i < plain.actions.size() && jstep < 0; ++i) {
      const uint8_t d = neml::dir_of_action(plain.actions[i]);
      if (d != neml::kDirNone && neml::open_moves(w) >= 3) { jx = w.x(); jy = w.y(); jd = d; jstep = int(i); }
      (void)w.apply(static_cast<MazeAction>(plain.actions[i]));
    }
    check(jstep >= 0, "memory prior: the seed 1 run passes a junction");
    if (jstep >= 0) {
      neml::Recorder r;
      for (int k = 0; k < 5; ++k) r.event(neml::kDeadEnd, 10 + k, 255, jx, jy, "dead_end_branch", jd);
      auto rc = r.finish(1700000000u, 40, false, "remembered dead end");
      rc.maze_key = HeadlessMaze(1).maze_key();
      neml::Ring ring;
      ring.append(rc);
      const neml::Snapshot snap = neml::Snapshot::from_bytes(ring.bytes());
      const Run m1 = run_maze(1, &quiet, false, &snap);
      const Run m2 = run_maze(1, &quiet, false, &snap);
      neml::Ring other;
      auto rc2 = rc;
      rc2.maze_key ^= 1;  // the same memory filed under another maze
      other.append(rc2);
      const neml::Snapshot snap_other = neml::Snapshot::from_bytes(other.bytes());
      const Run m3 = run_maze(1, &quiet, false, &snap_other);
      std::printf("[NEML] junction (%d,%d) step %d dir %s: plain steps=%zu, with memory steps=%zu nudges=%d\n", jx, jy, jstep,
                  neml::dir_name(jd), plain.actions.size(), m1.actions.size(), m1.res.neml_memory_nudges);
      const bool flipped = m1.actions.size() > size_t(jstep) &&
                           std::equal(plain.actions.begin(), plain.actions.begin() + jstep, m1.actions.begin()) &&
                           m1.actions[size_t(jstep)] != plain.actions[size_t(jstep)];
      check(m1.res.neml_memory_nudges > 0 && flipped, "memory prior: a remembered dead end at the junction changes the choice there");
      check(m1.actions == m2.actions && m1.head == m2.head && m1.world == m2.world, "memory prior: same snapshot, same run (reproducible)");
      check(m1.head != plain.head && m1.res.neml_snapshot_sha == snap.sha256, "provenance: the snapshot hash is in the run's chain");
      check(m3.actions == plain.actions && m3.res.neml_memory_nudges == 0, "memory prior: memory of another maze changes nothing here");
    }
  }
  std::printf("[NEML] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}