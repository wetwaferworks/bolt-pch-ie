// Bolt PCHIE - episodic memory ledger (NEML) checks. No model forward pass; the token cap is measured with Little
// Nemo's real tokenizer when the model file is found (otherwise with a stricter stand-in, and it says so).
// Every ledger here lives in a throwaway folder under the system temp folder; the user's ledger is never touched.
#include "pchie/HeadlessMaze.h"
#include "pchie/MazeGrid.h"
#include "pchie/ModelPath.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/neml/NemlInject.h"
#include "pchie/neml/NemlLedger.h"
#include "pchie/neml/NemlQuery.h"
#include "pchie/neml/NemlStore.h"
#include "pchie/neml/NemlTrack.h"
#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_gguf.h"

#include <algorithm>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <atomic>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

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
    const std::string d = neml::digest(ring.ordered(), ring.total(), ntok, 384, &nt, notes);
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
    std::vector<std::string> notes;
    check(neml::load_chat_notes(c, notes, &err) && notes.size() == 1 && notes[0].size() == 63, "chat write-back: a note is at most 64 bytes");
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
  // ---- journal: every save is chained; a damaged ledger is refused, never cleared ----
  {
    auto bytes_of = [](const std::string& p) { return neml::read_file(p); };
    auto put = [](const std::string& p, const std::vector<uint8_t>& b) { std::ofstream f(p, std::ios::binary | std::ios::trunc); f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size())); };
    auto put_s = [&](const std::string& p, const std::string& t) { put(p, std::vector<uint8_t>(t.begin(), t.end())); };
    auto str_of = [&](const std::string& p) { const auto b = bytes_of(p); return std::string(b.begin(), b.end()); };
    auto add = [](const neml::Config& c, int i, std::string* e = nullptr) { neml::Recorder r; r.event(neml::kDeadEnd, i, 255, i, 0, "t", neml::kDirN); return neml::write_receipt(c, r, 10 + i, false, "j" + std::to_string(i), neml::ReceiptInfo{}, e); };
    neml::Config c = cfg_in("journal");
    std::string e;
    bool all = true;
    for (int i = 0; i < 20; ++i) all = all && add(c, i, &e) == uint32_t(i + 1);
    neml::Ring r;
    neml::Loaded ld;
    check(all && neml::load_ring(c, r, &e, &ld) && r.total() == 20 && ld.has_journal && ld.records == 21 && !ld.pending,
          "journal: 20 saves give a starting record plus 20 chained records");
    const std::string ring_p = neml::ring_path(c), jp = neml::ring_journal_path(c);
    const std::vector<uint8_t> good_ring = bytes_of(ring_p);
    const std::string good_j = str_of(jp);

    // tamper with one byte of the ring (a note) -> refused, and a save does not touch either file
    std::vector<uint8_t> t = good_ring;
    t[32 + 3 * 256 + 230] ^= 1;
    put(ring_p, t);
    check(!neml::load_ring(c, r, &e) && e.find("does not match its journal") != std::string::npos && r.total() == 0,
          "tamper: an edited ring is refused, and the loaded ring is empty, not the edit");
    check(add(c, 99, &e) == 0 && bytes_of(ring_p) == t && str_of(jp) == good_j, "tamper: a save refuses and changes neither file");
    const neml::Snapshot ts = neml::Snapshot::take(c);
    check(!ts.ok && !ts.why.empty() && ts.runs.empty(), "tamper: a snapshot reports the refusal (ok=false, why) instead of an empty memory");
    size_t bt = 0;
    std::string berr;
    check(neml::memory_block(c, ntok, 384, &bt, &berr).empty() && !berr.empty(), "tamper: memory_block() refuses with a reason instead of an empty block");
    put(ring_p, good_ring);
    check(neml::load_ring(c, r, &e) && r.total() == 20, "tamper: putting the byte back loads again");

    // tamper with one journal record, delete a record, reorder -> refused
    {
      std::string j = good_j;
      const size_t at = j.find("|R|") + 3 + 64 + 1 + 100;
      j[at] = j[at] == 'a' ? 'b' : 'a';
      put_s(jp, j);
      check(!neml::load_ring(c, r, &e) && e.find("hash does not match") != std::string::npos, "tamper: an edited journal record is refused");
      std::string k = good_j;
      const size_t l1 = k.find('\n') + 1, l2 = k.find('\n', l1) + 1;
      k.erase(l1, l2 - l1);
      put_s(jp, k);
      check(!neml::load_ring(c, r, &e), "tamper: a removed journal record is refused");
      put_s(jp, good_j);
    }
    // the journal removed, the ring removed -> refused (the ring says it was journaled)
    fs::rename(jp, jp + ".away");
    check(!neml::load_ring(c, r, &e) && e.find("journal") != std::string::npos, "tamper: a journaled ring without its journal is refused");
    fs::rename(jp + ".away", jp);
    fs::rename(ring_p, ring_p + ".away");
    check(!neml::load_ring(c, r, &e), "tamper: a journal without its ring is refused");
    fs::rename(ring_p + ".away", ring_p);

    // a torn last line (a save cut short) -> refused; repair cuts only that, after checking the rest, with a backup
    put_s(jp, good_j + "NEJ1|21|R|" + std::string(40, 'a'));
    check(!neml::load_ring(c, r, &e) && e.find("torn") != std::string::npos, "torn line: refused, with the repair named");
    size_t removed = 0;
    std::string bak;
    check(neml::journal_repair_tail(neml::ring_save(c), removed, bak, e) && removed == 50 && str_of(jp) == good_j &&
              fs::exists(bak) && str_of(bak) == good_j + "NEJ1|21|R|" + std::string(40, 'a'),
          "torn line: repair cuts exactly the partial line and keeps a full backup");
    check(neml::load_ring(c, r, &e) && r.total() == 20, "torn line: the ledger loads after the repair");
    check(!neml::journal_repair_tail(neml::ring_save(c), removed, bak, e) && removed == 0, "repair: nothing torn, nothing changed");
    {
      std::string j = good_j;
      j[j.find("|R|") + 3] = j[j.find("|R|") + 3] == '0' ? '1' : '0';
      put_s(jp, j + "NEJ1|2");
      check(!neml::journal_repair_tail(neml::ring_save(c), removed, bak, e) && str_of(jp) == j + "NEJ1|2",
            "repair: refuses when a complete record is bad, and leaves the file alone");
      put_s(jp, good_j + "garbage");
      check(!neml::journal_repair_tail(neml::ring_save(c), removed, bak, e) && str_of(jp) == good_j + "garbage",
            "repair: refuses a tail that cannot be the start of a record");
      put_s(jp, good_j);
    }

    // an interrupted save: the journal has the record but the ring does not -> loads, and the next save completes it
    {
      neml::Config c2 = cfg_in("journal-pending");
      for (int i = 0; i < 3; ++i) (void)add(c2, i);
      const auto ring3 = bytes_of(neml::ring_path(c2));
      (void)add(c2, 3);
      put(neml::ring_path(c2), ring3);  // as if the ring rewrite never landed
      neml::Ring p;
      neml::Loaded pl;
      check(neml::load_ring(c2, p, &e, &pl) && pl.pending && p.total() == 4, "interrupted save: the journal's extra record is used");
      check(add(c2, 4) == 5 && neml::load_ring(c2, p, &e, &pl) && !pl.pending && p.total() == 5, "interrupted save: the next save completes it");
      put(neml::ring_path(c2), ring3);
      check(!neml::load_ring(c2, p, &e), "interrupted save: a ring two records behind is refused");
    }

    // an older ledger without a journal: loads; the first save keeps its bytes as the journal's starting record
    {
      neml::Config c3 = cfg_in("journal-legacy");
      fs::create_directories(c3.dir);
      neml::Ring old;
      for (int i = 0; i < 5; ++i) old.append(mk(i, true, 3));
      put(neml::ring_path(c3), old.bytes());
      neml::Ring p;
      neml::Loaded pl;
      check(neml::load_ring(c3, p, &e, &pl) && !pl.has_journal && p.total() == 5, "older ledger: loads without a journal");
      check(add(c3, 5) == 6, "older ledger: the next save works");
      const std::string j = str_of(neml::ring_journal_path(c3));
      check(j.rfind("NEJ1|0|G|" + neml::zero_sha() + "|" + neml::hex_of(old.bytes().data(), old.bytes().size()) + "|", 0) == 0,
            "older ledger: its bytes are the journal's starting record");
      std::vector<uint8_t> bad(100, 7);
      neml::Config c4 = cfg_in("journal-garbage");
      fs::create_directories(c4.dir);
      put(neml::ring_path(c4), bad);
      check(!neml::load_ring(c4, p, &e) && add(c4, 0, &e) == 0 && bytes_of(neml::ring_path(c4)) == bad && !fs::exists(neml::ring_journal_path(c4)),
            "unreadable ledger: refused, never reset to empty");
    }

    // chat notes: journaled the same way; a damaged notes file is refused, not reset
    {
      neml::Config cc = cfg_in("journal-chat");
      cc.chat_writeback = true;
      std::string err;
      bool ok = true;
      for (int i = 0; i < 18; ++i) ok = ok && neml::chat_writeback(cc, "note " + std::to_string(i), err);
      std::vector<std::string> notes;
      check(ok && neml::load_chat_notes(cc, notes, &err) && notes.size() == 16 && notes.front() == "note 2" && notes.back() == "note 17",
            "chat notes: 18 notes, newest 16 kept, journaled");
      auto b = bytes_of(neml::chat_path(cc));
      b[20] ^= 1;
      put(neml::chat_path(cc), b);
      check(!neml::load_chat_notes(cc, notes, &err) && !neml::chat_writeback(cc, "more", err) && bytes_of(neml::chat_path(cc)) == b,
            "chat notes: an edited notes file is refused and never reset");
      neml::Config cl = cfg_in("journal-chat-legacy");
      cl.chat_writeback = true;
      fs::create_directories(cl.dir);
      put(neml::chat_path(cl), std::vector<uint8_t>(50, 1));
      check(!neml::chat_writeback(cl, "x", err) && bytes_of(neml::chat_path(cl)) == std::vector<uint8_t>(50, 1),
            "chat notes: an unreadable older notes file is refused, not overwritten");
    }

    // pinned records: --verify --expect style anchoring through the loaded line hashes
    check(neml::load_ring(c, r, &e, &ld) && ld.line_sha.size() == 21 && ld.line_sha[20] == ld.head, "journal: every record's hash is available for pinning");
    { std::error_code ec; fs::remove_all(fs::path(c.dir).parent_path() / "journal-pending", ec); }
  }

  // ---- 64-bit counter: the lifetime count carries past 2^32 ----
  {
    neml::Ring ring;
    ring.set_total((uint64_t(1) << 32) - 2);
    for (int i = 0; i < 4; ++i) ring.append(mk(i, false, 1));
    neml::Ring back;
    check(ring.total() == (uint64_t(1) << 32) + 2 && ring.ordered().size() == 16 && back.parse(ring.bytes()) && back.total() == ring.total() &&
              ring.ordered().back().run_id == 2u,
          "counter: the lifetime count is 64-bit and survives a save");
  }

  // ---- save lock: an OS lock on the save file; a live holder is never taken over ----
  {
    neml::Config c = cfg_in("lock");
    fs::create_directories(c.dir);
    {
      std::optional<neml::SaveLock> a;
      a.emplace(c.dir);
      neml::SaveLock b(c.dir, "nemo_episodic.lock", 4);
      check(a->held() && !b.held(), "lock: a second lock waits and gives up while the first is held (no takeover)");
      { std::ofstream(neml::ring_path(c), std::ios::binary) << "x"; }
      std::string e;
      neml::Recorder rr;
      // a writer that would wait 10 s: use a thread so the test can release the first lock after a short while
      std::atomic<uint32_t> got{7};
      std::thread th([&] { neml::Config cw = c; got = neml::write_receipt(cw, rr, 1, false, "w", neml::ReceiptInfo{}, &e); });
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      check(got == 7, "lock: a writer waits while the lock is held");
      fs::remove(neml::ring_path(c));
      a.reset();
      th.join();
      check(got == 1, "lock: the waiting writer saves once the holder lets go");
    }
    {
      neml::SaveLock a(c.dir);
      const auto t0 = std::chrono::steady_clock::now();
      const neml::Snapshot busy = neml::Snapshot::take(c);
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      check(busy.busy && !busy.ok && busy.runs.empty() && secs > 5.0, "lock: take() waits, then reports busy; it never steals a live lock");
    }
    // threads in one process, no mutex: every save lands, in order, and the chain checks
    {
      neml::Config ct = cfg_in("lock-threads");
      std::vector<std::thread> ths;
      std::vector<int> ok(6, 0);
      for (int t = 0; t < 6; ++t)
        ths.emplace_back([&, t] {
          for (int i = 0; i < 10; ++i) { neml::Recorder rr; ok[size_t(t)] += neml::write_receipt(ct, rr, i, false, "t", neml::ReceiptInfo{}) != 0; }
        });
      for (auto& th : ths) th.join();
      int sum = 0;
      for (int v : ok) sum += v;
      neml::Ring r;
      neml::Loaded ld;
      std::string e;
      check(sum == 60 && neml::load_ring(ct, r, &e, &ld) && r.total() == 60 && ld.records == 61, "threads: 6 x 10 saves, all chained, none lost");
    }
#ifndef _WIN32
    // separate processes
    {
      neml::Config cp = cfg_in("lock-procs");
      fs::create_directories(cp.dir);
      std::vector<pid_t> kids;
      for (int k = 0; k < 4; ++k) {
        const pid_t pid = fork();
        if (pid == 0) {
          int n = 0;
          for (int i = 0; i < 10; ++i) { neml::Recorder rr; n += neml::write_receipt(cp, rr, i, false, "p", neml::ReceiptInfo{}) != 0; }
          _exit(n == 10 ? 0 : 1);
        }
        kids.push_back(pid);
      }
      bool kids_ok = true;
      for (pid_t p : kids) { int st = 0; kids_ok = kids_ok && waitpid(p, &st, 0) == p && WIFEXITED(st) && WEXITSTATUS(st) == 0; }
      neml::Ring r;
      neml::Loaded ld;
      std::string e;
      check(kids_ok && neml::load_ring(cp, r, &e, &ld) && r.total() == 40 && ld.records == 41, "processes: 4 x 10 saves, all chained, none lost");
    }
#endif
    std::error_code ec;
    fs::remove_all(fs::path(c.dir), ec);
  }
  // ---- --neml-inject: two independent gates, off by default ----
  {
    std::string why;
    const fs::path tmp = fs::temp_directory_path();
    check(neml::inject_allowed(false, false, "", why), "inject gate: not asked for, nothing to refuse (default runs unchanged)");
    check(!neml::inject_build_allowed(), "inject gate: the normal build does not allow it");
    check(!neml::inject_allowed(true, false, (tmp / "scratch-x" / "out").string(), why) && why.find("PCHIE_ALLOW_NEML_INJECT") != std::string::npos,
          "inject gate: refused by a build without the option, even in a scratch folder");
    check(!neml::inject_allowed(true, true, (tmp / "runs" / "official").string(), why) && why.find("scratch") != std::string::npos,
          "inject gate: refused for an output folder that is not under scratch*");
    check(!neml::inject_allowed(true, true, (tmp / "scratch" / ".." / "official").string(), why), "inject gate: '..' out of a scratch folder is refused");
    check(neml::inject_allowed(true, true, (tmp / "Scratch-2026" / "visnav").string(), why), "inject gate: both gates passed, scratch output, allowed");
  }
  std::printf("[NEML] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}