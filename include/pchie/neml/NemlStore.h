// Bolt PCHIE - NEML store: where the ledger lives, the on/off switches, receipt write-out at run teardown, the
// memory-block builder for the chat interface, and the chat write-back hook (off).
//
// Switches (process-wide, read once from the environment, then from command-line flags):
//   recording      ON by default.  PCHIE_NEML=0 or --neml off disables it.
//   memory prior   ON by default for the runs that ask for it (pilot, reasoning-layer --maze runs).
//                  PCHIE_NEML_REASON=0 or --neml-reason off disables it. Official pinned tests never ask.
//   chat write-back OFF by default. PCHIE_NEML_CHAT_WRITEBACK=1 or --neml-chat-writeback on enables it.
// Directory: PCHIE_NEML_DIR, or --neml DIR, else the user data folder
//   (Windows: %LOCALAPPDATA%\BoltPCHIE\neml; elsewhere: $XDG_DATA_HOME/bolt-pchie/neml or ~/.local/share/bolt-pchie/neml).
// The ledger is never written inside the repository.
#pragma once
#include "pchie/neml/NemlJournal.h"
#include "pchie/neml/NemlLedger.h"
#include <algorithm>
#include <cstddef>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace pchie {
namespace neml {

struct Config {
  bool record = true;           // write a receipt at every run teardown
  bool memory_prior = true;     // runs that opt in read a ledger snapshot as a prior
  bool chat_writeback = false;  // chat write-back hook (not approved; stays off)
  std::string dir;              // empty = default user data folder
};

inline bool env_flag(const char* name, bool dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  const std::string s(v);
  if (s == "0" || s == "off" || s == "OFF" || s == "false" || s == "no") return false;
  if (s == "1" || s == "on" || s == "ON" || s == "true" || s == "yes") return true;
  return dflt;
}

inline std::string default_dir() {
  namespace fs = std::filesystem;
#ifdef _WIN32
  if (const char* la = std::getenv("LOCALAPPDATA"); la && *la) return (fs::path(la) / "BoltPCHIE" / "neml").string();
#endif
  if (const char* x = std::getenv("XDG_DATA_HOME"); x && *x) return (fs::path(x) / "bolt-pchie" / "neml").string();
  if (const char* h = std::getenv("HOME"); h && *h) return (fs::path(h) / ".local" / "share" / "bolt-pchie" / "neml").string();
  std::error_code ec;
  return (fs::temp_directory_path(ec) / "bolt-pchie-neml").string();
}

inline Config config_from_env() {
  Config c;
  c.record = env_flag("PCHIE_NEML", true);
  c.memory_prior = env_flag("PCHIE_NEML_REASON", true);
  c.chat_writeback = env_flag("PCHIE_NEML_CHAT_WRITEBACK", false);
  if (const char* d = std::getenv("PCHIE_NEML_DIR"); d && *d) c.dir = d;
  return c;
}

inline std::string resolved_dir(const Config& c) { return c.dir.empty() ? default_dir() : c.dir; }

// Process-wide config. Set once at program start (before any run thread); runs read it at teardown.
inline Config& process_config() {
  static Config c = config_from_env();
  return c;
}

// Consumes the NEML flags it knows, at argv[i]; returns how many argv entries it used (0 = not a NEML flag,
// -1 = bad value). Flags: --neml on|off|DIR, --neml-reason on|off, --neml-chat-writeback on|off.
inline int parse_flag(int argc, char** argv, int i, Config& c) {
  const std::string k = argv[i];
  if (k != "--neml" && k != "--neml-reason" && k != "--neml-chat-writeback") return 0;
  if (i + 1 >= argc) return -1;
  const std::string v = argv[i + 1];
  const bool on = v == "on", off = v == "off";
  if (k == "--neml") {
    if (on) c.record = true;
    else if (off) c.record = false;
    else { c.record = true; c.dir = v; }
    return 2;
  }
  if (!on && !off) return -1;
  if (k == "--neml-reason") c.memory_prior = on;
  else c.chat_writeback = on;
  return 2;
}

// Applies every NEML flag in argv to the process config (for programs with a free-form argument list).
inline bool apply_cli(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const int n = parse_flag(argc, argv, i, process_config());
    if (n < 0) return false;
    if (n > 0) i += n - 1;
  }
  return true;
}

inline std::string ring_path(const Config& c) { return (std::filesystem::path(resolved_dir(c)) / "nemo_episodic.bin").string(); }
inline std::string ring_journal_path(const Config& c) { return (std::filesystem::path(resolved_dir(c)) / "nemo_episodic.journal").string(); }
inline std::string priors_path(const Config& c) { return (std::filesystem::path(resolved_dir(c)) / "nemo_priors.txt").string(); }
inline std::string chat_path(const Config& c) { return (std::filesystem::path(resolved_dir(c)) / "nemo_chat_notes.bin").string(); }
inline std::string chat_journal_path(const Config& c) { return (std::filesystem::path(resolved_dir(c)) / "nemo_chat_notes.journal").string(); }

inline std::vector<uint8_t> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---- The receipt ring and its journal. ----
// A receipt record's payload is the 256-byte receipt as stored, run_id included; it must be the next run.
inline bool ring_apply(char kind, const std::vector<uint8_t>& payload, std::vector<uint8_t>& state, std::string& why) {
  if (kind != 'R') { why = "not a receipt record"; return false; }
  if (payload.size() != sizeof(MazeRunReceipt)) { why = "a receipt record must hold 256 bytes"; return false; }
  Ring ring;
  if (!ring.parse(state)) { why = "the ring before this record is unreadable"; return false; }
  MazeRunReceipt r;
  std::memcpy(&r, payload.data(), sizeof r);
  if (r.run_id != uint32_t(ring.total() + 1)) { why = "the receipt is not the next run"; return false; }
  ring.append(r);
  ring.h.flags |= kRingJournaled;
  state = ring.bytes();
  return true;
}
inline SaveFile ring_save(const Config& c) {
  SaveFile sf;
  sf.file = ring_path(c);
  sf.journal = ring_journal_path(c);
  sf.apply = ring_apply;
  sf.valid = [](const std::vector<uint8_t>& b) { Ring r; return r.parse(b); };
  sf.journaled = [](const std::vector<uint8_t>& b) {
    return b.size() == kRingBytes && std::memcmp(b.data(), "NEML0002", 8) == 0 && (b[offsetof(RingHeader, flags)] & kRingJournaled);
  };
  return sf;
}

// Loads the ring and checks it against its journal (see load_checked). On refusal r is left empty and err says
// why; nothing on disk is changed. ld (optional) receives what a save needs.
inline bool load_ring(const Config& c, Ring& r, std::string* err = nullptr, Loaded* ld = nullptr) {
  Loaded tmp;
  Loaded& L = ld ? *ld : tmp;
  std::string why;
  r.clear();
  if (!load_checked(ring_save(c), L, why) || !r.parse(L.state)) {
    r.clear();
    if (err) *err = why.empty() ? "unreadable ledger " + ring_path(c) : why;
    return false;
  }
  return true;
}

inline const char* busy_text() { return "ledger busy: another PCHIE process is saving to it (waited 10 s)"; }

struct ReceiptInfo {
  uint64_t maze_key = 0;
  uint32_t maze_seed = 0;
  int maze_w = 0, maze_h = 0;
  uint8_t path_kind = kPathMaze;
  uint8_t flags = 0;
};

// Appends one receipt (journal first, then the ring) and rewrites the priors file. Returns the run id, or 0 when
// recording is off or the ledger was refused or could not be written (err says which). A ledger that fails its
// checks is left exactly as it was. Never touches a chain, an action, or a hash of the run.
inline uint32_t write_receipt(const Config& c, const Recorder& rec, int total_steps, bool exit_reached,
                              const std::string& note, const ReceiptInfo& info, std::string* err = nullptr) {
  if (!c.record) { if (err) *err = "recording off"; return 0; }
  const std::string dir = resolved_dir(c);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  SaveLock lock(dir);
  if (!lock.held()) { if (err) *err = busy_text(); return 0; }
  Ring ring;
  Loaded ld;
  if (!load_ring(c, ring, err, &ld)) return 0;
  MazeRunReceipt r = rec.finish(uint32_t(std::time(nullptr)), total_steps, exit_reached, note);
  r.maze_key = info.maze_key; r.maze_seed = info.maze_seed;
  r.maze_w = uint8_t(std::clamp(info.maze_w, 0, 255)); r.maze_h = uint8_t(std::clamp(info.maze_h, 0, 255));
  r.path_kind = info.path_kind; r.flags = info.flags;
  r.run_id = uint32_t(ring.total() + 1);
  std::vector<uint8_t> payload(sizeof r), next;
  std::memcpy(payload.data(), &r, sizeof r);
  std::string why;
  if (!save_checked(ring_save(c), ld, 'R', payload, next, why)) { if (err) *err = "ledger write failed: " + why; return 0; }
  Ring after;
  (void)after.parse(next);
  const std::string pt = priors_text(priors(after.ordered()));
  (void)write_file_atomic(priors_path(c), std::vector<uint8_t>(pt.begin(), pt.end()));
  return r.run_id;
}

// ---- Chat write-back hook: 64-byte notes Nemo writes after a chat. OFF by default (not approved). ----
// nemo_chat_notes.bin: "NEMLCHT1", the note count (low 32 bits at 8, high 16 bits at 12), flags at 14, then 16
// slots of 64 bytes. Journaled like the ring, in nemo_chat_notes.journal (record kind C, payload = the note).
static constexpr size_t kChatNoteBytes = 64;
static constexpr size_t kChatBytes = 16 + kCap * kChatNoteBytes;
inline uint64_t chat_total(const std::vector<uint8_t>& b) {
  uint32_t lo = 0; uint16_t hi = 0;
  std::memcpy(&lo, b.data() + 8, 4); std::memcpy(&hi, b.data() + 12, 2);
  return (uint64_t(hi) << 32) | lo;
}
inline bool chat_valid(const std::vector<uint8_t>& b) {
  return b.empty() || (b.size() == kChatBytes && std::memcmp(b.data(), "NEMLCHT1", 8) == 0);
}
inline bool chat_apply(char kind, const std::vector<uint8_t>& payload, std::vector<uint8_t>& state, std::string& why) {
  if (kind != 'C') { why = "not a chat note record"; return false; }
  if (payload.size() != kChatNoteBytes || payload.back() != 0) { why = "a chat note record must hold 64 bytes ending in 0"; return false; }
  if (!chat_valid(state)) { why = "the chat notes before this record are unreadable"; return false; }
  if (state.empty()) { state.assign(kChatBytes, 0); std::memcpy(state.data(), "NEMLCHT1", 8); }
  const uint64_t total = chat_total(state);
  if (total >= (uint64_t(1) << 48) - 1) { why = "the chat note count is full"; return false; }
  std::memcpy(state.data() + 16 + size_t(total % kCap) * kChatNoteBytes, payload.data(), kChatNoteBytes);
  const uint64_t t = total + 1;
  const uint32_t lo = uint32_t(t); const uint16_t hi = uint16_t(t >> 32);
  std::memcpy(state.data() + 8, &lo, 4); std::memcpy(state.data() + 12, &hi, 2);
  state[14] |= kRingJournaled;
  return true;
}
inline SaveFile chat_save(const Config& c) {
  SaveFile sf;
  sf.file = chat_path(c);
  sf.journal = chat_journal_path(c);
  sf.apply = chat_apply;
  sf.valid = chat_valid;
  sf.journaled = [](const std::vector<uint8_t>& b) { return b.size() == kChatBytes && (b[14] & kRingJournaled); };
  return sf;
}
// The chat notes, oldest first. Refuses (false, err) a notes file that fails its checks; never resets it.
inline bool load_chat_notes(const Config& c, std::vector<std::string>& out, std::string* err = nullptr) {
  out.clear();
  Loaded ld;
  std::string why;
  if (!load_checked(chat_save(c), ld, why)) { if (err) *err = why; return false; }
  const auto& b = ld.state;
  if (b.empty()) return true;
  const uint64_t total = chat_total(b), n = std::min<uint64_t>(total, kCap);
  for (uint64_t i = 0; i < n; ++i) {
    const size_t slot = size_t((total - n + i) % kCap);
    out.push_back(get_str(reinterpret_cast<const char*>(b.data() + 16 + slot * kChatNoteBytes), kChatNoteBytes));
  }
  return true;
}
inline bool chat_writeback(const Config& c, const std::string& note, std::string& err) {
  if (!c.chat_writeback) { err = "chat write-back is off"; return false; }
  const std::string dir = resolved_dir(c);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  SaveLock lock(dir, "nemo_chat_notes.lock");
  if (!lock.held()) { err = busy_text(); return false; }
  Loaded ld;
  if (!load_checked(chat_save(c), ld, err)) return false;
  std::vector<uint8_t> payload(kChatNoteBytes), next;
  put_str(reinterpret_cast<char*>(payload.data()), kChatNoteBytes, note);
  std::string why;
  if (!save_checked(chat_save(c), ld, 'C', payload, next, why)) { err = "chat note write failed: " + why; return false; }
  return true;
}

// ---- Memory block for the chat interface. ----
// The chat UI calls this to get the prefix it puts before the conversation. ntok must be the real tokenizer of
// the model in use (the cap is in its tokens). Reads only; returns the block and, in *tokens, its size. When the
// ledger or the chat notes fail their checks (or the ledger stays busy) it returns an empty string and says why
// in *err: a damaged memory is refused, never shown as an empty one.
inline std::string memory_block(const Config& c, const TokenCount& ntok, size_t max_tokens = 384, size_t* tokens = nullptr,
                                std::string* err = nullptr) {
  if (tokens) *tokens = 0;
  if (err) err->clear();
  const std::string dir = resolved_dir(c);
  Ring ring;
  std::vector<std::string> notes;
  std::string why;
  {
    std::error_code ec;
    if (std::filesystem::exists(ring_path(c), ec) || std::filesystem::exists(ring_journal_path(c), ec)) {
      SaveLock lock(dir);
      if (!lock.held()) { if (err) *err = busy_text(); return std::string(); }
      if (!load_ring(c, ring, &why)) { if (err) *err = why; return std::string(); }
    }
    if (std::filesystem::exists(chat_path(c), ec) || std::filesystem::exists(chat_journal_path(c), ec)) {
      SaveLock lock(dir, "nemo_chat_notes.lock");
      if (!lock.held()) { if (err) *err = busy_text(); return std::string(); }
      if (!load_chat_notes(c, notes, &why)) { if (err) *err = why; return std::string(); }
    }
  }
  return digest(ring.ordered(), ring.total(), ntok, max_tokens, tokens, notes);
}
inline std::string memory_block(const TokenCount& ntok, size_t max_tokens = 384, size_t* tokens = nullptr, std::string* err = nullptr) {
  return memory_block(process_config(), ntok, max_tokens, tokens, err);
}

}  // namespace neml
}  // namespace pchie