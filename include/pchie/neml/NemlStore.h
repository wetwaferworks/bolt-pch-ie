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
#include "pchie/OsRandom.h"
#include "pchie/neml/NemlLedger.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
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
inline std::string priors_path(const Config& c) { return (std::filesystem::path(resolved_dir(c)) / "nemo_priors.txt").string(); }
inline std::string chat_path(const Config& c) { return (std::filesystem::path(resolved_dir(c)) / "nemo_chat_notes.bin").string(); }

inline std::vector<uint8_t> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

inline bool write_file_atomic(const std::string& p, const std::vector<uint8_t>& b) {
  const std::string tmp = p + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    if (!f) return false;
  }
  std::error_code ec;
  std::filesystem::rename(tmp, p, ec);
  if (ec) { std::filesystem::remove(p, ec); ec.clear(); std::filesystem::rename(tmp, p, ec); }
  return !ec;
}

// Serializes ledger writers: one mutex in the process, and a lock directory across processes. The directory
// holds an owner token. A lock older than 60 s is taken over by renaming it aside (only one process can win
// that rename) and checking the token it carried is the one judged stale; a fresh lock that slipped in is put
// back. A holder only removes the lock if it still carries its own token.
class WriteLock {
 public:
  explicit WriteLock(const std::string& dir) : lk_(mu()) {
    namespace fs = std::filesystem;
    path_ = (fs::path(dir) / "nemo_episodic.lock").string();
    token_ = make_token();
    const std::string d = path_ + ".d";
    for (int i = 0; i < 400 && !held_; ++i) {
      std::error_code ec;
      if (fs::create_directory(d, ec)) {
        if (write_token(d, token_)) { held_ = true; break; }
        fs::remove_all(d, ec);
        continue;
      }
      const auto t = fs::last_write_time(d, ec);
      if (!ec && fs::file_time_type::clock::now() - t > std::chrono::seconds(60)) {
        const std::string seen = read_token(d);
        const std::string aside = d + ".stale." + token_;
        std::error_code rec;
        fs::rename(d, aside, rec);
        if (!rec) {
          const auto ta = fs::last_write_time(aside, rec);
          const bool still_stale = !rec && fs::file_time_type::clock::now() - ta > std::chrono::seconds(60);
          if (read_token(aside) == seen && still_stale) fs::remove_all(aside, rec);
          else { std::error_code bec; fs::rename(aside, d, bec); if (bec) fs::remove_all(aside, bec); }
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
  }
  ~WriteLock() {
    if (!held_) return;
    namespace fs = std::filesystem;
    const std::string d = path_ + ".d", aside = d + ".release." + token_;
    std::error_code ec;
    if (read_token(d) != token_) return;  // taken over while we held it; never delete someone else's lock
    fs::rename(d, aside, ec);
    if (ec) return;
    if (read_token(aside) == token_) fs::remove_all(aside, ec);
    else { std::error_code bec; fs::rename(aside, d, bec); }
  }
  WriteLock(const WriteLock&) = delete;
  WriteLock& operator=(const WriteLock&) = delete;
  bool held() const { return held_; }
 private:
  static std::mutex& mu() { static std::mutex m; return m; }
  static std::string make_token() {
    uint8_t r[16] = {};
    if (!pchie::os_random_bytes(r, sizeof r)) {
      const uint64_t t = uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
      std::memcpy(r, &t, sizeof t);
    }
    static const char* hx = "0123456789abcdef";
    std::string s;
    for (uint8_t b : r) { s += hx[b >> 4]; s += hx[b & 15]; }
    return s;
  }
  static bool write_token(const std::string& d, const std::string& tok) {
    std::ofstream f(std::filesystem::path(d) / "owner", std::ios::binary | std::ios::trunc);
    f << tok;
    return bool(f);
  }
  static std::string read_token(const std::string& d) {
    std::ifstream f(std::filesystem::path(d) / "owner", std::ios::binary);
    std::string s;
    if (f) std::getline(f, s);
    return s.size() > 64 ? std::string() : s;
  }
  std::unique_lock<std::mutex> lk_;
  std::string path_, token_;
  bool held_ = false;
};

inline bool load_ring(const Config& c, Ring& r, std::string* err = nullptr) {
  if (!r.parse(read_file(ring_path(c)))) { if (err) *err = "unreadable ledger " + ring_path(c); return false; }
  return true;
}

struct ReceiptInfo {
  uint64_t maze_key = 0;
  uint32_t maze_seed = 0;
  int maze_w = 0, maze_h = 0;
  uint8_t path_kind = kPathMaze;
  uint8_t flags = 0;
};

// Appends one receipt and rewrites the priors file. Returns the run id, or 0 when recording is off or the
// ledger could not be written (err says which). Never touches a chain, an action, or a hash of the run.
inline uint32_t write_receipt(const Config& c, const Recorder& rec, int total_steps, bool exit_reached,
                              const std::string& note, const ReceiptInfo& info, std::string* err = nullptr) {
  if (!c.record) { if (err) *err = "recording off"; return 0; }
  const std::string dir = resolved_dir(c);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  WriteLock lock(dir);
  if (!lock.held()) { if (err) *err = "ledger busy"; return 0; }
  Ring ring;
  if (!load_ring(c, ring, err)) return 0;
  MazeRunReceipt r = rec.finish(uint32_t(std::time(nullptr)), total_steps, exit_reached, note);
  r.maze_key = info.maze_key; r.maze_seed = info.maze_seed;
  r.maze_w = uint8_t(std::clamp(info.maze_w, 0, 255)); r.maze_h = uint8_t(std::clamp(info.maze_h, 0, 255));
  r.path_kind = info.path_kind; r.flags = info.flags;
  const uint32_t id = ring.append(r);
  if (!write_file_atomic(ring_path(c), ring.bytes())) { if (err) *err = "ledger write failed"; return 0; }
  const std::string pt = priors_text(priors(ring.ordered()));
  (void)write_file_atomic(priors_path(c), std::vector<uint8_t>(pt.begin(), pt.end()));
  return id;
}

// ---- Chat write-back hook: 64-byte notes Nemo writes after a chat. OFF by default (not approved). ----
static constexpr size_t kChatNoteBytes = 64;
inline std::vector<std::string> load_chat_notes(const Config& c) {
  std::vector<std::string> out;
  const auto b = read_file(chat_path(c));
  if (b.size() != 16 + kCap * kChatNoteBytes || std::memcmp(b.data(), "NEMLCHT1", 8) != 0) return out;
  uint32_t total = 0; std::memcpy(&total, b.data() + 8, 4);
  const uint32_t n = std::min(total, kCap);
  for (uint32_t i = 0; i < n; ++i) {
    const size_t slot = (total - n + i) % kCap;
    out.push_back(get_str(reinterpret_cast<const char*>(b.data() + 16 + slot * kChatNoteBytes), kChatNoteBytes));
  }
  return out;
}
inline bool chat_writeback(const Config& c, const std::string& note, std::string& err) {
  if (!c.chat_writeback) { err = "chat write-back is off"; return false; }
  const std::string dir = resolved_dir(c);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  WriteLock lock(dir);
  if (!lock.held()) { err = "ledger busy"; return false; }
  auto b = read_file(chat_path(c));
  if (b.size() != 16 + kCap * kChatNoteBytes || std::memcmp(b.data(), "NEMLCHT1", 8) != 0) {
    b.assign(16 + kCap * kChatNoteBytes, 0);
    std::memcpy(b.data(), "NEMLCHT1", 8);
  }
  uint32_t total = 0; std::memcpy(&total, b.data() + 8, 4);
  put_str(reinterpret_cast<char*>(b.data() + 16 + (total % kCap) * kChatNoteBytes), kChatNoteBytes, note);
  ++total; std::memcpy(b.data() + 8, &total, 4);
  if (!write_file_atomic(chat_path(c), b)) { err = "chat note write failed"; return false; }
  return true;
}

// ---- Memory block for the chat interface. ----
// The chat UI calls this to get the prefix it puts before the conversation. ntok must be the real tokenizer of
// the model in use (the cap is in its tokens). Reads only; returns the block and, in *tokens, its size.
inline std::string memory_block(const Config& c, const TokenCount& ntok, size_t max_tokens = 384, size_t* tokens = nullptr) {
  Ring ring;
  std::string err;
  if (!load_ring(c, ring, &err)) ring.clear();
  return digest(ring.ordered(), ring.h.total_written, ntok, max_tokens, tokens, load_chat_notes(c));
}
inline std::string memory_block(const TokenCount& ntok, size_t max_tokens = 384, size_t* tokens = nullptr) {
  return memory_block(process_config(), ntok, max_tokens, tokens);
}

}  // namespace neml
}  // namespace pchie