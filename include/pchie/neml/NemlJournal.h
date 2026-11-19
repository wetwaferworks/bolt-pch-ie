// Bolt PCHIE - NEML journal: the append-only, hash-chained record behind each NEML save file, and the save-file lock.
//
// Each save file (the receipt ring, the chat notes) is a small fixed-size file that is rewritten on every save. The
// journal next to it keeps every change ever made, one line per change, each line carrying the SHA-256 of the line
// before it, so a removed, edited or reordered record breaks the chain. Replaying the journal from its first line
// gives the exact bytes the save file must hold; a save file that does not match is refused, never cleared.
//
// Line format (text, one record per line, '\n' terminated):
//   NEJ1|<seq>|<kind>|<prev>|<payload>|<state>|<sha>
//     seq      record number, decimal, 0 for the first line, then +1 per line
//     kind     G for the first line (payload = the save file's bytes when the journal began, possibly empty),
//              R for a receipt (payload = the 256-byte receipt), C for a chat note (payload = the 64-byte note)
//     prev     SHA-256 (64 lowercase hex) of the line before, 64 zeros on the first line
//     payload  lowercase hex
//     state    SHA-256 of the save file's bytes after this record
//     sha      SHA-256 of "NEJ1|<seq>|<kind>|<prev>|<payload>|<state>"
//
// The chain has no key: anyone who can write the folder can rewrite both files from scratch. It catches damage,
// lost saves and edits of single records; pin a known record (pchie_neml --verify --expect SEQ:SHA) to catch more.
//
// The save lock is an operating-system file lock on a lock file in the ledger folder (flock on POSIX, a CRT byte
// lock on Windows). The system drops it when the holder exits, so it is never taken over from a live holder and
// never left behind by a dead one. Each lock is its own open file, so two threads in one process exclude each other
// too, without a mutex.
#pragma once
#include "pchie/Sha256.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/locking.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pchie {
namespace neml {

static constexpr size_t kJournalMaxPayload = 8192;        // bytes; the largest real payload is a 4128-byte ring
static constexpr size_t kJournalMaxBytes = size_t(1) << 30; // a journal over 1 GiB is refused rather than read

inline std::string hex_of(const uint8_t* p, size_t n) {
  static const char* hx = "0123456789abcdef";
  std::string s(n * 2, '0');
  for (size_t i = 0; i < n; ++i) { s[2 * i] = hx[p[i] >> 4]; s[2 * i + 1] = hx[p[i] & 15]; }
  return s;
}
inline std::string sha_hex(const uint8_t* p, size_t n) { const auto h = Sha256::hash(p, n); return hex_of(h.data(), h.size()); }
inline std::string sha_hex(const std::vector<uint8_t>& b) { return sha_hex(b.data(), b.size()); }
inline std::string sha_hex(const std::string& s) { return sha_hex(reinterpret_cast<const uint8_t*>(s.data()), s.size()); }
inline const std::string& zero_sha() { static const std::string z(64, '0'); return z; }

inline int hex_digit(char c) { return c >= '0' && c <= '9' ? c - '0' : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1); }
inline bool is_hex64(const char* s, size_t n) {
  if (n != 64) return false;
  for (size_t i = 0; i < n; ++i) if (hex_digit(s[i]) < 0) return false;
  return true;
}
inline bool unhex(const char* s, size_t n, std::vector<uint8_t>& out) {
  out.clear();
  if (n % 2) return false;
  out.resize(n / 2);
  for (size_t i = 0; i < n / 2; ++i) {
    const int a = hex_digit(s[2 * i]), b = hex_digit(s[2 * i + 1]);
    if (a < 0 || b < 0) return false;
    out[i] = uint8_t(a * 16 + b);
  }
  return true;
}

// The state of a save file after a record, given the state before. Returns false (with why) for a record that
// cannot apply. Never called for the first (G) record: its payload is the state.
using JournalApply = std::function<bool(char kind, const std::vector<uint8_t>& payload, std::vector<uint8_t>& state,
                                        std::string& why)>;

inline std::string journal_line(uint64_t seq, char kind, const std::string& prev, const std::vector<uint8_t>& payload,
                                const std::string& state_sha) {
  std::string body = "NEJ1|" + std::to_string(seq) + "|" + std::string(1, kind) + "|" + prev + "|" +
                     hex_of(payload.data(), payload.size()) + "|" + state_sha;
  const std::string sha = sha_hex(body);
  return body + "|" + sha + "\n";
}

struct JournalView {
  uint64_t records = 0;             // complete, checked lines
  std::string head = zero_sha();    // sha of the last line (zeros when there is none)
  std::vector<uint8_t> state;       // the save file's bytes after the last line
  std::vector<uint8_t> prev_state;  // and after the line before it (valid when records >= 2)
  std::string state_sha, prev_state_sha;
  std::vector<std::string> line_sha;  // sha of every line, by seq
  size_t good_bytes = 0;            // bytes of complete, checked lines
  bool torn = false;                // the file ends in a partial line (a save that was cut short)
};

// Checks one journal's bytes from the start. Returns false (with why) at the first line that is malformed, out of
// order, unchained, mis-hashed, or does not replay to the state it records. A partial last line sets torn and stops
// there (true is returned only when everything before it is good, and the partial line could be the start of a
// record; the caller decides whether a torn journal is acceptable).
inline bool journal_check(const std::vector<uint8_t>& b, const JournalApply& apply, JournalView& v, std::string& why) {
  v = JournalView{};
  size_t at = 0;
  const char* s = reinterpret_cast<const char*>(b.data());
  while (at < b.size()) {
    const void* nl = std::memchr(s + at, '\n', b.size() - at);
    if (!nl) {
      const size_t n = b.size() - at;
      static const char kMagic[] = "NEJ1|";
      const size_t m = n < 5 ? n : 5;
      if (std::memcmp(s + at, kMagic, m) != 0) { why = "the journal ends in bytes that are not part of a record"; return false; }
      v.torn = true;
      return true;
    }
    const size_t end = size_t(static_cast<const char*>(nl) - s);
    const char* p = s + at;
    const size_t n = end - at;
    const std::string where = "journal record " + std::to_string(v.records) + ": ";
    // split on '|': exactly 7 fields
    size_t f0[7], fl[7], nf = 0, st = 0;
    for (size_t i = 0; i <= n; ++i) {
      if (i == n || p[i] == '|') {
        if (nf == 7) { why = where + "too many fields"; return false; }
        f0[nf] = st; fl[nf] = i - st; ++nf; st = i + 1;
      }
    }
    if (nf != 7) { why = where + "wrong number of fields"; return false; }
    if (fl[0] != 4 || std::memcmp(p, "NEJ1", 4) != 0) { why = where + "not a NEJ1 record"; return false; }
    // seq: canonical decimal
    if (fl[1] == 0 || fl[1] > 20 || (fl[1] > 1 && p[f0[1]] == '0')) { why = where + "bad record number"; return false; }
    uint64_t seq = 0;
    for (size_t i = 0; i < fl[1]; ++i) {
      const char c = p[f0[1] + i];
      if (c < '0' || c > '9') { why = where + "bad record number"; return false; }
      const uint64_t d = uint64_t(c - '0');
      if (seq > (UINT64_MAX - d) / 10) { why = where + "bad record number"; return false; }
      seq = seq * 10 + d;
    }
    if (seq != v.records) { why = where + "out of order (says " + std::to_string(seq) + ")"; return false; }
    if (fl[2] != 1) { why = where + "bad kind"; return false; }
    const char kind = p[f0[2]];
    if ((v.records == 0) != (kind == 'G')) { why = where + (kind == 'G' ? "a second starting record" : "the first record must be the starting record"); return false; }
    if (kind != 'G' && kind != 'R' && kind != 'C') { why = where + "unknown kind"; return false; }
    if (!is_hex64(p + f0[3], fl[3]) || std::string(p + f0[3], 64) != v.head) { why = where + "does not chain to the record before"; return false; }
    if (fl[4] > 2 * kJournalMaxPayload) { why = where + "payload too large"; return false; }
    std::vector<uint8_t> payload;
    if (!unhex(p + f0[4], fl[4], payload)) { why = where + "bad payload"; return false; }
    if (!is_hex64(p + f0[5], fl[5]) || !is_hex64(p + f0[6], fl[6])) { why = where + "bad hash field"; return false; }
    const std::string body(p, f0[6] - 1), sha(p + f0[6], 64);
    if (sha_hex(body) != sha) { why = where + "its hash does not match its contents"; return false; }
    std::vector<uint8_t> next;
    if (kind == 'G') next = payload;
    else {
      next = v.state;
      std::string w;
      if (!apply(kind, payload, next, w)) { why = where + w; return false; }
    }
    const std::string ss = sha_hex(next);
    if (ss != std::string(p + f0[5], 64)) { why = where + "replaying it does not give the state it records"; return false; }
    v.prev_state.swap(v.state);
    v.prev_state_sha.swap(v.state_sha);
    v.state.swap(next);
    v.state_sha = ss;
    v.head = sha;
    v.line_sha.push_back(sha);
    ++v.records;
    at = end + 1;
    v.good_bytes = at;
  }
  return true;
}

// Reads a whole file. exists=false and true for a file that is not there; false for one that is there but could
// not be read.
inline bool read_file_checked(const std::string& p, std::vector<uint8_t>& out, bool& exists) {
  out.clear();
  std::error_code ec;
  exists = std::filesystem::exists(p, ec);
  if (ec) return false;
  if (!exists) return true;
  std::FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) return false;
  bool ok = true;
  uint8_t buf[65536];
  for (;;) {
    const size_t n = std::fread(buf, 1, sizeof buf, f);
    out.insert(out.end(), buf, buf + n);
    if (out.size() > kJournalMaxBytes) { ok = false; break; }
    if (n < sizeof buf) { ok = !std::ferror(f); break; }
  }
  std::fclose(f);
  return ok;
}

inline bool flush_and_close(std::FILE* f) {
  bool ok = std::fflush(f) == 0;
#ifdef _WIN32
  ok = ok && _commit(_fileno(f)) == 0;
#else
  ok = ok && ::fsync(fileno(f)) == 0;
#endif
  return (std::fclose(f) == 0) && ok;
}

// Writes to "<p>.tmp", flushes it to the disk, then renames over p (replacing it on POSIX and Windows alike). A
// failure leaves the old file as it was; there is no delete-then-rename fallback that could lose it.
inline bool write_file_atomic(const std::string& p, const std::vector<uint8_t>& b) {
  const std::string tmp = p + ".tmp";
  std::FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) return false;
  const bool ok = (b.empty() || std::fwrite(b.data(), 1, b.size(), f) == b.size());
  const bool closed = flush_and_close(f);
  std::error_code ec;
  if (!ok || !closed) { std::filesystem::remove(tmp, ec); return false; }
  std::filesystem::rename(tmp, p, ec);
  if (ec) { std::error_code rec; std::filesystem::remove(tmp, rec); }
  return !ec;
}

// Appends one line and flushes it to the disk. On a failed write it cuts the file back to where it was, so a
// failed save never leaves a partial line of its own behind. Call only under the save lock.
inline bool journal_append(const std::string& path, const std::string& line, std::string& why) {
#ifdef _WIN32
  int fd = -1;
  if (_sopen_s(&fd, path.c_str(), _O_WRONLY | _O_CREAT | _O_BINARY | _O_NOINHERIT, _SH_DENYNO, _S_IREAD | _S_IWRITE) != 0 || fd < 0) {
    why = "cannot open the journal " + path; return false;
  }
  const __int64 start = _lseeki64(fd, 0, SEEK_END);
  bool ok = start >= 0;
  size_t done = 0;
  while (ok && done < line.size()) {
    const int w = _write(fd, line.data() + done, unsigned(line.size() - done));
    if (w <= 0) ok = false; else done += size_t(w);
  }
  ok = ok && _commit(fd) == 0;
  if (!ok && start >= 0) { (void)_chsize_s(fd, start); (void)_commit(fd); }
  _close(fd);
#else
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) { why = "cannot open the journal " + path; return false; }
  const off_t start = ::lseek(fd, 0, SEEK_END);
  bool ok = start >= 0;
  size_t done = 0;
  while (ok && done < line.size()) {
    const ssize_t w = ::write(fd, line.data() + done, line.size() - done);
    if (w <= 0) ok = false; else done += size_t(w);
  }
  ok = ok && ::fsync(fd) == 0;
  if (!ok && start >= 0) { (void)::ftruncate(fd, start); (void)::fsync(fd); }
  ::close(fd);
#endif
  if (!ok) why = "writing the journal " + path + " failed; it was cut back to where it was";
  return ok;
}

// ---- One save file and its journal. ----
struct SaveFile {
  std::string file, journal;
  JournalApply apply;
  std::function<bool(const std::vector<uint8_t>&)> valid;      // bytes a save file may hold (empty = none yet)
  std::function<bool(const std::vector<uint8_t>&)> journaled;  // bytes written by a journaled save
};

struct Loaded {
  std::vector<uint8_t> state;   // the save file's bytes, as checked (or as the journal says, when pending)
  bool has_journal = false;
  bool pending = false;         // the journal holds one record the save file does not have yet (an interrupted save)
  uint64_t records = 0;
  std::string head = zero_sha();
  std::vector<std::string> line_sha;
};

// Loads a save file and checks it against its journal. Refuses (false, with why) and changes nothing when:
// either file cannot be read; the journal is broken or ends in a torn line; the save file does not match the
// journal, except that it may be exactly one record behind (a save cut short between the two writes); or the save
// file was written by a journaled save but its journal is gone. Never clears anything.
inline bool load_checked(const SaveFile& sf, Loaded& out, std::string& why) {
  out = Loaded{};
  std::vector<uint8_t> raw, jb;
  bool raw_exists = false, j_exists = false;
  if (!read_file_checked(sf.file, raw, raw_exists)) { why = "cannot read " + sf.file; return false; }
  if (!read_file_checked(sf.journal, jb, j_exists)) { why = "cannot read " + sf.journal; return false; }
  JournalView v;
  if (!journal_check(jb, sf.apply, v, why)) { why = sf.journal + ": " + why; return false; }
  if (v.torn) {
    why = sf.journal + " ends in a torn line (a save that was cut short); pchie_neml --repair-tail cuts it off after checking the rest";
    return false;
  }
  if (v.records == 0) {
    if (raw_exists && !raw.empty() && sf.journaled(raw)) { why = sf.file + " was written with a journal, but " + sf.journal + " is missing or empty"; return false; }
    if (!sf.valid(raw)) { why = "unreadable ledger " + sf.file; return false; }
    out.state = raw;
    return true;
  }
  out.has_journal = true;
  out.records = v.records;
  out.head = v.head;
  out.line_sha = v.line_sha;
  const std::string rs = sha_hex(raw);
  if (rs == v.state_sha) { out.state = raw; return true; }
  if (v.records >= 2 && rs == v.prev_state_sha) { out.state = v.state; out.pending = true; return true; }
  why = sf.file + " does not match its journal (" + std::to_string(v.records) + " records, last " + v.head.substr(0, 16) +
        "): it was changed, replaced or removed outside PCHIE";
  return false;
}

// Adds one record: starts the journal if there is none (its first record holds the save file as loaded), appends
// the record, then rewrites the save file. If the save file write fails the journal is one record ahead, which
// the next load accepts and the next save completes. Call only under the save lock, with what load_checked gave.
inline bool save_checked(const SaveFile& sf, Loaded& ld, char kind, const std::vector<uint8_t>& payload,
                         std::vector<uint8_t>& new_state, std::string& why) {
  new_state = ld.state;
  if (!sf.apply(kind, payload, new_state, why)) return false;
  if (!ld.has_journal) {
    const std::string g = journal_line(0, 'G', zero_sha(), ld.state, sha_hex(ld.state));
    if (!journal_append(sf.journal, g, why)) return false;
    ld.has_journal = true;
    ld.records = 1;
    ld.head = sha_hex(g.substr(0, g.size() - 66));
    ld.line_sha.push_back(ld.head);
  }
  const std::string line = journal_line(ld.records, kind, ld.head, payload, sha_hex(new_state));
  if (!journal_append(sf.journal, line, why)) return false;
  ld.head = sha_hex(line.substr(0, line.size() - 66));
  ld.line_sha.push_back(ld.head);
  ++ld.records;
  ld.state = new_state;
  ld.pending = true;
  if (!write_file_atomic(sf.file, new_state)) { why = "the journal has the record, but rewriting " + sf.file + " failed (the next save completes it)"; return false; }
  ld.pending = false;
  return true;
}

// Cuts a torn last line off a journal, after checking every complete line before it. The whole journal is first
// copied to "<journal>.torn-<time>[-N].bak" (never overwriting a file). Complete records are never changed.
// removed is the number of bytes cut. Call only under the save lock.
inline bool journal_repair_tail(const SaveFile& sf, size_t& removed, std::string& backup, std::string& why) {
  removed = 0;
  backup.clear();
  std::vector<uint8_t> jb;
  bool exists = false;
  if (!read_file_checked(sf.journal, jb, exists) || !exists) { why = "cannot read " + sf.journal; return false; }
  JournalView v;
  if (!journal_check(jb, sf.apply, v, why)) { why = sf.journal + ": " + why + "; not repaired"; return false; }
  if (!v.torn) { why = sf.journal + " has no torn line; nothing to repair"; return false; }
  const std::time_t t = std::time(nullptr);
  char stamp[32];
  std::tm tmv{};
#ifdef _WIN32
  gmtime_s(&tmv, &t);
#else
  gmtime_r(&t, &tmv);
#endif
  std::strftime(stamp, sizeof stamp, "%Y%m%dT%H%M%SZ", &tmv);
  for (int k = 0; k < 100 && backup.empty(); ++k) {
    const std::string name = sf.journal + ".torn-" + stamp + (k ? "-" + std::to_string(k) : std::string()) + ".bak";
    std::FILE* f = std::fopen(name.c_str(), "wbx");  // C11 exclusive create: never overwrites
    if (!f) continue;
    const bool wrote = std::fwrite(jb.data(), 1, jb.size(), f) == jb.size();
    const bool ok = flush_and_close(f) && wrote;
    if (!ok) { std::error_code ec; std::filesystem::remove(name, ec); why = "cannot write the backup " + name; return false; }
    backup = name;
  }
  if (backup.empty()) { why = "cannot make a backup next to " + sf.journal; return false; }
  std::error_code ec;
  std::filesystem::resize_file(sf.journal, v.good_bytes, ec);
  if (ec) { why = "cutting " + sf.journal + " failed: " + ec.message(); return false; }
  removed = jb.size() - v.good_bytes;
  return true;
}

// ---- The save lock. ----
class SaveLock {
 public:
  // Waits up to tries x 25 ms for the lock on "<dir>/<name>", then gives up (held() is false).
  SaveLock(const std::string& dir, const char* name = "nemo_episodic.lock", int tries = 400) {
    const std::string p = (std::filesystem::path(dir) / name).string();
#ifdef _WIN32
    if (_sopen_s(&fd_, p.c_str(), _O_RDWR | _O_CREAT | _O_BINARY | _O_NOINHERIT, _SH_DENYNO, _S_IREAD | _S_IWRITE) != 0) fd_ = -1;
#else
    fd_ = ::open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
#endif
    if (fd_ < 0) return;
    for (int i = 0; i < tries && !held_; ++i) {
#ifdef _WIN32
      _lseek(fd_, 0, SEEK_SET);
      held_ = _locking(fd_, _LK_NBLCK, 1) == 0;
#else
      held_ = ::flock(fd_, LOCK_EX | LOCK_NB) == 0;
#endif
      if (!held_ && i + 1 < tries) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
  }
  ~SaveLock() {
    if (fd_ < 0) return;
#ifdef _WIN32
    if (held_) { _lseek(fd_, 0, SEEK_SET); (void)_locking(fd_, _LK_UNLCK, 1); }
    _close(fd_);
#else
    ::close(fd_);  // closing the file drops the flock
#endif
  }
  SaveLock(const SaveLock&) = delete;
  SaveLock& operator=(const SaveLock&) = delete;
  bool held() const { return held_; }

 private:
  int fd_ = -1;
  bool held_ = false;
};

}  // namespace neml
}  // namespace pchie
