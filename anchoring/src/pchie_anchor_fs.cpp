#include "pchie_anchor_fs.h"
#include <cstdint>
#include <stdexcept>
#include "pchie/OsRandom.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pchie::anchor {

static std::string unique_suffix() {
  uint8_t r[12];
  if (!pchie::os_random_bytes(r, sizeof r)) throw std::runtime_error("no OS random bytes for a temp file name");
  static const char* d = "0123456789abcdef";
  std::string s = ".tmp.";
#if defined(_WIN32)
  s += std::to_string(GetCurrentProcessId());
#else
  s += std::to_string(long(getpid()));
#endif
  s += '.';
  for (uint8_t b : r) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}


void durable_write_file(const std::filesystem::path& path, const std::string& data) {
  std::filesystem::path tmpp = path;
  tmpp += unique_suffix();
#if defined(_WIN32)
  const std::wstring wt = tmpp.wstring(), wp = path.wstring();
  HANDLE h = CreateFileW(wt.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot create " + tmpp.string());
  size_t off = 0;
  bool ok = true;
  while (ok && off < data.size()) {
    DWORD w = 0;
    const DWORD chunk = DWORD(data.size() - off > 0x40000000u ? 0x40000000u : data.size() - off);
    ok = WriteFile(h, data.data() + off, chunk, &w, nullptr) && w == chunk;
    off += w;
  }
  ok = ok && FlushFileBuffers(h);
  CloseHandle(h);
  if (!ok || !MoveFileExW(wt.c_str(), wp.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    DeleteFileW(wt.c_str());
    throw std::runtime_error("write fail " + path.string());
  }
#else
  const std::string tmp = tmpp.string();
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0) throw std::runtime_error("cannot create " + tmp);
  size_t off = 0;
  bool ok = true;
  while (ok && off < data.size()) {
    const ssize_t w = ::write(fd, data.data() + off, data.size() - off);
    if (w < 0 && errno == EINTR) continue;
    ok = w > 0;
    if (ok) off += size_t(w);
  }
  ok = ok && ::fsync(fd) == 0;
  ok = (::close(fd) == 0) && ok;
  if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
    ::unlink(tmp.c_str());
    throw std::runtime_error("write fail " + path.string());
  }
  // Make the rename itself durable.
  const std::filesystem::path parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
  const int dfd = ::open(parent.c_str(), O_RDONLY);
  if (dfd >= 0) { (void)::fsync(dfd); ::close(dfd); }
#endif
}

void durable_append_file(const std::filesystem::path& path, const std::string& data) {
#if defined(_WIN32)
  const std::wstring wp = path.wstring();
  HANDLE h = CreateFileW(wp.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path.string() + " for append");
  size_t off = 0;
  bool ok = true;
  while (ok && off < data.size()) {
    DWORD w = 0;
    const DWORD chunk = DWORD(data.size() - off > 0x40000000u ? 0x40000000u : data.size() - off);
    ok = WriteFile(h, data.data() + off, chunk, &w, nullptr) && w == chunk;
    off += w;
  }
  ok = ok && FlushFileBuffers(h);
  CloseHandle(h);
  if (!ok) throw std::runtime_error("append fail " + path.string());
#else
  const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0600);
  if (fd < 0) throw std::runtime_error("cannot open " + path.string() + " for append");
  size_t off = 0;
  bool ok = true;
  while (ok && off < data.size()) {
    const ssize_t w = ::write(fd, data.data() + off, data.size() - off);
    if (w < 0 && errno == EINTR) continue;
    ok = w > 0;
    if (ok) off += size_t(w);
  }
  ok = ok && ::fsync(fd) == 0;
  ok = (::close(fd) == 0) && ok;
  if (!ok) throw std::runtime_error("append fail " + path.string());
#endif
}

ExclusiveFileLock::ExclusiveFileLock(const std::filesystem::path& lock_path) {
#if defined(_WIN32)
  const std::wstring w = lock_path.wstring();
  HANDLE h = CreateFileW(w.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open lock " + lock_path.string());
  OVERLAPPED ov{};
  if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &ov)) {
    CloseHandle(h);
    throw std::runtime_error("cannot lock " + lock_path.string());
  }
  h_ = h;
#else
  fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd_ < 0) throw std::runtime_error("cannot open lock " + lock_path.string());
  int r;
  do { r = ::flock(fd_, LOCK_EX); } while (r != 0 && errno == EINTR);
  if (r != 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("cannot lock " + lock_path.string());
  }
#endif
}

ExclusiveFileLock::~ExclusiveFileLock() {
#if defined(_WIN32)
  if (h_) {
    OVERLAPPED ov{};
    UnlockFileEx(static_cast<HANDLE>(h_), 0, 1, 0, &ov);
    CloseHandle(static_cast<HANDLE>(h_));
  }
#else
  if (fd_ >= 0) {
    ::flock(fd_, LOCK_UN);
    ::close(fd_);
  }
#endif
}

void wipe_string(std::string& s) {
  volatile char* p = s.empty() ? nullptr : &s[0];
  for (size_t i = 0; i < s.size(); ++i) p[i] = 0;
  s.clear();
}

}  // namespace pchie::anchor
