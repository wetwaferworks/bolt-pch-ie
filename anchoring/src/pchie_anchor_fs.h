// pchie_anchor: crash-safe file writes and an inter-process lock for key and witness state.
#pragma once
#include <filesystem>
#include <string>

namespace pchie::anchor {

// Writes data to a unique temp file next to path, flushes it to disk, then atomically replaces path.
// A crash leaves either the old file or the new one, never a torn file, and concurrent writers never share a temp name.
void durable_write_file(const std::filesystem::path& path, const std::string& data);

// Exclusive lock on lock_path, held until destruction. Blocks until the lock is free. The OS drops it if the
// process dies, so a crash never leaves a stale lock behind.
class ExclusiveFileLock {
 public:
  explicit ExclusiveFileLock(const std::filesystem::path& lock_path);
  ~ExclusiveFileLock();
  ExclusiveFileLock(const ExclusiveFileLock&) = delete;
  ExclusiveFileLock& operator=(const ExclusiveFileLock&) = delete;

 private:
#if defined(_WIN32)
  void* h_ = nullptr;
#else
  int fd_ = -1;
#endif
};

// Best-effort wipe of a secret held in a string.
void wipe_string(std::string& s);

}  // namespace pchie::anchor
