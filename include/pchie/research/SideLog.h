// Append-only JSONL log with its own SHA-256 chain: every line carries the previous line's hash and its own.
// Used by the research layers for their side chains; never the official chain.
#pragma once

#include <string>

namespace pchie {
namespace research {

class SideLog {
 public:
  SideLog() = default;
  explicit SideLog(std::string path) : path_(std::move(path)) {}
  bool enabled() const { return !path_.empty(); }
  /// body is a JSON object without its closing brace; prev and sha256 are added here.
  bool append(const std::string& body, std::string& err);
  const std::string& head() const { return head_; }
  size_t lines() const { return lines_; }
  /// Re-walks a log file; head receives the last hash (64 zeros when empty).
  static bool verify(const std::string& path, std::string& head, size_t& lines, std::string& err);

 private:
  std::string path_;
  std::string head_ = std::string(64, '0');
  size_t lines_ = 0;
};

}  // namespace research
}  // namespace pchie
