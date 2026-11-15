#include "pchie/research/SideLog.h"

#include "pchie/research/Digest.h"

#include <fstream>

namespace pchie {
namespace research {

bool SideLog::append(const std::string& body, std::string& err) {
  const std::string full = body + ",\"prev\":\"" + head_ + "\"";
  const std::string sha = sha256_hex(full);
  if (enabled()) {
    std::ofstream f(path_, std::ios::binary | std::ios::app);
    if (!f) {
      err = "cannot open the research log";
      return false;
    }
    f << full << ",\"sha256\":\"" << sha << "\"}\n";
    if (!f) {
      err = "cannot write the research log";
      return false;
    }
  }
  head_ = sha;
  ++lines_;
  return true;
}

bool SideLog::verify(const std::string& path, std::string& head, size_t& lines, std::string& err) {
  head = std::string(64, '0');
  lines = 0;
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "no research log";
    return false;
  }
  const std::string tag = ",\"sha256\":\"";
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const size_t p = line.rfind(tag);
    if (p == std::string::npos || line.size() < p + tag.size() + 66) {
      err = "malformed research log line";
      return false;
    }
    const std::string full = line.substr(0, p);
    const std::string want = ",\"prev\":\"" + head + "\"";
    if (full.size() < want.size() || full.compare(full.size() - want.size(), want.size(), want) != 0 ||
        sha256_hex(full) != line.substr(p + tag.size(), 64)) {
      err = "research log chain broken at line " + std::to_string(lines + 1);
      return false;
    }
    head = line.substr(p + tag.size(), 64);
    ++lines;
  }
  return true;
}

}  // namespace research
}  // namespace pchie
