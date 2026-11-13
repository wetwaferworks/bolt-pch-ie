// Scratch hash-move ledger for L1–L6. JSONL only, under scratch/rl/.
// Never written to the official IHashChain. Flags off => 0 moves.
#pragma once

#include "pchie/rl/RlCommon.h"

#include <fstream>
#include <stdexcept>
#include <string>

namespace pchie {
namespace rl {

struct HashMoveRecord {
  std::string run_id;
  std::string log;
  std::string token;
  std::string variable;
  std::string old_value;
  std::string new_value;
  std::string baseline;
  std::string change_id;
  std::string cause;
  std::string attribution;  // "single-flag ablation" | "bisect" | "inferred" | "unknown" | "interaction"
  bool official_token = false;
};

inline std::string hash_move_to_jsonl(const HashMoveRecord& r) {
  std::string s = "{";
  auto field = [&](const char* k, const std::string& v, bool last = false) {
    s += "\"";
    s += k;
    s += "\":\"";
    s += json_escape(v);
    s += last ? "\"" : "\",";
  };
  field("run_id", r.run_id);
  field("log", r.log);
  field("token", r.token);
  field("variable", r.variable);
  field("old_value", r.old_value);
  field("new_value", r.new_value);
  field("baseline", r.baseline);
  field("change_id", r.change_id);
  field("cause", r.cause);
  field("attribution", r.attribution);
  s += "\"official_token\":";
  s += r.official_token ? "true" : "false";
  s += "}";
  return s;
}

// Returns false and writes nothing if path is not under a scratch/rl directory
// (path must contain "/scratch/rl/" or "\\scratch\\rl\\"). Official chain paths
// are refused by construction.
inline bool append_hash_move(const std::string& path, const HashMoveRecord& r, std::string& err) {
  err.clear();
  const bool ok_path = path.find("/scratch/rl/") != std::string::npos ||
                       path.find("\\scratch\\rl\\") != std::string::npos;
  if (!ok_path) {
    err = "hash-move log must live under scratch/rl/ (official chain refused)";
    return false;
  }
  if (path.find("hash_moves.jsonl") == std::string::npos &&
      path.find("hash_moves") == std::string::npos) {
    // still allow other jsonl names under scratch/rl/, but prefer hash_moves.jsonl
  }
  std::ofstream out(path, std::ios::app | std::ios::binary);
  if (!out) {
    err = "cannot open hash-move log: " + path;
    return false;
  }
  const std::string line = hash_move_to_jsonl(r);
  out << line << '\n';
  if (!out) {
    err = "write failed";
    return false;
  }
  return true;
}

// Count existing records (0 if missing).
inline size_t hash_move_count(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return 0;
  size_t n = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) ++n;
  }
  return n;
}

}  // namespace rl
}  // namespace pchie
