// L2. Query-before-plan memory (UML Q1–Q2).
// A hit is a proposal, not a bypass: it still goes through PchGate (and L6
// verifier once that exists). Writes happen between runs only.
#pragma once

#include "pchie/rl/RlCommon.h"
#include "pchie/rl/RlFlags.h"

#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

enum class TruthState : uint8_t { Unverified = 0, SimVerified = 1, HumanVerified = 2, Refuted = 3 };

struct SituationKey {
  uint32_t task_id = 0;
  std::vector<int32_t> obs_tokens;
  std::array<uint8_t, 32> belief_digest{};  // zeros when L4 is off
  std::array<uint8_t, 32> weights_hash{};
  uint32_t domain_version = 0;
};

inline std::array<uint8_t, 32> situation_hash(const SituationKey& k) {
  std::vector<uint8_t> b;
  b.reserve(64 + k.obs_tokens.size() * 4);
  append_u32_le(b, k.task_id);
  append_u32_le(b, uint32_t(k.obs_tokens.size()));
  for (int32_t t : k.obs_tokens) append_i32_le(b, t);
  b.insert(b.end(), k.belief_digest.begin(), k.belief_digest.end());
  b.insert(b.end(), k.weights_hash.begin(), k.weights_hash.end());
  append_u32_le(b, k.domain_version);
  return sha256_bytes(b.data(), b.size());
}

struct MemoryRow {
  std::array<uint8_t, 32> situation_hash{};
  std::vector<int32_t> plan_fragment;  // action / option ids, not free text
  std::array<uint8_t, 32> provenance_run{};
  std::array<uint8_t, 32> row_hash{};  // SHA256(prev_head || body), chained
  bool chain_valid = false;
  TruthState truth = TruthState::Unverified;
};

inline std::vector<uint8_t> memory_row_body(const MemoryRow& r) {
  std::vector<uint8_t> b;
  b.insert(b.end(), r.situation_hash.begin(), r.situation_hash.end());
  append_u32_le(b, uint32_t(r.plan_fragment.size()));
  for (int32_t t : r.plan_fragment) append_i32_le(b, t);
  b.insert(b.end(), r.provenance_run.begin(), r.provenance_run.end());
  b.push_back(uint8_t(r.truth));
  return b;
}

inline std::array<uint8_t, 32> memory_row_hash(const std::array<uint8_t, 32>& prev_head,
                                               const MemoryRow& r) {
  const auto body = memory_row_body(r);
  std::vector<uint8_t> cat;
  cat.reserve(32 + body.size());
  cat.insert(cat.end(), prev_head.begin(), prev_head.end());
  cat.insert(cat.end(), body.begin(), body.end());
  return sha256_bytes(cat.data(), cat.size());
}

class IPlanMemory {
 public:
  virtual ~IPlanMemory() = default;
  // Returns a row only when chain_valid and truth meets policy (Sim or Human).
  // Unverified and Refuted are never returned (Refuted is not ">= Human").
  virtual std::optional<MemoryRow> query_verified(const SituationKey& key) = 0;
  // Refused while a run is live. Returns false on refuse or I/O error.
  virtual bool append_between_runs(const MemoryRow& row) = 0;
};

class FilePlanMemory final : public IPlanMemory {
 public:
  FilePlanMemory(std::string path, MinTruth min_truth) : path_(std::move(path)), min_truth_(min_truth) {
    load();
  }

  void set_run_live(bool live) { run_live_ = live; }
  bool run_live() const { return run_live_; }
  bool load_ok() const { return load_ok_; }
  const std::string& load_err() const { return load_err_; }
  size_t size() const { return rows_.size(); }
  size_t miss_count() const { return miss_count_; }
  size_t hit_count() const { return hit_count_; }
  bool broken_link() const { return broken_link_; }
  std::array<uint8_t, 32> head() const { return head_; }

  std::optional<MemoryRow> query_verified(const SituationKey& key) override {
    const auto h = situation_hash(key);
    if (broken_link_) {
      ++miss_count_;
      return std::nullopt;
    }
    for (const MemoryRow& r : rows_) {
      if (r.situation_hash != h) continue;
      if (!r.chain_valid) {
        ++miss_count_;
        return std::nullopt;
      }
      if (!truth_ok(r.truth)) {
        ++miss_count_;
        return std::nullopt;
      }
      ++hit_count_;
      last_hit_ = r;
      return r;
    }
    ++miss_count_;
    return std::nullopt;
  }

  bool append_between_runs(const MemoryRow& row_in) override {
    if (run_live_) {
      last_append_refused_ = true;
      return false;
    }
    last_append_refused_ = false;
    if (broken_link_) return false;
    MemoryRow row = row_in;
    row.row_hash = memory_row_hash(head_, row);
    row.chain_valid = true;
    if (!append_file(row)) return false;
    rows_.push_back(row);
    head_ = row.row_hash;
    return true;
  }

  bool last_append_refused() const { return last_append_refused_; }
  const std::optional<MemoryRow>& last_hit() const { return last_hit_; }

 private:
  bool truth_ok(TruthState t) const {
    if (t == TruthState::Unverified || t == TruthState::Refuted) return false;
    if (min_truth_ == MinTruth::Human) return t == TruthState::HumanVerified;
    return t == TruthState::SimVerified || t == TruthState::HumanVerified;
  }

  static std::string hex32(const std::array<uint8_t, 32>& a) { return to_hex(a); }

  std::string row_to_jsonl(const MemoryRow& r) const {
    std::ostringstream o;
    o << "{\"situation_hash\":\"" << hex32(r.situation_hash) << "\",\"plan_fragment\":[";
    for (size_t i = 0; i < r.plan_fragment.size(); ++i) {
      if (i) o << ",";
      o << r.plan_fragment[i];
    }
    o << "],\"provenance_run\":\"" << hex32(r.provenance_run) << "\",\"row_hash\":\""
      << hex32(r.row_hash) << "\",\"chain_valid\":" << (r.chain_valid ? "true" : "false")
      << ",\"truth\":" << int(r.truth) << "}";
    return o.str();
  }

  // Minimal parser for our own JSONL (not a general JSON parser).
  bool parse_row(const std::string& line, MemoryRow& r) const {
    auto grab = [&](const char* key) -> std::string {
      const std::string k = std::string("\"") + key + "\":";
      auto p = line.find(k);
      if (p == std::string::npos) return {};
      p += k.size();
      if (p < line.size() && line[p] == '"') {
        auto e = line.find('"', p + 1);
        if (e == std::string::npos) return {};
        return line.substr(p + 1, e - (p + 1));
      }
      auto e = line.find_first_of(",}", p);
      if (e == std::string::npos) return {};
      return line.substr(p, e - p);
    };
    if (!from_hex(grab("situation_hash"), r.situation_hash)) return false;
    if (!from_hex(grab("provenance_run"), r.provenance_run)) return false;
    if (!from_hex(grab("row_hash"), r.row_hash)) return false;
    const std::string cv = grab("chain_valid");
    r.chain_valid = (cv == "true");
    r.truth = TruthState(std::atoi(grab("truth").c_str()));
    r.plan_fragment.clear();
    auto fp = line.find("\"plan_fragment\":[");
    if (fp == std::string::npos) return false;
    fp = line.find('[', fp);
    auto fe = line.find(']', fp);
    if (fe == std::string::npos) return false;
    const std::string inside = line.substr(fp + 1, fe - fp - 1);
    if (!inside.empty()) {
      std::stringstream ss(inside);
      std::string tok;
      while (std::getline(ss, tok, ',')) r.plan_fragment.push_back(int32_t(std::atoi(tok.c_str())));
    }
    return true;
  }

  void load() {
    load_ok_ = true;
    load_err_.clear();
    rows_.clear();
    head_.fill(0);
    broken_link_ = false;
    std::ifstream in(path_, std::ios::binary);
    if (!in) return;  // empty store is fine
    std::string line;
    std::array<uint8_t, 32> prev{};
    size_t lineno = 0;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      ++lineno;
      MemoryRow r;
      if (!parse_row(line, r)) {
        load_ok_ = false;
        load_err_ = "parse error at row " + std::to_string(lineno);
        broken_link_ = true;
        return;
      }
      const auto expect = memory_row_hash(prev, r);
      if (r.row_hash != expect) {
        r.chain_valid = false;
        broken_link_ = true;
        load_ok_ = false;
        load_err_ = "broken link at row " + std::to_string(lineno);
        rows_.push_back(r);
        return;
      }
      r.chain_valid = true;
      rows_.push_back(r);
      prev = r.row_hash;
      head_ = r.row_hash;
    }
  }

  bool append_file(const MemoryRow& r) {
    const bool ok_path = path_.find("/scratch/rl/") != std::string::npos ||
                         path_.find("\\scratch\\rl\\") != std::string::npos;
    if (!ok_path) return false;
    std::ofstream out(path_, std::ios::app | std::ios::binary);
    if (!out) return false;
    out << row_to_jsonl(r) << '\n';
    return bool(out);
  }

  std::string path_;
  MinTruth min_truth_;
  std::vector<MemoryRow> rows_;
  std::array<uint8_t, 32> head_{};
  bool run_live_ = false;
  bool load_ok_ = true;
  bool broken_link_ = false;
  bool last_append_refused_ = false;
  std::string load_err_;
  size_t miss_count_ = 0;
  size_t hit_count_ = 0;
  std::optional<MemoryRow> last_hit_;
};

}  // namespace rl
}  // namespace pchie
