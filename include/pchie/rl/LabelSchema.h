// L6 label schema v0. Draft awaiting review.
// Stored as scratch/rl/labels/v0/*.json. Labels teach method preferences and
// calibrate the verifier. They are not a training corpus for the weights.
// Stubs (status=STUB_AWAITING_JD_REVIEW) are not ground truth and must not be
// used for any run.
#pragma once

#include "pchie/rl/RlCommon.h"
#include "pchie/rl/SymbolicPlanner.h"

#include <optional>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

constexpr const char* kLabelSchemaVersion = "l6-label-v0";
constexpr const char* kStubStatus = "STUB_AWAITING_JD_REVIEW";
constexpr const char* kLabeledUnreviewed = "LABELED_CORTANA_UNREVIEWED";
constexpr const char* kReviewedJd = "REVIEWED_JD";
constexpr const char* kRejected = "REJECTED";
constexpr const char* kFixtureBanner = "LABELED UNIT-TEST FIXTURE ONLY";

struct LabelObservation {
  int32_t radius = 1;
  std::vector<std::string> cells;
  Pose pose{};
};

struct LabelExample {
  std::string example_id;
  std::string status;
  std::string schema_version = kLabelSchemaVersion;
  std::string world;
  LabelObservation observation;
  std::string subgoal;
  std::vector<std::string> history_tail;
  std::vector<std::string> proposed_plan;
  std::string proposal_source;
  std::string coherence_label;
  std::vector<std::string> reason_codes;
  std::string preferred_method;  // empty => JSON null
  bool preferred_method_is_null = false;
  std::string labeler;
  std::string reviewed_by;  // empty + reviewed_by_is_null => JSON null
  bool reviewed_by_is_null = true;
  std::string notes;
  std::string label_hash;  // hex, not part of the hashed canonical form
};

// Canonical JSON (UTF-8, no whitespace, keys in schema order, label_hash omitted).
std::string label_canonical_json(const LabelExample& e);

inline std::array<uint8_t, 32> label_hash_bytes(const LabelExample& e) {
  const std::string c = label_canonical_json(e);
  return sha256_bytes(c);
}

inline std::string label_hash_hex(const LabelExample& e) { return to_hex(label_hash_bytes(e)); }

// Pretty-ish one-object JSON including label_hash as the last field.
std::string label_file_json(const LabelExample& e);

bool parse_label_json(const std::string& text, LabelExample& out, std::string& err);

struct LabelLoadResult {
  bool ok = false;
  std::string err;
  LabelExample example;
};

// Schema-test load: allowed for stubs. Verifies label_hash if present.
LabelLoadResult load_label_file(const std::string& path);

// Run load: stubs and unreviewed labels are refused. Nothing in v0 is for runs
// until the maintainer replaces the stubs.
LabelLoadResult load_label_for_run(const std::string& path);

inline bool is_stub_status(const std::string& s) { return s == kStubStatus; }

}  // namespace rl
}  // namespace pchie
