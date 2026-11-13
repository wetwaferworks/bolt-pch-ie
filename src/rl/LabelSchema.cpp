#include "pchie/rl/LabelSchema.h"

#include <cctype>
#include <fstream>
#include <sstream>

namespace pchie {
namespace rl {

static std::string jq(const std::string& s) { return std::string("\"") + json_escape(s) + "\""; }

static std::string jarr(const std::vector<std::string>& v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) o += ",";
    o += jq(v[i]);
  }
  o += "]";
  return o;
}

static char heading_char(Heading h) {
  switch (h) {
    case Heading::E: return 'E';
    case Heading::S: return 'S';
    case Heading::W: return 'W';
    default: return 'N';
  }
}

static bool heading_from_str(const std::string& s, Heading& h) {
  if (s == "N") { h = Heading::N; return true; }
  if (s == "E") { h = Heading::E; return true; }
  if (s == "S") { h = Heading::S; return true; }
  if (s == "W") { h = Heading::W; return true; }
  return false;
}

std::string label_canonical_json(const LabelExample& e) {
  std::ostringstream o;
  o << "{";
  o << "\"example_id\":" << jq(e.example_id) << ",";
  o << "\"status\":" << jq(e.status) << ",";
  o << "\"schema_version\":" << jq(e.schema_version) << ",";
  o << "\"world\":" << jq(e.world) << ",";
  o << "\"observation\":{";
  o << "\"radius\":" << e.observation.radius << ",";
  o << "\"cells\":" << jarr(e.observation.cells) << ",";
  o << "\"pose\":{";
  o << "\"x\":" << e.observation.pose.x << ",";
  o << "\"y\":" << e.observation.pose.y << ",";
  o << "\"heading\":" << jq(std::string(1, heading_char(e.observation.pose.heading)));
  o << "}}";
  o << ",";
  o << "\"subgoal\":" << jq(e.subgoal) << ",";
  o << "\"history_tail\":" << jarr(e.history_tail) << ",";
  o << "\"proposed_plan\":" << jarr(e.proposed_plan) << ",";
  o << "\"proposal_source\":" << jq(e.proposal_source) << ",";
  o << "\"coherence_label\":" << jq(e.coherence_label) << ",";
  o << "\"reason_codes\":" << jarr(e.reason_codes) << ",";
  if (e.preferred_method_is_null) o << "\"preferred_method\":null,";
  else o << "\"preferred_method\":" << jq(e.preferred_method) << ",";
  o << "\"labeler\":" << jq(e.labeler) << ",";
  if (e.reviewed_by_is_null) o << "\"reviewed_by\":null,";
  else o << "\"reviewed_by\":" << jq(e.reviewed_by) << ",";
  o << "\"notes\":" << jq(e.notes);
  o << "}";
  return o.str();
}

std::string label_file_json(const LabelExample& e) {
  LabelExample tmp = e;
  if (tmp.label_hash.empty()) tmp.label_hash = label_hash_hex(tmp);
  // Compact single-line file form, hash last.
  std::string c = label_canonical_json(tmp);
  // Insert label_hash before the final }.
  if (!c.empty() && c.back() == '}') {
    c.pop_back();
    c += ",\"label_hash\":" + jq(tmp.label_hash) + "}";
  }
  return c;
}

// Tiny extractor for our own compact JSON (strings, ints, null, string arrays, nested objects).
static bool extract_string(const std::string& t, const std::string& key, std::string& out, bool* is_null) {
  const std::string pat = "\"" + key + "\":";
  auto p = t.find(pat);
  if (p == std::string::npos) return false;
  p += pat.size();
  if (is_null) *is_null = false;
  if (p + 4 <= t.size() && t.compare(p, 4, "null") == 0) {
    out.clear();
    if (is_null) *is_null = true;
    return true;
  }
  if (p >= t.size() || t[p] != '"') return false;
  ++p;
  std::string v;
  while (p < t.size()) {
    char c = t[p++];
    if (c == '\\') {
      if (p >= t.size()) return false;
      char e = t[p++];
      if (e == 'n') v += '\n';
      else if (e == 't') v += '\t';
      else if (e == 'r') v += '\r';
      else v += e;
    } else if (c == '"') {
      out = v;
      return true;
    } else {
      v += c;
    }
  }
  return false;
}

static bool extract_int(const std::string& t, const std::string& key, int32_t& out) {
  const std::string pat = "\"" + key + "\":";
  auto p = t.find(pat);
  if (p == std::string::npos) return false;
  p += pat.size();
  bool neg = false;
  if (p < t.size() && t[p] == '-') { neg = true; ++p; }
  if (p >= t.size() || !std::isdigit(static_cast<unsigned char>(t[p]))) return false;
  int32_t v = 0;
  while (p < t.size() && std::isdigit(static_cast<unsigned char>(t[p]))) {
    v = v * 10 + (t[p] - '0');
    ++p;
  }
  out = neg ? -v : v;
  return true;
}

static bool extract_string_array(const std::string& t, const std::string& key, std::vector<std::string>& out) {
  const std::string pat = "\"" + key + "\":";
  auto p = t.find(pat);
  if (p == std::string::npos) return false;
  p += pat.size();
  if (p >= t.size() || t[p] != '[') return false;
  ++p;
  out.clear();
  while (p < t.size() && t[p] != ']') {
    if (t[p] == ',') { ++p; continue; }
    if (t[p] == '"') {
      ++p;
      std::string v;
      while (p < t.size()) {
        char c = t[p++];
        if (c == '\\') {
          if (p >= t.size()) return false;
          v += t[p++];
        } else if (c == '"') {
          out.push_back(v);
          break;
        } else {
          v += c;
        }
      }
    } else {
      return false;
    }
  }
  return p < t.size() && t[p] == ']';
}

static std::string slice_object(const std::string& t, const std::string& key) {
  const std::string pat = "\"" + key + "\":";
  auto p = t.find(pat);
  if (p == std::string::npos) return {};
  p += pat.size();
  if (p >= t.size() || t[p] != '{') return {};
  int depth = 0;
  auto b = p;
  for (; p < t.size(); ++p) {
    if (t[p] == '{') ++depth;
    else if (t[p] == '}') {
      --depth;
      if (depth == 0) return t.substr(b, p - b + 1);
    }
  }
  return {};
}

bool parse_label_json(const std::string& text, LabelExample& out, std::string& err) {
  err.clear();
  out = LabelExample{};
  // Strip whitespace outside strings so pretty or compact both parse.
  std::string t;
  t.reserve(text.size());
  bool in_str = false, esc = false;
  for (char c : text) {
    if (in_str) {
      t += c;
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
    } else {
      if (c == '"') { in_str = true; t += c; }
      else if (!std::isspace(static_cast<unsigned char>(c))) t += c;
    }
  }
  if (!extract_string(t, "example_id", out.example_id, nullptr) ||
      !extract_string(t, "status", out.status, nullptr) ||
      !extract_string(t, "schema_version", out.schema_version, nullptr) ||
      !extract_string(t, "world", out.world, nullptr) ||
      !extract_string(t, "subgoal", out.subgoal, nullptr) ||
      !extract_string(t, "proposal_source", out.proposal_source, nullptr) ||
      !extract_string(t, "coherence_label", out.coherence_label, nullptr) ||
      !extract_string(t, "labeler", out.labeler, nullptr) ||
      !extract_string(t, "notes", out.notes, nullptr)) {
    err = "missing required string field";
    return false;
  }
  bool pref_null = false, rev_null = false;
  if (!extract_string(t, "preferred_method", out.preferred_method, &pref_null)) {
    err = "preferred_method";
    return false;
  }
  out.preferred_method_is_null = pref_null;
  if (!extract_string(t, "reviewed_by", out.reviewed_by, &rev_null)) {
    err = "reviewed_by";
    return false;
  }
  out.reviewed_by_is_null = rev_null;
  if (!extract_string_array(t, "history_tail", out.history_tail) ||
      !extract_string_array(t, "proposed_plan", out.proposed_plan) ||
      !extract_string_array(t, "reason_codes", out.reason_codes)) {
    err = "missing array field";
    return false;
  }
  const std::string obs = slice_object(t, "observation");
  if (obs.empty()) {
    err = "observation";
    return false;
  }
  if (!extract_int(obs, "radius", out.observation.radius) ||
      !extract_string_array(obs, "cells", out.observation.cells)) {
    err = "observation.cells/radius";
    return false;
  }
  const std::string pose = slice_object(obs, "pose");
  if (pose.empty() || !extract_int(pose, "x", out.observation.pose.x) ||
      !extract_int(pose, "y", out.observation.pose.y)) {
    err = "observation.pose";
    return false;
  }
  std::string hs;
  if (!extract_string(pose, "heading", hs, nullptr) || !heading_from_str(hs, out.observation.pose.heading)) {
    err = "observation.pose.heading";
    return false;
  }
  extract_string(t, "label_hash", out.label_hash, nullptr);  // optional at parse
  return true;
}

static LabelLoadResult load_path(const std::string& path) {
  LabelLoadResult r;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    r.err = "cannot open " + path;
    return r;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (!parse_label_json(ss.str(), r.example, r.err)) return r;
  if (r.example.schema_version != kLabelSchemaVersion) {
    r.err = "schema_version mismatch";
    return r;
  }
  const std::string expect = label_hash_hex(r.example);
  if (!r.example.label_hash.empty() && r.example.label_hash != expect) {
    r.err = "label_hash mismatch for " + r.example.example_id;
    return r;
  }
  r.example.label_hash = expect;
  r.ok = true;
  return r;
}

LabelLoadResult load_label_file(const std::string& path) { return load_path(path); }

LabelLoadResult load_label_for_run(const std::string& path) {
  auto r = load_path(path);
  if (!r.ok) return r;
  if (is_stub_status(r.example.status) || r.example.status == kLabeledUnreviewed ||
      r.example.status == kRejected) {
    r.ok = false;
    r.err = "label " + r.example.example_id + " status=" + r.example.status +
            " is not for any run (stubs await review)";
  }
  return r;
}

}  // namespace rl
}  // namespace pchie
