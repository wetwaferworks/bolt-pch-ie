#include "pchie/harness/Transcript.h"

#include "pchie/research/Digest.h"

#include <cstdio>

namespace pchie {
namespace harness {

const char* event_kind_name(EventKind k) {
  switch (k) {
    case EventKind::Input: return "input";
    case EventKind::Observation: return "observation";
    case EventKind::Action: return "action";
    case EventKind::Model: return "model";
    case EventKind::Score: return "score";
    case EventKind::Note: return "note";
    case EventKind::Error: return "error";
  }
  return "note";
}

std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (const char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(c));
          o += buf;
        } else {
          o.push_back(ch);
        }
    }
  }
  return o;
}

void Transcript::add(EventKind kind, uint32_t step, std::string text) {
  TranscriptEvent e;
  e.kind = kind;
  e.step = step;
  e.text = std::move(text);
  events_.push_back(std::move(e));
}

std::string Transcript::canonical() const {
  std::string s;
  for (const auto& e : events_) {
    s += event_kind_name(e.kind);
    s += '|';
    s += std::to_string(e.step);
    s += '|';
    s += json_escape(e.text);
    s += '\n';
  }
  return s;
}

std::string Transcript::sha256() const { return research::sha256_hex(canonical()); }

}  // namespace harness
}  // namespace pchie
