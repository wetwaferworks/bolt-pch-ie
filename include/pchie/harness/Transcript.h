// Harness transcript: every event of one sample, in order (what he was asked, what he saw, what he did,
// what the model said, how it was scored). The text is canonical, so its SHA-256 is the same on every
// machine and at every thread count. The eval runner writes each event onto a research side chain.
// Scratch research only: never the official chain.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace harness {

enum class EventKind : uint8_t { Input = 0, Observation = 1, Action = 2, Model = 3, Score = 4, Note = 5, Error = 6 };

const char* event_kind_name(EventKind k);

struct TranscriptEvent {
  EventKind kind = EventKind::Note;
  uint32_t step = 0;
  std::string text;
};

class Transcript {
 public:
  void add(EventKind kind, uint32_t step, std::string text);
  const std::vector<TranscriptEvent>& events() const { return events_; }
  size_t size() const { return events_.size(); }
  void clear() { events_.clear(); }

  /// One line per event: kind|step|escaped text.
  std::string canonical() const;
  std::string sha256() const;

 private:
  std::vector<TranscriptEvent> events_;
};

/// JSON string escaping (quotes, backslash, control bytes as \u00XX). Bytes >= 0x80 pass through.
std::string json_escape(const std::string& s);

}  // namespace harness
}  // namespace pchie
