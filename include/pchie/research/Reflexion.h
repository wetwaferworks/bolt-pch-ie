// Reflexion (Shinn et al., 2023): after a failed or messy run, Little Nemo writes short lessons from
// a fixed template ("Moving east from (3, 4) is blocked; I bumped it 3 times.").
// Lessons come only from his own trace. They are stored as receipts in a lesson ledger with its own
// SHA-256 side chain, separate from the episodic ring and from every official chain.
// Putting lessons into his prompt is not wired: kLessonInjectionAllowed stays false until JD's go.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace research {

struct StepRecord {
  int32_t x = 0;            // odometry before the action
  int32_t y = 0;
  int action = 0;           // kNorth..kWest, kPickup, kUse
  bool succeeded = false;
  bool fell = false;        // he ended up in a pit
};

struct RunTrace {
  std::string run_id;
  std::vector<StepRecord> steps;
  bool reached_exit = false;
  uint32_t step_budget = 0;
};

enum class LessonKind : uint8_t { Pit = 0, Bump = 1, Loop = 2, DoorStall = 3, Timeout = 4 };

struct Lesson {
  LessonKind kind = LessonKind::Bump;
  int32_t x = 0;
  int32_t y = 0;
  int move = -1;
  uint32_t count = 0;
  uint32_t first_step = 0;
  std::string text;
};

class Reflexion {
 public:
  /// Deterministic: sorted by kind, then first step, then place.
  static std::vector<Lesson> reflect(const RunTrace& trace);
  static const char* kind_name(LessonKind k);
  /// Prompt text for the lessons, capped at max_chars. Not used anywhere while injection is off.
  static std::string prompt_block(const std::vector<Lesson>& lessons, size_t max_chars);
};

class LessonLedger {
 public:
  explicit LessonLedger(std::string path) : path_(std::move(path)) {}
  /// Appends one JSON line per lesson; each line carries prev and its own SHA-256 (side chain).
  bool append(const std::string& run_id, const std::vector<Lesson>& lessons, std::string& err);
  /// Re-walks the chain. head receives the last SHA-256 (64 zeros when empty).
  bool verify(std::string& head, size_t& lines, std::string& err) const;
  static std::string genesis() { return std::string(64, '0'); }

 private:
  std::string path_;
};

}  // namespace research
}  // namespace pchie
