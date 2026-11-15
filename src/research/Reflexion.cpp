#include "pchie/research/Reflexion.h"

#include "pchie/research/Digest.h"
#include "pchie/research/SelfObservation.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <tuple>

namespace pchie {
namespace research {

namespace {
const char* move_word(int m) {
  switch (m) {
    case kNorth: return "north";
    case kEast: return "east";
    case kSouth: return "south";
    case kWest: return "west";
    case kPickup: return "pick up";
    case kUse: return "use";
  }
  return "?";
}
std::string at(int32_t x, int32_t y) {
  std::ostringstream o;
  o << '(' << x << ", " << y << ')';
  return o.str();
}
}  // namespace

const char* Reflexion::kind_name(LessonKind k) {
  switch (k) {
    case LessonKind::Pit: return "pit";
    case LessonKind::Bump: return "bump";
    case LessonKind::Loop: return "loop";
    case LessonKind::DoorStall: return "door_stall";
    case LessonKind::Timeout: return "timeout";
  }
  return "?";
}

std::vector<Lesson> Reflexion::reflect(const RunTrace& t) {
  std::vector<Lesson> out;
  struct Count {
    uint32_t n = 0;
    uint32_t first = 0;
  };
  std::map<std::tuple<int32_t, int32_t, int>, Count> bumps, failed_use;
  std::map<std::pair<int32_t, int32_t>, Count> visits;
  std::map<std::pair<int32_t, int32_t>, bool> places;
  for (uint32_t i = 0; i < t.steps.size(); ++i) {
    const StepRecord& s = t.steps[i];
    places[{s.x, s.y}] = true;
    auto& v = visits[{s.x, s.y}];
    if (v.n++ == 0) v.first = i;
    if (s.fell) {
      Lesson l;
      l.kind = LessonKind::Pit;
      l.x = s.x;
      l.y = s.y;
      l.move = s.action;
      l.count = 1;
      l.first_step = i;
      l.text = std::string("Stepping ") + move_word(s.action) + " from " + at(s.x, s.y) + " dropped me into a pit.";
      out.push_back(l);
    }
    if (s.action < kMoves && !s.succeeded) {
      auto& c = bumps[{s.x, s.y, s.action}];
      if (c.n++ == 0) c.first = i;
    }
    if (s.action == kUse && !s.succeeded) {
      auto& c = failed_use[{s.x, s.y, s.action}];
      if (c.n++ == 0) c.first = i;
    }
  }
  for (const auto& kv : bumps) {
    if (kv.second.n < 2) continue;
    Lesson l;
    l.kind = LessonKind::Bump;
    std::tie(l.x, l.y, l.move) = kv.first;
    l.count = kv.second.n;
    l.first_step = kv.second.first;
    l.text = std::string("Moving ") + move_word(l.move) + " from " + at(l.x, l.y) + " is blocked; I bumped it " +
             std::to_string(l.count) + " times.";
    out.push_back(l);
  }
  for (const auto& kv : visits) {
    if (kv.second.n < 4) continue;
    Lesson l;
    l.kind = LessonKind::Loop;
    l.x = kv.first.first;
    l.y = kv.first.second;
    l.count = kv.second.n;
    l.first_step = kv.second.first;
    l.text = "I came back to " + at(l.x, l.y) + " " + std::to_string(l.count) +
             " times; next time I should leave it by a way I have not tried.";
    out.push_back(l);
  }
  for (const auto& kv : failed_use) {
    if (kv.second.n < 2) continue;
    Lesson l;
    l.kind = LessonKind::DoorStall;
    std::tie(l.x, l.y, l.move) = kv.first;
    l.count = kv.second.n;
    l.first_step = kv.second.first;
    l.text = "Using the door at " + at(l.x, l.y) + " failed " + std::to_string(l.count) +
             " times; I need the key before I try again.";
    out.push_back(l);
  }
  if (!t.reached_exit && t.step_budget > 0 && t.steps.size() >= t.step_budget) {
    Lesson l;
    l.kind = LessonKind::Timeout;
    l.count = uint32_t(t.steps.size());
    l.first_step = uint32_t(t.steps.size());
    l.text = "I ran out of steps after " + std::to_string(t.steps.size()) + " moves and " +
             std::to_string(places.size()) + " places without finding the exit.";
    out.push_back(l);
  }
  std::stable_sort(out.begin(), out.end(), [](const Lesson& a, const Lesson& b) {
    return std::make_tuple(int(a.kind), a.first_step, a.x, a.y, a.move) <
           std::make_tuple(int(b.kind), b.first_step, b.x, b.y, b.move);
  });
  return out;
}

std::string Reflexion::prompt_block(const std::vector<Lesson>& lessons, size_t max_chars) {
  std::string s = "Lessons from my earlier runs:\n";
  for (const Lesson& l : lessons) {
    const std::string line = "- " + l.text + "\n";
    if (s.size() + line.size() > max_chars) break;
    s += line;
  }
  return s;
}

namespace {
std::string json_escape(const std::string& in) {
  std::string o;
  for (char c : in) {
    if (c == '"' || c == '\\') {
      o.push_back('\\');
      o.push_back(c);
    } else if (c == '\n') {
      o += "\\n";
    } else {
      o.push_back(c);
    }
  }
  return o;
}
std::string body_of(const std::string& run_id, const Lesson& l, const std::string& prev) {
  std::ostringstream o;
  o << "{\"format\":\"pchie.lesson.v1\",\"run\":\"" << json_escape(run_id) << "\",\"kind\":\""
    << Reflexion::kind_name(l.kind) << "\",\"x\":" << l.x << ",\"y\":" << l.y << ",\"move\":" << l.move
    << ",\"count\":" << l.count << ",\"first_step\":" << l.first_step << ",\"text\":\"" << json_escape(l.text)
    << "\",\"prev\":\"" << prev << "\"";
  return o.str();
}
}  // namespace

bool LessonLedger::append(const std::string& run_id, const std::vector<Lesson>& lessons, std::string& err) {
  std::string head;
  size_t lines = 0;
  if (!verify(head, lines, err)) return false;
  std::ofstream f(path_, std::ios::binary | std::ios::app);
  if (!f) {
    err = "cannot open the lesson ledger";
    return false;
  }
  for (const Lesson& l : lessons) {
    const std::string body = body_of(run_id, l, head);
    head = sha256_hex(body);
    f << body << ",\"sha256\":\"" << head << "\"}\n";
  }
  return bool(f);
}

bool LessonLedger::verify(std::string& head, size_t& lines, std::string& err) const {
  head = genesis();
  lines = 0;
  std::ifstream f(path_, std::ios::binary);
  if (!f) return true;  // no ledger yet
  std::string line;
  const std::string tag = ",\"sha256\":\"";
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const size_t p = line.rfind(tag);
    if (p == std::string::npos || line.size() < p + tag.size() + 66) {
      err = "malformed lesson line";
      return false;
    }
    const std::string body = line.substr(0, p);
    const std::string sha = line.substr(p + tag.size(), 64);
    if (body.find("\"prev\":\"" + head + "\"") == std::string::npos || sha256_hex(body) != sha) {
      err = "lesson chain broken at line " + std::to_string(lines + 1);
      return false;
    }
    head = sha;
    ++lines;
  }
  return true;
}

}  // namespace research
}  // namespace pchie
