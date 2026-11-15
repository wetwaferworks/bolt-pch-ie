#include "pchie/research/ExperienceMap.h"

#include <deque>
#include <sstream>

namespace pchie {
namespace research {

Place& ExperienceMap::at(int32_t x, int32_t y) {
  Place& p = places_[place_key(x, y)];
  p.x = x;
  p.y = y;
  return p;
}

void ExperienceMap::stand(const SelfObservation& o) {
  Place& p = at(o.x, o.y);
  ++p.visits;
  p.saw_key = p.saw_key || o.key_here;
  p.saw_door = p.saw_door || o.door_near;
  p.saw_exit = p.saw_exit || o.at_exit;
}

void ExperienceMap::tried(int32_t x, int32_t y, int move, bool moved) {
  if (move < 0 || move >= kMoves) return;
  Place& p = at(x, y);
  ++p.tries[move];
  p.edge[move] = moved ? EdgeState::Open : EdgeState::Blocked;
  if (moved) {
    Place& q = at(x + move_dx(move), y + move_dy(move));
    q.edge[reverse_move(move)] = EdgeState::Open;
  }
}

void ExperienceMap::forget_blocks(int32_t x, int32_t y) {
  auto it = places_.find(place_key(x, y));
  if (it == places_.end()) return;
  for (int m = 0; m < kMoves; ++m)
    if (it->second.edge[m] == EdgeState::Blocked) it->second.edge[m] = EdgeState::Unknown;
}

void ExperienceMap::forget_door_blocks() {
  for (auto& kv : places_)
    if (kv.second.saw_door)
      for (int m = 0; m < kMoves; ++m)
        if (kv.second.edge[m] == EdgeState::Blocked) kv.second.edge[m] = EdgeState::Unknown;
}

const Place* ExperienceMap::find(int32_t x, int32_t y) const {
  auto it = places_.find(place_key(x, y));
  return it == places_.end() ? nullptr : &it->second;
}

EdgeState ExperienceMap::edge(int32_t x, int32_t y, int move) const {
  const Place* p = find(x, y);
  return (p && move >= 0 && move < kMoves) ? p->edge[move] : EdgeState::Unknown;
}

namespace {
struct Back {
  uint64_t from = 0;
  uint8_t move = 0;
};
}  // namespace

std::vector<uint8_t> ExperienceMap::path(int32_t fx, int32_t fy, int32_t tx, int32_t ty, bool& ok) const {
  ok = false;
  std::vector<uint8_t> out;
  const uint64_t start = place_key(fx, fy), goal = place_key(tx, ty);
  if (start == goal) {
    ok = true;
    return out;
  }
  if (!find(fx, fy) || !find(tx, ty)) return out;
  std::map<uint64_t, Back> seen;
  std::deque<uint64_t> q;
  seen[start] = Back{start, 0};
  q.push_back(start);
  while (!q.empty()) {
    const uint64_t k = q.front();
    q.pop_front();
    if (k == goal) break;
    const Place& p = places_.at(k);
    for (int m = 0; m < kMoves; ++m) {
      if (p.edge[m] != EdgeState::Open) continue;
      const uint64_t n = place_key(p.x + move_dx(m), p.y + move_dy(m));
      if (seen.count(n) || !places_.count(n)) continue;
      seen[n] = Back{k, uint8_t(m)};
      q.push_back(n);
    }
  }
  if (!seen.count(goal)) return out;
  for (uint64_t k = goal; k != start; k = seen[k].from) out.push_back(seen[k].move);
  std::vector<uint8_t> fwd(out.rbegin(), out.rend());
  ok = true;
  return fwd;
}

bool ExperienceMap::nearest_frontier(int32_t fx, int32_t fy, int32_t& tx, int32_t& ty) const {
  const uint64_t start = place_key(fx, fy);
  if (!places_.count(start)) return false;
  std::map<uint64_t, bool> seen;
  std::deque<uint64_t> q{start};
  seen[start] = true;
  while (!q.empty()) {
    const uint64_t k = q.front();
    q.pop_front();
    const Place& p = places_.at(k);
    for (int m = 0; m < kMoves; ++m)
      if (p.edge[m] == EdgeState::Unknown) {
        tx = p.x;
        ty = p.y;
        return true;
      }
    for (int m = 0; m < kMoves; ++m) {
      if (p.edge[m] != EdgeState::Open) continue;
      const uint64_t n = place_key(p.x + move_dx(m), p.y + move_dy(m));
      if (seen.count(n) || !places_.count(n)) continue;
      seen[n] = true;
      q.push_back(n);
    }
  }
  return false;
}

std::string ExperienceMap::canonical() const {
  std::ostringstream o;
  o << "experience_map.v1|" << places_.size();
  for (const auto& kv : places_) {
    const Place& p = kv.second;
    o << '|' << p.x << ',' << p.y << ',' << p.visits << ',';
    for (int m = 0; m < kMoves; ++m) o << int(p.edge[m]);
    o << ',' << int(p.saw_key) << int(p.saw_door) << int(p.saw_exit);
  }
  return o.str();
}

}  // namespace research
}  // namespace pchie
