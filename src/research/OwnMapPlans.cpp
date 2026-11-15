#include "pchie/research/OwnMapPlans.h"

#include "pchie/research/StepScorer.h"

#include <sstream>

namespace pchie {
namespace research {

namespace {

const int32_t kQ = 1 << 20;

int32_t key_x(uint64_t k) { return int32_t((k >> 16) & 0xFFFF) - 32768; }
int32_t key_y(uint64_t k) { return int32_t(k & 0xFFFF) - 32768; }

bool next_to_any(const std::set<uint64_t>& cells, int32_t x, int32_t y) {
  for (int m = 0; m < kMoves; ++m)
    if (cells.count(place_key(x + move_dx(m), y + move_dy(m)))) return true;
  return false;
}

}  // namespace

void OwnSight::see(int32_t x, int32_t y, uint8_t danger_mask, uint8_t key_mask, uint8_t door_mask, uint8_t exit_mask,
                   bool key_here, bool at_exit, bool have_key, bool door_opened) {
  for (int m = 0; m < kMoves; ++m) {
    const uint64_t n = place_key(x + move_dx(m), y + move_dy(m));
    if ((danger_mask >> m) & 1) pits.insert(n);
    if ((key_mask >> m) & 1) keys.insert(n);
    if ((door_mask >> m) & 1) doors.insert(n);
    if ((exit_mask >> m) & 1) exits.insert(n);
  }
  if (key_here) keys.insert(place_key(x, y));
  if (at_exit) exits.insert(place_key(x, y));
  if (have_key) keys.clear();
  if (door_opened) doors.clear();
}

std::string OwnSight::canonical() const {
  std::ostringstream o;
  auto put = [&](const char* tag, const std::set<uint64_t>& s) {
    o << tag;
    for (uint64_t k : s) o << ' ' << key_x(k) << ',' << key_y(k);
    o << ';';
  };
  put("pits", pits);
  put("keys", keys);
  put("doors", doors);
  put("exits", exits);
  return o.str();
}

std::string OwnGraph::canonical() const {
  std::ostringstream o;
  o << "own_graph.v1|" << goal_kind << '|' << g.nodes << '|' << g.start << '|' << g.goal << '|' << g.doors << '|'
    << int(key_held);
  for (size_t i = 0; i < coords.size(); ++i) o << "|n" << i << '=' << coords[i].first << ',' << coords[i].second;
  for (const GatedEdge& e : g.edges) o << "|e" << e.a << '-' << e.b << ':' << e.door;
  for (const KeySpot& k : g.keys) o << "|k" << k.node << ':' << k.door;
  return o.str();
}

OwnGraph build_own_graph(const ExperienceMap& map, const OwnSight& sight, int32_t x, int32_t y, bool have_key,
                         bool door_opened) {
  OwnGraph og;
  std::map<uint64_t, int> id;  // place -> node, in key order (fixed)
  for (const auto& kv : map.all())
    if (!sight.pits.count(kv.first)) id[kv.first] = 0;
  const bool gated = !door_opened && !sight.doors.empty();
  if (gated)
    for (uint64_t d : sight.doors) id[d] = 0;
  for (uint64_t k : sight.keys) id[k] = 0;
  for (uint64_t e : sight.exits) id[e] = 0;
  const uint64_t here = place_key(x, y);
  id[here] = 0;
  int n = 0;
  for (auto& kv : id) {
    kv.second = n++;
    og.coords.push_back({key_x(kv.first), key_y(kv.first)});
  }
  og.g.nodes = n;
  og.g.start = id[here];
  // Open ways he saw or walked (never into a pit he saw).
  std::set<std::pair<int, int>> seen_edges;
  auto add_edge = [&](int a, int b, int door) {
    const std::pair<int, int> k = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
    if (seen_edges.insert(k).second) og.g.edges.push_back(GatedEdge{k.first, k.second, door});
  };
  for (const auto& kv : map.all()) {
    const Place& p = kv.second;
    if (sight.pits.count(kv.first)) continue;
    for (int m = 0; m < kMoves; ++m) {
      if (p.edge[m] != EdgeState::Open) continue;
      const uint64_t q = place_key(p.x + move_dx(m), p.y + move_dy(m));
      if (sight.pits.count(q) || !id.count(q)) continue;
      add_edge(id[kv.first], id[q], -1);
    }
  }
  // The shut doors he saw: passable once opened, from any place he knows next to them.
  if (gated) {
    og.g.doors = 1;
    for (uint64_t d : sight.doors)
      for (int m = 0; m < kMoves; ++m) {
        const uint64_t q = place_key(key_x(d) + move_dx(m), key_y(d) + move_dy(m));
        if (id.count(q) && !sight.doors.count(q) && map.find(key_x(q), key_y(q))) add_edge(id[q], id[d], 0);
      }
    if (have_key) {
      og.g.keys.push_back(KeySpot{og.g.start, 0});
      og.key_held = true;
    } else {
      for (uint64_t k : sight.keys) og.g.keys.push_back(KeySpot{id[k], 0});
    }
  }
  if (!sight.exits.empty()) {
    og.g.goal = id[*sight.exits.begin()];
    og.goal_kind = "exit";
    og.asked = true;
  } else if (gated) {
    og.g.goal = id[*sight.doors.begin()];
    og.goal_kind = "door";
    og.asked = true;
  } else {
    og.goal_kind = "-";
  }
  return og;
}

int first_plan_act(const OwnGraph& og, const GatedPlan& plan, int32_t x, int32_t y) {
  for (const PlanAction& a : plan.actions) {
    if (a.kind == PlanStep::Pickup) {
      if (og.key_held) continue;  // his own key, modelled as lying at the start
      return kPickup;
    }
    if (a.kind == PlanStep::Open) return kUse;
    if (a.value < 0 || a.value >= int(og.coords.size())) return -1;
    const int32_t dx = og.coords[size_t(a.value)].first - x, dy = og.coords[size_t(a.value)].second - y;
    for (int m = 0; m < kMoves; ++m)
      if (move_dx(m) == dx && move_dy(m) == dy) return m;
    return -1;
  }
  return -1;
}

TotResult own_map_tree(const ExperienceMap& map, const OwnSight& sight, const SelfObservation& o, bool door_opened,
                       const TotConfig& cfg, IJobDispatcher* jobs) {
  static const char* kWord[kMoves] = {"N", "E", "S", "W"};
  auto fresh = [&](int32_t x, int32_t y) {
    const Place* p = map.find(x, y);
    return !p || p->visits == 0;
  };
  auto end_of = [&](const Thought& t, int32_t& x, int32_t& y) {
    x = o.x;
    y = o.y;
    for (uint8_t m : t.moves) {
      x += move_dx(m);
      y += move_dy(m);
    }
  };
  ProposeFn propose = [&](const Thought& parent, uint32_t breadth) {
    std::vector<Thought> out;
    int32_t x, y;
    end_of(parent, x, y);
    if (!parent.moves.empty() && fresh(x, y)) return out;  // he cannot see past new ground
    const int back = parent.moves.empty() ? -1 : reverse_move(parent.moves.back());
    for (int pass = 0; pass < 2 && out.empty(); ++pass)
      for (int m = 0; m < kMoves && out.size() < breadth; ++m) {
        if (map.edge(x, y, m) != EdgeState::Open) continue;
        if (sight.pits.count(place_key(x + move_dx(m), y + move_dy(m)))) continue;
        if (pass == 0 && m == back) continue;  // turning straight back only when nothing else is open
        Thought t = parent;
        t.moves.push_back(uint8_t(m));
        t.text = parent.text + kWord[m];
        out.push_back(t);
      }
    return out;
  };
  EvaluateFn evaluate = [&](const Thought& t) -> int32_t {
    int32_t x = o.x, y = o.y;
    int64_t s = 0;
    std::set<uint64_t> been{place_key(x, y)};
    for (uint8_t m : t.moves) {
      x += move_dx(m);
      y += move_dy(m);
      const uint64_t k = place_key(x, y);
      if (sight.pits.count(k)) return -2 * kQ;
      if (!been.insert(k).second) s -= kQ / 8;
      if (fresh(x, y)) {
        s += kQ / 2;
      } else {
        const uint32_t v = map.find(x, y)->visits;
        s -= int64_t(v > 8 ? 8 : v) * (kQ / 32);
      }
      if (sight.exits.count(k)) s += 2 * kQ;
      if (!o.have_key && sight.keys.count(k)) s += kQ;
      if (o.have_key && !door_opened && next_to_any(sight.doors, x, y)) s += kQ;
    }
    s -= int64_t(t.moves.size()) * (kQ / 64);
    return int32_t(s);
  };
  Thought root;
  root.text = "";
  return TreeOfThoughts(cfg).run(root, propose, evaluate, jobs);
}

}  // namespace research
}  // namespace pchie
