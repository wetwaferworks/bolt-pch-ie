#include "pchie/research/GoExplore.h"

#include "pchie/research/Digest.h"

#include <sstream>

namespace pchie {
namespace research {

void GoExploreArchive::observe(const SelfObservation& o, uint32_t step) {
  const uint64_t k = cell_key(o, use_heading_);
  auto it = cells_.find(k);
  if (it == cells_.end()) {
    ArchiveCell c;
    c.x = o.x;
    c.y = o.y;
    c.have_key = o.have_key;
    c.first_step = step;
    it = cells_.emplace(k, c).first;
  }
  ++it->second.visits;
}

bool GoExploreArchive::select(const ExperienceMap& map, bool have_key, uint64_t& out_key) {
  for (int pass = 0; pass < 2; ++pass) {
    bool found = false;
    uint64_t best_w = 0, best_k = 0;
    uint32_t best_first = 0;
    for (const auto& kv : cells_) {
      const ArchiveCell& c = kv.second;
      if (pass == 0 && c.have_key != have_key) continue;
      uint64_t w = (uint64_t(1) << 20) / (1 + c.chosen) + (uint64_t(1) << 19) / (1 + c.visits);
      if (const Place* p = map.find(c.x, c.y)) {
        for (int m = 0; m < kMoves; ++m)
          if (p->edge[m] == EdgeState::Unknown) {
            w += uint64_t(1) << 20;
            break;
          }
      }
      const bool better = !found || w > best_w || (w == best_w && (c.first_step < best_first ||
                                                                   (c.first_step == best_first && kv.first < best_k)));
      if (better) {
        found = true;
        best_w = w;
        best_k = kv.first;
        best_first = c.first_step;
      }
    }
    if (found) {
      out_key = best_k;
      return true;
    }
  }
  return false;
}

const ArchiveCell* GoExploreArchive::find(uint64_t key) const {
  auto it = cells_.find(key);
  return it == cells_.end() ? nullptr : &it->second;
}

void GoExploreArchive::mark_chosen(uint64_t key) {
  auto it = cells_.find(key);
  if (it != cells_.end()) ++it->second.chosen;
}

std::string GoExploreArchive::canonical() const {
  std::ostringstream o;
  o << "go_explore_archive.v1|" << int(use_heading_) << '|' << cells_.size();
  for (const auto& kv : cells_)
    o << '|' << kv.first << ',' << kv.second.visits << ',' << kv.second.chosen << ',' << kv.second.first_step;
  return o.str();
}

GoExplorePolicy::GoExplorePolicy(GoExploreConfig cfg) : cfg_(cfg), archive_(cfg.use_heading) {
  budget_ = cfg_.explore_steps;
}

bool GoExplorePolicy::usable(const SelfObservation& o, int m) const {
  if (o.danger_mask & (1u << m)) return false;
  return map_.edge(o.x, o.y, m) != EdgeState::Blocked;
}

int GoExplorePolicy::explore_move(const SelfObservation& o) {
  // 1) an untried edge: keep going the way he faces if he can, else compass order.
  if (map_.edge(o.x, o.y, o.heading) == EdgeState::Unknown && usable(o, o.heading)) return o.heading;
  for (int m = 0; m < kMoves; ++m)
    if (map_.edge(o.x, o.y, m) == EdgeState::Unknown && usable(o, m)) return m;
  // 2) the least-visited open neighbour; going straight back is the last choice.
  int best = -1;
  uint64_t best_cost = 0;
  for (int m = 0; m < kMoves; ++m) {
    if (map_.edge(o.x, o.y, m) != EdgeState::Open || !usable(o, m)) continue;
    const Place* q = map_.find(o.x + move_dx(m), o.y + move_dy(m));
    uint64_t cost = (q ? uint64_t(q->visits) : 0) * 4;
    if (step_ > 0 && m == reverse_move(o.heading)) cost += 2;
    if (best < 0 || cost < best_cost) {
      best = m;
      best_cost = cost;
    }
  }
  return best;
}

void GoExplorePolicy::plan_return(const SelfObservation& o) {
  path_.clear();
  uint64_t target = 0;
  bool ok = false;
  if (archive_.select(map_, o.have_key, target)) {
    archive_.mark_chosen(target);
    const ArchiveCell* c = archive_.find(target);
    const auto p = map_.path(o.x, o.y, c->x, c->y, ok);
    if (ok) path_.assign(p.begin(), p.end());
  }
  if (!ok) {
    // Fall back to the nearest place with an untried edge.
    int32_t tx = 0, ty = 0;
    if (map_.nearest_frontier(o.x, o.y, tx, ty)) {
      const auto p = map_.path(o.x, o.y, tx, ty, ok);
      if (ok) path_.assign(p.begin(), p.end());
    }
  }
  ++returns_;
  phase_ = path_.empty() ? GoExplorePhase::Explore : GoExplorePhase::Return;
  budget_ = cfg_.explore_steps;
}

int GoExplorePolicy::decide(const SelfObservation& o) {
  last_ = o;
  map_.stand(o);
  archive_.observe(o, step_);
  ++step_;
  if (o.at_exit) return -1;
  if (o.key_here && !o.have_key) return kPickup;
  if (o.door_near && o.have_key && !used_at_.count(place_key(o.x, o.y))) return kUse;

  for (int guard = 0; guard < 3; ++guard) {
    if (phase_ == GoExplorePhase::Return) {
      if (!path_.empty()) {
        const int m = path_.front();
        if (usable(o, m)) {
          path_.pop_front();
          return m;
        }
        ++failed_returns_;
      }
      phase_ = GoExplorePhase::Explore;
      budget_ = cfg_.explore_steps;
    }
    if (budget_ > 0) {
      const int m = explore_move(o);
      if (m >= 0) {
        --budget_;
        return m;
      }
    }
    plan_return(o);
  }
  // Nothing left to try from here.
  return explore_move(o);
}

void GoExplorePolicy::outcome(int action, bool succeeded) {
  if (action >= 0 && action < kMoves) {
    map_.tried(last_.x, last_.y, action, succeeded);
    if (!succeeded && phase_ == GoExplorePhase::Return) {
      ++failed_returns_;
      path_.clear();
      phase_ = GoExplorePhase::Explore;
    }
  } else if (action == kUse) {
    used_at_.insert(place_key(last_.x, last_.y));
    if (succeeded) map_.forget_door_blocks();
  }
}

std::string GoExplorePolicy::digest() const {
  return sha256_hex(archive_.canonical() + "\n" + map_.canonical());
}

}  // namespace research
}  // namespace pchie
