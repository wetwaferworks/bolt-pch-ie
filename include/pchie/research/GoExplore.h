// L7 Go-Explore for Little Nemo (Ecoffet et al., 2021), on his own experience only.
// The archive remembers every cell he has reached (odometry place + key state + view bucket,
// optionally heading). When a short exploration burst ends, he picks a promising cell from the
// archive, walks back to it over edges he has already walked (return), and explores from there.
// Selection is integer-only with a fixed tie order, so a run is identical on every machine.
#pragma once

#include "pchie/research/ExperienceMap.h"
#include "pchie/research/SelfObservation.h"

#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>

namespace pchie {
namespace research {

struct GoExploreConfig {
  uint32_t explore_steps = 8;  // moves per exploration burst
  bool use_heading = false;    // include heading in the archive cell
};

struct ArchiveCell {
  int32_t x = 0;
  int32_t y = 0;
  bool have_key = false;
  uint32_t visits = 0;
  uint32_t chosen = 0;       // times it was picked as a return target
  uint32_t first_step = 0;   // step he first reached it
};

class GoExploreArchive {
 public:
  explicit GoExploreArchive(bool use_heading = false) : use_heading_(use_heading) {}

  void observe(const SelfObservation& o, uint32_t step);

  /// Pick the next return target. Cells with the same key state as now come first.
  /// Weight = 2^20/(1+chosen) + 2^19/(1+visits) + 2^20 if the place still has an untried edge.
  /// Ties: earlier first_step, then lower cell key.
  bool select(const ExperienceMap& map, bool have_key, uint64_t& out_key);

  const ArchiveCell* find(uint64_t key) const;
  void mark_chosen(uint64_t key);
  size_t size() const { return cells_.size(); }
  std::string canonical() const;

 private:
  bool use_heading_;
  std::map<uint64_t, ArchiveCell> cells_;
};

enum class GoExplorePhase : uint8_t { Explore = 0, Return = 1 };

class GoExplorePolicy {
 public:
  explicit GoExplorePolicy(GoExploreConfig cfg = {});

  /// Next action for this observation (kNorth..kWest, kPickup, kUse), or -1 at the exit.
  int decide(const SelfObservation& o);

  /// What happened after decide(): the action actually taken (it may differ from the one decide()
  /// returned when another layer chose) and whether it worked. A different action ends a return trip.
  void outcome(int action, bool succeeded);

  /// Feed what he sees around him (open moves) at the current observation, before decide().
  void sense(const SelfObservation& o, uint8_t open_mask) { map_.sense(o.x, o.y, open_mask); }

  const ExperienceMap& map() const { return map_; }
  const GoExploreArchive& archive() const { return archive_; }
  uint32_t steps() const { return step_; }
  uint32_t returns() const { return returns_; }
  uint32_t failed_returns() const { return failed_returns_; }
  GoExplorePhase phase() const { return phase_; }

  /// SHA-256 over the archive and the map.
  std::string digest() const;

 private:
  int choose(const SelfObservation& o);
  int explore_move(const SelfObservation& o);
  void plan_return(const SelfObservation& o);
  bool usable(const SelfObservation& o, int m) const;

  GoExploreConfig cfg_;
  ExperienceMap map_;
  GoExploreArchive archive_;
  GoExplorePhase phase_ = GoExplorePhase::Explore;
  std::deque<uint8_t> path_;
  uint32_t budget_ = 0;
  uint32_t step_ = 0;
  uint32_t returns_ = 0;
  uint32_t failed_returns_ = 0;
  SelfObservation last_{};
  int last_choice_ = -2;
  std::set<uint64_t> used_at_;
};

}  // namespace research
}  // namespace pchie
