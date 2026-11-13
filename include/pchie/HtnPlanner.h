// Bolt PCHIE - HtnPlanner: locked goal -> ordered primitive tasks.
// Concrete final. Coordinator-thread only: no mutex, no atomic.
// World text is untrusted (B6 test-scenario) and is never used to set or change
// the goal or the task order. Optional IHashChain append is coordinator-only,
// ChainRecordKind::Custom, opaque payload (an open review question).
#pragma once

#include "pchie/ChainRecord.h"
#include "pchie/HtnDecomposition.h"
#include "pchie/HtnDomain.h"
#include "pchie/HtnState.h"
#include "pchie/HtnTask.h"
#include "pchie/IHashChain.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class HtnPlanner final {
 public:
  void set_goal(uint32_t goal_id) {
    goal_id_ = goal_id;
    goal_set_ = true;
  }
  uint32_t goal() const noexcept { return goal_id_; }
  bool goal_is_set() const noexcept { return goal_set_; }

  // world_text is accepted so callers can pass untrusted observation; it is ignored.
  std::vector<HtnTask> plan(const HtnDomain& domain, const std::string& world_text,
                            IHashChain* chain, std::string& err) const {
    (void)world_text;
    err.clear();
    if (!goal_set_) {
      err = "goal not set";
      return {};
    }
    std::vector<HtnTask> tasks = domain.decompose(goal_id_, err);
    if (tasks.empty()) {
      if (err.empty()) err = "empty plan";
      return {};
    }
    append_plan(tasks, chain);
    return tasks;
  }

  // same contract, but method choice uses preconditions and
  // priority against `start` (trusted state from the grid/harness, never parsed
  // from world_text). The chain record is byte-for-byte the same shape as plan():
  // one Custom record of little-endian primitive task ids. Chosen method ids are
  // returned in `detail` only and are not appended (official chain unchanged).
  std::vector<HtnTask> plan_with_state(const HtnDomain& domain, const HtnState& start,
                                       const std::string& world_text, IHashChain* chain,
                                       std::string& err, HtnDecomposition* detail = nullptr) const {
    (void)world_text;
    err.clear();
    if (detail) *detail = HtnDecomposition{};
    if (!goal_set_) {
      err = "goal not set";
      return {};
    }
    HtnDecomposition d;
    if (!domain.decompose_with_state(goal_id_, start, d, err)) {
      if (err.empty()) err = "empty plan";
      if (detail) *detail = std::move(d);
      return {};
    }
    append_plan(d.tasks, chain);
    std::vector<HtnTask> tasks = d.tasks;
    if (detail) *detail = std::move(d);
    return tasks;
  }

 private:
  static void append_plan(const std::vector<HtnTask>& tasks, IHashChain* chain) {
    if (!chain) return;
    std::vector<uint8_t> payload(tasks.size() * 4);
    for (size_t i = 0; i < tasks.size(); ++i) {
      const uint32_t id = tasks[i].id();
      payload[4 * i] = uint8_t(id);
      payload[4 * i + 1] = uint8_t(id >> 8);
      payload[4 * i + 2] = uint8_t(id >> 16);
      payload[4 * i + 3] = uint8_t(id >> 24);
    }
    chain->append(ChainRecord{ChainRecordKind::Custom, std::move(payload)});
  }

  uint32_t goal_id_ = 0;
  bool goal_set_ = false;
};

}  // namespace pchie