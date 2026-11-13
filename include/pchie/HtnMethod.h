// Bolt PCHIE - one HTN method: a compound head and an ordered subtask list.
// Concrete final. No mutex.
// optional HtnCondition precondition and an integer priority.
// Defaults (empty condition, priority 0) keep the original rule: lowest method id wins.
#pragma once

#include "pchie/HtnCondition.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class HtnMethod final {
 public:
  HtnMethod() = default;
  HtnMethod(uint32_t id, std::string name, uint32_t head_task_id,
            std::vector<uint32_t> subtask_ids)
      : id_(id),
        name_(std::move(name)),
        head_task_id_(head_task_id),
        subtask_ids_(std::move(subtask_ids)) {}

  uint32_t id() const noexcept { return id_; }
  const std::string& name() const noexcept { return name_; }
  uint32_t head_task_id() const noexcept { return head_task_id_; }
  const std::vector<uint32_t>& subtask_ids() const noexcept { return subtask_ids_; }
  const HtnCondition& precondition() const noexcept { return pre_; }
  int32_t priority() const noexcept { return priority_; }

  // builders. Higher priority is tried first; equal priority falls
  // back to lowest method id.
  HtnMethod& with_precondition(HtnCondition c) {
    pre_ = std::move(c);
    return *this;
  }
  HtnMethod& with_priority(int32_t p) {
    priority_ = p;
    return *this;
  }

 private:
  uint32_t id_ = 0;
  std::string name_;
  uint32_t head_task_id_ = 0;
  std::vector<uint32_t> subtask_ids_;
  HtnCondition pre_;
  int32_t priority_ = 0;
};

}  // namespace pchie