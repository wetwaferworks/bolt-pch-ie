// Bolt PCHIE - one HTN task (primitive or compound). Concrete final. No mutex.
// optional HtnEffect on primitives; default empty = the original behavior.
#pragma once

#include "pchie/HtnEffect.h"

#include <cstdint>
#include <string>

namespace pchie {

class HtnTask final {
 public:
  HtnTask() = default;
  HtnTask(uint32_t id, std::string name, bool primitive)
      : id_(id), name_(std::move(name)), primitive_(primitive) {}

  uint32_t id() const noexcept { return id_; }
  const std::string& name() const noexcept { return name_; }
  bool primitive() const noexcept { return primitive_; }
  const HtnEffect& effect() const noexcept { return effect_; }
  // builder-style; only meaningful on primitives (add_task rejects
  // an effect on a compound task).
  HtnTask& with_effect(HtnEffect e) {
    effect_ = std::move(e);
    return *this;
  }

  bool operator==(const HtnTask& o) const {
    return id_ == o.id_ && name_ == o.name_ && primitive_ == o.primitive_ && effect_ == o.effect_;
  }
  bool operator!=(const HtnTask& o) const { return !(*this == o); }

 private:
  uint32_t id_ = 0;
  std::string name_;
  bool primitive_ = true;
  HtnEffect effect_;
};

}  // namespace pchie