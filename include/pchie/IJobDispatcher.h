// Bolt PCHIE - job hook at the IModel runtime boundary (Archie amendment 4).
// Concrete dispatchers are used by the model; workers never touch the hash chain.
#pragma once

#include <functional>

namespace pchie {

class IJobDispatcher {
 public:
  virtual ~IJobDispatcher() = default;
  // Invoke fn(0)..fn(count-1). SerialJobDispatcher runs them in increasing index order.
  virtual void parallel_for(int count, const std::function<void(int)>& fn) = 0;
  virtual int size() const noexcept = 0;
};

}  // namespace pchie