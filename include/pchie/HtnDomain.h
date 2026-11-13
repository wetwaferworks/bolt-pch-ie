// Bolt PCHIE - HTN task/method registry and deterministic decompose.
// Concrete final. Coordinator-thread only: no mutex, no atomic.
// Multiple methods for one head: lowest method id wins (provisional, see
// an open review question).
// decompose_with_state() picks, at each compound task, the
// highest-priority method whose precondition holds in the current state (ties:
// lowest id), threads primitive effects forward, and backtracks to the next
// applicable method if a subtree fails. The legacy expand_once()/decompose()
// are unchanged (they ignore preconditions/priority) so official B1/B4-B7 and
// Stage B plans cannot move.
#pragma once

#include "pchie/HtnDecomposition.h"
#include "pchie/HtnMethod.h"
#include "pchie/HtnState.h"
#include "pchie/HtnTask.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class HtnDomain final {
 public:
  static constexpr size_t kDefaultMaxExpansions = 4096;

  bool add_task(HtnTask task) {
    if (find_task(task.id()) != nullptr) return false;
    if (!task.primitive() && !task.effect().empty()) return false;  // effects are primitive-only
    tasks_.push_back(std::move(task));
    return true;
  }

  bool add_method(HtnMethod method) {
    if (find_method(method.id()) != nullptr) return false;
    if (find_task(method.head_task_id()) == nullptr) return false;
    if (method.precondition().contradictory()) return false;  // could never apply
    for (uint32_t s : method.subtask_ids()) {
      if (find_task(s) == nullptr) return false;
    }
    methods_.push_back(std::move(method));
    return true;
  }

  bool has_method(uint32_t id) const { return find_method(id) != nullptr; }

  const HtnTask* find_task(uint32_t id) const {
    for (const HtnTask& t : tasks_) {
      if (t.id() == id) return &t;
    }
    return nullptr;
  }

  std::vector<const HtnMethod*> methods_for(uint32_t head_task_id) const {
    std::vector<const HtnMethod*> out;
    for (const HtnMethod& m : methods_) {
      if (m.head_task_id() == head_task_id) out.push_back(&m);
    }
    std::sort(out.begin(), out.end(),
              [](const HtnMethod* a, const HtnMethod* b) { return a->id() < b->id(); });
    return out;
  }

  // methods for a head in try order: priority descending, then id ascending.
  std::vector<const HtnMethod*> ranked_methods(uint32_t head_task_id) const {
    std::vector<const HtnMethod*> out = methods_for(head_task_id);  // already id-ascending
    std::stable_sort(out.begin(), out.end(), [](const HtnMethod* a, const HtnMethod* b) {
      return a->priority() > b->priority();
    });
    return out;
  }

  // first ranked method whose precondition holds, or nullptr.
  const HtnMethod* select_method(uint32_t head_task_id, const HtnState& state) const {
    for (const HtnMethod* m : ranked_methods(head_task_id))
      if (m->precondition().holds(state)) return m;
    return nullptr;
  }

  // state-aware recursive decomposition with backtracking.
  // Fails closed: on any error `out` is cleared and err says why. Hard errors
  // (unknown task/subtask, expansion budget) stop at once; soft ones (no
  // applicable method, cycle) let the parent try its next applicable method.
  bool decompose_with_state(uint32_t goal_id, const HtnState& start, HtnDecomposition& out,
                            std::string& err,
                            size_t max_expansions = kDefaultMaxExpansions) const {
    err.clear();
    out = HtnDecomposition{};
    out.end_state = start;
    std::vector<uint32_t> stack;
    bool hard = false;
    if (!decompose_state_into(goal_id, out, stack, err, hard, max_expansions)) {
      const size_t e = out.expansions, b = out.backtracks;
      out = HtnDecomposition{};
      out.expansions = e;
      out.backtracks = b;
      if (err.empty()) err = "decompose failed";
      return false;
    }
    if (out.tasks.empty()) {
      err = "empty plan";
      return false;
    }
    return true;
  }

  // One level: primitive => {task}; compound => body of the lowest-id method.
  std::vector<HtnTask> expand_once(uint32_t task_id, std::string& err) const {
    err.clear();
    const HtnTask* t = find_task(task_id);
    if (!t) {
      err = "unknown task";
      return {};
    }
    if (t->primitive()) return {*t};
    const auto ms = methods_for(task_id);
    if (ms.empty()) {
      err = "no method";
      return {};
    }
    std::vector<HtnTask> out;
    for (uint32_t s : ms[0]->subtask_ids()) {
      const HtnTask* st = find_task(s);
      if (!st) {
        err = "unknown subtask";
        return {};
      }
      out.push_back(*st);
    }
    return out;
  }

  // Recursive expansion to primitives, method-body order, lowest method id at each head.
  std::vector<HtnTask> decompose(uint32_t goal_id, std::string& err) const {
    err.clear();
    std::vector<HtnTask> out;
    std::vector<uint32_t> stack;
    if (!decompose_into(goal_id, out, stack, err)) return {};
    return out;
  }

 private:
  const HtnMethod* find_method(uint32_t id) const {
    for (const HtnMethod& m : methods_) {
      if (m.id() == id) return &m;
    }
    return nullptr;
  }

  bool decompose_into(uint32_t task_id, std::vector<HtnTask>& out,
                      std::vector<uint32_t>& stack, std::string& err) const {
    for (uint32_t s : stack) {
      if (s == task_id) {
        err = "cycle";
        return false;
      }
    }
    const HtnTask* t = find_task(task_id);
    if (!t) {
      err = "unknown task";
      return false;
    }
    if (t->primitive()) {
      out.push_back(*t);
      return true;
    }
    const auto ms = methods_for(task_id);
    if (ms.empty()) {
      err = "no method";
      return false;
    }
    stack.push_back(task_id);
    for (uint32_t s : ms[0]->subtask_ids()) {
      if (!decompose_into(s, out, stack, err)) return false;
    }
    stack.pop_back();
    return true;
  }

  bool decompose_state_into(uint32_t task_id, HtnDecomposition& out,
                            std::vector<uint32_t>& stack, std::string& err, bool& hard,
                            size_t max_expansions) const {
    for (uint32_t s : stack) {
      if (s == task_id) {
        err = "cycle";
        return false;  // soft
      }
    }
    const HtnTask* t = find_task(task_id);
    if (!t) {
      err = "unknown task";
      hard = true;
      return false;
    }
    if (t->primitive()) {
      out.tasks.push_back(*t);
      t->effect().apply(out.end_state);
      return true;
    }
    const auto ms = ranked_methods(task_id);
    if (ms.empty()) {
      err = "no method";
      return false;  // soft: a sibling method of the parent may avoid this task
    }
    bool any_applicable = false;
    std::string last_err;
    stack.push_back(task_id);
    for (const HtnMethod* m : ms) {
      if (!m->precondition().holds(out.end_state)) continue;
      any_applicable = true;
      if (out.expansions >= max_expansions) {
        err = "expansion budget";
        hard = true;
        stack.pop_back();
        return false;
      }
      ++out.expansions;
      const size_t n_tasks = out.tasks.size();
      const size_t n_methods = out.methods.size();
      const HtnState saved = out.end_state;
      out.methods.push_back(m->id());
      bool ok = true;
      for (uint32_t s : m->subtask_ids()) {
        if (!decompose_state_into(s, out, stack, err, hard, max_expansions)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        stack.pop_back();
        err.clear();
        return true;
      }
      if (hard) {
        stack.pop_back();
        return false;
      }
      // Undo this attempt and try the next applicable method.
      last_err = err;
      out.tasks.resize(n_tasks);
      out.methods.resize(n_methods);
      out.end_state = saved;
      ++out.backtracks;
    }
    stack.pop_back();
    err = any_applicable ? ("all applicable methods failed: " + last_err) : "no applicable method";
    return false;
  }

  std::vector<HtnTask> tasks_;
  std::vector<HtnMethod> methods_;
};

}  // namespace pchie