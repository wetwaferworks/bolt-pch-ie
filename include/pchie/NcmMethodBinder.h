// Bolt PCHIE - NcmMethodBinder: an NCM concept becomes an HTN method.
// Concrete final. Coordinator-thread only: no mutex, no atomic.
//
// Mapping (default, logged for review; not a locked schema):
//   concept c (span of 1..16 tokens) on compound head H
//     -> primitive "concept task"  id = kNcmConceptTaskBase + c.id, name "ncm.concept:<id>"
//     -> HtnMethod                 id = kNcmMethodBase + c.id, head H, body {concept task}
// The concept task is opaque to the planner (its chain payload is still just a task id).
// PchStack (span mode) turns a concept task into a search whose root is forced along the
// concept's token span. Reserved id ranges keep bound methods from colliding with authored
// ids, and a bound method only beats an authored one through priority or preconditions.
// Fails closed: nothing is added to the domain unless every check passes.
#pragma once

#include "pchie/HtnCondition.h"
#include "pchie/HtnDomain.h"
#include "pchie/HtnMethod.h"
#include "pchie/HtnTask.h"
#include "pchie/NcmCandidateGenerator.h"
#include "pchie/NcmConcept.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class NcmMethodBinder final {
 public:
  static constexpr uint32_t kNcmConceptTaskBase = 0x40000000u;
  static constexpr uint32_t kNcmMethodBase = 0x60000000u;
  static constexpr uint32_t kNcmMaxConceptId = 0x1FFFFFFFu;

  static bool is_concept_task(uint32_t task_id) noexcept {
    return task_id >= kNcmConceptTaskBase && task_id <= kNcmConceptTaskBase + kNcmMaxConceptId;
  }
  static uint32_t concept_task_id(uint32_t concept_id) noexcept { return kNcmConceptTaskBase + concept_id; }
  static uint32_t method_id(uint32_t concept_id) noexcept { return kNcmMethodBase + concept_id; }

  // Bind `c` as a method on compound `head_task_id`. On success the concept is also kept
  // in this binder so a driver can look its span up by concept task id.
  bool bind(HtnDomain& d, const NcmConcept& c, uint32_t head_task_id, std::string& err,
            HtnCondition pre = HtnCondition(), int32_t priority = 0) {
    err.clear();
    if (c.tokens().empty() || c.tokens().size() > kNcmMaxSpan) { err = "span length"; return false; }
    if (std::isnan(c.score())) { err = "nan score"; return false; }
    for (int32_t t : c.tokens())
      if (t < 0) { err = "negative token"; return false; }
    if (c.id() > kNcmMaxConceptId) { err = "concept id out of range"; return false; }
    if (is_concept_task(head_task_id)) { err = "head is a concept task"; return false; }
    const HtnTask* head = d.find_task(head_task_id);
    if (!head) { err = "unknown head"; return false; }
    if (head->primitive()) { err = "head is primitive"; return false; }
    if (pre.contradictory()) { err = "contradictory precondition"; return false; }
    const uint32_t tid = concept_task_id(c.id());
    const uint32_t mid = method_id(c.id());
    if (d.has_method(mid)) { err = "method id taken"; return false; }
    const std::string tname = "ncm.concept:" + std::to_string(c.id());
    const HtnTask* existing = d.find_task(tid);
    if (existing && (!existing->primitive() || existing->name() != tname)) { err = "task id taken"; return false; }
    const NcmConcept* known = find(c.id());
    if (known && *known != c) { err = "concept id rebound with a different span"; return false; }
    // All checks passed: mutate.
    if (!existing && !d.add_task(HtnTask(tid, tname, true))) { err = "add_task"; return false; }
    const std::string mname = "ncm:" + (c.name().empty() ? std::to_string(c.id()) : c.name());
    if (!d.add_method(HtnMethod(mid, mname, head_task_id, {tid}).with_precondition(std::move(pre))
                          .with_priority(priority))) {
      err = "add_method";
      return false;  // unreachable after the checks above; the concept task stays (harmless, primitive)
    }
    if (!known) concepts_.push_back(c);
    return true;
  }

  // Span lookup for a concept task id, or nullptr.
  const NcmConcept* concept_for_task(uint32_t task_id) const {
    if (!is_concept_task(task_id)) return nullptr;
    return find(task_id - kNcmConceptTaskBase);
  }
  const std::vector<NcmConcept>& concepts() const noexcept { return concepts_; }

 private:
  const NcmConcept* find(uint32_t concept_id) const {
    for (const NcmConcept& c : concepts_)
      if (c.id() == concept_id) return &c;
    return nullptr;
  }
  std::vector<NcmConcept> concepts_;  // bind order
};

}  // namespace pchie
