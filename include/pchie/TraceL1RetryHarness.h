// Bolt PCHIE failed-actuation retry - TraceL1RetryHarness: the Trace L1 placement
// proposal with a primed, reusable prompt branch and ranked candidates. Concrete final. No mutex.
// prime(): the SAME prompt as TraceL1Harness (TraceL1Harness::prompt_for via IModel::tokenize, one
// leading BOS) on a fresh branch that is kept. rank(trace): with an empty trace the logits are the
// primed ones (so attempt 1 == TraceL1Harness exactly); otherwise the BOS-free FailureTrace ids are
// forwarded on a FORK of the primed branch (scratch context; the fork is released), never on a
// fresh branch. Candidates: dividor, multiplexor, ranked by logit desc then id asc (the
// TraceL1Harness tie rule). Each carries its vocab path for PchGate. No chain is touched.
#pragma once

#include "pchie/Dividor.h"
#include "pchie/IModel.h"
#include "pchie/ModelBranch.h"
#include "pchie/Multiplexor.h"
#include "pchie/TraceL1Harness.h"
#include "pchie/TraceL1Task.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class TraceL1RetryHarness final {
 public:
  struct Candidate {
    const char* name = "";
    int32_t id = -1;
    float logit = 0.0f;
    std::vector<int32_t> path;  // vocab-id path (tokenize_no_bos), checked by PchGate
  };

  bool prime(IModel& m, const TraceL1Task& task, std::string& err) {
    err.clear();
    if (primed_) release(m);
    const auto ids = m.tokenize(TraceL1Harness::prompt_for(task));
    if (ids.empty()) {
      err = "empty prompt";
      return false;
    }
    br_ = m.new_branch();
    for (int32_t t : ids) m.forward(br_, t);
    base_logits_ = m.logits();
    base_len_ = ids.size();
    base_first_ = ids[0];
    primed_ = true;
    if (base_logits_.empty()) {
      err = "no logits";
      return false;
    }
    return true;
  }

  bool rank(IModel& m, const std::vector<int32_t>& trace, std::vector<Candidate>& out, std::string& err) {
    out.clear();
    err.clear();
    if (!primed_) {
      err = "trace without a primed frame";  // BOS-free ids never start a fresh branch
      return false;
    }
    std::vector<float> lg;
    if (trace.empty()) {
      lg = base_logits_;
    } else {
      ModelBranch f = m.fork(br_);
      for (int32_t t : trace) m.forward(f, t);
      lg = m.logits();
      m.release(f);
    }
    if (lg.empty()) {
      err = "no logits";
      return false;
    }
    const char* names[2] = {Dividor::kName, Multiplexor::kName};
    for (const char* name : names) {
      Candidate c;
      c.name = name;
      c.path = m.tokenize_no_bos(name);  // vocab-id lookup, as TraceL1Harness
      if (c.path.empty()) continue;
      c.id = c.path[0];
      if (c.id < 0 || size_t(c.id) >= lg.size()) continue;
      c.logit = lg[size_t(c.id)];
      out.push_back(std::move(c));
    }
    std::stable_sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
      return a.logit != b.logit ? a.logit > b.logit : a.id < b.id;
    });
    return true;
  }

  void release(IModel& m) {
    if (primed_) m.release(br_);
    primed_ = false;
  }
  bool primed() const noexcept { return primed_; }
  size_t base_len() const noexcept { return base_len_; }
  int32_t base_first_id() const noexcept { return base_first_; }

 private:
  ModelBranch br_;
  std::vector<float> base_logits_;
  size_t base_len_ = 0;
  int32_t base_first_ = -1;
  bool primed_ = false;
};

}  // namespace pchie