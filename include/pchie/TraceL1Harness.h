// Bolt PCHIE - minimal Little Nemo L1 placement harness.
// Concrete final. The model proposes dividor vs multiplexor from logits on a
// prompt that does not name the answer. The proposal is checked by PchGate
// and scored by SimTraceGrader against the draft key. Honest pass/fail;
// not tuned. No official chain append. No mutex.
#pragma once

#include "pchie/Dividor.h"
#include "pchie/IModel.h"
#include "pchie/Multiplexor.h"
#include "pchie/PchGate.h"
#include "pchie/SimTraceGrader.h"
#include "pchie/TraceCircuit.h"
#include "pchie/TraceL1Task.h"

#include <string>
#include <vector>

namespace pchie {

class TraceL1Harness final {
 public:
  struct Result {
    TraceCircuit circuit;
    bool gate_pass = false;
    TraceGrade grader;
  };

  static std::string prompt_for(const TraceL1Task& task) {
    std::string s = "L1 task ";
    s += task.id ? task.id : "";
    s += ". Place one component facing north. south=";
    s += std::to_string(task.south);
    s += " west=";
    s += std::to_string(task.west);
    s += " east=";
    s += std::to_string(task.east);
    s += ". Goal output at north=";
    s += std::to_string(task.goal);
    s += ".";
    return s;
  }

  static bool run(IModel& model, const TraceL1Task& task, Result& out, std::string& err) {
    out = {};
    err.clear();
    out.circuit.facing = task.facing;
    out.circuit.south = task.south;
    out.circuit.west = task.west;
    out.circuit.east = task.east;
    out.circuit.north = 0;

    const std::string prompt = prompt_for(task);
    const auto ids = model.tokenize(prompt);
    ModelBranch br = model.new_branch();
    for (int32_t tok : ids) model.forward(br, tok);
    const std::vector<float> logits = model.logits();
    model.release(br);
    if (logits.empty()) {
      err = "no logits";
      return false;
    }

    const char* names[2] = {Dividor::kName, Multiplexor::kName};
    const char* best_name = nullptr;
    int32_t best_id = -1;
    float best_logit = 0.0f;
    std::vector<int32_t> best_path;
    for (const char* name : names) {
      const auto w = model.tokenize_no_bos(name);  // vocab-id lookup, no BOS (BOS policy)
      if (w.empty()) continue;
      const int32_t id = w[0];
      if (id < 0 || size_t(id) >= logits.size()) continue;
      const float lg = logits[size_t(id)];
      const bool better =
          !best_name || lg > best_logit || (lg == best_logit && (best_id < 0 || id < best_id));
      if (better) {
        best_name = name;
        best_id = id;
        best_logit = lg;
        best_path = w;
      }
    }
    if (!best_name) {
      err = "no proposal";
      return false;
    }
    out.circuit.component = best_name;
    PchGate gate;
    out.gate_pass = gate.Pass(ProposalTokens::from_proposal(best_path));  // proposal (component name) only
    if (!out.gate_pass) {
      out.grader.expected = task.goal;
      out.grader.reason = "gate reject";
      return true;
    }
    out.grader = SimTraceGrader().grade_against_key(out.circuit, task);
    return true;
  }
};

}  // namespace pchie
