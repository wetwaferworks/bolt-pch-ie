// Bolt PCHIE Stage B - L1 logic-gate door (Q57/Q60). Concrete final.
// Opens exactly one door, only when: the door id is valid and closed, the
// task is that door's task, the agent is adjacent, PchGate passed the
// proposal, and StageBDoorGrader passes it. Anything else leaves the world
// unchanged. Opening a door is never a win. The model path reuses
// TraceL1Harness unchanged (its draft-key grade is ignored; Stage B ids are
// not in the draft key). No official chain append. No mutex.
#pragma once

#include "pchie/IModel.h"
#include "pchie/StageBDoorGrader.h"
#include "pchie/StageBDoorGrid.h"
#include "pchie/StageBDoorKey.h"
#include "pchie/TraceCircuit.h"
#include "pchie/TraceGrade.h"
#include "pchie/TraceL1Harness.h"
#include "pchie/TraceL1Task.h"

#include <cstring>
#include <string>

namespace pchie {

struct StageBDoorAttempt {
  int door = -1;
  const char* task_id = "";
  const char* proposed = "";
  bool gate_pass = false;
  TraceGrade grade;
  bool opened = false;
  const char* reason = "";
};

class StageBLogicDoor final {
 public:
  static StageBDoorAttempt resolve(StageBDoorGrid& g, int door, const TraceL1Task& task,
                                   const char* component, bool gate_pass) {
    StageBDoorAttempt a;
    a.door = door;
    a.task_id = task.id ? task.id : "";
    a.proposed = component ? component : "";
    a.gate_pass = gate_pass;
    a.grade.expected = task.goal;
    TraceL1Task want;
    if (!StageBDoorKey::task_for_door(door, want)) {
      a.reason = "bad door";
      return a;
    }
    if (!task.id || std::strcmp(task.id, want.id) != 0) {
      a.reason = "wrong task";
      return a;
    }
    if (g.door_open(door)) {
      a.reason = "already open";
      return a;
    }
    if (!g.adjacent_to_door(door)) {
      a.reason = "not adjacent";
      return a;
    }
    if (!component) {
      a.reason = "no proposal";
      return a;
    }
    if (!gate_pass) {
      a.grade.reason = "gate reject";
      a.reason = "gate reject";
      return a;
    }
    TraceCircuit c;
    c.component = component;
    c.facing = want.facing;
    c.south = want.south;
    c.west = want.west;
    c.east = want.east;
    c.north = 0;
    a.grade = StageBDoorGrader().grade(c, want);
    if (!a.grade.pass) {
      a.reason = a.grade.reason;
      return a;
    }
    a.opened = g.open_door(door);
    a.reason = a.opened ? "opened" : "open failed";
    return a;
  }

  // Real-model attempt: the unchanged Trace L1 harness proposes the component.
  static bool attempt(IModel& model, StageBDoorGrid& g, int door, StageBDoorAttempt& out,
                      std::string& err) {
    out = {};
    err.clear();
    TraceL1Task t;
    if (!StageBDoorKey::task_for_door(door, t)) {
      err = "bad door";
      return false;
    }
    TraceL1Harness::Result r;
    if (!TraceL1Harness::run(model, t, r, err)) return false;
    out = resolve(g, door, t, r.circuit.component, r.gate_pass);
    return true;
  }
};

}  // namespace pchie