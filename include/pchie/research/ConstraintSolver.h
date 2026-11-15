// Exact constraint peer beside L6 (the verifier). Two kinds of question Little Nemo stalls on:
//   1. Stage B logic-gate doors: which component makes the door's output equal its goal?
//   2. Doors, keys and shut edges on his own map: in what order must he fetch keys and open
//      doors to reach a place, and how many actions does it take at least?
// Every question has a built-in exact solver (the reference). The same question can also be
// written as SMT-LIB2 and handed to a pinned Z3 binary: the binary's SHA-256 is checked before it
// runs, it uses smt.random_seed=0 and a step-based rlimit (never wall time), and its answer must
// agree with the built-in one. The peer only reports; it never moves Nemo by itself.
#pragma once

#include "pchie/TraceL1Task.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace research {

// ---- Stage B logic-gate doors -------------------------------------------------------------

struct GateAnswer {
  bool solvable = false;
  std::string component;          // first component (in name order) whose output equals the goal
  std::vector<std::string> all;   // every component that works
};

class GateDoorSolver {
 public:
  static GateAnswer solve(const TraceL1Task& task);
  /// SMT-LIB2 for the same question; the model's value of comp is 0 (dividor) or 1 (multiplexor).
  static std::string smt2(const TraceL1Task& task);
};

// ---- Doors, keys and shut edges ----------------------------------------------------------

struct GatedEdge {
  int a = 0;
  int b = 0;
  int door = -1;  // -1: always open; otherwise passable only once that door is open
};

struct KeySpot {
  int node = 0;
  int door = 0;   // the door this key opens
};

struct GatedGraph {
  int nodes = 0;
  int start = 0;
  int goal = 0;
  int doors = 0;  // at most 8
  std::vector<GatedEdge> edges;
  std::vector<KeySpot> keys;
};

enum class PlanStep : uint8_t { Move = 0, Pickup = 1, Open = 2 };

struct PlanAction {
  PlanStep kind = PlanStep::Move;
  int value = 0;  // Move: node moved to; Pickup: door of the key; Open: door id
};

struct GatedPlan {
  bool reachable = false;
  std::vector<PlanAction> actions;  // shortest; ties broken by a fixed action order
};

class GatedGraphSolver {
 public:
  /// Exact breadth-first search over (node, keys held, doors open).
  static GatedPlan solve(const GatedGraph& g);
  /// Bounded reachability in exactly `horizon` or fewer actions, as SMT-LIB2.
  static std::string smt2(const GatedGraph& g, int horizon);
  static std::string describe(const GatedPlan& p);
};

// ---- Pinned Z3 -----------------------------------------------------------------------------

struct PinnedFile {
  std::string path;
  std::string sha256_hex;
};

struct PinnedSolver {
  std::string binary;       // path to the z3 executable
  std::string sha256_hex;   // expected SHA-256 of that file
  uint64_t rlimit = 20000000;
  std::vector<PinnedFile> companions;  // libraries it loads (libz3.dll on Windows), each pinned too
};

enum class SmtVerdict : uint8_t { Sat = 0, Unsat = 1, Unknown = 2, Refused = 3 };

struct SmtResult {
  SmtVerdict verdict = SmtVerdict::Refused;
  std::string output;   // solver stdout (model text for sat)
  std::string reason;   // why it was refused, if it was
};

class PinnedZ3 {
 public:
  explicit PinnedZ3(PinnedSolver pin) : pin_(std::move(pin)) {}
  /// SHA-256 of the binary on disk (empty when unreadable).
  static std::string file_sha256(const std::string& path);
  /// Refuses to run when the binary or a companion is missing or its SHA-256 differs from the pin.
  SmtResult check(const std::string& smt2_text, const std::string& scratch_file) const;
  /// Adds the determinism header (random seeds 0, rlimit) to a problem.
  std::string with_header(const std::string& smt2_text) const;

 private:
  PinnedSolver pin_;
};

const char* verdict_name(SmtVerdict v);

}  // namespace research
}  // namespace pchie
