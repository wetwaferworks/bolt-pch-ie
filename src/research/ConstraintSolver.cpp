#include "pchie/research/ConstraintSolver.h"

#include "pchie/Dividor.h"
#include "pchie/Multiplexor.h"
#include "pchie/SimTraceGrader.h"
#include "pchie/TraceStrength.h"
#include "pchie/research/Digest.h"

#include <cstdio>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>

#ifdef _WIN32
#define PCHIE_POPEN _popen
#define PCHIE_PCLOSE _pclose
#else
#define PCHIE_POPEN popen
#define PCHIE_PCLOSE pclose
#endif

namespace pchie {
namespace research {

// ---- gate doors ----

GateAnswer GateDoorSolver::solve(const TraceL1Task& task) {
  GateAnswer a;
  const char* names[] = {Dividor::kName, Multiplexor::kName};
  const SimTraceGrader grader;
  for (const char* n : names) {
    TraceCircuit c;
    c.component = n;
    c.facing = task.facing;
    c.south = task.south;
    c.west = task.west;
    c.east = task.east;
    c.north = 0;
    if (grader.grade_circuit(c, task.goal).pass) a.all.emplace_back(n);
  }
  a.solvable = !a.all.empty();
  if (a.solvable) a.component = a.all.front();
  return a;
}

std::string GateDoorSolver::smt2(const TraceL1Task& t) {
  // Dividor reads its back face (south for a north-facing part) and halves it, rounding down.
  // Multiplexor (north-facing only): east > 0 selects west, otherwise south.
  int back = t.south;
  switch (t.facing) {
    case TraceFacing::North: back = t.south; break;
    case TraceFacing::South: back = 0; break;  // back face is north, which is unpowered here
    case TraceFacing::East: back = t.west; break;
    case TraceFacing::West: back = t.east; break;
  }
  const bool ports_ok = TraceStrength::ok(t.south) && TraceStrength::ok(t.west) && TraceStrength::ok(t.east);
  std::ostringstream o;
  o << "(set-logic QF_LIA)\n"
    << "(declare-const comp Int)\n"
    << "(assert (or (= comp 0) (= comp 1)))\n"
    << "(define-fun south () Int " << t.south << ")\n"
    << "(define-fun west () Int " << t.west << ")\n"
    << "(define-fun east () Int " << t.east << ")\n"
    << "(define-fun back () Int " << back << ")\n"
    << "(define-fun mux_ok () Bool " << ((t.facing == TraceFacing::North && ports_ok) ? "true" : "false") << ")\n"
    << "(define-fun div_ok () Bool (and (>= back " << TraceStrength::kMin << ") (<= back " << TraceStrength::kMax
    << ")))\n"
    << "(assert (=> (= comp 0) (and div_ok (= (div back 2) " << t.goal << "))))\n"
    << "(assert (=> (= comp 1) (and mux_ok (= (ite (> east 0) west south) " << t.goal << "))))\n"
    << "(check-sat)\n(get-value (comp))\n";
  return o.str();
}

// ---- gated graphs ----

namespace {
struct State {
  int node;
  uint32_t keys;
  uint32_t open;
  bool operator<(const State& o) const {
    if (node != o.node) return node < o.node;
    if (keys != o.keys) return keys < o.keys;
    return open < o.open;
  }
};
struct Came {
  State from;
  PlanAction act;
};
bool touches_door(const GatedGraph& g, int node, int door) {
  for (const auto& e : g.edges)
    if (e.door == door && (e.a == node || e.b == node)) return true;
  return false;
}
}  // namespace

GatedPlan GatedGraphSolver::solve(const GatedGraph& g) {
  GatedPlan plan;
  if (g.nodes <= 0 || g.doors < 0 || g.doors > 8 || g.start < 0 || g.start >= g.nodes) return plan;
  std::map<State, Came> seen;
  std::deque<State> q;
  const State s0{g.start, 0, 0};
  seen[s0] = Came{s0, {}};
  q.push_back(s0);
  bool found = false;
  State end{};
  while (!q.empty() && !found) {
    const State s = q.front();
    q.pop_front();
    if (s.node == g.goal) {
      found = true;
      end = s;
      break;
    }
    std::vector<std::pair<State, PlanAction>> next;
    // Fixed order: pick up keys, open doors, then moves by neighbour id.
    for (const auto& k : g.keys)
      if (k.node == s.node && !(s.keys & (1u << k.door)))
        next.push_back({State{s.node, s.keys | (1u << k.door), s.open}, PlanAction{PlanStep::Pickup, k.door}});
    for (int d = 0; d < g.doors; ++d)
      if ((s.keys & (1u << d)) && !(s.open & (1u << d)) && touches_door(g, s.node, d))
        next.push_back({State{s.node, s.keys, s.open | (1u << d)}, PlanAction{PlanStep::Open, d}});
    std::map<int, bool> nbr;
    for (const auto& e : g.edges) {
      if (e.door >= 0 && !(s.open & (1u << e.door))) continue;
      if (e.a == s.node) nbr[e.b] = true;
      if (e.b == s.node) nbr[e.a] = true;
    }
    for (const auto& kv : nbr) next.push_back({State{kv.first, s.keys, s.open}, PlanAction{PlanStep::Move, kv.first}});
    for (const auto& n : next) {
      if (seen.count(n.first)) continue;
      seen[n.first] = Came{s, n.second};
      q.push_back(n.first);
    }
  }
  if (!found) return plan;
  std::vector<PlanAction> rev;
  for (State s = end; !(s.node == s0.node && s.keys == s0.keys && s.open == s0.open); s = seen[s].from)
    rev.push_back(seen[s].act);
  plan.actions.assign(rev.rbegin(), rev.rend());
  plan.reachable = true;
  return plan;
}

std::string GatedGraphSolver::smt2(const GatedGraph& g, int horizon) {
  std::ostringstream o;
  o << "(set-logic QF_LIA)\n";
  const int T = horizon < 0 ? 0 : horizon;
  for (int t = 0; t <= T; ++t) {
    o << "(declare-const p" << t << " Int)\n(assert (and (>= p" << t << " 0) (< p" << t << ' ' << g.nodes << ")))\n";
    for (int d = 0; d < g.doors; ++d)
      o << "(declare-const k" << t << '_' << d << " Bool)\n(declare-const o" << t << '_' << d << " Bool)\n";
  }
  o << "(assert (= p0 " << g.start << "))\n";
  for (int d = 0; d < g.doors; ++d) o << "(assert (and (not k0_" << d << ") (not o0_" << d << ")))\n";
  auto frame_keys = [&](int t, int except_key, int except_open) {
    std::ostringstream f;
    for (int d = 0; d < g.doors; ++d) {
      if (d != except_key) f << " (= k" << t + 1 << '_' << d << " k" << t << '_' << d << ")";
      if (d != except_open) f << " (= o" << t + 1 << '_' << d << " o" << t << '_' << d << ")";
    }
    return f.str();
  };
  for (int t = 0; t < T; ++t) {
    o << "(assert (or";
    // stay (lets a shorter plan fit the horizon)
    o << " (and (= p" << t + 1 << " p" << t << ")" << frame_keys(t, -1, -1) << ")";
    for (const auto& e : g.edges) {
      const std::string gate = e.door >= 0 ? (" o" + std::to_string(t) + "_" + std::to_string(e.door)) : "";
      for (int dir = 0; dir < 2; ++dir) {
        const int from = dir ? e.b : e.a, to = dir ? e.a : e.b;
        o << " (and (= p" << t << ' ' << from << ") (= p" << t + 1 << ' ' << to << ")" << gate << frame_keys(t, -1, -1)
          << ")";
      }
    }
    for (const auto& k : g.keys)
      o << " (and (= p" << t << ' ' << k.node << ") (= p" << t + 1 << " p" << t << ") k" << t + 1 << '_' << k.door
        << frame_keys(t, k.door, -1) << ")";
    for (int d = 0; d < g.doors; ++d) {
      std::ostringstream at;
      for (const auto& e : g.edges)
        if (e.door == d) at << " (= p" << t << ' ' << e.a << ") (= p" << t << ' ' << e.b << ")";
      if (at.str().empty()) continue;
      o << " (and (or" << at.str() << ") k" << t << '_' << d << " (= p" << t + 1 << " p" << t << ") o" << t + 1 << '_'
        << d << frame_keys(t, -1, d) << ")";
    }
    o << "))\n";
  }
  o << "(assert (= p" << T << ' ' << g.goal << "))\n(check-sat)\n";
  return o.str();
}

std::string GatedGraphSolver::describe(const GatedPlan& p) {
  if (!p.reachable) return "unreachable";
  std::ostringstream o;
  for (size_t i = 0; i < p.actions.size(); ++i) {
    if (i) o << ' ';
    const auto& a = p.actions[i];
    o << (a.kind == PlanStep::Move ? "go" : (a.kind == PlanStep::Pickup ? "key" : "open")) << a.value;
  }
  return o.str();
}

// ---- pinned Z3 ----

const char* verdict_name(SmtVerdict v) {
  switch (v) {
    case SmtVerdict::Sat: return "sat";
    case SmtVerdict::Unsat: return "unsat";
    case SmtVerdict::Unknown: return "unknown";
    case SmtVerdict::Refused: return "refused";
  }
  return "refused";
}

std::string PinnedZ3::file_sha256(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return sha256_hex(bytes);
}

std::string PinnedZ3::with_header(const std::string& smt2_text) const {
  std::ostringstream o;
  o << "(set-option :smt.random_seed 0)\n(set-option :sat.random_seed 0)\n(set-option :rlimit " << pin_.rlimit
    << ")\n(set-option :produce-models true)\n"
    << smt2_text;
  return o.str();
}

SmtResult PinnedZ3::check(const std::string& smt2_text, const std::string& scratch_file) const {
  SmtResult r;
  if (pin_.binary.empty()) {
    r.reason = "no solver binary";
    return r;
  }
  const std::string have = file_sha256(pin_.binary);
  if (have.empty()) {
    r.reason = "solver binary not readable";
    return r;
  }
  if (have != pin_.sha256_hex) {
    r.reason = "solver binary does not match its pin";
    return r;
  }
  for (const PinnedFile& c : pin_.companions)
    if (file_sha256(c.path) != c.sha256_hex) {
      r.reason = "solver library does not match its pin: " + c.path;
      return r;
    }
  {
    std::ofstream f(scratch_file, std::ios::binary | std::ios::trunc);
    if (!f) {
      r.reason = "cannot write the problem file";
      return r;
    }
    f << with_header(smt2_text);
  }
  const std::string cmd = "\"\"" + pin_.binary + "\" -smt2 \"" + scratch_file + "\"\"";
#ifdef _WIN32
  const std::string run = cmd;
#else
  const std::string run = cmd.substr(1, cmd.size() - 2);
#endif
  FILE* p = PCHIE_POPEN(run.c_str(), "r");
  if (!p) {
    r.reason = "cannot start the solver";
    return r;
  }
  char buf[4096];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) r.output.append(buf, n);
  PCHIE_PCLOSE(p);
  std::string first = r.output.substr(0, r.output.find('\n'));
  while (!first.empty() && (first.back() == '\r' || first.back() == ' ')) first.pop_back();
  if (first == "sat") r.verdict = SmtVerdict::Sat;
  else if (first == "unsat") r.verdict = SmtVerdict::Unsat;
  else r.verdict = SmtVerdict::Unknown;
  r.reason = "ran";
  return r;
}

}  // namespace research
}  // namespace pchie
