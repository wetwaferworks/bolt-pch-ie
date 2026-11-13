// Reads the [RLON] lines that pchie_rl_l1l2_test prints for a maze run and judges two checklist items.
//   Planner comparison  L1 model-driven PUCT vs null vs oracle at the same steps and budget. The bar is that L1 must beat null.
//        If it does not (same walk, or a different walk that gets no further), the rule is stop and report:
//        no higher-budget runs until the maintainer has seen the result.
//   Query-before-plan  L2 query-before-plan: L1 with a seeded store vs L1 with an empty store vs null with a seeded store.
//        Checks that L2 was engaged, stayed on the scratch side chain, and reports whether hits changed the walk.
// Post-hoc reading of finished logs only. Nothing here feeds a run or touches a chain.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pchie::rl {

using KV = std::map<std::string, std::string>;

struct RlonRun {
  bool have_start = false, have_result = false, have_tokens = false, have_stats = false, have_g5 = false;
  KV start, flags, result, tokens, stats, g5;
  std::string get(const KV& m, const std::string& k) const { auto it = m.find(k); return it == m.end() ? std::string() : it->second; }
  long long num(const KV& m, const std::string& k, long long dflt = -1) const {
    const std::string v = get(m, k); if (v.empty()) return dflt; char* e = nullptr; const long long x = std::strtoll(v.c_str(), &e, 10); return (e && *e == 0) ? x : dflt; }
  bool complete() const { return have_start && have_result && have_tokens && have_stats && have_g5; }
  std::string label() const { return get(start, "label"); }
  // Progress toward the goal, measured after the run: BFS steps closed on the grid mazes, doors and exit on Stage B.
  long long progress() const {
    if (get(result, "maze") == "stageb") return num(result, "door0", 0) + num(result, "door1", 0) + 2 * num(result, "exit", 0);
    const long long s = num(result, "bfs_start"), r = num(result, "bfs_remaining_posthoc");
    return (s < 0 || r < 0) ? -1 : s - r;
  }
};

// key=value tokens separated by spaces. A token without '=' belongs to the previous value ("err=step budget").
inline KV parse_kv(const std::string& s) {
  KV m; std::istringstream in(s); std::string tok, last;
  while (in >> tok) {
    const auto eq = tok.find('=');
    if (eq == std::string::npos || eq == 0) { if (!last.empty()) m[last] += " " + tok; continue; }
    last = tok.substr(0, eq); m[last] = tok.substr(eq + 1);
  }
  return m;
}

inline RlonRun parse_rlon_log(const std::string& text) {
  RlonRun r; std::istringstream in(text); std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::string p = "[RLON] ";
    if (line.compare(0, p.size(), p) != 0) continue;
    std::string rest = line.substr(p.size());
    const auto sp = rest.find(' '); if (sp == std::string::npos) continue;
    const std::string kind = rest.substr(0, sp); rest = rest.substr(sp + 1);
    if (kind == "start") {
      const auto a = rest.find("flags={"), b = rest.find('}', a == std::string::npos ? 0 : a);
      if (a != std::string::npos && b != std::string::npos) { r.flags = parse_kv(rest.substr(a + 7, b - a - 7)); rest = rest.substr(0, a) + rest.substr(b + 1); }
      r.start = parse_kv(rest); r.have_start = true;
    } else if (kind == "result") { r.result = parse_kv(rest); r.have_result = true; }
    else if (kind == "tokens") { r.tokens = parse_kv(rest); r.have_tokens = true; }
    else if (kind == "stats") { r.stats = parse_kv(rest); r.have_stats = true; }
    else if (kind == "g5") { r.g5 = parse_kv(rest); r.have_g5 = true; }
  }
  return r;
}

struct Verdict {
  std::string verdict;          // short word for logs and status files
  bool valid = false;           // false: the runs cannot be compared (missing lines, wrong flags, mismatched settings)
  bool stop_and_report = false; // Planner comparison only: L1 did not beat null, so stop before any higher-budget run
  int exit_code = 2;            // 0 = passed the bar / comparable, 10 = stop and report, 2 = invalid
  std::vector<std::string> lines;
};

namespace detail {
inline const std::string kZero64(64, '0');
inline bool same_settings(const RlonRun& a, const RlonRun& b, std::vector<std::string>& why) {
  bool ok = true;
  for (const char* k : {"maze", "steps", "budget", "ctx", "poison", "loop_guard"})
    if (a.get(a.start, k) != b.get(b.start, k)) { ok = false; why.push_back(std::string("setting differs: ") + k + " " + a.label() + "=" + a.get(a.start, k) + " " + b.label() + "=" + b.get(b.start, k)); }
  return ok;
}
inline bool flag_is(const RlonRun& r, const char* k, const char* v, std::vector<std::string>& why) {
  if (r.get(r.flags, k) == v) return true;
  why.push_back(r.label() + ": expected " + k + "=" + v + ", got " + (r.get(r.flags, k).empty() ? "(missing)" : r.get(r.flags, k)));
  return false;
}
inline bool common_ok(const RlonRun& r, const char* role, std::vector<std::string>& why) {
  if (!r.complete()) { why.push_back(std::string(role) + " log is incomplete (no result/tokens/stats/g5 line; cancelled or still running?)"); return false; }
  bool ok = true;
  if (r.get(r.g5, "tuples_rehash") != "PASS") { ok = false; why.push_back(std::string(role) + ": g5 tuples_rehash is not PASS"); }
  if (r.num(r.g5, "live_pages_end") != 0) { ok = false; why.push_back(std::string(role) + ": KV pages left live at the end"); }
  return ok;
}
inline std::string pct(long long a, long long b) { if (b <= 0) return "n/a"; std::ostringstream o; o.setf(std::ios::fixed); o.precision(1); o << (100.0 * double(a - b) / double(b)) << "%"; return o.str(); }
inline std::string line_for(const RlonRun& r, const char* role) {
  std::ostringstream o;
  o << role << " " << r.label() << ": exit=" << r.get(r.result, "exit") << " path=" << r.get(r.result, "path") << " progress=" << r.progress()
    << " unique_cells=" << r.get(r.result, "unique_cells") << " actions=" << r.get(r.tokens, "actions_sha256").substr(0, 8)
    << " forwards=" << r.get(r.stats, "model_forwards") << " oracle_calls=" << r.get(r.stats, "oracle_calls") << " wall_s=" << r.get(r.stats, "wall_s");
  return o.str();
}
}  // namespace detail

// Planner comparison: does L1 beat null at this budget? Oracle at the same budget is the reference ceiling.
inline Verdict judge_l1_separation(const RlonRun& l1, const RlonRun& nul, const RlonRun& orc) {
  using namespace detail;
  Verdict v; std::vector<std::string> why; bool ok = true;
  ok &= common_ok(l1, "L1", why); ok &= common_ok(nul, "null", why); ok &= common_ok(orc, "oracle", why);
  if (ok) {
    ok &= same_settings(l1, nul, why); ok &= same_settings(l1, orc, why);
    ok &= flag_is(l1, "puct_model", "1", why); ok &= flag_is(l1, "puct_prior", "model", why); ok &= flag_is(l1, "puct_value", "model", why);
    ok &= flag_is(nul, "puct_prior", "model", why); ok &= flag_is(nul, "puct_value", "null", why);
    ok &= flag_is(orc, "puct_value", "oracle", why); ok &= flag_is(orc, "puct_prior", "current", why);
    for (const RlonRun* r : {&l1, &nul, &orc}) ok &= flag_is(*r, "qbp", "0", why);
    if (l1.get(l1.start, "engaged") != "1" || nul.get(nul.start, "engaged") != "1") { ok = false; why.push_back("L1 and null must both run engaged=1"); }
    if (orc.get(orc.start, "engaged") != "0") { ok = false; why.push_back("oracle reference must run engaged=0 (the official search path)"); }
    if (l1.num(l1.stats, "oracle_calls") != 0) { ok = false; why.push_back("L1 made oracle calls; it must not use the BFS oracle in the tree"); }
    if (l1.progress() < 0 || nul.progress() < 0 || orc.progress() < 0) { ok = false; why.push_back("progress could not be read from a result line"); }
  }
  if (!ok) { v.verdict = "INVALID"; v.lines = why; v.lines.insert(v.lines.begin(), "planner comparison verdict: INVALID (runs cannot be compared)"); return v; }
  v.valid = true;
  const bool same_walk = l1.get(l1.tokens, "actions_sha256") == nul.get(nul.tokens, "actions_sha256");
  const long long pl = l1.progress(), pn = nul.progress(), po = orc.progress();
  const bool l1_exit = l1.get(l1.result, "exit") == "1", n_exit = nul.get(nul.result, "exit") == "1";
  const bool beats = !same_walk && (pl > pn || (l1_exit && !n_exit));
  if (beats) { v.verdict = "SEPARATED"; v.exit_code = 0; }
  else { v.verdict = same_walk ? "NO_SEPARATION_SAME_WALK" : "NO_SEPARATION_NOT_BETTER"; v.stop_and_report = true; v.exit_code = 10; }
  std::ostringstream h; h << "planner comparison verdict: " << v.verdict << " (maze=" << l1.get(l1.start, "maze") << " steps=" << l1.get(l1.start, "steps") << " budget=" << l1.get(l1.start, "budget") << ")";
  v.lines.push_back(h.str());
  v.lines.push_back(line_for(l1, "L1    ")); v.lines.push_back(line_for(nul, "null  ")); v.lines.push_back(line_for(orc, "oracle"));
  std::ostringstream c; c << "L1 progress " << pl << " vs null " << pn << " vs oracle " << po << (same_walk ? "; L1 and null committed the same walk" : "; L1 and null walks differ");
  v.lines.push_back(c.str());
  v.lines.push_back(v.stop_and_report ? "STOP AND REPORT: L1 did not beat null at this budget. Do not start a higher-budget run; report it first."
                                      : "L1 beat null at this budget. Report it before choosing the next budget.");
  return v;
}

// Query-before-plan: L2 query-before-plan engagement and effect. No stop rule; reports what memory hits did.
inline Verdict judge_l2_qbp(const RlonRun& seeded, const RlonRun& empty, const RlonRun& nul_seeded) {
  using namespace detail;
  Verdict v; std::vector<std::string> why; bool ok = true;
  ok &= common_ok(seeded, "L1 seeded", why); ok &= common_ok(empty, "L1 empty", why); ok &= common_ok(nul_seeded, "null seeded", why);
  if (ok) {
    ok &= same_settings(seeded, empty, why); ok &= same_settings(seeded, nul_seeded, why);
    for (const RlonRun* r : {&seeded, &empty, &nul_seeded}) {
      ok &= flag_is(*r, "qbp", "1", why); ok &= flag_is(*r, "qbp_write", "0", why);
      if (r->get(r->stats, "qbp_side_head").empty() || r->get(r->stats, "qbp_side_head") == kZero64) { ok = false; why.push_back(r->label() + ": qbp_side_head is zero; L2 left no record on the scratch side chain"); }
      if (r->num(r->stats, "mem_q") != r->num(r->stats, "decisions")) { ok = false; why.push_back(r->label() + ": memory was not queried once per decision"); }
    }
    ok &= flag_is(seeded, "puct_model", "1", why); ok &= flag_is(empty, "puct_model", "1", why); ok &= flag_is(nul_seeded, "puct_value", "null", why);
    if (seeded.num(seeded.stats, "mem_hit") <= 0) { ok = false; why.push_back("L1 seeded: no memory hits; the seeded store was not used"); }
    if (empty.num(empty.stats, "mem_hit") != 0) { ok = false; why.push_back("L1 empty: memory hits from a store that should be empty"); }
  }
  if (!ok) { v.verdict = "INVALID"; v.lines = why; v.lines.insert(v.lines.begin(), "query-before-plan verdict: INVALID (runs cannot be compared)"); return v; }
  v.valid = true; v.exit_code = 0;
  const bool changed = seeded.get(seeded.tokens, "actions_sha256") != empty.get(empty.tokens, "actions_sha256");
  v.verdict = changed ? "L2_ENGAGED_WALK_CHANGED" : "L2_ENGAGED_WALK_UNCHANGED";
  v.lines.push_back("query-before-plan verdict: " + v.verdict + " (maze=" + seeded.get(seeded.start, "maze") + " steps=" + seeded.get(seeded.start, "steps") + " budget=" + seeded.get(seeded.start, "budget") + ")");
  const std::pair<const RlonRun*, const char*> rows[] = {{&seeded, "L1 seeded  "}, {&empty, "L1 empty   "}, {&nul_seeded, "null seeded"}};
  for (const auto& p : rows) {
    const RlonRun& r = *p.first;
    v.lines.push_back(line_for(r, p.second) + " mem_hit=" + r.get(r.stats, "mem_hit") + "/" + r.get(r.stats, "mem_q") + " mem_rej=" + r.get(r.stats, "mem_rej") +
                      " gate_fail_closed=" + r.get(r.stats, "gate_fail_closed") + " qbp_side_head=" + r.get(r.stats, "qbp_side_head").substr(0, 8));
  }
  v.lines.push_back("L1 seeded vs empty: progress " + std::to_string(seeded.progress()) + " vs " + std::to_string(empty.progress()) + ", model forwards " +
                    seeded.get(seeded.stats, "model_forwards") + " vs " + empty.get(empty.stats, "model_forwards") + " (" + pct(seeded.num(seeded.stats, "model_forwards"), empty.num(empty.stats, "model_forwards")) + ")");
  v.lines.push_back(changed ? "Memory hits changed the committed walk (hits are prior boosts after the gate and legality checks, never a bypass)."
                            : "Memory hits were used as proposals but did not change the committed walk at this budget.");
  return v;
}

// L2 store paths must sit under <root>/scratch/rl/mem/ (same rule as --seed-mem), so a run can never read or write an official store.
inline bool mem_path_under_scratch(const std::string& root, const std::string& path, std::string& why) {
  auto norm = [](std::string s) { for (char& c : s) { if (c == '\\') c = '/'; if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a'); } while (s.size() > 1 && s.back() == '/') s.pop_back(); return s; };
  const std::string r = norm(root), p = norm(path), want = r + "/scratch/rl/mem/";
  if (path.empty()) { why = "empty store path"; return false; }
  if (p.find("/../") != std::string::npos || p.find("..") == 0) { why = "store path may not contain .."; return false; }
  if (p.compare(0, want.size(), want) != 0 || p.size() == want.size()) { why = "store path must be under " + root + "/scratch/rl/mem/"; return false; }
  return true;
}

}  // namespace pchie::rl
