#include "pchie/research/ResearchFlags.h"

#include <cstdlib>
#include <sstream>

namespace pchie {
namespace research {

bool ResearchFlags::any_on() const {
  return go_explore || smt || !smt_binary.empty() || step_scorer || tree_of_thoughts || reflexion || lookahead ||
         ctx != 1024;
}

std::string ResearchFlags::identity_line() const {
  std::ostringstream o;
  o << "research.v1 go_explore=" << int(go_explore) << " smt=" << int(smt)
    << " smt_binary=" << (smt_binary.empty() ? "-" : "set") << " step_scorer=" << int(step_scorer)
    << " step_scorer_mode=" << (step_scorer_mode == ScorerMode::Enforce ? "enforce" : "record")
    << " tot=" << int(tree_of_thoughts) << " tot_breadth=" << tot_breadth << " tot_depth=" << tot_depth
    << " reflexion=" << int(reflexion) << " lookahead=" << int(lookahead) << " lookahead_depth=" << lookahead_depth
    << " ctx=" << ctx;
  return o.str();
}

bool research_runner_clean(const ResearchFlags& f) {
  return !f.any_on() && f.identity_line() == ResearchFlags{}.identity_line();
}

namespace {
bool starts_with(const std::string& s, const char* p) {
  const std::string q(p);
  return s.size() >= q.size() && s.compare(0, q.size(), q) == 0;
}
bool take(const std::string& a, const char* key, std::string& val) {
  const std::string k = std::string(key) + "=";
  if (!starts_with(a, k.c_str())) return false;
  val = a.substr(k.size());
  return true;
}
bool small_number(const std::string& v, unsigned long lo, unsigned long hi, uint32_t& out) {
  char* end = nullptr;
  const unsigned long n = std::strtoul(v.c_str(), &end, 10);
  if (v.empty() || !end || *end || n < lo || n > hi) return false;
  out = uint32_t(n);
  return true;
}
}  // namespace

ResearchParseResult parse_research_flags(const std::vector<std::string>& args) {
  ResearchParseResult r;
  auto fail = [&](const std::string& why) {
    r.ok = false;
    r.err = why;
    return r;
  };
  for (const std::string& a : args) {
    if (!starts_with(a, "--research-")) continue;
    std::string v;
    if (a == "--research-go-explore") r.flags.go_explore = true;
    else if (a == "--research-smt") r.flags.smt = true;
    else if (take(a, "--research-smt-binary", v)) {
      if (v.empty()) return fail("--research-smt-binary needs a path");
      r.flags.smt = true;
      r.flags.smt_binary = v;
    } else if (a == "--research-step-scorer") r.flags.step_scorer = true;
    else if (take(a, "--research-step-scorer", v)) {
      if (v == "record") r.flags.step_scorer_mode = ScorerMode::Record;
      else if (v == "enforce") r.flags.step_scorer_mode = ScorerMode::Enforce;
      else return fail("bad --research-step-scorer (want record|enforce)");
      r.flags.step_scorer = true;
    } else if (a == "--research-tree-of-thoughts") r.flags.tree_of_thoughts = true;
    else if (take(a, "--research-tot-breadth", v)) {
      if (!small_number(v, 1, 8, r.flags.tot_breadth)) return fail("bad --research-tot-breadth (1..8)");
    } else if (take(a, "--research-tot-depth", v)) {
      if (!small_number(v, 1, 6, r.flags.tot_depth)) return fail("bad --research-tot-depth (1..6)");
    } else if (a == "--research-reflexion") r.flags.reflexion = true;
    else if (a == "--research-lookahead") r.flags.lookahead = true;
    else if (take(a, "--research-lookahead", v)) {
      if (!small_number(v, 1, 6, r.flags.lookahead_depth)) return fail("bad --research-lookahead (1..6)");
      r.flags.lookahead = true;
    } else if (take(a, "--research-ctx", v)) {
      if (v == "1024") r.flags.ctx = 1024;
      else if (v == "2048") r.flags.ctx = 2048;
      else return fail("bad --research-ctx (want 1024|2048)");
    } else {
      return fail("unknown research flag: " + a);
    }
  }
  return r;
}

ResearchParseResult parse_research_flags(int argc, char** argv) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i)
    if (argv[i]) args.emplace_back(argv[i]);
  return parse_research_flags(args);
}

}  // namespace research
}  // namespace pchie
