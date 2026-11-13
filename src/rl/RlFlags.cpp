#include "pchie/rl/RlFlags.h"

#include "pchie/rl/RlCommon.h"

#include <cstdlib>
#include <sstream>

namespace pchie {
namespace rl {

bool RlFlags::any_on() const {
  return puct_model || puct_prior != PriorSrc::Current || puct_value != ValueSrc::Oracle ||
         qbp || qbp_write || fork_score || belief || vision_only || options || symbolic_peer ||
         symbolic_weight_q20 != 0 || verifier;
  // qbp_min_truth, fork_horizon, fork_budget, options_set, verifier_mode are inert when their
  // parent flags are off; they are still recorded in identity_line().
}

std::string RlFlags::identity_line() const {
  std::ostringstream o;
  o << "puct_model=" << (puct_model ? 1 : 0)
    << " puct_prior=" << (puct_prior == PriorSrc::Model ? "model" : "current")
    << " puct_value="
    << (puct_value == ValueSrc::Model ? "model" : puct_value == ValueSrc::Null ? "null" : "oracle")
    << " qbp=" << (qbp ? 1 : 0) << " qbp_write=" << (qbp_write ? 1 : 0)
    << " qbp_min_truth=" << (qbp_min_truth == MinTruth::Human ? "human" : "sim")
    << " fork_score=" << (fork_score ? 1 : 0) << " fork_horizon=" << fork_horizon
    << " fork_budget=" << fork_budget << " belief=" << (belief ? 1 : 0)
    << " vision_only=" << (vision_only ? 1 : 0) << " options=" << (options ? 1 : 0)
    << " options_set=" << options_set << " symbolic_peer=" << (symbolic_peer ? 1 : 0)
    << " symbolic_weight_q20=" << symbolic_weight_q20 << " verifier=" << (verifier ? 1 : 0)
    << " verifier_mode=" << (verifier_mode == VerifierMode::Enforce ? "enforce" : "record");
  return o.str();
}

bool official_runner_clean(const RlFlags& f) {
  const RlFlags d{};
  return f.identity_line() == d.identity_line() && !f.any_on();
}

static bool starts_with(const std::string& s, const char* p) {
  const size_t n = std::char_traits<char>::length(p);
  return s.size() >= n && s.compare(0, n, p) == 0;
}

static bool take_eq(const std::string& a, const char* key, std::string& val) {
  const std::string k = std::string(key) + "=";
  if (starts_with(a, k.c_str())) {
    val = a.substr(k.size());
    return true;
  }
  return false;
}

RlParseResult parse_rl_flags(const std::vector<std::string>& args) {
  RlParseResult r;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (!starts_with(a, "--rl-")) continue;
    std::string val;
    if (a == "--rl-puct-model") {
      r.flags.puct_model = true;
      r.flags.puct_prior = PriorSrc::Model;
      r.flags.puct_value = ValueSrc::Model;
    } else if (take_eq(a, "--rl-puct-prior", val)) {
      if (val == "model") r.flags.puct_prior = PriorSrc::Model;
      else if (val == "current") r.flags.puct_prior = PriorSrc::Current;
      else {
        r.ok = false;
        r.err = "bad --rl-puct-prior (want model|current)";
        return r;
      }
    } else if (take_eq(a, "--rl-puct-value", val)) {
      if (val == "model") r.flags.puct_value = ValueSrc::Model;
      else if (val == "oracle") r.flags.puct_value = ValueSrc::Oracle;
      else if (val == "null") r.flags.puct_value = ValueSrc::Null;
      else {
        r.ok = false;
        r.err = "bad --rl-puct-value (want model|oracle|null)";
        return r;
      }
    } else if (a == "--rl-qbp") {
      r.flags.qbp = true;
    } else if (a == "--rl-qbp-write") {
      r.flags.qbp_write = true;
    } else if (take_eq(a, "--rl-qbp-min-truth", val)) {
      if (val == "sim") r.flags.qbp_min_truth = MinTruth::Sim;
      else if (val == "human") r.flags.qbp_min_truth = MinTruth::Human;
      else {
        r.ok = false;
        r.err = "bad --rl-qbp-min-truth (want sim|human)";
        return r;
      }
    } else if (a == "--rl-fork-score") {
      r.flags.fork_score = true;
    } else if (take_eq(a, "--rl-fork-horizon", val)) {
      char* end = nullptr;
      const unsigned long n = std::strtoul(val.c_str(), &end, 10);
      if (!end || *end || n == 0 || n > 16) {
        r.ok = false;
        r.err = "bad --rl-fork-horizon";
        return r;
      }
      r.flags.fork_horizon = uint32_t(n);
    } else if (take_eq(a, "--rl-fork-budget", val)) {
      char* end = nullptr;
      const unsigned long n = std::strtoul(val.c_str(), &end, 10);
      if (!end || *end || n > 1000000ul) {
        r.ok = false;
        r.err = "bad --rl-fork-budget";
        return r;
      }
      r.flags.fork_budget = uint32_t(n);
    } else if (a == "--rl-belief") {
      r.flags.belief = true;
    } else if (a == "--rl-vision-only") {
      r.flags.vision_only = true;
      r.flags.belief = true;
    } else if (a == "--rl-options") {
      r.flags.options = true;
    } else if (take_eq(a, "--rl-options-set", val)) {
      r.flags.options_set = val;
    } else if (a == "--rl-symbolic-peer") {
      r.flags.symbolic_peer = true;
    } else if (take_eq(a, "--rl-symbolic-weight", val)) {
      char* end = nullptr;
      const long n = std::strtol(val.c_str(), &end, 10);
      if (!end || *end || n < 0 || n > kQ20) {
        r.ok = false;
        r.err = "bad --rl-symbolic-weight (Q20)";
        return r;
      }
      r.flags.symbolic_weight_q20 = int32_t(n);
    } else if (a == "--rl-verifier") {
      r.flags.verifier = true;
    } else if (take_eq(a, "--rl-verifier-mode", val)) {
      if (val == "record") r.flags.verifier_mode = VerifierMode::Record;
      else if (val == "enforce") r.flags.verifier_mode = VerifierMode::Enforce;
      else {
        r.ok = false;
        r.err = "bad --rl-verifier-mode (want record|enforce)";
        return r;
      }
    } else {
      r.ok = false;
      r.err = "unknown reasoning-layer flag: " + a;
      return r;
    }
  }
  return r;
}

RlParseResult parse_rl_flags(int argc, char** argv) {
  std::vector<std::string> args;
  // skip argv[0]
  for (int i = 1; i < argc; ++i) {
    if (argv[i]) args.emplace_back(argv[i]);
  }
  return parse_rl_flags(args);
}

}  // namespace rl
}  // namespace pchie
