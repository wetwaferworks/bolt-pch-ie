// Bolt PCHIE - the gates on --neml-inject (pchie_lk_visnav), which puts the NEML memory block in front of the
// trusted rules prefix. The block holds free text, including notes a model wrote, so the flag is a path from
// earlier model output into the trusted prefix. It stays off by default and needs two independent gates:
//   1. the build: only a build configured with -DPCHIE_ALLOW_NEML_INJECT=ON (off by default) accepts the flag;
//   2. the output: the run's --out folder must sit under a folder whose name starts with "scratch", so an
//      injected run can never write where official records live.
// A run that passes both records neml_inject=1 and the SHA-256 of the injected block as the first link of its
// side chain and in its report. Without the flag nothing changes.
#pragma once
#include <cctype>
#include <filesystem>
#include <string>
#include <system_error>

namespace pchie {
namespace neml {

inline bool inject_build_allowed() {
#ifdef PCHIE_ALLOW_NEML_INJECT
  return true;
#else
  return false;
#endif
}

// True when some folder in the absolute, normalized form of out (not counting out's own last part when out is a
// plain file name) is named scratch* (any case). ".." is resolved before the check, so "scratch/../official" is
// not scratch.
inline bool is_scratch_path(const std::string& out) {
  if (out.empty()) return false;
  std::error_code ec;
  std::filesystem::path p = std::filesystem::weakly_canonical(std::filesystem::absolute(out, ec), ec);
  if (ec) return false;
  for (const auto& part : p) {
    std::string s = part.string();
    for (auto& ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    if (s.rfind("scratch", 0) == 0) return true;
  }
  return false;
}

// Both gates. requested=false always passes (the default run is untouched).
inline bool inject_allowed(bool requested, bool build_allowed, const std::string& out, std::string& why) {
  if (!requested) return true;
  if (!build_allowed) {
    why = "--neml-inject is refused: this build was made without PCHIE_ALLOW_NEML_INJECT (it is off by default)";
    return false;
  }
  if (!is_scratch_path(out)) {
    why = "--neml-inject is refused: --out must be under a folder named scratch*, never an official output folder";
    return false;
  }
  return true;
}

}  // namespace neml
}  // namespace pchie
