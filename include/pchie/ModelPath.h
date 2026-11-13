#pragma once
// Resolve the Little Nemo GGUF from the environment, the same way the launchers do.
// Set PCHIE_MODEL to the weights file, or PCHIE_MODELS to the directory that holds
// Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf. There is no machine-specific default; a missing
// or empty setting is an error the caller must report.
#include <cstdlib>
#include <filesystem>
#include <string>

namespace pchie {

inline constexpr const char* kDefaultGgufName = "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf";

inline bool resolve_model_path(std::string& path, std::string& err) {
  path.clear();
  err.clear();
  const char* file = std::getenv("PCHIE_MODEL");
  const char* dir = std::getenv("PCHIE_MODELS");
  std::filesystem::path p;
  if (file && file[0]) p = file;
  else if (dir && dir[0]) p = std::filesystem::path(dir) / kDefaultGgufName;
  else {
    err = "Set PCHIE_MODEL to your GGUF weights, or set PCHIE_MODELS to the directory that holds Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf.";
    return false;
  }
  std::error_code ec;
  if (!std::filesystem::exists(p, ec) || !std::filesystem::is_regular_file(p, ec)) {
    err = "Model file not found: " + p.string() + ". Set PCHIE_MODEL or PCHIE_MODELS to a valid path.";
    return false;
  }
  path = std::filesystem::weakly_canonical(p, ec).string();
  if (path.empty()) path = p.string();
  return true;
}

}  // namespace pchie
