// Bolt PCHIE observation contract - UntrustedText: a distinct type for raw /
// observation / failure-trace text. Concrete final. No conversion to or from std::string or
// const char*, so it cannot reach IModel::tokenize() / tokenize_no_bos() (trusted framing) by
// accident; FailureTrace tokenizes it only through IModel::tokenize_untrusted().
#pragma once

#include <string>
#include <utility>

namespace pchie {

class UntrustedText final {
 public:
  explicit UntrustedText(std::string s) : s_(std::move(s)) {}
  explicit UntrustedText(const char* s) : s_(s ? s : "") {}
  const std::string& str() const noexcept { return s_; }

 private:
  std::string s_;
};

}  // namespace pchie