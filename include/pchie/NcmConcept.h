// Bolt PCHIE - one NCM concept: a named, bounded token span with a stable id and a score.
// Concrete final. No mutex. Token expansion of a concept is the tokens() sequence itself.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class NcmConcept final {
 public:
  NcmConcept() = default;
  NcmConcept(uint32_t id, std::string name, std::vector<int32_t> tokens, double score)
      : id_(id), name_(std::move(name)), tokens_(std::move(tokens)), score_(score) {}

  uint32_t id() const noexcept { return id_; }
  const std::string& name() const noexcept { return name_; }
  const std::vector<int32_t>& tokens() const noexcept { return tokens_; }
  double score() const noexcept { return score_; }

  bool operator==(const NcmConcept& o) const {
    return id_ == o.id_ && name_ == o.name_ && tokens_ == o.tokens_ && score_ == o.score_;
  }
  bool operator!=(const NcmConcept& o) const { return !(*this == o); }

 private:
  uint32_t id_ = 0;
  std::string name_;
  std::vector<int32_t> tokens_;
  double score_ = 0.0;
};

}  // namespace pchie