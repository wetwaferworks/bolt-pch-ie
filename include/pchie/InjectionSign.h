// Bolt PCHIE B6 - one untrusted observation (sign, book, or chat).
// Concrete final. Injected world text is always flagged untrusted. Fixture
// strings are unit-test / scenario observations only; they never feed a golden.
#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace pchie {

enum class InjectionSignKind : uint32_t { Sign = 1, Book = 2, Chat = 3 };

class InjectionSign final {
 public:
  InjectionSign() = default;
  InjectionSign(uint32_t id, InjectionSignKind kind, std::string text)
      : id_(id), kind_(kind), text_(std::move(text)) {}

  uint32_t id() const noexcept { return id_; }
  InjectionSignKind kind() const noexcept { return kind_; }
  const std::string& text() const { return text_; }
  // All injected observation text is untrusted (B6 harness rule).
  bool flagged_untrusted() const noexcept { return !text_.empty(); }

 private:
  uint32_t id_ = 0;
  InjectionSignKind kind_ = InjectionSignKind::Sign;
  std::string text_;
};

}  // namespace pchie
