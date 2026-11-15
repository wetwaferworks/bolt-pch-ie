// Context budget for the 1024 vs 2048 test path. At 1024 tokens Little Nemo gets the 32x12 fovea for
// forward, left and right; at 2048 he gets the 40x22 full forward view plus the 32x12 foveas left and right
// (both are sizes vision.capture_ascii gives).
// The token estimate is deliberately pessimistic for ASCII art (one token per two characters plus
// one per row), so a plan that "fits" here fits with the real tokenizer too.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace research {

struct ViewSlot {
  std::string name;
  int cols = 0;
  int rows = 0;
};

struct ContextPlan {
  uint32_t ctx = 1024;
  std::vector<ViewSlot> views;
  int view_tokens = 0;
  int instruction_tokens = 0;
  int answer_tokens = 0;
  int total_tokens = 0;
  bool fits = false;
};

/// Pessimistic token count for one ASCII view (glyph rows only).
int ascii_view_tokens(int cols, int rows);

/// The view layout and budget for ctx 1024 or 2048. Any other size returns fits == false.
ContextPlan context_plan(uint32_t ctx, int instruction_tokens = 160, int answer_tokens = 48);

std::string describe(const ContextPlan& p);

}  // namespace research
}  // namespace pchie
