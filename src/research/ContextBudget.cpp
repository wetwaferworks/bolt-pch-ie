#include "pchie/research/ContextBudget.h"

#include <sstream>

namespace pchie {
namespace research {

int ascii_view_tokens(int cols, int rows) {
  if (cols <= 0 || rows <= 0) return 0;
  return (cols * rows + 1) / 2 + rows + 8;  // + 8 for the view header line
}

ContextPlan context_plan(uint32_t ctx, int instruction_tokens, int answer_tokens) {
  ContextPlan p;
  p.ctx = ctx;
  if (ctx == 1024) {
    p.views = {{"fovea_forward", 32, 12}, {"fovea_left", 32, 12}, {"fovea_right", 32, 12}};
  } else if (ctx == 2048) {
    p.views = {{"forward", 40, 22}, {"fovea_left", 32, 12}, {"fovea_right", 32, 12}};  // the views vision.capture_ascii gives
  } else {
    return p;
  }
  for (const ViewSlot& v : p.views) p.view_tokens += ascii_view_tokens(v.cols, v.rows);
  p.instruction_tokens = instruction_tokens;
  p.answer_tokens = answer_tokens;
  p.total_tokens = p.view_tokens + instruction_tokens + answer_tokens;
  p.fits = p.total_tokens <= int(ctx);
  return p;
}

std::string describe(const ContextPlan& p) {
  std::ostringstream o;
  o << "ctx=" << p.ctx << " views=";
  for (size_t i = 0; i < p.views.size(); ++i)
    o << (i ? "," : "") << p.views[i].name << ':' << p.views[i].cols << 'x' << p.views[i].rows;
  o << " view_tokens=" << p.view_tokens << " total=" << p.total_tokens << " fits=" << (p.fits ? "yes" : "no");
  return o.str();
}

}  // namespace research
}  // namespace pchie
