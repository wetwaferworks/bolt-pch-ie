#include "pchie/LogProbLeafEvaluator.h"
#include "pchie/DetLog.h"

namespace pchie {

double LogProbLeafEvaluator::EvalLeaf(const LeafContext& ctx) const {
    if (!ctx.gate_pass) return 0.0;
    const double L = ctx.depth > 0 ? double(ctx.depth) : 1.0;
    const double x = ctx.sum_log_prob / DetLog::Pow(L, alpha_) - beta_ * ctx.entropy;
    return DetLog::Sigmoid(x);
}

} // namespace pchie