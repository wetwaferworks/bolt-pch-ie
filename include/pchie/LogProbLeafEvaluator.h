// Bolt PCHIE - default leaf value (Archie ruling 1), concrete and non-virtual (search hot path):
//   V = sigmoid( (1 / L^alpha) * sum ln P - beta * H ) * G.
// Host CPU only; exp/log/pow/sigmoid come from DetMath (our own polynomials), never std or CUDA.
// alpha and beta are supplied by the caller: the maintainer locks the values.
#pragma once
#include "pchie/LeafContext.h"

namespace pchie {

class LogProbLeafEvaluator final {
public:
    LogProbLeafEvaluator(double alpha, double beta) : alpha_(alpha), beta_(beta) {}
    double EvalLeaf(const LeafContext& ctx) const;   // value in [0, 1]
    double Alpha() const noexcept { return alpha_; }
    double Beta() const noexcept { return beta_; }
private:
    double alpha_, beta_;
};

} // namespace pchie