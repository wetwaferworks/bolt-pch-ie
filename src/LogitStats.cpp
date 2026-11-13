#include "pchie/LogitStats.h"
#include "pchie/DetLog.h"
#include "pchie/pchie_det_math.h"

namespace pchie {

LogitStats::LogitStats(const std::vector<float>& logits) : logits_(logits) {
    max_ = logits.empty() ? 0.0f : logits[0];
    for (float f : logits) if (f > max_) max_ = f;
    double z = 0.0, s = 0.0;
    for (float f : logits) {
        const float d = f - max_;
        const double e = double(exp_deterministic(d));
        z = z + e; s = s + e * double(d);
    }
    log_z_ = DetLog::Ln(z);
    entropy_ = z > 0.0 ? log_z_ - s / z : 0.0;
    if (entropy_ < 0.0) entropy_ = 0.0;
}

double LogitStats::LogProb(int32_t token) const {
    return double(logits_[size_t(token)] - max_) - log_z_;
}

} // namespace pchie