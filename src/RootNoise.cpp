#include "pchie/RootNoise.h"
#include "pchie/DetLog.h"
#include "pchie/NodeSeed.h"

namespace pchie {

void RootNoise::Apply(std::vector<PuctEdge>& edges, uint64_t root_seed) const {
    if (num_ == 0 || den_ == 0 || edges.empty()) return;
    std::vector<double> g(edges.size()); double sum = 0.0;
    for (size_t i = 0; i < edges.size(); ++i) {           // edges are token-sorted: fixed order
        g[i] = -DetLog::Ln(NodeSeed::Uniform(NodeSeed::Splitmix64(root_seed ^ uint64_t(uint32_t(edges[i].token)))));
        sum = sum + g[i];
    }
    for (size_t i = 0; i < edges.size(); ++i) {
        const uint64_t noise_q20 = uint64_t((g[i] / sum) * double(PUCT_Q20));
        edges[i].prior_q20 = uint32_t((uint64_t(den_ - num_) * edges[i].prior_q20 + uint64_t(num_) * noise_q20) / den_);
    }
}

} // namespace pchie