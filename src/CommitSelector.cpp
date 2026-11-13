#include "pchie/CommitSelector.h"
#include "pchie/DetLog.h"
#include "pchie/NodeSeed.h"
#include <vector>

namespace pchie {

int CommitSelector::Select(const PuctSearch& s, uint32_t node) const {
    const PuctNode& nd = s.nodes[node];
    int best = -1; uint32_t bn = 0;
    for (size_t i = 0; i < nd.edges.size(); ++i) {           // token-sorted: first max = lowest token
        const uint32_t c = nd.edges[i].child;
        if (c != 0xFFFFFFFFu && s.nodes[c].n > bn) { bn = s.nodes[c].n; best = int(i); }
    }
    if (best < 0 || !(temp_ > 0.0)) return best;
    std::vector<double> w(nd.edges.size(), 0.0); double sum = 0.0;
    for (size_t i = 0; i < nd.edges.size(); ++i) {
        const uint32_t c = nd.edges[i].child;
        if (c != 0xFFFFFFFFu && s.nodes[c].n) w[i] = DetLog::Pow(double(s.nodes[c].n), 1.0 / temp_);
        sum = sum + w[i];
    }
    double u = NodeSeed::Uniform(NodeSeed::Splitmix64(nd.seed)) * sum;
    for (size_t i = 0; i < w.size(); ++i) { if (w[i] > 0.0 && u < w[i]) return int(i); u = u - w[i]; }
    return best;
}

} // namespace pchie