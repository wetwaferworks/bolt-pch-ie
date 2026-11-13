// Bolt PCHIE - per-node seed tied to the tree coordinate (Archie ruling 5):
//   seed(child) = splitmix64(parent_seed ^ token_id ^ depth)
// Used only for root exploration noise, exact-tie breaks in quantized logits, and commit-time
// selection at T > 0. Expansion itself is deterministic top-K: nothing is sampled from it.
#pragma once
#include <cstdint>

namespace pchie {

class NodeSeed final {
public:
    static uint64_t Splitmix64(uint64_t x) noexcept {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }
    static uint64_t Child(uint64_t parent_seed, int32_t token, uint32_t depth) noexcept {
        return Splitmix64(parent_seed ^ uint64_t(uint32_t(token)) ^ uint64_t(depth));
    }
    // Uniform in (0, 1): 53 bits, never exactly 0.
    static double Uniform(uint64_t x) noexcept { return (double(x >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
};

} // namespace pchie