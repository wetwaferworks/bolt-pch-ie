// Bolt PCHIE - ternary I2 row kernels (clean-room). This first step ships the scalar reference only; the SSSE3,
// AVX2 and NEON kernels come next and must match it bit for bit. Until then nothing here is dispatched.
#include "pchie_ternary_internal.h"

namespace pchie {
namespace ternary {
namespace detail {

float row_dot_i2_ssse3(const View&, uint64_t, const ActRow&) noexcept { return 0.0f; }   // never dispatched
float row_dot_i2_avx2(const View&, uint64_t, const ActRow&) noexcept { return 0.0f; }
float row_dot_i2_neon(const View&, uint64_t, const ActRow&) noexcept { return 0.0f; }

bool simd_compiled(Tier t) noexcept { return t == Tier::Scalar; }

}  // namespace detail
}  // namespace ternary
}  // namespace pchie
