// Bolt PCHIE - ternary converter without the CUDA backend (the build found no CUDA compiler, or PCHIE_CUDA is OFF).
// The CPU path is the whole converter; these entry points only say that CUDA is not there.
#include "pchie/pchie_ternary_convert.h"

namespace pchie {
namespace ternary {
namespace convert {

namespace { CudaCounters g_counters; }

CudaInfo cuda_info() { return CudaInfo(); }
const CudaCounters& cuda_counters() noexcept { return g_counters; }
void cuda_release() noexcept {}
Status fit_groups_cuda(const float*, uint64_t, FitKind, int8_t*, float*, std::string& err) {
    err = "this build has no CUDA backend";
    return Status::BadArgument;
}

} // namespace convert
} // namespace ternary
} // namespace pchie
