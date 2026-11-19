/* Bolt PCHIE - ternary converter without the CUDA backend (the build found no CUDA compiler, or PCHIE_CUDA is OFF).
 * The CPU path is the whole converter; these entry points only say that CUDA is not there.
 */
#include "pchie/pchie_ternary_convert.h"
#include "pchie_ternary_convert_internal.h"
#include <string.h>

static ternary_cuda_counters g_counters;

ternary_cuda_info ternary_query_cuda_info(void) {
    ternary_cuda_info info;
    memset(&info, 0, sizeof(info));
    return info;
}

const ternary_cuda_counters *ternary_query_cuda_counters(void) { return &g_counters; }
void ternary_cuda_release(void) {}

ternary_convert_status ternary_fit_groups_cuda(const float *weights, uint64_t group_count, ternary_group_fit_kind fit_kind,
                                               int8_t *codes, float *scales, char *error, size_t error_capacity) {
    (void)weights;
    (void)group_count;
    (void)fit_kind;
    (void)codes;
    (void)scales;
    ternary_set_error(error, error_capacity, "this build has no CUDA backend");
    return TERNARY_CONVERT_BAD_ARGUMENT;
}
