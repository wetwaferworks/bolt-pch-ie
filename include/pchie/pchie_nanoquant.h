/* Bolt PCHIE - low-rank binary factorization (NanoQuant LB-ADMM initialization).
 *
 * Clean-room C++ from the published method: Chong, Kim, Kim and Choi, "NanoQuant: Efficient
 * Sub-1-Bit Quantization of Large Language Models", ICML 2026, PMLR 306:20335-20364,
 * arXiv:2602.06694, official code at github.com/SamsungLabs/NanoQuant (Apache 2.0). The
 * official repository was read only as a reference for solver details that the paper
 * states at a high level (Sign-Value Independent Decomposition, the penalty schedule,
 * the stabilized Cholesky shift, and how the row and column scales are taken from mean
 * absolute values). No source from that tree is copied here.
 *
 * Each linear weight W (rows x cols, row-major) is approximated as
 *     W_hat[i,j] = s1[i] * (U[i,:] dot V[j,:]) * s2[j]
 * with U, V in {+1,-1} and full-precision scales. See docs/ternary/nanoquant.md.
 */
#ifndef PCHIE_NANOQUANT_H
#define PCHIE_NANOQUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { PCHIE_NQ_ERROR_CAPACITY = 512 };

typedef enum pchie_nq_status {
    PCHIE_NQ_OK = 0,
    PCHIE_NQ_BAD_ARGUMENT = 1,
    PCHIE_NQ_TOO_LARGE = 2,
    PCHIE_NQ_NO_MEMORY = 3
} pchie_nq_status;

/* use_cuda: 0 = CPU only; 1 = CUDA if a device is present, otherwise CPU (default);
 * 2 = CUDA required (fails if this build has no CUDA or the tensor does not fit). */
typedef struct pchie_nq_config {
    float bits;
    int rank;
    int outer_iters;
    int inner_iters;
    float reg;
    float eps;
    int rho_cubic;
    uint64_t seed;
    int transpose_wide;
    int use_cuda;
} pchie_nq_config;

typedef struct pchie_nq_stats {
    int rows;
    int cols;
    int rank;
    uint64_t weight_count;
    uint64_t packed_bytes;
    double bits_per_weight;
    double bits_per_weight_fp16;
    double cosine;
    double relative_error;
    int transposed;
    int backend;            /* 0 = CPU, 1 = CUDA */
    uint64_t device_bytes;  /* peak device memory this call held, 0 on CPU */
} pchie_nq_stats;

typedef struct pchie_nq_cuda_info {
    int built;              /* compiled with the CUDA factorizer */
    int device_count;
    char device_name[256];
    uint64_t total_bytes;
    uint64_t free_bytes;
} pchie_nq_cuda_info;

void pchie_nq_config_init(pchie_nq_config *config);
int pchie_nq_choose_rank(int rows, int cols, float bits);
uint64_t pchie_nq_packed_bytes(int rows, int cols, int rank, int scale_bytes);
pchie_nq_cuda_info pchie_nq_query_cuda(void);
int pchie_nq_cuda_available(void);

pchie_nq_status pchie_nq_factorize(const float *w, int rows, int cols, const float *d_in, const float *d_out,
                                   const pchie_nq_config *config, int8_t *u_sign, int8_t *v_sign, float *s1,
                                   float *s2, pchie_nq_stats *stats, char *error, size_t error_capacity);
void pchie_nq_accumulate_error(const float *w, const int8_t *u_sign, const int8_t *v_sign, const float *s1,
                               const float *s2, int rows, int cols, int rank, pchie_nq_stats *stats);
void pchie_nq_pack_signs(const int8_t *signs, uint64_t count, uint8_t *bytes);
void pchie_nq_unpack_signs(const uint8_t *bytes, uint64_t count, int8_t *signs);

#ifdef __cplusplus
}
#endif

#endif
