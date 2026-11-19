/* Bolt PCHIE - NanoQuant LB-ADMM initialization (clean-room C++). See pchie_nanoquant.h. */
#include "pchie/pchie_nanoquant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#ifdef PCHIE_NANOQUANT_CUDA
#include <cublas_v2.h>
#endif

static void set_error(char *error, size_t cap, const char *msg) {
    if (!error || cap == 0) return;
    std::snprintf(error, cap, "%s", msg ? msg : "");
}

static void zero_outputs(int8_t *u_sign, int8_t *v_sign, float *s1, float *s2, int rows, int cols, int rank,
                         pchie_nq_stats *stats) {
    const uint64_t ur = (uint64_t)rows * (uint64_t)(rank > 0 ? rank : 0);
    const uint64_t vr = (uint64_t)cols * (uint64_t)(rank > 0 ? rank : 0);
    if (u_sign && ur) std::memset(u_sign, 0, (size_t)ur);
    if (v_sign && vr) std::memset(v_sign, 0, (size_t)vr);
    if (s1 && rows > 0) std::memset(s1, 0, (size_t)rows * sizeof(float));
    if (s2 && cols > 0) std::memset(s2, 0, (size_t)cols * sizeof(float));
    if (stats) std::memset(stats, 0, sizeof(*stats));
}

void pchie_nq_config_init(pchie_nq_config *config) {
    if (!config) return;
    std::memset(config, 0, sizeof(*config));
    config->bits = 1.0f;
    config->rank = 0;
    config->outer_iters = 400;
    config->inner_iters = 5;
    config->reg = 0.03f;
    config->eps = 1e-12f;
    config->rho_cubic = 0;
    config->seed = 0;
    config->transpose_wide = 1;
}

int pchie_nq_choose_rank(int rows, int cols, float bits) {
    if (rows <= 0 || cols <= 0 || !(bits > 0.0f) || !std::isfinite(bits)) return 0;
    const double a = (double)rows;
    const double b = (double)cols;
    double rank = (bits * a * b) / (a + b) - 16.0;
    int r = (int)rank;
    if (r < 0) r = 0;
    r = (r / 32) * 32;
    if (r == 0) r = 32;
    const int cap = rows < cols ? rows : cols;
    if (r > cap) r = cap;
    if (r <= 0) r = cap < 32 ? cap : 32;
    return r;
}

uint64_t pchie_nq_packed_bytes(int rows, int cols, int rank, int scale_bytes) {
    if (rows <= 0 || cols <= 0 || rank <= 0 || scale_bytes <= 0) return 0;
    const uint64_t ubits = (uint64_t)rows * (uint64_t)rank;
    const uint64_t vbits = (uint64_t)cols * (uint64_t)rank;
    const uint64_t ubytes = (ubits + 7) / 8;
    const uint64_t vbytes = (vbits + 7) / 8;
    const uint64_t scales = (uint64_t)scale_bytes * ((uint64_t)rows + (uint64_t)cols);
    return ubytes + vbytes + scales;
}

void pchie_nq_pack_signs(const int8_t *signs, uint64_t count, uint8_t *bytes) {
    if (!signs || !bytes) return;
    const uint64_t nbytes = (count + 7) / 8;
    std::memset(bytes, 0, (size_t)nbytes);
    for (uint64_t i = 0; i < count; ++i) {
        if (signs[i] > 0) bytes[i / 8] = (uint8_t)(bytes[i / 8] | (uint8_t)(1u << (i % 8)));
    }
}

void pchie_nq_unpack_signs(const uint8_t *bytes, uint64_t count, int8_t *signs) {
    if (!bytes || !signs) return;
    for (uint64_t i = 0; i < count; ++i) {
        signs[i] = (bytes[i / 8] & (uint8_t)(1u << (i % 8))) ? (int8_t)1 : (int8_t)-1;
    }
}

/* ---------- RNG (SplitMix64 + Box-Muller) ---------- */

struct NqRng {
    uint64_t state;
};

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double rng_uniform(NqRng *rng) {
    return (splitmix64(&rng->state) >> 11) * (1.0 / 9007199254740992.0);
}

static float rng_normal(NqRng *rng) {
    double u, v, s;
    do {
        u = 2.0 * rng_uniform(rng) - 1.0;
        v = 2.0 * rng_uniform(rng) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    return (float)(u * std::sqrt(-2.0 * std::log(s) / s));
}

static void fill_randn(float *a, uint64_t n, NqRng *rng) {
    for (uint64_t i = 0; i < n; ++i) a[i] = rng_normal(rng);
}

/* ---------- small dense linear algebra (row-major) ---------- */

static float dot(const float *a, const float *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

static float nrm2(const float *a, int n) {
    return std::sqrt(dot(a, a, n));
}

/* C[m,n] = alpha * op(A) * op(B) + beta * C. All row-major.
 * transA/transB 0: A is m x k / B is k x n; 1: A is k x m / B is n x k. */
static void gemm(int transA, int transB, int m, int n, int k, float alpha,
                 const float *A, const float *B, float beta, float *C) {
#ifdef PCHIE_NANOQUANT_CUDA
    /* Used only when a handle is set; otherwise fall through to CPU. */
#endif
    if (beta == 0.0f) {
        std::memset(C, 0, (size_t)m * (size_t)n * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0, t = m * n; i < t; ++i) C[i] *= beta;
    }
    if (alpha == 0.0f) return;

    const int lda = transA ? m : k;
    const int ldb = transB ? k : n;
    /* Tiled over i,j to keep the inner product in cache. */
    const int TS = 64;
    for (int i0 = 0; i0 < m; i0 += TS) {
        const int i1 = i0 + TS < m ? i0 + TS : m;
        for (int j0 = 0; j0 < n; j0 += TS) {
            const int j1 = j0 + TS < n ? j0 + TS : n;
            for (int p0 = 0; p0 < k; p0 += TS) {
                const int p1 = p0 + TS < k ? p0 + TS : k;
                for (int i = i0; i < i1; ++i) {
                    for (int p = p0; p < p1; ++p) {
                        const float a = alpha * (transA ? A[p * lda + i] : A[i * lda + p]);
                        float *crow = C + (size_t)i * (size_t)n;
                        if (!transB) {
                            const float *brow = B + (size_t)p * (size_t)ldb;
                            for (int j = j0; j < j1; ++j) crow[j] += a * brow[j];
                        } else {
                            for (int j = j0; j < j1; ++j) crow[j] += a * B[(size_t)j * (size_t)ldb + (size_t)p];
                        }
                    }
                }
            }
        }
    }
}

#ifdef PCHIE_NANOQUANT_CUDA
static cublasHandle_t g_blas = nullptr;
static int cublas_ok() {
    if (g_blas) return 1;
    if (cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) {
        g_blas = nullptr;
        return 0;
    }
    return 1;
}

static int gemm_cuda(int transA, int transB, int m, int n, int k, float alpha,
                     const float *A, const float *B, float beta, float *C) {
    if (!cublas_ok()) return 0;
    /* cublas is column-major; row-major (m x n) = col-major (n x m) of the transpose.
     * C_rm = opA(A) opB(B)  <=>  C_cm^T = opB(B)^T opA(A)^T. */
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t as = (size_t)(transA ? k : m) * (size_t)(transA ? m : k) * sizeof(float);
    const size_t bs = (size_t)(transB ? n : k) * (size_t)(transB ? k : n) * sizeof(float);
    const size_t cs = (size_t)m * (size_t)n * sizeof(float);
    if (cudaMalloc(&dA, as) != cudaSuccess) return 0;
    if (cudaMalloc(&dB, bs) != cudaSuccess) { cudaFree(dA); return 0; }
    if (cudaMalloc(&dC, cs) != cudaSuccess) { cudaFree(dA); cudaFree(dB); return 0; }
    cudaMemcpy(dA, A, as, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bs, cudaMemcpyHostToDevice);
    if (beta != 0.0f) cudaMemcpy(dC, C, cs, cudaMemcpyHostToDevice);
    const cublasOperation_t opA = transA ? CUBLAS_OP_N : CUBLAS_OP_T;
    const cublasOperation_t opB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int lda = transA ? m : k;
    const int ldb = transB ? k : n;
    /* We compute n x m in column-major: C^T = op(B^T) * op(A^T). */
    const cublasStatus_t st = cublasSgemm(g_blas, opB, opA, n, m, k, &alpha, dB, ldb, dA, lda, &beta, dC, n);
    int ok = (st == CUBLAS_STATUS_SUCCESS);
    if (ok) cudaMemcpy(C, dC, cs, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok;
}
#endif

static void gemm_auto(int transA, int transB, int m, int n, int k, float alpha,
                      const float *A, const float *B, float beta, float *C) {
#ifdef PCHIE_NANOQUANT_CUDA
    const uint64_t flops = 2ull * (uint64_t)m * (uint64_t)n * (uint64_t)k;
    if (flops >= 8ull * 1000000ull && gemm_cuda(transA, transB, m, n, k, alpha, A, B, beta, C)) return;
#endif
    gemm(transA, transB, m, n, k, alpha, A, B, beta, C);
}

/* Cholesky solve (A + stab I) X = B, A is r x r symmetric (we form it as XtX), B and X are r x nrhs, row-major. */
static int cholesky_solve(int r, float *A, int nrhs, float *B, float stab, float eps) {
    /* Symmetrize and add stabilizer on the diagonal. */
    for (int i = 0; i < r; ++i) {
        for (int j = i + 1; j < r; ++j) {
            const float s = 0.5f * (A[i * r + j] + A[j * r + i]);
            A[i * r + j] = A[j * r + i] = s;
        }
        float d = A[i * r + i] + stab;
        if (d < eps) d = eps;
        A[i * r + i] = d;
    }
    /* In-place Cholesky (lower). */
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i * r + j];
            for (int p = 0; p < j; ++p) s -= (double)A[i * r + p] * (double)A[j * r + p];
            if (i == j) {
                if (s < (double)eps) s = (double)eps;
                A[i * r + i] = (float)std::sqrt(s);
            } else {
                A[i * r + j] = (float)(s / (double)A[j * r + j]);
            }
        }
        for (int j = i + 1; j < r; ++j) A[i * r + j] = 0.0f;
    }
    /* Solve L Y = B, then L^T X = Y, overwriting B. */
    for (int c = 0; c < nrhs; ++c) {
        for (int i = 0; i < r; ++i) {
            double s = B[i * nrhs + c];
            for (int p = 0; p < i; ++p) s -= (double)A[i * r + p] * (double)B[p * nrhs + c];
            B[i * nrhs + c] = (float)(s / (double)A[i * r + i]);
        }
        for (int i = r - 1; i >= 0; --i) {
            double s = B[i * nrhs + c];
            for (int p = i + 1; p < r; ++p) s -= (double)A[p * r + i] * (double)B[p * nrhs + c];
            B[i * nrhs + c] = (float)(s / (double)A[i * r + i]);
        }
    }
    return 1;
}

/* Solve (X^T X + stab I) Factor = X^T Y + rho (Z - U)
 * X is n x r, Y is n x m, Z,U,Factor are r x m? In the official code Factor has the same shape as Z (which is
 * passed as A.T so r x out). We'll keep:
 *   X: n x r  (row-major)
 *   Y: n x m
 *   Z, U, Out: r x m   ... actually official Z is mid x out if we pass A.T, but then they transpose back.
 * Easier: work with factors as (dim x rank) row-major, matching U (rows x r).
 *
 * Official: (X^T X + stab I) Factor = X^T Y + rho (Z - U) with Factor, Z, U all (k x m) if Y is (n x m), X is (n x k).
 * Then A = Factor^T is (m x k).
 *
 * We'll implement that and transpose around it.
 */
static void admm_solve_factor(const float *X, int n, int r, const float *Y, int m,
                              const float *Z, const float *U, float rho, float reg, float eps, float *Out) {
    /* system r x r = X^T X */
    std::vector<float> sys((size_t)r * (size_t)r, 0.0f);
    gemm_auto(1, 0, r, r, n, 1.0f, X, X, 0.0f, sys.data());
    double diag_mean = 0.0;
    for (int i = 0; i < r; ++i) diag_mean += std::fabs((double)sys[i * r + i]);
    diag_mean = (r > 0) ? diag_mean / (double)r : 0.0;
    float stab = (float)(rho * diag_mean + (double)reg);
    if (stab < eps) stab = eps;

    /* rhs = X^T Y + rho (Z - U), shape r x m */
    std::vector<float> rhs((size_t)r * (size_t)m, 0.0f);
    gemm_auto(1, 0, r, m, n, 1.0f, X, Y, 0.0f, rhs.data());
    const float prho = rho;
    for (int i = 0, t = r * m; i < t; ++i) rhs[i] += prho * (Z[i] - U[i]);

    cholesky_solve(r, sys.data(), m, rhs.data(), stab, eps);
    std::memcpy(Out, rhs.data(), (size_t)r * (size_t)m * sizeof(float));
}

/* SVID rank-1 approximation of a p x q matrix: sign(W) * outer(u, v) of |W|. */
static void svid_rank1(const float *W, int p, int q, int iters, NqRng *rng, float eps, float *Z) {
    std::vector<float> absW((size_t)p * (size_t)q);
    for (int i = 0, t = p * q; i < t; ++i) absW[i] = std::fabs(W[i]);
    std::vector<float> v((size_t)q), u((size_t)p);
    fill_randn(v.data(), (uint64_t)q, rng);
    float vn = nrm2(v.data(), q);
    if (vn < eps) v[0] = 1.0f;
    else for (int j = 0; j < q; ++j) v[j] /= vn;
    for (int t = 0; t < iters; ++t) {
        /* u = absW * v */
        std::memset(u.data(), 0, (size_t)p * sizeof(float));
        for (int i = 0; i < p; ++i) {
            const float *row = absW.data() + (size_t)i * (size_t)q;
            u[i] = dot(row, v.data(), q);
        }
        float un = nrm2(u.data(), p);
        if (un < eps) u[0] = 1.0f;
        else for (int i = 0; i < p; ++i) u[i] /= un;
        /* v = absW^T * u */
        std::memset(v.data(), 0, (size_t)q * sizeof(float));
        for (int i = 0; i < p; ++i) {
            const float *row = absW.data() + (size_t)i * (size_t)q;
            const float ui = u[i];
            for (int j = 0; j < q; ++j) v[j] += ui * row[j];
        }
        vn = nrm2(v.data(), q);
        if (vn < eps) v[0] = 1.0f;
        else for (int j = 0; j < q; ++j) v[j] /= vn;
    }
    /* sigma = |absW v|, u_scaled = (absW v) */
    std::vector<float> Av((size_t)p, 0.0f);
    for (int i = 0; i < p; ++i) Av[i] = dot(absW.data() + (size_t)i * (size_t)q, v.data(), q);
    const float sigma = nrm2(Av.data(), p);
    for (int i = 0; i < p; ++i) {
        const float ui = Av[i]; /* already u * sigma */
        const float *row = W + (size_t)i * (size_t)q;
        float *zrow = Z + (size_t)i * (size_t)q;
        for (int j = 0; j < q; ++j) {
            const float sg = (row[j] >= 0.0f) ? 1.0f : -1.0f;
            zrow[j] = ui * v[j] * sg;
        }
    }
    (void)sigma;
}

static void col_normalize_transpose(const float *B, int r, int n, float eps, float *X) {
    /* B is r x n (rows = rank components). X is n x r, X[:,k] = B[k,:]^T / ||B[k,:]|| */
    for (int k = 0; k < r; ++k) {
        const float *brow = B + (size_t)k * (size_t)n;
        float nrm = nrm2(brow, n);
        if (nrm < eps) nrm = eps;
        const float inv = 1.0f / nrm;
        for (int j = 0; j < n; ++j) X[(size_t)j * (size_t)r + (size_t)k] = brow[j] * inv;
    }
}

static void row_normalize_as_X(const float *A, int m, int r, float eps, float *X) {
    /* A is m x r. X is m x r, X[:,k] = A[:,k] / ||A[:,k]|| */
    std::vector<float> nrm((size_t)r, 0.0f);
    for (int i = 0; i < m; ++i) {
        const float *arow = A + (size_t)i * (size_t)r;
        for (int k = 0; k < r; ++k) nrm[k] += arow[k] * arow[k];
    }
    for (int k = 0; k < r; ++k) {
        float s = std::sqrt(nrm[k]);
        if (s < eps) s = eps;
        nrm[k] = 1.0f / s;
    }
    for (int i = 0; i < m; ++i) {
        const float *arow = A + (size_t)i * (size_t)r;
        float *xrow = X + (size_t)i * (size_t)r;
        for (int k = 0; k < r; ++k) xrow[k] = arow[k] * nrm[k];
    }
}

static void transpose(const float *A, int m, int n, float *At) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            At[(size_t)j * (size_t)m + (size_t)i] = A[(size_t)i * (size_t)n + (size_t)j];
}

static float frobenius(const float *A, uint64_t n) {
    double s = 0.0;
    for (uint64_t i = 0; i < n; ++i) s += (double)A[i] * (double)A[i];
    return (float)std::sqrt(s);
}

void pchie_nq_accumulate_error(const float *w, const int8_t *u_sign, const int8_t *v_sign, const float *s1,
                               const float *s2, int rows, int cols, int rank, pchie_nq_stats *stats) {
    if (!w || !u_sign || !v_sign || !s1 || !s2 || !stats || rows <= 0 || cols <= 0 || rank <= 0) return;
    double sw2 = 0.0, se2 = 0.0, swr = 0.0, sr2 = 0.0;
    for (int i = 0; i < rows; ++i) {
        const int8_t *ui = u_sign + (size_t)i * (size_t)rank;
        const float si = s1[i];
        for (int j = 0; j < cols; ++j) {
            const int8_t *vj = v_sign + (size_t)j * (size_t)rank;
            int acc = 0;
            for (int k = 0; k < rank; ++k) acc += (int)ui[k] * (int)vj[k];
            const double what = (double)si * (double)acc * (double)s2[j];
            const double ww = (double)w[(size_t)i * (size_t)cols + (size_t)j];
            const double e = ww - what;
            sw2 += ww * ww;
            se2 += e * e;
            swr += ww * what;
            sr2 += what * what;
        }
    }
    stats->weight_count = (uint64_t)rows * (uint64_t)cols;
    const double den = std::sqrt(sw2) * std::sqrt(sr2);
    stats->cosine = (den > 0.0) ? (swr / den) : 0.0;
    stats->relative_error = (sw2 > 0.0) ? std::sqrt(se2 / sw2) : 0.0;
}

static float rho_at(const pchie_nq_config *c, int it, int n) {
    if (n <= 0) return 1.0f;
    const double x = (double)it / (double)n;
    if (c->rho_cubic) {
        const double y = x < 1.0 ? x : 1.0;
        return (float)(y * y * y);
    }
    return (float)x;
}

static pchie_nq_status factorize_core(const float *W, int rows, int cols, const float *d_in, const float *d_out,
                                      const pchie_nq_config *cfg, int rank, int8_t *u_sign, int8_t *v_sign,
                                      float *s1, float *s2, pchie_nq_stats *stats, char *error, size_t errcap) {
    const float eps = cfg->eps > 0.0f ? cfg->eps : 1e-12f;
    const int r = rank;
    const int m = rows;
    const int n = cols;
    std::vector<float> Wn((size_t)m * (size_t)n);
    std::vector<float> norm_i((size_t)n, 1.0f), norm_o((size_t)m, 1.0f);
    for (int j = 0; j < n; ++j) {
        const float v = d_in ? d_in[j] : 1.0f;
        if (!(v >= 0.0f) || !std::isfinite(v)) {
            set_error(error, errcap, "a Hessian diagonal entry is not a finite non-negative number");
            return PCHIE_NQ_BAD_ARGUMENT;
        }
        float s = std::sqrt(v);
        if (s < eps) s = eps;
        norm_i[j] = s;
    }
    for (int i = 0; i < m; ++i) {
        const float v = d_out ? d_out[i] : 1.0f;
        if (!(v >= 0.0f) || !std::isfinite(v)) {
            set_error(error, errcap, "a Hessian diagonal entry is not a finite non-negative number");
            return PCHIE_NQ_BAD_ARGUMENT;
        }
        float s = std::sqrt(v);
        if (s < eps) s = eps;
        norm_o[i] = s;
    }
    for (int i = 0; i < m; ++i) {
        const float no = norm_o[i];
        for (int j = 0; j < n; ++j) Wn[(size_t)i * (size_t)n + (size_t)j] = W[(size_t)i * (size_t)n + (size_t)j] * no * norm_i[j];
    }

    NqRng rng{cfg->seed ? cfg->seed : 0xA5F1C3D2E4B60789ULL};
    /* Advance a little so seed 0 is still mixed. */
    splitmix64(&rng.state);

    std::vector<float> A((size_t)m * (size_t)r), B((size_t)r * (size_t)n);
    std::vector<float> Az((size_t)m * (size_t)r), Bz((size_t)r * (size_t)n);
    std::vector<float> Au((size_t)m * (size_t)r), Bu((size_t)r * (size_t)n);
    fill_randn(A.data(), (uint64_t)m * (uint64_t)r, &rng);
    fill_randn(B.data(), (uint64_t)r * (uint64_t)n, &rng);
    if (cfg->outer_iters > 0) {
        svid_rank1(A.data(), m, r, cfg->inner_iters, &rng, eps, Az.data());
        /* B is stored r x n; SVID on that layout. */
        svid_rank1(B.data(), r, n, cfg->inner_iters, &rng, eps, Bz.data());
    } else {
        Az = A;
        Bz = B;
    }
    for (int i = 0, t = m * r; i < t; ++i) Au[i] = A[i] - Az[i];
    for (int i = 0, t = r * n; i < t; ++i) Bu[i] = B[i] - Bz[i];

    std::vector<float> XA((size_t)n * (size_t)r);
    std::vector<float> XB((size_t)m * (size_t)r);
    std::vector<float> WT((size_t)n * (size_t)m);
    transpose(Wn.data(), m, n, WT.data());
    std::vector<float> AzT((size_t)r * (size_t)m), AuT((size_t)r * (size_t)m), AlsT((size_t)r * (size_t)m);

    const int iters = cfg->outer_iters < 0 ? 0 : cfg->outer_iters;
    for (int it = 0; it < iters; ++it) {
        const float rho = rho_at(cfg, it, iters);
        /* X_A = Bz^T / row-norms of Bz  (n x r) */
        col_normalize_transpose(Bz.data(), r, n, eps, XA.data());
        transpose(Az.data(), m, r, AzT.data());
        transpose(Au.data(), m, r, AuT.data());
        /* A_ls^T = solve(X_A, W^T, Az^T, Au^T) */
        admm_solve_factor(XA.data(), n, r, WT.data(), m, AzT.data(), AuT.data(), rho, cfg->reg, eps, AlsT.data());
        transpose(AlsT.data(), r, m, A.data());

        /* X_B = Az / col-norms (m x r), B_ls = solve(X_B, W, Bz, Bu) with B as r x n so
         * official: X_B is (out, mid), Y is W (out, in), Z is B (mid, in)? 
         * Official B_ls = solve(X_B, W_norm, B_z, B_u) with B all (mid, in).
         * X is (out, mid)= (m, r), Y is (m, n), Z,U are (r, n)? That would be Factor (k x m) if Y is n x m...
         * Official: X_B = A_z / mid_norm (out, mid), Y = W (out, in), Z=B_z (mid, in).
         * _admm_solve_step: (X^T X + stab I) Factor = X^T Y + rho(Z-U)
         * X is (out, mid) so n_x=out, k=mid; Y is (out, in) so m=in; Factor is (mid, in). Yes B is r x n.
         */
        row_normalize_as_X(Az.data(), m, r, eps, XB.data());
        admm_solve_factor(XB.data(), m, r, Wn.data(), n, Bz.data(), Bu.data(), rho, cfg->reg, eps, B.data());

        /* Z = SVID(A + Au), SVID(B + Bu) */
        std::vector<float> PA((size_t)m * (size_t)r), PB((size_t)r * (size_t)n);
        for (int i = 0, t = m * r; i < t; ++i) PA[i] = A[i] + Au[i];
        for (int i = 0, t = r * n; i < t; ++i) PB[i] = B[i] + Bu[i];
        svid_rank1(PA.data(), m, r, cfg->inner_iters, &rng, eps, Az.data());
        svid_rank1(PB.data(), r, n, cfg->inner_iters, &rng, eps, Bz.data());
        for (int i = 0, t = m * r; i < t; ++i) Au[i] += A[i] - Az[i];
        for (int i = 0, t = r * n; i < t; ++i) Bu[i] += B[i] - Bz[i];
    }

    /* Undo preconditioning: A_unb[i,k] = Az[i,k] / norm_o[i]; B_unb[k,j] = Bz[k,j] / norm_i[j] */
    std::vector<float> Aun((size_t)m * (size_t)r), Bun((size_t)r * (size_t)n);
    std::vector<float> Alat((size_t)m * (size_t)r), Blat((size_t)r * (size_t)n);
    for (int i = 0; i < m; ++i) {
        const float inv = 1.0f / norm_o[i];
        for (int k = 0; k < r; ++k) {
            Aun[(size_t)i * (size_t)r + (size_t)k] = Az[(size_t)i * (size_t)r + (size_t)k] * inv;
            Alat[(size_t)i * (size_t)r + (size_t)k] = (A[(size_t)i * (size_t)r + (size_t)k] + Au[(size_t)i * (size_t)r + (size_t)k]) * inv;
        }
    }
    for (int k = 0; k < r; ++k) {
        for (int j = 0; j < n; ++j) {
            const float inv = 1.0f / norm_i[j];
            Bun[(size_t)k * (size_t)n + (size_t)j] = Bz[(size_t)k * (size_t)n + (size_t)j] * inv;
            Blat[(size_t)k * (size_t)n + (size_t)j] = (B[(size_t)k * (size_t)n + (size_t)j] + Bu[(size_t)k * (size_t)n + (size_t)j]) * inv;
        }
    }

    const float nA = frobenius(Aun.data(), (uint64_t)m * (uint64_t)r);
    const float nB = frobenius(Bun.data(), (uint64_t)r * (uint64_t)n);
    float balance = std::sqrt((nB < eps ? eps : nB) / (nA < eps ? eps : nA));
    for (int i = 0, t = m * r; i < t; ++i) Aun[i] *= balance;
    for (int i = 0, t = r * n; i < t; ++i) Bun[i] /= balance;

    /* Column rescale of A from Az column norms (before undo? Official uses A_z.norm(dim=0) = col norms of Az). */
    if (iters > 0) {
        std::vector<float> coln((size_t)r, 0.0f);
        for (int i = 0; i < m; ++i)
            for (int k = 0; k < r; ++k)
                coln[k] += Az[(size_t)i * (size_t)r + (size_t)k] * Az[(size_t)i * (size_t)r + (size_t)k];
        for (int k = 0; k < r; ++k) {
            float s = std::sqrt(coln[k]);
            if (s < eps) s = eps;
            coln[k] = 1.0f / s;
        }
        for (int i = 0; i < m; ++i)
            for (int k = 0; k < r; ++k)
                Aun[(size_t)i * (size_t)r + (size_t)k] *= coln[k];
    }

    /* Scales: s2[j] = mean_k |B[k,j]|, s1[i] = mean_k |A[i,k]| */
    std::vector<float> scale1((size_t)m, 0.0f), scale2((size_t)n, 0.0f);
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int k = 0; k < r; ++k) s += std::fabs((double)Aun[(size_t)i * (size_t)r + (size_t)k]);
        scale1[i] = (float)(s / (double)r);
    }
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int k = 0; k < r; ++k) s += std::fabs((double)Bun[(size_t)k * (size_t)n + (size_t)j]);
        scale2[j] = (float)(s / (double)r);
    }

    std::vector<int8_t> us((size_t)m * (size_t)r), vs((size_t)n * (size_t)r);
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < r; ++k) {
            const float v = Aun[(size_t)i * (size_t)r + (size_t)k];
            us[(size_t)i * (size_t)r + (size_t)k] = (v >= 0.0f) ? (int8_t)1 : (int8_t)-1;
        }
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < r; ++k) {
            const float v = Bun[(size_t)k * (size_t)n + (size_t)j];
            vs[(size_t)j * (size_t)r + (size_t)k] = (v >= 0.0f) ? (int8_t)1 : (int8_t)-1;
        }

    /* Polish s1 and s2 with the signs frozen: a global least-squares scale, then a few
     * alternating updates. This does not use calibration data; it only fits the already
     * chosen binary factors to W. */
    auto gdot = [&](int i, int j) {
        const int8_t *ui = us.data() + (size_t)i * (size_t)r;
        const int8_t *vj = vs.data() + (size_t)j * (size_t)r;
        int acc = 0;
        for (int k = 0; k < r; ++k) acc += (int)ui[k] * (int)vj[k];
        return acc;
    };
    double num = 0.0, den = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const double what = (double)scale1[i] * (double)gdot(i, j) * (double)scale2[j];
            const double ww = (double)W[(size_t)i * (size_t)n + (size_t)j];
            num += ww * what;
            den += what * what;
        }
    }
    if (den > 0.0) {
        const float c = (float)(num / den);
        for (int i = 0; i < m; ++i) scale1[i] *= c;
    }
    for (int pass = 0; pass < 8; ++pass) {
        for (int i = 0; i < m; ++i) {
            double nrm = 0.0, pr = 0.0;
            for (int j = 0; j < n; ++j) {
                const double a = (double)gdot(i, j) * (double)scale2[j];
                pr += (double)W[(size_t)i * (size_t)n + (size_t)j] * a;
                nrm += a * a;
            }
            if (nrm > 0.0) scale1[i] = (float)(pr / nrm);
        }
        for (int j = 0; j < n; ++j) {
            double nrm = 0.0, pr = 0.0;
            for (int i = 0; i < m; ++i) {
                const double a = (double)gdot(i, j) * (double)scale1[i];
                pr += (double)W[(size_t)i * (size_t)n + (size_t)j] * a;
                nrm += a * a;
            }
            if (nrm > 0.0) scale2[j] = (float)(pr / nrm);
        }
    }

    if (u_sign) std::memcpy(u_sign, us.data(), (size_t)m * (size_t)r);
    if (v_sign) std::memcpy(v_sign, vs.data(), (size_t)n * (size_t)r);
    if (s1) std::memcpy(s1, scale1.data(), (size_t)m * sizeof(float));
    if (s2) std::memcpy(s2, scale2.data(), (size_t)n * sizeof(float));

    if (stats) {
        std::memset(stats, 0, sizeof(*stats));
        stats->rows = rows;
        stats->cols = cols;
        stats->rank = r;
        stats->weight_count = (uint64_t)rows * (uint64_t)cols;
        stats->packed_bytes = pchie_nq_packed_bytes(rows, cols, r, 4);
        if (stats->weight_count) {
            stats->bits_per_weight = (8.0 * (double)stats->packed_bytes) / (double)stats->weight_count;
            stats->bits_per_weight_fp16 = (8.0 * (double)pchie_nq_packed_bytes(rows, cols, r, 2)) / (double)stats->weight_count;
        }
        if (u_sign && v_sign && s1 && s2) {
            pchie_nq_accumulate_error(W, u_sign, v_sign, s1, s2, rows, cols, r, stats);
        }
    }
    return PCHIE_NQ_OK;
}

pchie_nq_status pchie_nq_factorize(const float *w, int rows, int cols, const float *d_in, const float *d_out,
                                   const pchie_nq_config *config, int8_t *u_sign, int8_t *v_sign, float *s1,
                                   float *s2, pchie_nq_stats *stats, char *error, size_t error_capacity) {
    pchie_nq_config local;
    if (!config) {
        pchie_nq_config_init(&local);
        config = &local;
    }
    if (!w || rows <= 0 || cols <= 0) {
        zero_outputs(u_sign, v_sign, s1, s2, rows, cols, config->rank, stats);
        set_error(error, error_capacity, "a weight matrix and a positive shape are required");
        return PCHIE_NQ_BAD_ARGUMENT;
    }
    int rank = config->rank > 0 ? config->rank : pchie_nq_choose_rank(rows, cols, config->bits > 0.0f ? config->bits : 1.0f);
    if (rank <= 0) {
        zero_outputs(u_sign, v_sign, s1, s2, rows, cols, 0, stats);
        set_error(error, error_capacity, "could not choose a positive rank for this shape");
        return PCHIE_NQ_BAD_ARGUMENT;
    }
    const int cap = rows < cols ? rows : cols;
    if (rank > cap) rank = cap;

    const int do_t = config->transpose_wide && rows < cols;
    pchie_nq_status st;
    if (do_t) {
        std::vector<float> WT((size_t)cols * (size_t)rows);
        transpose(w, rows, cols, WT.data());
        /* swap d_in / d_out and U / V / s1 / s2 */
        std::vector<int8_t> uT((size_t)cols * (size_t)rank), vT((size_t)rows * (size_t)rank);
        std::vector<float> s1T((size_t)cols), s2T((size_t)rows);
        st = factorize_core(WT.data(), cols, rows, d_out, d_in, config, rank, uT.data(), vT.data(), s1T.data(),
                            s2T.data(), stats, error, error_capacity);
        if (st != PCHIE_NQ_OK) {
            zero_outputs(u_sign, v_sign, s1, s2, rows, cols, rank, stats);
            return st;
        }
        /* After transpose, u_sign was for W^T so it is cols x rank = V of W; v_sign is rows x rank = U of W. */
        if (u_sign) std::memcpy(u_sign, vT.data(), (size_t)rows * (size_t)rank);
        if (v_sign) std::memcpy(v_sign, uT.data(), (size_t)cols * (size_t)rank);
        if (s1) std::memcpy(s1, s2T.data(), (size_t)rows * sizeof(float));
        if (s2) std::memcpy(s2, s1T.data(), (size_t)cols * sizeof(float));
        if (stats) {
            stats->rows = rows;
            stats->cols = cols;
            stats->transposed = 1;
            stats->packed_bytes = pchie_nq_packed_bytes(rows, cols, rank, 4);
            if (stats->weight_count) {
                stats->bits_per_weight = (8.0 * (double)stats->packed_bytes) / (double)stats->weight_count;
                stats->bits_per_weight_fp16 = (8.0 * (double)pchie_nq_packed_bytes(rows, cols, rank, 2)) / (double)stats->weight_count;
            }
            if (u_sign && v_sign && s1 && s2) pchie_nq_accumulate_error(w, u_sign, v_sign, s1, s2, rows, cols, rank, stats);
        }
        return PCHIE_NQ_OK;
    }

    st = factorize_core(w, rows, cols, d_in, d_out, config, rank, u_sign, v_sign, s1, s2, stats, error, error_capacity);
    if (st != PCHIE_NQ_OK) {
        zero_outputs(u_sign, v_sign, s1, s2, rows, cols, rank, stats);
        return st;
    }
    if (stats) stats->transposed = 0;
    return PCHIE_NQ_OK;
}
