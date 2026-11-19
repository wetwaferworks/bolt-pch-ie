/* Bolt PCHIE - CUDA LB-ADMM inner loop for NanoQuant initialization.
 * One tensor at a time. Factors and the ADMM state stay on the device; the weight
 * matrix itself is kept in pinned host memory and streamed in double-buffered tiles
 * with cudaMemcpyAsync on a pair of copy streams that overlap the cuBLAS work.
 * Tile size is chosen from cudaMemGetInfo at the start of the call, leaving a 256 MiB
 * headroom so a 4 GB RTX 3050 Ti is never asked to page through WDDM shared memory.
 *
 * Matmuls go through cuBLAS (IEEE / pedantic, no TF32). Sign-Value Independent
 * Decomposition, the dual update, the precondition undo and the scale polish run as
 * simple kernels. The small Cholesky solve of the r x r system is done on the host
 * (r is hundreds to a few thousand). See include/pchie/pchie_nanoquant.h.
 */
#include "pchie/pchie_nanoquant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

static void set_error(char *error, size_t cap, const char *msg) {
    if (!error || cap == 0) return;
    std::snprintf(error, cap, "%s", msg ? msg : "");
}

static void set_errorf(char *error, size_t cap, const char *fmt, const char *detail) {
    if (!error || cap == 0) return;
    std::snprintf(error, cap, fmt, detail ? detail : "");
}

pchie_nq_cuda_info pchie_nq_query_cuda(void) {
    pchie_nq_cuda_info info;
    std::memset(&info, 0, sizeof info);
    info.built = 1;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return info;
    info.device_count = n;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::snprintf(info.device_name, sizeof info.device_name, "%s", prop.name);
        info.total_bytes = (uint64_t)prop.totalGlobalMem;
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        info.free_bytes = (uint64_t)free_b;
        if (!info.total_bytes) info.total_bytes = (uint64_t)total_b;
    }
    return info;
}

int pchie_nq_cuda_available(void) {
    const pchie_nq_cuda_info info = pchie_nq_query_cuda();
    return info.built && info.device_count > 0;
}

/* ---------- host helpers (same math as the CPU path) ---------- */

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

static int cholesky_solve(int r, float *A, int nrhs, float *B, float stab, float eps) {
    for (int i = 0; i < r; ++i) {
        for (int j = i + 1; j < r; ++j) {
            const float s = 0.5f * (A[i * r + j] + A[j * r + i]);
            A[i * r + j] = A[j * r + i] = s;
        }
        float d = A[i * r + i] + stab;
        if (d < eps) d = eps;
        A[i * r + i] = d;
    }
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

static float rho_at(const pchie_nq_config *c, int it, int n) {
    if (n <= 0) return 1.0f;
    const double x = (double)it / (double)n;
    if (c->rho_cubic) {
        const double y = x < 1.0 ? x : 1.0;
        return (float)(y * y * y);
    }
    return (float)x;
}

/* ---------- device kernels ---------- */

__global__ void nq_k_abs(const float *x, float *y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fabsf(x[i]);
}

__global__ void nq_k_add(const float *a, const float *b, float *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void nq_k_dual(float *lambda, const float *factor, const float *z, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lambda[i] += factor[i] - z[i];
}

__global__ void nq_k_init_dual(const float *factor, const float *z, float *lambda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lambda[i] = factor[i] - z[i];
}

__global__ void nq_k_rhs_add(float *rhs, const float *z, const float *u, float rho, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) rhs[i] += rho * (z[i] - u[i]);
}

__global__ void nq_k_sign_outer(const float *W, const float *u, const float *v, float *Z, int p, int q) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p || j >= q) return;
    const float w = W[(size_t)i * (size_t)q + (size_t)j];
    const float sg = (w >= 0.0f) ? 1.0f : -1.0f;
    Z[(size_t)i * (size_t)q + (size_t)j] = u[i] * v[j] * sg;
}

__global__ void nq_k_scale_vec(float *x, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void nq_k_row_scale(const float *A, const float *inv, float *out, int m, int r) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || k >= r) return;
    out[(size_t)i * (size_t)r + (size_t)k] = A[(size_t)i * (size_t)r + (size_t)k] * inv[i];
}

__global__ void nq_k_col_scale_rn(const float *B, const float *inv, float *out, int r, int n) {
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= r || j >= n) return;
    out[(size_t)k * (size_t)n + (size_t)j] = B[(size_t)k * (size_t)n + (size_t)j] * inv[j];
}

__global__ void nq_k_scale_cols_mr(float *A, const float *inv, int m, int r) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || k >= r) return;
    A[(size_t)i * (size_t)r + (size_t)k] *= inv[k];
}

/* B is r x n; X is n x r with X[j,k] = B[k,j] / ||B[k,:]|| */
__global__ void nq_k_col_norm_T(const float *B, const float *inv_row, float *X, int r, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n || k >= r) return;
    X[(size_t)j * (size_t)r + (size_t)k] = B[(size_t)k * (size_t)n + (size_t)j] * inv_row[k];
}

/* A is m x r; X is m x r with columns of A normalized. */
__global__ void nq_k_row_norm(const float *A, const float *inv_col, float *X, int m, int r) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || k >= r) return;
    X[(size_t)i * (size_t)r + (size_t)k] = A[(size_t)i * (size_t)r + (size_t)k] * inv_col[k];
}

__global__ void nq_k_row_ssq(const float *B, float *ssq, int r, int n) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= r) return;
    double s = 0.0;
    const float *row = B + (size_t)k * (size_t)n;
    for (int j = 0; j < n; ++j) s += (double)row[j] * (double)row[j];
    ssq[k] = (float)s;
}

__global__ void nq_k_col_ssq_mr(const float *A, float *ssq, int m, int r) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= r) return;
    double s = 0.0;
    for (int i = 0; i < m; ++i) {
        const float v = A[(size_t)i * (size_t)r + (size_t)k];
        s += (double)v * (double)v;
    }
    ssq[k] = (float)s;
}

__global__ void nq_k_inv_sqrt_eps(float *ssq, int n, float eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = sqrtf(ssq[i]);
    if (s < eps) s = eps;
    ssq[i] = 1.0f / s;
}

__global__ void nq_k_mean_abs_rows(const float *A, float *s1, int m, int r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double s = 0.0;
    const float *row = A + (size_t)i * (size_t)r;
    for (int k = 0; k < r; ++k) s += fabs((double)row[k]);
    s1[i] = (float)(s / (double)r);
}

__global__ void nq_k_mean_abs_cols_rn(const float *B, float *s2, int r, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double s = 0.0;
    for (int k = 0; k < r; ++k) s += fabs((double)B[(size_t)k * (size_t)n + (size_t)j]);
    s2[j] = (float)(s / (double)r);
}

__global__ void nq_k_sign_mr(const float *A, int8_t *S, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) S[i] = (A[i] >= 0.0f) ? (int8_t)1 : (int8_t)-1;
}

/* v_sign is n x r, B is r x n: v[j,k] = sign(B[k,j]) */
__global__ void nq_k_sign_B_as_nr(const float *B, int8_t *V, int r, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n || k >= r) return;
    const float v = B[(size_t)k * (size_t)n + (size_t)j];
    V[(size_t)j * (size_t)r + (size_t)k] = (v >= 0.0f) ? (int8_t)1 : (int8_t)-1;
}

/* One ALS half: s1[i] = <W_i, G_i * s2> / ||G_i * s2||^2, G_ij = dot(U_i, V_j). */
__global__ void nq_k_als_s1(const float *W, const int8_t *U, const int8_t *V, const float *s2, float *s1,
                            int m, int n, int r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const int8_t *ui = U + (size_t)i * (size_t)r;
    const float *wrow = W + (size_t)i * (size_t)n;
    double pr = 0.0, nrm = 0.0;
    for (int j = 0; j < n; ++j) {
        const int8_t *vj = V + (size_t)j * (size_t)r;
        int acc = 0;
        for (int k = 0; k < r; ++k) acc += (int)ui[k] * (int)vj[k];
        const double a = (double)acc * (double)s2[j];
        pr += (double)wrow[j] * a;
        nrm += a * a;
    }
    if (nrm > 0.0) s1[i] = (float)(pr / nrm);
}

__global__ void nq_k_als_s2(const float *W, const int8_t *U, const int8_t *V, const float *s1, float *s2,
                            int m, int n, int r) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int8_t *vj = V + (size_t)j * (size_t)r;
    double pr = 0.0, nrm = 0.0;
    for (int i = 0; i < m; ++i) {
        const int8_t *ui = U + (size_t)i * (size_t)r;
        int acc = 0;
        for (int k = 0; k < r; ++k) acc += (int)ui[k] * (int)vj[k];
        const double a = (double)acc * (double)s1[i];
        pr += (double)W[(size_t)i * (size_t)n + (size_t)j] * a;
        nrm += a * a;
    }
    if (nrm > 0.0) s2[j] = (float)(pr / nrm);
}

__global__ void nq_k_global_scale_numden(const float *W, const int8_t *U, const int8_t *V, const float *s1,
                                         const float *s2, int m, int n, int r, double *num, double *den) {
    /* one block reduction over a 2D grid is awkward; each thread does a row and atomics. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const int8_t *ui = U + (size_t)i * (size_t)r;
    const float *wrow = W + (size_t)i * (size_t)n;
    double nloc = 0.0, dloc = 0.0;
    for (int j = 0; j < n; ++j) {
        const int8_t *vj = V + (size_t)j * (size_t)r;
        int acc = 0;
        for (int k = 0; k < r; ++k) acc += (int)ui[k] * (int)vj[k];
        const double what = (double)s1[i] * (double)acc * (double)s2[j];
        nloc += (double)wrow[j] * what;
        dloc += what * what;
    }
    atomicAdd(num, nloc);
    atomicAdd(den, dloc);
}

/* Partial s2 update over a row-tile of W: accumulate <W, G*s1> and ||G*s1||^2 per column. */
__global__ void nq_k_als_s2_acc(const float *Wtile, const int8_t *Utile, const int8_t *V, const float *s1tile,
                                int trows, int n, int r, double *pr, double *nrm) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int8_t *vj = V + (size_t)j * (size_t)r;
    double prl = pr[j], nrml = nrm[j];
    for (int i = 0; i < trows; ++i) {
        const int8_t *ui = Utile + (size_t)i * (size_t)r;
        int acc = 0;
        for (int k = 0; k < r; ++k) acc += (int)ui[k] * (int)vj[k];
        const double a = (double)acc * (double)s1tile[i];
        prl += (double)Wtile[(size_t)i * (size_t)n + (size_t)j] * a;
        nrml += a * a;
    }
    pr[j] = prl;
    nrm[j] = nrml;
}

__global__ void nq_k_als_s2_final(double *pr, double *nrm, float *s2, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    if (nrm[j] > 0.0) s2[j] = (float)(pr[j] / nrm[j]);
}

/* ---------- launch helpers ---------- */

static int div_up(int a, int b) { return (a + b - 1) / b; }

static void launch_abs(cudaStream_t st, const float *x, float *y, int n) {
    if (n > 0) nq_k_abs<<<div_up(n, 256), 256, 0, st>>>(x, y, n);
}
static void launch_add(cudaStream_t st, const float *a, const float *b, float *out, int n) {
    if (n > 0) nq_k_add<<<div_up(n, 256), 256, 0, st>>>(a, b, out, n);
}
static void launch_dual(cudaStream_t st, float *lambda, const float *factor, const float *z, int n) {
    if (n > 0) nq_k_dual<<<div_up(n, 256), 256, 0, st>>>(lambda, factor, z, n);
}
static void launch_init_dual(cudaStream_t st, const float *factor, const float *z, float *lambda, int n) {
    if (n > 0) nq_k_init_dual<<<div_up(n, 256), 256, 0, st>>>(factor, z, lambda, n);
}
static void launch_rhs_add(cudaStream_t st, float *rhs, const float *z, const float *u, float rho, int n) {
    if (n > 0) nq_k_rhs_add<<<div_up(n, 256), 256, 0, st>>>(rhs, z, u, rho, n);
}

__global__ void nq_k_ssq_one(const float *x, int n, float *out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)x[i] * (double)x[i];
    *out = (float)sqrt(s);
}

static int device_nrm2(cudaStream_t st, const float *x, int n, float *scratch, float *host) {
    if (n <= 0) {
        *host = 0.0f;
        return 1;
    }
    nq_k_ssq_one<<<1, 1, 0, st>>>(x, n, scratch);
    return cudaMemcpyAsync(host, scratch, sizeof(float), cudaMemcpyDeviceToHost, st) == cudaSuccess &&
           cudaStreamSynchronize(st) == cudaSuccess;
}

static int scale_or_unit0(cudaStream_t st, float *x, int n, float nrm, float eps) {
    if (nrm < eps) {
        const float one = 1.0f;
        if (n > 0 && cudaMemcpyAsync(x, &one, sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess) return 0;
        return cudaStreamSynchronize(st) == cudaSuccess;
    }
    nq_k_scale_vec<<<div_up(n, 256), 256, 0, st>>>(x, 1.0f / nrm, n);
    return 1;
}

static dim3 g2(int x, int y) { return dim3(div_up(x, 16), div_up(y, 16)); }
static dim3 b2() { return dim3(16, 16); }

/* Row-major SGEMM on the device through cuBLAS. The handle's stream must already be set. */
static int dgemm(cublasHandle_t h, int transA, int transB, int m, int n, int k, float alpha,
                 const float *A, const float *B, float beta, float *C) {
    const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int ldb = transB ? k : n;
    const int lda = transA ? m : k;
    const cublasStatus_t st = cublasSgemm(h, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta, C, n);
    return st == CUBLAS_STATUS_SUCCESS;
}

static int dgemv_A_v(cublasHandle_t h, const float *A, int p, int q, const float *v, float *u) {
    const float one = 1.0f, zero = 0.0f;
    return cublasSgemv(h, CUBLAS_OP_T, q, p, &one, A, q, v, 1, &zero, u, 1) == CUBLAS_STATUS_SUCCESS;
}

static int dgemv_AT_u(cublasHandle_t h, const float *A, int p, int q, const float *u, float *v) {
    const float one = 1.0f, zero = 0.0f;
    return cublasSgemv(h, CUBLAS_OP_N, q, p, &one, A, q, u, 1, &zero, v, 1) == CUBLAS_STATUS_SUCCESS;
}

struct NqMem {
    float *A, *Az, *Au, *B, *Bz, *Bu, *XA, *XB, *PA, *PB, *sys, *rhs;
    float *absW, *u, *v, *tmp, *inv, *s1, *s2, *norm_i, *norm_o;
    int8_t *Us, *Vs;
    double *num, *den, *acc_pr, *acc_nrm;
    float *tile[2];
    float *h_stage[2];
    float *h_Wn, *h_src, *h_sys, *h_rhs;
    int src_registered;
    int ev_used_recorded[2];
    cudaStream_t compute, copy[2];
    cudaEvent_t ev_copied[2], ev_used[2];
    size_t tile_elems;
    size_t bytes;
    size_t host_bytes;
};

static void mem_clear(NqMem *m) { std::memset(m, 0, sizeof(*m)); }

static void mem_free(NqMem *m) {
    if (!m) return;
    void *dptrs[] = {m->A, m->Az, m->Au, m->B, m->Bz, m->Bu, m->XA, m->XB, m->PA, m->PB,
                     m->sys, m->rhs, m->absW, m->u, m->v, m->tmp, m->inv, m->s1, m->s2,
                     m->norm_i, m->norm_o, m->Us, m->Vs, m->num, m->den, m->acc_pr, m->acc_nrm,
                     m->tile[0], m->tile[1]};
    for (size_t i = 0; i < sizeof dptrs / sizeof dptrs[0]; ++i) {
        if (dptrs[i]) cudaFree(dptrs[i]);
    }
    if (m->h_Wn) cudaFreeHost(m->h_Wn);
    if (m->h_sys) cudaFreeHost(m->h_sys);
    if (m->h_rhs) cudaFreeHost(m->h_rhs);
    if (m->h_stage[0]) cudaFreeHost(m->h_stage[0]);
    if (m->h_stage[1]) cudaFreeHost(m->h_stage[1]);
    if (m->src_registered && m->h_src) cudaHostUnregister(m->h_src);
    if (m->compute) cudaStreamDestroy(m->compute);
    for (int i = 0; i < 2; ++i) {
        if (m->copy[i]) cudaStreamDestroy(m->copy[i]);
        if (m->ev_copied[i]) cudaEventDestroy(m->ev_copied[i]);
        if (m->ev_used[i]) cudaEventDestroy(m->ev_used[i]);
    }
    mem_clear(m);
}

static int malloc_f(float **p, size_t n, size_t *bytes) {
    if (n == 0) {
        *p = nullptr;
        return 1;
    }
    if (cudaMalloc(p, n * sizeof(float)) != cudaSuccess) return 0;
    *bytes += n * sizeof(float);
    return 1;
}
static int malloc_i8(int8_t **p, size_t n, size_t *bytes) {
    if (n == 0) {
        *p = nullptr;
        return 1;
    }
    if (cudaMalloc(p, n) != cudaSuccess) return 0;
    *bytes += n;
    return 1;
}
static int malloc_d(double **p, size_t n, size_t *bytes) {
    if (n == 0) {
        *p = nullptr;
        return 1;
    }
    if (cudaMalloc(p, n * sizeof(double)) != cudaSuccess) return 0;
    *bytes += n * sizeof(double);
    return 1;
}

static uint64_t persistent_bytes(int m, int n, int r) {
    const uint64_t mr = (uint64_t)m * (uint64_t)r;
    const uint64_t rn = (uint64_t)r * (uint64_t)n;
    const uint64_t rr = (uint64_t)r * (uint64_t)r;
    const uint64_t absmax = mr > rn ? mr : rn;
    const uint64_t rhs = mr > rn ? mr : rn;
    /* A,Az,Au, B,Bz,Bu, XA, XB, PA, PB, sys, rhs, absW, u,v,tmp,inv,s1,s2,norm_i,norm_o */
    const uint64_t f = 3 * mr + 3 * rn + (uint64_t)n * (uint64_t)r + mr + mr + rn + rr + rhs + absmax +
                       (uint64_t)(m > r ? m : r) + (uint64_t)(n > r ? n : r) +
                       (uint64_t)(m > n ? m : n) + (uint64_t)r + (uint64_t)(m > n ? m : n) + (uint64_t)r +
                       (uint64_t)m + (uint64_t)n + (uint64_t)n + (uint64_t)m;
    const uint64_t i8 = mr + (uint64_t)n * (uint64_t)r;
    const uint64_t acc = 2ull * (uint64_t)(m > n ? m : n) * 8ull;
    return f * 4ull + i8 + 16ull + acc;
}

static int create_streams(NqMem *m) {
    if (cudaStreamCreateWithFlags(&m->compute, cudaStreamNonBlocking) != cudaSuccess) return 0;
    for (int i = 0; i < 2; ++i) {
        if (cudaStreamCreateWithFlags(&m->copy[i], cudaStreamNonBlocking) != cudaSuccess) return 0;
        if (cudaEventCreateWithFlags(&m->ev_copied[i], cudaEventDisableTiming) != cudaSuccess) return 0;
        if (cudaEventCreateWithFlags(&m->ev_used[i], cudaEventDisableTiming) != cudaSuccess) return 0;
    }
    return 1;
}

static int mem_alloc_persistent(NqMem *m, int rows, int cols, int r, char *error, size_t errcap) {
    const int absm = rows * r > r * cols ? rows * r : r * cols;
    const int rhsn = rows * r > r * cols ? rows * r : r * cols;
    const int accn = rows > cols ? rows : cols;
    int ok = 1;
    ok = ok && malloc_f(&m->A, (size_t)rows * (size_t)r, &m->bytes);
    ok = ok && malloc_f(&m->Az, (size_t)rows * (size_t)r, &m->bytes);
    ok = ok && malloc_f(&m->Au, (size_t)rows * (size_t)r, &m->bytes);
    ok = ok && malloc_f(&m->B, (size_t)r * (size_t)cols, &m->bytes);
    ok = ok && malloc_f(&m->Bz, (size_t)r * (size_t)cols, &m->bytes);
    ok = ok && malloc_f(&m->Bu, (size_t)r * (size_t)cols, &m->bytes);
    ok = ok && malloc_f(&m->XA, (size_t)cols * (size_t)r, &m->bytes);
    ok = ok && malloc_f(&m->XB, (size_t)rows * (size_t)r, &m->bytes);
    ok = ok && malloc_f(&m->PA, (size_t)rows * (size_t)r, &m->bytes);
    ok = ok && malloc_f(&m->PB, (size_t)r * (size_t)cols, &m->bytes);
    ok = ok && malloc_f(&m->sys, (size_t)r * (size_t)r, &m->bytes);
    ok = ok && malloc_f(&m->rhs, (size_t)rhsn, &m->bytes);
    ok = ok && malloc_f(&m->absW, (size_t)absm, &m->bytes);
    ok = ok && malloc_f(&m->u, (size_t)(rows > r ? rows : r), &m->bytes);
    ok = ok && malloc_f(&m->v, (size_t)(cols > r ? cols : r), &m->bytes);
    ok = ok && malloc_f(&m->tmp, (size_t)(rows > cols ? rows : cols) + (size_t)r, &m->bytes);
    ok = ok && malloc_f(&m->inv, (size_t)(rows > cols ? rows : cols) + (size_t)r, &m->bytes);
    ok = ok && malloc_f(&m->s1, (size_t)rows, &m->bytes);
    ok = ok && malloc_f(&m->s2, (size_t)cols, &m->bytes);
    ok = ok && malloc_f(&m->norm_i, (size_t)cols, &m->bytes);
    ok = ok && malloc_f(&m->norm_o, (size_t)rows, &m->bytes);
    ok = ok && malloc_i8(&m->Us, (size_t)rows * (size_t)r, &m->bytes);
    ok = ok && malloc_i8(&m->Vs, (size_t)cols * (size_t)r, &m->bytes);
    ok = ok && malloc_d(&m->num, 1, &m->bytes);
    ok = ok && malloc_d(&m->den, 1, &m->bytes);
    ok = ok && malloc_d(&m->acc_pr, (size_t)accn, &m->bytes);
    ok = ok && malloc_d(&m->acc_nrm, (size_t)accn, &m->bytes);
    if (ok && cudaHostAlloc(&m->h_sys, (size_t)r * (size_t)r * sizeof(float), cudaHostAllocDefault) != cudaSuccess)
        ok = 0;
    else if (ok) m->host_bytes += (size_t)r * (size_t)r * sizeof(float);
    const size_t rhs_b = (size_t)rhsn * sizeof(float);
    if (ok && cudaHostAlloc(&m->h_rhs, rhs_b, cudaHostAllocDefault) != cudaSuccess) ok = 0;
    else if (ok) m->host_bytes += rhs_b;
    if (!ok) {
        set_error(error, errcap, "not enough device memory for the factor state of this layer");
        mem_free(m);
        return 0;
    }
    return 1;
}

/* Size the two streaming tiles from the free memory that is actually left. */
static int mem_alloc_tiles(NqMem *m, int rows, int cols, char *error, size_t errcap) {
    const uint64_t headroom = 256ull << 20;
    const uint64_t min_elems = (uint64_t)(rows > cols ? rows : cols);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        set_error(error, errcap, "could not query free device memory");
        return 0;
    }
    if (free_b <= headroom) {
        set_error(error, errcap, "less than 256 MiB of device memory is free; refusing to run on a tight card");
        return 0;
    }
    uint64_t budget = (uint64_t)free_b - headroom;
    uint64_t per_tile = budget / 2;
    const uint64_t cap = 64ull << 20; /* 64 MiB per tile is already a wide GEMM panel */
    if (per_tile > cap) per_tile = cap;
    if (per_tile < min_elems * sizeof(float)) {
        set_error(error, errcap,
                  "this layer does not leave enough free device memory for a streamed tile with 256 MiB still in reserve");
        return 0;
    }
    m->tile_elems = (size_t)(per_tile / sizeof(float));
    /* At least one full row of W and one full column of W must fit in a tile. */
    if (m->tile_elems < (size_t)cols || m->tile_elems < (size_t)rows) {
        set_error(error, errcap, "a single row or column of this layer does not fit in a streaming tile");
        return 0;
    }
    int ok = 1;
    for (int i = 0; i < 2; ++i) {
        ok = ok && malloc_f(&m->tile[i], m->tile_elems, &m->bytes);
        if (ok && cudaHostAlloc(&m->h_stage[i], m->tile_elems * sizeof(float), cudaHostAllocDefault) != cudaSuccess)
            ok = 0;
        else if (ok) m->host_bytes += m->tile_elems * sizeof(float);
    }
    const size_t wn = (size_t)rows * (size_t)cols * sizeof(float);
    if (ok && cudaHostAlloc(&m->h_Wn, wn, cudaHostAllocDefault) != cudaSuccess) ok = 0;
    else if (ok) m->host_bytes += wn;
    if (!ok) {
        set_error(error, errcap, "could not allocate the pinned host buffers or the device tiles for this layer");
        return 0;
    }
    return 1;
}

static void pack_w_cols(const float *W, int m, int n, int j0, int jcount, float *out) {
    for (int i = 0; i < m; ++i) {
        std::memcpy(out + (size_t)i * (size_t)jcount, W + (size_t)i * (size_t)n + (size_t)j0,
                    (size_t)jcount * sizeof(float));
    }
}

/* C (r x m) += X^T W^T, with W (m x n) in pinned host memory. X (n x r) is on the device.
 * Columns of W are packed into a pinned staging buffer and copied asynchronously; the next
 * pack overlaps the previous GEMM. */
static int stream_Xt_WT(cublasHandle_t h, NqMem *d, const float *X, int n, int r, const float *W, int m, float *C) {
    const int max_cols = (int)(d->tile_elems / (size_t)m);
    if (max_cols < 1) return 0;
    if (cublasSetStream(h, d->compute) != CUBLAS_STATUS_SUCCESS) return 0;
    float beta = 0.0f;
    int buf = 0;
    for (int j0 = 0; j0 < n;) {
        const int jcount = (j0 + max_cols > n) ? (n - j0) : max_cols;
        if (d->ev_used_recorded[buf]) {
            if (cudaStreamWaitEvent(d->copy[buf], d->ev_used[buf], 0) != cudaSuccess) return 0;
        }
        pack_w_cols(W, m, n, j0, jcount, d->h_stage[buf]);
        if (cudaMemcpyAsync(d->tile[buf], d->h_stage[buf], (size_t)m * (size_t)jcount * sizeof(float),
                            cudaMemcpyHostToDevice, d->copy[buf]) != cudaSuccess)
            return 0;
        if (cudaEventRecord(d->ev_copied[buf], d->copy[buf]) != cudaSuccess) return 0;
        if (cudaStreamWaitEvent(d->compute, d->ev_copied[buf], 0) != cudaSuccess) return 0;
        if (!dgemm(h, 1, 1, r, m, jcount, 1.0f, X + (size_t)j0 * (size_t)r, d->tile[buf], beta, C)) return 0;
        if (cudaEventRecord(d->ev_used[buf], d->compute) != cudaSuccess) return 0;
        d->ev_used_recorded[buf] = 1;
        beta = 1.0f;
        buf ^= 1;
        j0 += jcount;
    }
    return cudaStreamSynchronize(d->compute) == cudaSuccess;
}

/* C (r x n) += X^T W, with W (m x n) pinned. X (m x r) is on the device. Row tiles of W
 * are copied directly from the pinned buffer. */
static int stream_Xt_W(cublasHandle_t h, NqMem *d, const float *X, int m, int r, const float *W, int n, float *C) {
    const int max_rows = (int)(d->tile_elems / (size_t)n);
    if (max_rows < 1) return 0;
    if (cublasSetStream(h, d->compute) != CUBLAS_STATUS_SUCCESS) return 0;
    float beta = 0.0f;
    int buf = 0;
    for (int i0 = 0; i0 < m;) {
        const int trows = (i0 + max_rows > m) ? (m - i0) : max_rows;
        if (d->ev_used_recorded[buf]) {
            if (cudaStreamWaitEvent(d->copy[buf], d->ev_used[buf], 0) != cudaSuccess) return 0;
        }
        if (cudaMemcpyAsync(d->tile[buf], W + (size_t)i0 * (size_t)n, (size_t)trows * (size_t)n * sizeof(float),
                            cudaMemcpyHostToDevice, d->copy[buf]) != cudaSuccess)
            return 0;
        if (cudaEventRecord(d->ev_copied[buf], d->copy[buf]) != cudaSuccess) return 0;
        if (cudaStreamWaitEvent(d->compute, d->ev_copied[buf], 0) != cudaSuccess) return 0;
        if (!dgemm(h, 1, 0, r, n, trows, 1.0f, X + (size_t)i0 * (size_t)r, d->tile[buf], beta, C)) return 0;
        if (cudaEventRecord(d->ev_used[buf], d->compute) != cudaSuccess) return 0;
        d->ev_used_recorded[buf] = 1;
        beta = 1.0f;
        buf ^= 1;
        i0 += trows;
    }
    return cudaStreamSynchronize(d->compute) == cudaSuccess;
}

static int svid_device(cublasHandle_t h, cudaStream_t st, const float *W, int p, int q, int iters, NqRng *rng,
                       float eps, float *Z, float *absW, float *u, float *v, float *tmp) {
    if (cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS) return 0;
    launch_abs(st, W, absW, p * q);
    std::vector<float> vh((size_t)q);
    fill_randn(vh.data(), (uint64_t)q, rng);
    double s = 0.0;
    for (int j = 0; j < q; ++j) s += (double)vh[j] * (double)vh[j];
    float vn = (float)std::sqrt(s);
    if (vn < eps) vh[0] = 1.0f;
    else
        for (int j = 0; j < q; ++j) vh[j] /= vn;
    if (cudaMemcpyAsync(v, vh.data(), (size_t)q * sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess)
        return 0;
    if (cudaStreamSynchronize(st) != cudaSuccess) return 0;
    for (int t = 0; t < iters; ++t) {
        if (!dgemv_A_v(h, absW, p, q, v, u)) return 0;
        float un = 0.0f;
        if (!device_nrm2(st, u, p, tmp, &un) || !scale_or_unit0(st, u, p, un, eps)) return 0;
        if (!dgemv_AT_u(h, absW, p, q, u, v)) return 0;
        vn = 0.0f;
        if (!device_nrm2(st, v, q, tmp, &vn) || !scale_or_unit0(st, v, q, vn, eps)) return 0;
    }
    if (!dgemv_A_v(h, absW, p, q, v, u)) return 0;
    nq_k_sign_outer<<<g2(q, p), b2(), 0, st>>>(W, u, v, Z, p, q);
    return cudaStreamSynchronize(st) == cudaSuccess && cudaGetLastError() == cudaSuccess;
}

/* y_is_wt: 0 = Y is W (m_y = n_W) streamed as row tiles; 1 = Y is W^T streamed as packed columns. */
static int admm_solve_streamed(cublasHandle_t h, NqMem *d, const float *X, int n, int r, const float *W_host, int m_w,
                               int n_w, int y_is_wt, const float *Z, const float *U, int m, float rho, float reg,
                               float eps, float *Out) {
    if (cublasSetStream(h, d->compute) != CUBLAS_STATUS_SUCCESS) return 0;
    if (!dgemm(h, 1, 0, r, r, n, 1.0f, X, X, 0.0f, d->sys)) return 0;
    if (cudaMemcpyAsync(d->h_sys, d->sys, (size_t)r * (size_t)r * sizeof(float), cudaMemcpyDeviceToHost, d->compute) !=
        cudaSuccess)
        return 0;
    if (cudaStreamSynchronize(d->compute) != cudaSuccess) return 0;
    double diag_mean = 0.0;
    for (int i = 0; i < r; ++i) diag_mean += std::fabs((double)d->h_sys[i * r + i]);
    diag_mean = (r > 0) ? diag_mean / (double)r : 0.0;
    float stab = (float)(rho * diag_mean + (double)reg);
    if (stab < eps) stab = eps;
    d->ev_used_recorded[0] = d->ev_used_recorded[1] = 0;
    int ok = 0;
    if (y_is_wt)
        ok = stream_Xt_WT(h, d, X, n, r, W_host, m_w, d->rhs);
    else
        ok = stream_Xt_W(h, d, X, n, r, W_host, n_w, d->rhs);
    if (!ok) return 0;
    launch_rhs_add(d->compute, d->rhs, Z, U, rho, r * m);
    if (cudaMemcpyAsync(d->h_rhs, d->rhs, (size_t)r * (size_t)m * sizeof(float), cudaMemcpyDeviceToHost, d->compute) !=
        cudaSuccess)
        return 0;
    if (cudaStreamSynchronize(d->compute) != cudaSuccess) return 0;
    cholesky_solve(r, d->h_sys, m, d->h_rhs, stab, eps);
    if (cudaMemcpyAsync(Out, d->h_rhs, (size_t)r * (size_t)m * sizeof(float), cudaMemcpyHostToDevice, d->compute) !=
        cudaSuccess)
        return 0;
    return cudaStreamSynchronize(d->compute) == cudaSuccess;
}

__global__ void nq_k_transpose(const float *A, float *At, int m, int n) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    At[(size_t)j * (size_t)m + (size_t)i] = A[(size_t)i * (size_t)n + (size_t)j];
}

static void dtranspose(cudaStream_t st, const float *A, float *At, int m, int n) {
    if (m > 0 && n > 0) nq_k_transpose<<<g2(n, m), b2(), 0, st>>>(A, At, m, n);
}

static int col_normalize_T_dev(cudaStream_t st, const float *B, int r, int n, float eps, float *X, float *inv) {
    nq_k_row_ssq<<<div_up(r, 256), 256, 0, st>>>(B, inv, r, n);
    nq_k_inv_sqrt_eps<<<div_up(r, 256), 256, 0, st>>>(inv, r, eps);
    nq_k_col_norm_T<<<g2(n, r), b2(), 0, st>>>(B, inv, X, r, n);
    return cudaGetLastError() == cudaSuccess;
}

static int row_normalize_dev(cudaStream_t st, const float *A, int m, int r, float eps, float *X, float *inv) {
    nq_k_col_ssq_mr<<<div_up(r, 256), 256, 0, st>>>(A, inv, m, r);
    nq_k_inv_sqrt_eps<<<div_up(r, 256), 256, 0, st>>>(inv, r, eps);
    nq_k_row_norm<<<g2(r, m), b2(), 0, st>>>(A, inv, X, m, r);
    return cudaGetLastError() == cudaSuccess;
}

static int polish_scales(NqMem *d, const float *W, int m, int n, int r) {
    const cudaStream_t st = d->compute;
    const int max_rows = (int)(d->tile_elems / (size_t)n);
    if (max_rows < 1) return 0;
    if (cudaMemsetAsync(d->num, 0, sizeof(double), st) != cudaSuccess) return 0;
    if (cudaMemsetAsync(d->den, 0, sizeof(double), st) != cudaSuccess) return 0;
    d->ev_used_recorded[0] = d->ev_used_recorded[1] = 0;
    int buf = 0;
    for (int i0 = 0; i0 < m;) {
        const int trows = (i0 + max_rows > m) ? (m - i0) : max_rows;
        if (d->ev_used_recorded[buf] && cudaStreamWaitEvent(d->copy[buf], d->ev_used[buf], 0) != cudaSuccess) return 0;
        if (cudaMemcpyAsync(d->tile[buf], W + (size_t)i0 * (size_t)n, (size_t)trows * (size_t)n * sizeof(float),
                            cudaMemcpyHostToDevice, d->copy[buf]) != cudaSuccess)
            return 0;
        if (cudaEventRecord(d->ev_copied[buf], d->copy[buf]) != cudaSuccess) return 0;
        if (cudaStreamWaitEvent(st, d->ev_copied[buf], 0) != cudaSuccess) return 0;
        nq_k_global_scale_numden<<<div_up(trows, 128), 128, 0, st>>>(d->tile[buf], d->Us + (size_t)i0 * (size_t)r,
                                                                     d->Vs, d->s1 + i0, d->s2, trows, n, r, d->num,
                                                                     d->den);
        if (cudaEventRecord(d->ev_used[buf], st) != cudaSuccess) return 0;
        d->ev_used_recorded[buf] = 1;
        buf ^= 1;
        i0 += trows;
    }
    double num = 0, den = 0;
    if (cudaMemcpyAsync(&num, d->num, sizeof(double), cudaMemcpyDeviceToHost, st) != cudaSuccess) return 0;
    if (cudaMemcpyAsync(&den, d->den, sizeof(double), cudaMemcpyDeviceToHost, st) != cudaSuccess) return 0;
    if (cudaStreamSynchronize(st) != cudaSuccess) return 0;
    if (den > 0.0) nq_k_scale_vec<<<div_up(m, 256), 256, 0, st>>>(d->s1, (float)(num / den), m);

    auto stream_rows = [&](auto kernel_on_tile) -> int {
        d->ev_used_recorded[0] = d->ev_used_recorded[1] = 0;
        int b = 0;
        for (int i0 = 0; i0 < m;) {
            const int trows = (i0 + max_rows > m) ? (m - i0) : max_rows;
            if (d->ev_used_recorded[b] && cudaStreamWaitEvent(d->copy[b], d->ev_used[b], 0) != cudaSuccess) return 0;
            if (cudaMemcpyAsync(d->tile[b], W + (size_t)i0 * (size_t)n, (size_t)trows * (size_t)n * sizeof(float),
                                cudaMemcpyHostToDevice, d->copy[b]) != cudaSuccess)
                return 0;
            if (cudaEventRecord(d->ev_copied[b], d->copy[b]) != cudaSuccess) return 0;
            if (cudaStreamWaitEvent(st, d->ev_copied[b], 0) != cudaSuccess) return 0;
            if (!kernel_on_tile(d->tile[b], trows, i0, b)) return 0;
            if (cudaEventRecord(d->ev_used[b], st) != cudaSuccess) return 0;
            d->ev_used_recorded[b] = 1;
            b ^= 1;
            i0 += trows;
        }
        return cudaStreamSynchronize(st) == cudaSuccess;
    };

    for (int pass = 0; pass < 8; ++pass) {
        if (!stream_rows([&](float *tile, int trows, int i0, int) {
                nq_k_als_s1<<<div_up(trows, 128), 128, 0, st>>>(tile, d->Us + (size_t)i0 * (size_t)r, d->Vs, d->s2,
                                                                d->s1 + i0, trows, n, r);
                return 1;
            }))
            return 0;
        if (cudaMemsetAsync(d->acc_pr, 0, (size_t)n * sizeof(double), st) != cudaSuccess) return 0;
        if (cudaMemsetAsync(d->acc_nrm, 0, (size_t)n * sizeof(double), st) != cudaSuccess) return 0;
        if (!stream_rows([&](float *tile, int trows, int i0, int) {
                nq_k_als_s2_acc<<<div_up(n, 128), 128, 0, st>>>(tile, d->Us + (size_t)i0 * (size_t)r, d->Vs, d->s1 + i0,
                                                                trows, n, r, d->acc_pr, d->acc_nrm);
                return 1;
            }))
            return 0;
        nq_k_als_s2_final<<<div_up(n, 256), 256, 0, st>>>(d->acc_pr, d->acc_nrm, d->s2, n);
        if (cudaStreamSynchronize(st) != cudaSuccess) return 0;
    }
    return 1;
}

extern "C" pchie_nq_status pchie_nq_factorize_cuda(const float *w, int rows, int cols, const float *d_in,
                                                   const float *d_out, const pchie_nq_config *cfg, int rank,
                                                   int8_t *u_sign, int8_t *v_sign, float *s1, float *s2,
                                                   pchie_nq_stats *stats, char *error, size_t errcap) {
    const float eps = cfg->eps > 0.0f ? cfg->eps : 1e-12f;
    const int r = rank;
    const int m = rows;
    const int n = cols;

    if (cudaSetDevice(0) != cudaSuccess) {
        set_error(error, errcap, "could not select the first CUDA device");
        return PCHIE_NQ_BAD_ARGUMENT;
    }
    pchie_nq_cuda_info info = pchie_nq_query_cuda();
    const uint64_t headroom = 256ull << 20;
    const uint64_t need = persistent_bytes(m, n, r) + 2ull * (uint64_t)(m > n ? m : n) * 4ull;
    if (info.free_bytes && need + headroom > info.free_bytes) {
        set_error(error, errcap,
                  "this layer's factor state does not fit in the free device memory with 256 MiB left in reserve");
        return PCHIE_NQ_TOO_LARGE;
    }

    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        set_error(error, errcap, "could not create a cuBLAS handle");
        return PCHIE_NQ_BAD_ARGUMENT;
    }
    cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH);

    NqMem d;
    mem_clear(&d);
    if (!create_streams(&d)) {
        set_error(error, errcap, "could not create CUDA streams");
        cublasDestroy(blas);
        mem_free(&d);
        return PCHIE_NQ_BAD_ARGUMENT;
    }
    if (!mem_alloc_persistent(&d, m, n, r, error, errcap)) {
        cublasDestroy(blas);
        return PCHIE_NQ_NO_MEMORY;
    }
    if (!mem_alloc_tiles(&d, m, n, error, errcap)) {
        cublasDestroy(blas);
        mem_free(&d);
        return PCHIE_NQ_TOO_LARGE;
    }
    if (cublasSetStream(blas, d.compute) != CUBLAS_STATUS_SUCCESS) {
        set_error(error, errcap, "could not bind cuBLAS to the compute stream");
        mem_free(&d);
        cublasDestroy(blas);
        return PCHIE_NQ_BAD_ARGUMENT;
    }

    std::vector<float> norm_i((size_t)n, 1.0f), norm_o((size_t)m, 1.0f);
    for (int j = 0; j < n; ++j) {
        const float v = d_in ? d_in[j] : 1.0f;
        if (!(v >= 0.0f) || !std::isfinite(v)) {
            set_error(error, errcap, "a Hessian diagonal entry is not a finite non-negative number");
            mem_free(&d);
            cublasDestroy(blas);
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
            mem_free(&d);
            cublasDestroy(blas);
            return PCHIE_NQ_BAD_ARGUMENT;
        }
        float s = std::sqrt(v);
        if (s < eps) s = eps;
        norm_o[i] = s;
    }
    cudaMemcpyAsync(d.norm_i, norm_i.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice, d.compute);
    cudaMemcpyAsync(d.norm_o, norm_o.data(), (size_t)m * sizeof(float), cudaMemcpyHostToDevice, d.compute);

    /* Precondition into pinned host memory: W_norm[i,j] = W[i,j] * norm_o[i] * norm_i[j]. */
    for (int i = 0; i < m; ++i) {
        const float no = norm_o[i];
        const float *in = w + (size_t)i * (size_t)n;
        float *out = d.h_Wn + (size_t)i * (size_t)n;
        for (int j = 0; j < n; ++j) out[j] = in[j] * no * norm_i[j];
    }

    /* Prefer a pinned view of the caller's W for the scale polish; fall back to pageable copies. */
    if (cudaHostRegister((void *)w, (size_t)m * (size_t)n * sizeof(float), cudaHostRegisterDefault) == cudaSuccess) {
        d.h_src = (float *)(void *)w;
        d.src_registered = 1;
    } else {
        d.h_src = (float *)(void *)w;
        d.src_registered = 0;
    }

    NqRng rng{cfg->seed ? cfg->seed : 0xA5F1C3D2E4B60789ULL};
    splitmix64(&rng.state);

    std::vector<float> Ah((size_t)m * (size_t)r), Bh((size_t)r * (size_t)n);
    fill_randn(Ah.data(), (uint64_t)m * (uint64_t)r, &rng);
    fill_randn(Bh.data(), (uint64_t)r * (uint64_t)n, &rng);
    cudaMemcpyAsync(d.A, Ah.data(), Ah.size() * sizeof(float), cudaMemcpyHostToDevice, d.compute);
    cudaMemcpyAsync(d.B, Bh.data(), Bh.size() * sizeof(float), cudaMemcpyHostToDevice, d.compute);
    if (cudaStreamSynchronize(d.compute) != cudaSuccess) {
        set_error(error, errcap, "failed to upload the initial factors");
        mem_free(&d);
        cublasDestroy(blas);
        return PCHIE_NQ_BAD_ARGUMENT;
    }

    if (cfg->outer_iters > 0) {
        if (!svid_device(blas, d.compute, d.A, m, r, cfg->inner_iters, &rng, eps, d.Az, d.absW, d.u, d.v, d.tmp) ||
            !svid_device(blas, d.compute, d.B, r, n, cfg->inner_iters, &rng, eps, d.Bz, d.absW, d.u, d.v, d.tmp)) {
            set_error(error, errcap, "SVID failed on the device");
            mem_free(&d);
            cublasDestroy(blas);
            return PCHIE_NQ_BAD_ARGUMENT;
        }
    } else {
        cudaMemcpyAsync(d.Az, d.A, (size_t)m * (size_t)r * sizeof(float), cudaMemcpyDeviceToDevice, d.compute);
        cudaMemcpyAsync(d.Bz, d.B, (size_t)r * (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, d.compute);
    }
    launch_init_dual(d.compute, d.A, d.Az, d.Au, m * r);
    launch_init_dual(d.compute, d.B, d.Bz, d.Bu, r * n);

    const int iters = cfg->outer_iters < 0 ? 0 : cfg->outer_iters;
    for (int it = 0; it < iters; ++it) {
        const float rho = rho_at(cfg, it, iters);
        if (!col_normalize_T_dev(d.compute, d.Bz, r, n, eps, d.XA, d.inv)) {
            set_error(error, errcap, "column normalize failed");
            mem_free(&d);
            cublasDestroy(blas);
            return PCHIE_NQ_BAD_ARGUMENT;
        }
        dtranspose(d.compute, d.Az, d.PA, m, r);
        dtranspose(d.compute, d.Au, d.XB, m, r);
        if (cudaStreamSynchronize(d.compute) != cudaSuccess) {
            set_error(error, errcap, "failed to prepare the U-update");
            mem_free(&d);
            cublasDestroy(blas);
            return PCHIE_NQ_BAD_ARGUMENT;
        }
        /* X is n x r, Y is W^T (n x m), Out is r x m. */
        if (!admm_solve_streamed(blas, &d, d.XA, n, r, d.h_Wn, m, n, 1, d.PA, d.XB, m, rho, cfg->reg, eps, d.rhs)) {
            set_error(error, errcap, "ADMM U-update failed");
            mem_free(&d);
            cublasDestroy(blas);
            return PCHIE_NQ_BAD_ARGUMENT;
        }
        dtranspose(d.compute, d.rhs, d.A, r, m);

        if (!row_normalize_dev(d.compute, d.Az, m, r, eps, d.XB, d.inv)) {
            set_error(error, errcap, "row normalize failed");
            mem_free(&d);
            cublasDestroy(blas);
            return PCHIE_NQ_BAD_ARGUMENT;
        }
        if (cudaStreamSynchronize(d.compute) != cudaSuccess) {
            set_error(error, errcap, "failed to prepare the V-update");
            mem_free(&d);
            cublasDestroy(blas);
            return PCHIE_NQ_BAD_ARGUMENT;
        }
        /* X is m x r, Y is W (m x n), Out is r x n. */
        if (!admm_solve_streamed(blas, &d, d.XB, m, r, d.h_Wn, m, n, 0, d.Bz, d.Bu, n, rho, cfg->reg, eps, d.B)) {
            set_error(error, errcap, "ADMM V-update failed");
            mem_free(&d);
            cublasDestroy(blas);
            return PCHIE_NQ_BAD_ARGUMENT;
        }

        launch_add(d.compute, d.A, d.Au, d.PA, m * r);
        launch_add(d.compute, d.B, d.Bu, d.PB, r * n);
        if (!svid_device(blas, d.compute, d.PA, m, r, cfg->inner_iters, &rng, eps, d.Az, d.absW, d.u, d.v, d.tmp) ||
            !svid_device(blas, d.compute, d.PB, r, n, cfg->inner_iters, &rng, eps, d.Bz, d.absW, d.u, d.v, d.tmp)) {
            set_error(error, errcap, "SVID failed during ADMM");
            mem_free(&d);
            cublasDestroy(blas);
            return PCHIE_NQ_BAD_ARGUMENT;
        }
        launch_dual(d.compute, d.Au, d.A, d.Az, m * r);
        launch_dual(d.compute, d.Bu, d.B, d.Bz, r * n);
    }

    std::vector<float> inv_o((size_t)m), inv_i((size_t)n);
    for (int i = 0; i < m; ++i) inv_o[i] = 1.0f / norm_o[i];
    for (int j = 0; j < n; ++j) inv_i[j] = 1.0f / norm_i[j];
    cudaMemcpyAsync(d.norm_o, inv_o.data(), (size_t)m * sizeof(float), cudaMemcpyHostToDevice, d.compute);
    cudaMemcpyAsync(d.norm_i, inv_i.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice, d.compute);
    nq_k_row_scale<<<g2(r, m), b2(), 0, d.compute>>>(d.Az, d.norm_o, d.A, m, r);
    nq_k_col_scale_rn<<<g2(n, r), b2(), 0, d.compute>>>(d.Bz, d.norm_i, d.B, r, n);

    float nA = 0.0f, nB = 0.0f;
    if (!device_nrm2(d.compute, d.A, m * r, d.tmp, &nA) || !device_nrm2(d.compute, d.B, r * n, d.tmp, &nB)) {
        set_error(error, errcap, "failed to measure factor norms on the device");
        mem_free(&d);
        cublasDestroy(blas);
        return PCHIE_NQ_BAD_ARGUMENT;
    }
    float balance = std::sqrt((nB < eps ? eps : nB) / (nA < eps ? eps : nA));
    nq_k_scale_vec<<<div_up(m * r, 256), 256, 0, d.compute>>>(d.A, balance, m * r);
    nq_k_scale_vec<<<div_up(r * n, 256), 256, 0, d.compute>>>(d.B, 1.0f / balance, r * n);

    if (iters > 0) {
        nq_k_col_ssq_mr<<<div_up(r, 256), 256, 0, d.compute>>>(d.Az, d.inv, m, r);
        nq_k_inv_sqrt_eps<<<div_up(r, 256), 256, 0, d.compute>>>(d.inv, r, eps);
        nq_k_scale_cols_mr<<<g2(r, m), b2(), 0, d.compute>>>(d.A, d.inv, m, r);
    }

    nq_k_mean_abs_rows<<<div_up(m, 256), 256, 0, d.compute>>>(d.A, d.s1, m, r);
    nq_k_mean_abs_cols_rn<<<div_up(n, 256), 256, 0, d.compute>>>(d.B, d.s2, r, n);
    nq_k_sign_mr<<<div_up(m * r, 256), 256, 0, d.compute>>>(d.A, d.Us, m * r);
    nq_k_sign_B_as_nr<<<g2(n, r), b2(), 0, d.compute>>>(d.B, d.Vs, r, n);
    if (cudaStreamSynchronize(d.compute) != cudaSuccess) {
        set_error(error, errcap, "a CUDA kernel failed after the ADMM loop");
        mem_free(&d);
        cublasDestroy(blas);
        return PCHIE_NQ_BAD_ARGUMENT;
    }

    if (!polish_scales(&d, d.h_src, m, n, r) || cudaGetLastError() != cudaSuccess) {
        set_error(error, errcap, "a CUDA kernel failed during the scale polish");
        mem_free(&d);
        cublasDestroy(blas);
        return PCHIE_NQ_BAD_ARGUMENT;
    }

    if (u_sign) cudaMemcpy(u_sign, d.Us, (size_t)m * (size_t)r, cudaMemcpyDeviceToHost);
    if (v_sign) cudaMemcpy(v_sign, d.Vs, (size_t)n * (size_t)r, cudaMemcpyDeviceToHost);
    if (s1) cudaMemcpy(s1, d.s1, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost);
    if (s2) cudaMemcpy(s2, d.s2, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

    if (stats) {
        std::memset(stats, 0, sizeof(*stats));
        stats->rows = rows;
        stats->cols = cols;
        stats->rank = r;
        stats->weight_count = (uint64_t)rows * (uint64_t)cols;
        stats->packed_bytes = pchie_nq_packed_bytes(rows, cols, r, 4);
        stats->backend = 1;
        stats->device_bytes = (uint64_t)d.bytes;
        if (stats->weight_count) {
            stats->bits_per_weight = (8.0 * (double)stats->packed_bytes) / (double)stats->weight_count;
            stats->bits_per_weight_fp16 =
                (8.0 * (double)pchie_nq_packed_bytes(rows, cols, r, 2)) / (double)stats->weight_count;
        }
        if (u_sign && v_sign && s1 && s2) {
            pchie_nq_accumulate_error(w, u_sign, v_sign, s1, s2, rows, cols, r, stats);
        }
    }

    mem_free(&d);
    cublasDestroy(blas);
    return PCHIE_NQ_OK;
}
