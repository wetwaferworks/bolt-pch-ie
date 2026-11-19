/* Bolt PCHIE - NanoQuant block and model reconstruction. See pchie_nanoquant_recon.h. */
#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif
#include "pchie/pchie_nanoquant_recon.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

/* ---------- errors, paths, SHA-256 ---------- */

static void set_error(char *error, size_t cap, const char *msg) {
    if (!error || cap == 0) return;
    std::snprintf(error, cap, "%s", msg ? msg : "");
}

static int mkdir_one(const char *path) {
#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, 0755);
#endif
}

static int mkdir_p(const char *path) {
    if (!path || !path[0]) return -1;
    std::string s(path);
    for (size_t i = 1; i < s.size(); ++i) {
        if (s[i] == '/' || s[i] == '\\') {
            const char c = s[i];
            s[i] = 0;
            mkdir_one(s.c_str());
            s[i] = c;
        }
    }
    mkdir_one(s.c_str());
    return 0;
}

static int fsync_file(FILE *f) {
    if (!f) return 0;
    if (std::fflush(f) != 0) return 0;
#ifdef _WIN32
    const int fd = _fileno(f);
    if (fd < 0) return 0;
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    return FlushFileBuffers(h) ? 1 : 0;
#else
    return fsync(fileno(f)) == 0;
#endif
}

static int atomic_replace(const char *partial, const char *dest) {
#ifdef _WIN32
    if (MoveFileExA(partial, dest, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return 1;
    /* Some file systems refuse WRITE_THROUGH; try without it. */
    if (MoveFileExA(partial, dest, MOVEFILE_REPLACE_EXISTING)) return 1;
    return 0;
#else
    return rename(partial, dest) == 0;
#endif
}

/* FIPS 180-4 SHA-256, compact, used only to stamp checkpoints. */
struct Sha256 {
    uint32_t h[8];
    uint64_t nbits;
    uint8_t buf[64];
    size_t off;
};

static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_init(Sha256 *s) {
    static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::memcpy(s->h, iv, sizeof iv);
    s->nbits = 0;
    s->off = 0;
}

static void sha256_block(Sha256 *s, const uint8_t b[64]) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = ((uint32_t)b[4 * i] << 24) | ((uint32_t)b[4 * i + 1] << 16) | ((uint32_t)b[4 * i + 2] << 8) |
               (uint32_t)b[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
        const uint32_t x = w[i - 15], y = w[i - 2];
        const uint32_t s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
        const uint32_t s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = s->h[0], c = s->h[2], d = s->h[3], e = s->h[4], f = s->h[5], g = s->h[6], hh = s->h[7];
    uint32_t b2 = s->h[1];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b2) ^ (a & c) ^ (b2 & c);
        const uint32_t t2 = S0 + maj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b2;
        b2 = a;
        a = t1 + t2;
    }
    s->h[0] += a;
    s->h[1] += b2;
    s->h[2] += c;
    s->h[3] += d;
    s->h[4] += e;
    s->h[5] += f;
    s->h[6] += g;
    s->h[7] += hh;
}

static void sha256_update(Sha256 *s, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    s->nbits += (uint64_t)n * 8ull;
    while (n) {
        const size_t c = n < 64 - s->off ? n : 64 - s->off;
        std::memcpy(s->buf + s->off, p, c);
        s->off += c;
        p += c;
        n -= c;
        if (s->off == 64) {
            sha256_block(s, s->buf);
            s->off = 0;
        }
    }
}

static void sha256_final(Sha256 *s, uint8_t out[32]) {
    s->buf[s->off++] = 0x80;
    if (s->off > 56) {
        while (s->off < 64) s->buf[s->off++] = 0;
        sha256_block(s, s->buf);
        s->off = 0;
    }
    while (s->off < 56) s->buf[s->off++] = 0;
    for (int i = 0; i < 8; ++i) s->buf[63 - i] = (uint8_t)(s->nbits >> (8 * i));
    sha256_block(s, s->buf);
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = (uint8_t)(s->h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(s->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(s->h[i] >> 8);
        out[4 * i + 3] = (uint8_t)s->h[i];
    }
}

static void sha256_bytes(const void *data, size_t n, uint8_t out[32]) {
    Sha256 s;
    sha256_init(&s);
    sha256_update(&s, data, n);
    sha256_final(&s, out);
}

static void hex32(const uint8_t h[32], char out[65]) {
    static const char *d = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out[2 * i] = d[h[i] >> 4];
        out[2 * i + 1] = d[h[i] & 15];
    }
    out[64] = 0;
}

static int parse_hex32(const char *s, uint8_t out[32]) {
    if (!s) return 0;
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (int i = 0; i < 32; ++i) {
        const int a = nib(s[2 * i]), b = nib(s[2 * i + 1]);
        if (a < 0 || b < 0) return 0;
        out[i] = (uint8_t)((a << 4) | b);
    }
    return 1;
}

/* ---------- RNG, Adam, math ---------- */

struct Rng {
    uint64_t state;
};

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void rng_seed(Rng *r, uint64_t seed) { r->state = seed ? seed : 0xA5F1C3D2E4B60789ULL; }

static double rng_u01(Rng *r) { return (splitmix64(&r->state) >> 11) * (1.0 / 9007199254740992.0); }

static float rng_normal(Rng *r) {
    double u, v, s;
    do {
        u = 2.0 * rng_u01(r) - 1.0;
        v = 2.0 * rng_u01(r) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    return (float)(u * std::sqrt(-2.0 * std::log(s) / s));
}

static void rng_perm(Rng *r, int n, int *idx) {
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int i = n - 1; i > 0; --i) {
        const int j = (int)(splitmix64(&r->state) % (uint64_t)(i + 1));
        const int t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
}

struct Adam {
    std::vector<float> m, v;
    int t;
    float b1, b2, eps;
};

static void adam_init(Adam *a, size_t n) {
    a->m.assign(n, 0.0f);
    a->v.assign(n, 0.0f);
    a->t = 0;
    a->b1 = 0.9f;
    a->b2 = 0.999f;
    a->eps = 1e-8f;
}

static void adam_step(Adam *a, float *p, const float *g, size_t n, float lr) {
    ++a->t;
    const double bc1 = 1.0 - std::pow((double)a->b1, (double)a->t);
    const double bc2 = 1.0 - std::pow((double)a->b2, (double)a->t);
    for (size_t i = 0; i < n; ++i) {
        a->m[i] = a->b1 * a->m[i] + (1.0f - a->b1) * g[i];
        a->v[i] = a->b2 * a->v[i] + (1.0f - a->b2) * g[i] * g[i];
        const double mhat = (double)a->m[i] / bc1;
        const double vhat = (double)a->v[i] / bc2;
        p[i] = (float)((double)p[i] - (double)lr * mhat / (std::sqrt(vhat) + (double)a->eps));
    }
}

static float cosine_lr(float lr, float lr_min, int step, int total) {
    if (total <= 1) return lr_min;
    const int s = step < total ? step : (total - 1);
    const double x = (double)s / (double)(total - 1);
    return (float)(lr_min + 0.5 * ((double)lr - (double)lr_min) * (1.0 + std::cos(3.14159265358979323846 * x)));
}

static float gelu(float x) {
    const float k = 0.7978845608028654f; /* sqrt(2/pi) */
    const float u = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(u));
}

static float gelu_dx(float x) {
    const float k = 0.7978845608028654f;
    const float u = k * (x + 0.044715f * x * x * x);
    const float th = std::tanh(u);
    const float du = k * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * du;
}

/* Y[T,out] = X[T,in] * W[out,in]^T */
static void gemm_xtwt(const float *X, const float *W, float *Y, int T, int out, int in) {
    for (int t = 0; t < T; ++t) {
        const float *x = X + (size_t)t * (size_t)in;
        float *y = Y + (size_t)t * (size_t)out;
        for (int i = 0; i < out; ++i) {
            const float *w = W + (size_t)i * (size_t)in;
            double s = 0.0;
            for (int j = 0; j < in; ++j) s += (double)x[j] * (double)w[j];
            y[i] = (float)s;
        }
    }
}

static void gemm_dW(const float *X, const float *dY, float *dW, int T, int out, int in, int acc) {
    if (!acc) std::memset(dW, 0, (size_t)out * (size_t)in * sizeof(float));
    for (int t = 0; t < T; ++t) {
        const float *x = X + (size_t)t * (size_t)in;
        const float *dy = dY + (size_t)t * (size_t)out;
        for (int i = 0; i < out; ++i) {
            float *dw = dW + (size_t)i * (size_t)in;
            const float e = dy[i];
            for (int j = 0; j < in; ++j) dw[j] += e * x[j];
        }
    }
}

static void gemm_dX(const float *W, const float *dY, float *dX, int T, int out, int in, int acc) {
    if (!acc) std::memset(dX, 0, (size_t)T * (size_t)in * sizeof(float));
    for (int t = 0; t < T; ++t) {
        const float *dy = dY + (size_t)t * (size_t)out;
        float *dx = dX + (size_t)t * (size_t)in;
        for (int i = 0; i < out; ++i) {
            const float *w = W + (size_t)i * (size_t)in;
            const float e = dy[i];
            for (int j = 0; j < in; ++j) dx[j] += e * w[j];
        }
    }
}

/* ---------- model ---------- */

struct Linear {
    int rows, cols, rank, factored, finalized;
    std::vector<float> W, s1, s2, Ulat, Vlat, din, dout;
    std::vector<int8_t> U, V;
};

struct Block {
    Linear up, down; /* up: mid x hidden, down: hidden x mid */
};

struct Recon {
    int n_blocks, hidden, mid, vocab, samples, seq;
    uint64_t seed;
    std::vector<Block> blocks;
    std::vector<float> teacher_up, teacher_down, head; /* packed teacher copies */
    std::vector<float> calib, student_act, teacher_act;
    pchie_nq_recon_config cfg;
};

static size_t up_n(const Recon *m) { return (size_t)m->mid * (size_t)m->hidden; }
static size_t down_n(const Recon *m) { return (size_t)m->hidden * (size_t)m->mid; }
static size_t act_n(const Recon *m) { return (size_t)m->samples * (size_t)m->seq * (size_t)m->hidden; }
static size_t head_n(const Recon *m) { return (size_t)m->vocab * (size_t)m->hidden; }

static void linear_init_dense(Linear *L, int rows, int cols) {
    L->rows = rows;
    L->cols = cols;
    L->rank = 0;
    L->factored = 0;
    L->finalized = 0;
    L->W.assign((size_t)rows * (size_t)cols, 0.0f);
    L->din.assign((size_t)cols, 1.0f);
    L->dout.assign((size_t)rows, 1.0f);
    L->s1.clear();
    L->s2.clear();
    L->Ulat.clear();
    L->Vlat.clear();
    L->U.clear();
    L->V.clear();
}

static void shrink_diag(std::vector<float> *d, float a) {
    if (a <= 0.0f || a >= 1.0f || d->empty()) return;
    double m = 0.0;
    for (float v : *d) m += (double)v;
    m /= (double)d->size();
    for (float &v : *d) v = (float)((1.0 - (double)a) * (double)v + (double)a * m);
}

static void make_tiny(Recon *m, int blocks, int hidden, int mid, int vocab, const pchie_nq_recon_config *cfg) {
    m->n_blocks = blocks;
    m->hidden = hidden;
    m->mid = mid;
    m->vocab = vocab;
    m->samples = cfg->samples;
    m->seq = cfg->seq;
    m->seed = cfg->seed;
    m->cfg = *cfg;
    m->blocks.resize((size_t)blocks);
    Rng rng;
    rng_seed(&rng, cfg->seed ^ 0xC0FFEEULL);
    const float scale = 1.0f / std::sqrt((float)hidden);
    m->teacher_up.assign((size_t)blocks * up_n(m), 0.0f);
    m->teacher_down.assign((size_t)blocks * down_n(m), 0.0f);
    m->head.assign(head_n(m), 0.0f);
    for (int b = 0; b < blocks; ++b) {
        linear_init_dense(&m->blocks[b].up, mid, hidden);
        linear_init_dense(&m->blocks[b].down, hidden, mid);
        for (size_t i = 0, n = up_n(m); i < n; ++i) {
            const float v = rng_normal(&rng) * scale;
            m->blocks[b].up.W[i] = v;
            m->teacher_up[(size_t)b * n + i] = v;
        }
        for (size_t i = 0, n = down_n(m); i < n; ++i) {
            const float v = rng_normal(&rng) * scale;
            m->blocks[b].down.W[i] = v;
            m->teacher_down[(size_t)b * n + i] = v;
        }
    }
    for (size_t i = 0, n = head_n(m); i < n; ++i) m->head[i] = rng_normal(&rng) * scale;
    m->calib.assign(act_n(m), 0.0f);
    for (float &v : m->calib) v = rng_normal(&rng) * 0.5f;
    m->student_act = m->calib;
    m->teacher_act = m->calib;
}

/* Correct factored forward using signed latents (STE). */
static void factored_W(const Linear *L, std::vector<float> *W) {
    if (!L->factored) {
        *W = L->W;
        return;
    }
    const int r = L->rank;
    W->assign((size_t)L->rows * (size_t)L->cols, 0.0f);
    for (int i = 0; i < L->rows; ++i) {
        for (int j = 0; j < L->cols; ++j) {
            double acc = 0.0;
            if (L->finalized) {
                const int8_t *u = L->U.data() + (size_t)i * (size_t)r;
                const int8_t *v = L->V.data() + (size_t)j * (size_t)r;
                for (int k = 0; k < r; ++k) acc += (double)u[k] * (double)v[k];
            } else {
                const float *u = L->Ulat.data() + (size_t)i * (size_t)r;
                const float *v = L->Vlat.data() + (size_t)j * (size_t)r;
                for (int k = 0; k < r; ++k) {
                    const double su = u[k] >= 0.0f ? 1.0 : -1.0;
                    const double sv = v[k] >= 0.0f ? 1.0 : -1.0;
                    acc += su * sv;
                }
            }
            (*W)[(size_t)i * (size_t)L->cols + (size_t)j] = (float)((double)L->s1[i] * acc * (double)L->s2[j]);
        }
    }
}

struct BlockCache {
    std::vector<float> x, z, h, y, W1, W2, dW1, dW2, dz, dh, dy, dx;
};

static void block_forward(const Block *B, const float *X, int T, int hidden, int mid, BlockCache *c, int need_bwd) {
    c->W1.clear();
    c->W2.clear();
    factored_W(&B->up, &c->W1);
    factored_W(&B->down, &c->W2);
    if (need_bwd) {
        c->x.assign(X, X + (size_t)T * (size_t)hidden);
        c->z.assign((size_t)T * (size_t)mid, 0.0f);
        c->h.assign((size_t)T * (size_t)mid, 0.0f);
    } else {
        c->z.resize((size_t)T * (size_t)mid);
        c->h.resize((size_t)T * (size_t)mid);
    }
    c->y.resize((size_t)T * (size_t)hidden);
    gemm_xtwt(X, c->W1.data(), c->z.data(), T, mid, hidden);
    for (int i = 0, n = T * mid; i < n; ++i) c->h[i] = gelu(c->z[i]);
    gemm_xtwt(c->h.data(), c->W2.data(), c->y.data(), T, hidden, mid);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < hidden; ++i)
            c->y[(size_t)t * (size_t)hidden + (size_t)i] += X[(size_t)t * (size_t)hidden + (size_t)i];
}

/* dY is incoming gradient on the residual output. Accumulates dW into dense or STE latents/scales. */
static void block_backward(Block *B, BlockCache *c, const float *dY, int T, int hidden, int mid, int train_dense,
                           int train_fact) {
    c->dy.assign(dY, dY + (size_t)T * (size_t)hidden);
    c->dW2.assign((size_t)hidden * (size_t)mid, 0.0f);
    c->dW1.assign((size_t)mid * (size_t)hidden, 0.0f);
    c->dh.assign((size_t)T * (size_t)mid, 0.0f);
    c->dz.assign((size_t)T * (size_t)mid, 0.0f);
    c->dx.assign((size_t)T * (size_t)hidden, 0.0f);
    gemm_dW(c->h.data(), c->dy.data(), c->dW2.data(), T, hidden, mid, 0);
    gemm_dX(c->W2.data(), c->dy.data(), c->dh.data(), T, hidden, mid, 0);
    for (int i = 0, n = T * mid; i < n; ++i) c->dz[i] = c->dh[i] * gelu_dx(c->z[i]);
    gemm_dW(c->x.data(), c->dz.data(), c->dW1.data(), T, mid, hidden, 0);
    gemm_dX(c->W1.data(), c->dz.data(), c->dx.data(), T, mid, hidden, 0);
    /* residual */
    for (int i = 0, n = T * hidden; i < n; ++i) c->dx[i] += c->dy[i];

    auto apply_dense_or_ste = [&](Linear *L, const float *dW) {
        if (!L->factored && train_dense) {
            /* caller uses dW with Adam on L->W */
            return;
        }
        if (L->factored && train_fact && !L->finalized) {
            const int r = L->rank, out = L->rows, in = L->cols;
            /* dW[i,j] = d(s1[i]*acc*s2[j]); acc = sum_k sign(Uik) sign(Vjk)
             * STE: d acc / d Uik = sign(Vjk)  (treating d sign(U)/dU = 1) */
            for (int i = 0; i < out; ++i) {
                double gs1 = 0.0;
                for (int j = 0; j < in; ++j) {
                    double acc = 0.0;
                    const float *u = L->Ulat.data() + (size_t)i * (size_t)r;
                    const float *v = L->Vlat.data() + (size_t)j * (size_t)r;
                    for (int k = 0; k < r; ++k) {
                        const double su = u[k] >= 0.0f ? 1.0 : -1.0;
                        const double sv = v[k] >= 0.0f ? 1.0 : -1.0;
                        acc += su * sv;
                    }
                    const double gw = (double)dW[(size_t)i * (size_t)in + (size_t)j];
                    gs1 += gw * acc * (double)L->s2[j];
                }
                /* store in dW buffer's extra? we accumulate into Ulat/s via separate grads later */
                (void)gs1;
            }
        }
    };
    apply_dense_or_ste(&B->down, c->dW2.data());
    apply_dense_or_ste(&B->up, c->dW1.data());
}

static void ste_grads(const Linear *L, const float *dW, std::vector<float> *gU, std::vector<float> *gV,
                      std::vector<float> *gs1, std::vector<float> *gs2) {
    const int r = L->rank, out = L->rows, in = L->cols;
    gU->assign((size_t)out * (size_t)r, 0.0f);
    gV->assign((size_t)in * (size_t)r, 0.0f);
    gs1->assign((size_t)out, 0.0f);
    gs2->assign((size_t)in, 0.0f);
    auto su_at = [&](int i, int k) -> double {
        if (L->finalized && !L->U.empty()) return L->U[(size_t)i * (size_t)r + (size_t)k] >= 0 ? 1.0 : -1.0;
        if (!L->Ulat.empty()) return L->Ulat[(size_t)i * (size_t)r + (size_t)k] >= 0.0f ? 1.0 : -1.0;
        return 1.0;
    };
    auto sv_at = [&](int j, int k) -> double {
        if (L->finalized && !L->V.empty()) return L->V[(size_t)j * (size_t)r + (size_t)k] >= 0 ? 1.0 : -1.0;
        if (!L->Vlat.empty()) return L->Vlat[(size_t)j * (size_t)r + (size_t)k] >= 0.0f ? 1.0 : -1.0;
        return 1.0;
    };
    for (int i = 0; i < out; ++i) {
        for (int j = 0; j < in; ++j) {
            double acc = 0.0;
            for (int k = 0; k < r; ++k) acc += su_at(i, k) * sv_at(j, k);
            const double gw = (double)dW[(size_t)i * (size_t)in + (size_t)j];
            (*gs1)[i] += (float)(gw * acc * (double)L->s2[j]);
            (*gs2)[j] += (float)(gw * (double)L->s1[i] * acc);
            const double c = gw * (double)L->s1[i] * (double)L->s2[j];
            for (int k = 0; k < r; ++k) {
                const double su = su_at(i, k);
                const double sv = sv_at(j, k);
                (*gU)[(size_t)i * (size_t)r + (size_t)k] += (float)(c * sv); /* STE on U */
                (*gV)[(size_t)j * (size_t)r + (size_t)k] += (float)(c * su); /* STE on V */
            }
        }
    }
}

/* ---------- pause / busy ---------- */

uint32_t pchie_nq_last_input_age_ms(void) {
#ifdef _WIN32
    LASTINPUTINFO info;
    info.cbSize = sizeof info;
    if (!GetLastInputInfo(&info)) return 0x7fffffffu;
    return (uint32_t)(GetTickCount() - info.dwTime);
#else
    return 0x7fffffffu;
#endif
}

#ifdef _WIN32
static int name_matches_conflict(const char *exe) {
    if (!exe) return 0;
    char buf[MAX_PATH];
    size_t n = 0;
    for (; exe[n] && n + 1 < sizeof buf; ++n) {
        char c = exe[n];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        buf[n] = c;
    }
    buf[n] = 0;
    const char *keys[] = {"pchie_nanoquant.exe", "playmutexfree", "play-g1", "play_g1", "playg1",
                          "playa9", "playb9", "playc9", "playd9", "playe9", "play-a9", "play-b9", "play-c9", "play-d9",
                          "play-e9", nullptr};
    for (int i = 0; keys[i]; ++i) {
        if (std::strstr(buf, keys[i])) return 1;
    }
    return 0;
}
#endif

#ifdef _WIN32
static int nvidia_other_compute(char *why, size_t cap) {
    /* Best-effort: nvidia-smi is enough and does not need an administrator. */
    const char *cmd = "cmd.exe /C nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>NUL";
    FILE *f = _popen(cmd, "r");
    if (!f) return 0;
    char line[1024];
    int busy = 0;
    const DWORD self = GetCurrentProcessId();
    while (fgets(line, sizeof line, f)) {
        unsigned long pid = 0;
        char name[512] = {0};
        if (sscanf(line, "%lu, %511[^\r\n]", &pid, name) < 1) continue;
        if (pid == 0 || pid == self) continue;
        busy = 1;
        if (why && cap) std::snprintf(why, cap, "GPU already in use by pid %lu (%s)", pid, name);
        break;
    }
    _pclose(f);
    return busy;
}
#endif

int pchie_nq_gpu_is_busy(char *why, size_t cap) {
#ifdef _WIN32
    const DWORD self = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32 pe;
        pe.dwSize = sizeof pe;
        if (Process32First(snap, &pe)) {
            do {
                if (pe.th32ProcessID == self) continue;
                char exe[MAX_PATH];
#ifdef UNICODE
                WideCharToMultiByte(CP_ACP, 0, pe.szExeFile, -1, exe, MAX_PATH, nullptr, nullptr);
#else
                std::snprintf(exe, sizeof exe, "%s", pe.szExeFile);
#endif
                if (name_matches_conflict(exe)) {
                    if (why && cap)
                        std::snprintf(why, cap, "another job or a Play build is running (%s, pid %lu)", exe,
                                      (unsigned long)pe.th32ProcessID);
                    CloseHandle(snap);
                    return 1;
                }
            } while (Process32Next(snap, &pe));
        }
        CloseHandle(snap);
    }
    return nvidia_other_compute(why, cap);
#else
    if (why && cap) std::snprintf(why, cap, "busy-query is only implemented on Windows");
    return 0;
#endif
}

static int should_pause(const pchie_nq_recon_config *cfg, char *error, size_t cap, const char *why_pause) {
    if (cfg->pause_if_input_ms > 0) {
        const uint32_t age = pchie_nq_last_input_age_ms();
        if (age < (uint32_t)cfg->pause_if_input_ms) {
            if (error && cap)
                std::snprintf(error, cap, "paused: the user used the machine %u ms ago (%s)", (unsigned)age,
                               why_pause ? why_pause : "check");
            return 1;
        }
    }
    if (cfg->respect_gpu_busy) {
        char why[256];
        if (pchie_nq_gpu_is_busy(why, sizeof why)) {
            if (error && cap)
                std::snprintf(error, cap, "paused: %s (%s)", why, why_pause ? why_pause : "check");
            return 1;
        }
    }
    return 0;
}

/* ---------- I/O: journal and checkpoints ---------- */

static std::string join_path(const char *a, const char *b) {
    std::string s(a);
    if (!s.empty() && s.back() != '/' && s.back() != '\\') s.push_back('/');
    s += b;
    return s;
}

struct Journal {
    int n_blocks, hidden, mid, vocab, samples, seq, completed, kd_done;
    uint64_t seed;
    char teacher_sha[65], calib_sha[65], kd_sha[65];
    std::vector<std::string> block_sha;
};

static int write_bytes_atomic(const char *path, const std::vector<uint8_t> &body, uint8_t sha[32], char *error,
                              size_t cap) {
    sha256_bytes(body.data(), body.size(), sha);
    const std::string part = std::string(path) + ".partial";
    FILE *f = std::fopen(part.c_str(), "wb");
    if (!f) {
        set_error(error, cap, "could not open a temporary checkpoint");
        return 0;
    }
    int ok = std::fwrite(body.data(), 1, body.size(), f) == body.size();
    ok = ok && std::fwrite(sha, 1, 32, f) == 32;
    ok = ok && fsync_file(f);
    std::fclose(f);
    if (!ok || !atomic_replace(part.c_str(), path)) {
        std::remove(part.c_str());
        set_error(error, cap, "could not finish an atomic checkpoint write");
        return 0;
    }
    return 1;
}

static int read_file(const char *path, std::vector<uint8_t> *out) {
    FILE *f = std::fopen(path, "rb");
    if (!f) return 0;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < 0) {
        std::fclose(f);
        return 0;
    }
    out->resize((size_t)n);
    const int ok = std::fread(out->data(), 1, (size_t)n, f) == (size_t)n;
    std::fclose(f);
    return ok;
}

static int read_checkpoint(const char *path, std::vector<uint8_t> *payload, uint8_t sha[32], char *error, size_t cap) {
    std::vector<uint8_t> all;
    if (!read_file(path, &all) || all.size() < 32) {
        set_error(error, cap, "could not read a checkpoint");
        return 0;
    }
    std::memcpy(sha, all.data() + all.size() - 32, 32);
    payload->assign(all.begin(), all.end() - 32);
    uint8_t got[32];
    sha256_bytes(payload->data(), payload->size(), got);
    if (std::memcmp(got, sha, 32) != 0) {
        set_error(error, cap, "a checkpoint's SHA-256 did not match its payload");
        return 0;
    }
    return 1;
}

static void put_u32(std::vector<uint8_t> *o, uint32_t v) {
    o->push_back((uint8_t)v);
    o->push_back((uint8_t)(v >> 8));
    o->push_back((uint8_t)(v >> 16));
    o->push_back((uint8_t)(v >> 24));
}
static void put_f32s(std::vector<uint8_t> *o, const float *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    o->insert(o->end(), b, b + n * sizeof(float));
}
static void put_i8s(std::vector<uint8_t> *o, const int8_t *p, size_t n) {
    o->insert(o->end(), (const uint8_t *)p, (const uint8_t *)p + n);
}

static int get_u32(const std::vector<uint8_t> &b, size_t *off, uint32_t *v) {
    if (*off + 4 > b.size()) return 0;
    *v = (uint32_t)b[*off] | ((uint32_t)b[*off + 1] << 8) | ((uint32_t)b[*off + 2] << 16) | ((uint32_t)b[*off + 3] << 24);
    *off += 4;
    return 1;
}
static int get_f32s(const std::vector<uint8_t> &b, size_t *off, float *p, size_t n) {
    const size_t bytes = n * sizeof(float);
    if (*off + bytes > b.size()) return 0;
    std::memcpy(p, b.data() + *off, bytes);
    *off += bytes;
    return 1;
}
static int get_i8s(const std::vector<uint8_t> &b, size_t *off, int8_t *p, size_t n) {
    if (*off + n > b.size()) return 0;
    std::memcpy(p, b.data() + *off, n);
    *off += n;
    return 1;
}

static void pack_linear(std::vector<uint8_t> *o, const Linear *L) {
    put_u32(o, (uint32_t)L->rows);
    put_u32(o, (uint32_t)L->cols);
    put_u32(o, (uint32_t)L->rank);
    put_u32(o, (uint32_t)L->factored);
    put_u32(o, (uint32_t)L->finalized);
    if (!L->factored) {
        put_f32s(o, L->W.data(), (size_t)L->rows * (size_t)L->cols);
    } else {
        put_f32s(o, L->s1.data(), (size_t)L->rows);
        put_f32s(o, L->s2.data(), (size_t)L->cols);
        if (L->finalized) {
            put_i8s(o, L->U.data(), (size_t)L->rows * (size_t)L->rank);
            put_i8s(o, L->V.data(), (size_t)L->cols * (size_t)L->rank);
        } else {
            put_f32s(o, L->Ulat.data(), (size_t)L->rows * (size_t)L->rank);
            put_f32s(o, L->Vlat.data(), (size_t)L->cols * (size_t)L->rank);
        }
    }
}

static int unpack_linear(const std::vector<uint8_t> &b, size_t *off, Linear *L) {
    uint32_t rows, cols, rank, fact, fin;
    if (!get_u32(b, off, &rows) || !get_u32(b, off, &cols) || !get_u32(b, off, &rank) || !get_u32(b, off, &fact) ||
        !get_u32(b, off, &fin))
        return 0;
    linear_init_dense(L, (int)rows, (int)cols);
    L->rank = (int)rank;
    L->factored = (int)fact;
    L->finalized = (int)fin;
    if (!L->factored) return get_f32s(b, off, L->W.data(), (size_t)rows * (size_t)cols);
    L->s1.assign(rows, 0.0f);
    L->s2.assign(cols, 0.0f);
    if (!get_f32s(b, off, L->s1.data(), rows) || !get_f32s(b, off, L->s2.data(), cols)) return 0;
    if (L->finalized) {
        L->U.assign((size_t)rows * (size_t)rank, 0);
        L->V.assign((size_t)cols * (size_t)rank, 0);
        if (!get_i8s(b, off, L->U.data(), (size_t)rows * (size_t)rank) ||
            !get_i8s(b, off, L->V.data(), (size_t)cols * (size_t)rank))
            return 0;
        L->Ulat.resize((size_t)rows * (size_t)rank);
        L->Vlat.resize((size_t)cols * (size_t)rank);
        for (size_t i = 0, n = L->Ulat.size(); i < n; ++i) L->Ulat[i] = (float)L->U[i];
        for (size_t i = 0, n = L->Vlat.size(); i < n; ++i) L->Vlat[i] = (float)L->V[i];
        return 1;
    }
    L->Ulat.assign((size_t)rows * (size_t)rank, 0.0f);
    L->Vlat.assign((size_t)cols * (size_t)rank, 0.0f);
    return get_f32s(b, off, L->Ulat.data(), (size_t)rows * (size_t)rank) &&
           get_f32s(b, off, L->Vlat.data(), (size_t)cols * (size_t)rank);
}

static int save_journal(const char *out_dir, const Journal *j, char *error, size_t cap) {
    std::string body;
    body += "NQJ1\n";
    char line[256];
    std::snprintf(line, sizeof line, "blocks %d\nhidden %d\nmid %d\nvocab %d\nsamples %d\nseq %d\nseed %llu\n",
                  j->n_blocks, j->hidden, j->mid, j->vocab, j->samples, j->seq, (unsigned long long)j->seed);
    body += line;
    body += "teacher_sha ";
    body += j->teacher_sha;
    body += "\ncalib_sha ";
    body += j->calib_sha;
    body += "\ncompleted ";
    std::snprintf(line, sizeof line, "%d\nkd %d\n", j->completed, j->kd_done);
    body += line;
    if (j->kd_done && j->kd_sha[0]) {
        body += "kd_sha ";
        body += j->kd_sha;
        body += "\n";
    }
    for (int i = 0; i < j->completed && i < (int)j->block_sha.size(); ++i) {
        std::snprintf(line, sizeof line, "block %d sha ", i);
        body += line;
        body += j->block_sha[i];
        body += "\n";
    }
    const std::string path = join_path(out_dir, "journal.nqj");
    uint8_t sha[32];
    std::vector<uint8_t> raw(body.begin(), body.end());
    return write_bytes_atomic(path.c_str(), raw, sha, error, cap);
}

static int load_journal(const char *out_dir, Journal *j, char *error, size_t cap) {
    std::vector<uint8_t> raw;
    uint8_t sha[32];
    const std::string path = join_path(out_dir, "journal.nqj");
    if (!read_checkpoint(path.c_str(), &raw, sha, error, cap)) {
        set_error(error, cap, "could not read the progress journal");
        return 0;
    }
    raw.push_back(0);
    const char *p = (const char *)raw.data();
    if (std::strncmp(p, "NQJ1", 4) != 0) {
        set_error(error, cap, "the progress journal is not in the NQJ1 format");
        return 0;
    }
    *j = Journal();
    unsigned long long seed = 0;
    for (const char *line = p; *line;) {
        const char *nl = std::strchr(line, '\n');
        std::string s(line, nl ? (size_t)(nl - line) : std::strlen(line));
        if (s.rfind("blocks ", 0) == 0) j->n_blocks = std::atoi(s.c_str() + 7);
        else if (s.rfind("hidden ", 0) == 0) j->hidden = std::atoi(s.c_str() + 7);
        else if (s.rfind("mid ", 0) == 0) j->mid = std::atoi(s.c_str() + 4);
        else if (s.rfind("vocab ", 0) == 0) j->vocab = std::atoi(s.c_str() + 6);
        else if (s.rfind("samples ", 0) == 0) j->samples = std::atoi(s.c_str() + 8);
        else if (s.rfind("seq ", 0) == 0) j->seq = std::atoi(s.c_str() + 4);
        else if (s.rfind("seed ", 0) == 0) {
            seed = std::strtoull(s.c_str() + 5, nullptr, 10);
            j->seed = (uint64_t)seed;
        } else if (s.rfind("teacher_sha ", 0) == 0) {
            std::snprintf(j->teacher_sha, sizeof j->teacher_sha, "%s", s.c_str() + 12);
        } else if (s.rfind("calib_sha ", 0) == 0) {
            std::snprintf(j->calib_sha, sizeof j->calib_sha, "%s", s.c_str() + 10);
        } else if (s.rfind("completed ", 0) == 0) j->completed = std::atoi(s.c_str() + 10);
        else if (s.rfind("kd_sha ", 0) == 0) {
            std::snprintf(j->kd_sha, sizeof j->kd_sha, "%s", s.c_str() + 7);
        } else if (s.rfind("kd ", 0) == 0) j->kd_done = std::atoi(s.c_str() + 3);
        else if (s.rfind("block ", 0) == 0) {
            int idx = 0;
            char blockhex[65] = {0};
            if (std::sscanf(s.c_str(), "block %d sha %64s", &idx, blockhex) == 2) {
                if ((int)j->block_sha.size() <= idx) j->block_sha.resize((size_t)idx + 1);
                j->block_sha[idx] = blockhex;
            }
        }
        if (!nl) break;
        line = nl + 1;
    }
    return 1;
}

static void pack_teacher(const Recon *m, std::vector<uint8_t> *o) {
    o->clear();
    o->insert(o->end(), {'N', 'Q', 'T', '1'});
    put_u32(o, (uint32_t)m->n_blocks);
    put_u32(o, (uint32_t)m->hidden);
    put_u32(o, (uint32_t)m->mid);
    put_u32(o, (uint32_t)m->vocab);
    put_f32s(o, m->teacher_up.data(), m->teacher_up.size());
    put_f32s(o, m->teacher_down.data(), m->teacher_down.size());
    put_f32s(o, m->head.data(), m->head.size());
}

static int unpack_teacher(Recon *m, const std::vector<uint8_t> &b, char *error, size_t cap) {
    if (b.size() < 4 || std::memcmp(b.data(), "NQT1", 4) != 0) {
        set_error(error, cap, "the teacher snapshot is not in the NQT1 format");
        return 0;
    }
    size_t off = 4;
    uint32_t nb, h, mid, v;
    if (!get_u32(b, &off, &nb) || !get_u32(b, &off, &h) || !get_u32(b, &off, &mid) || !get_u32(b, &off, &v)) return 0;
    m->n_blocks = (int)nb;
    m->hidden = (int)h;
    m->mid = (int)mid;
    m->vocab = (int)v;
    m->teacher_up.assign((size_t)nb * (size_t)mid * (size_t)h, 0.0f);
    m->teacher_down.assign((size_t)nb * (size_t)h * (size_t)mid, 0.0f);
    m->head.assign((size_t)v * (size_t)h, 0.0f);
    if (!get_f32s(b, &off, m->teacher_up.data(), m->teacher_up.size()) ||
        !get_f32s(b, &off, m->teacher_down.data(), m->teacher_down.size()) ||
        !get_f32s(b, &off, m->head.data(), m->head.size())) {
        set_error(error, cap, "the teacher snapshot was truncated");
        return 0;
    }
    return 1;
}

static void pack_block_ckpt(const Recon *m, int bi, std::vector<uint8_t> *o) {
    o->clear();
    o->insert(o->end(), {'N', 'Q', 'B', '1'});
    put_u32(o, 1);
    put_u32(o, (uint32_t)bi);
    put_u32(o, (uint32_t)m->n_blocks);
    put_u32(o, (uint32_t)m->hidden);
    put_u32(o, (uint32_t)m->mid);
    put_u32(o, (uint32_t)m->samples);
    put_u32(o, (uint32_t)m->seq);
    for (int i = 0; i <= bi; ++i) {
        pack_linear(o, &m->blocks[i].up);
        pack_linear(o, &m->blocks[i].down);
    }
    put_f32s(o, m->student_act.data(), act_n(m));
    put_f32s(o, m->teacher_act.data(), act_n(m));
}

static int unpack_block_ckpt(Recon *m, const std::vector<uint8_t> &b, int expected, char *error, size_t cap) {
    if (b.size() < 4 || std::memcmp(b.data(), "NQB1", 4) != 0) {
        set_error(error, cap, "a block checkpoint is not in the NQB1 format");
        return 0;
    }
    size_t off = 4;
    uint32_t ver, bi, nb, h, mid, samples, seq;
    if (!get_u32(b, &off, &ver) || !get_u32(b, &off, &bi) || !get_u32(b, &off, &nb) || !get_u32(b, &off, &h) ||
        !get_u32(b, &off, &mid) || !get_u32(b, &off, &samples) || !get_u32(b, &off, &seq))
        return 0;
    if ((int)bi != expected) {
        set_error(error, cap, "a block checkpoint's index did not match the journal");
        return 0;
    }
    m->samples = (int)samples;
    m->seq = (int)seq;
    if ((int)m->blocks.size() < (int)nb) m->blocks.resize(nb);
    m->n_blocks = (int)nb;
    m->hidden = (int)h;
    m->mid = (int)mid;
    for (int i = 0; i <= expected; ++i) {
        if (!unpack_linear(b, &off, &m->blocks[i].up) || !unpack_linear(b, &off, &m->blocks[i].down)) {
            set_error(error, cap, "a block checkpoint was truncated");
            return 0;
        }
    }
    m->student_act.assign(act_n(m), 0.0f);
    m->teacher_act.assign(act_n(m), 0.0f);
    if (!get_f32s(b, &off, m->student_act.data(), act_n(m)) || !get_f32s(b, &off, m->teacher_act.data(), act_n(m))) {
        set_error(error, cap, "a block checkpoint was missing activation caches");
        return 0;
    }
    return 1;
}

static void pack_kd_final(const Recon *m, std::vector<uint8_t> *o) {
    o->clear();
    o->insert(o->end(), {'N', 'Q', 'K', '1'});
    put_u32(o, (uint32_t)m->n_blocks);
    for (int i = 0; i < m->n_blocks; ++i) {
        pack_linear(o, &m->blocks[i].up);
        pack_linear(o, &m->blocks[i].down);
    }
}

static int unpack_kd_final(Recon *m, const std::vector<uint8_t> &b, char *error, size_t cap) {
    if (b.size() < 4 || std::memcmp(b.data(), "NQK1", 4) != 0) {
        set_error(error, cap, "the distillation checkpoint is not in the NQK1 format");
        return 0;
    }
    size_t off = 4;
    uint32_t nb = 0;
    if (!get_u32(b, &off, &nb) || (int)nb != m->n_blocks) {
        set_error(error, cap, "the distillation checkpoint's block count did not match");
        return 0;
    }
    if ((int)m->blocks.size() < m->n_blocks) m->blocks.resize((size_t)m->n_blocks);
    for (int i = 0; i < m->n_blocks; ++i) {
        if (!unpack_linear(b, &off, &m->blocks[i].up) || !unpack_linear(b, &off, &m->blocks[i].down)) {
            set_error(error, cap, "the distillation checkpoint was truncated");
            return 0;
        }
    }
    return 1;
}

static int commit_kd_final(const char *out_dir, Recon *m, Journal *j, char *error, size_t cap) {
    std::vector<uint8_t> body;
    pack_kd_final(m, &body);
    uint8_t sha[32];
    if (!write_bytes_atomic(join_path(out_dir, "kd-final.nqk").c_str(), body, sha, error, cap)) return 0;
    hex32(sha, j->kd_sha);
    j->kd_done = 1;
    return save_journal(out_dir, j, error, cap);
}

static int load_kd_final(const char *out_dir, Recon *m, const Journal *j, char *error, size_t cap) {
    if (!j->kd_done) return 1;
    std::vector<uint8_t> payload;
    uint8_t sha[32], expect[32];
    if (!j->kd_sha[0] || !parse_hex32(j->kd_sha, expect)) {
        set_error(error, cap, "the journal is missing the distillation SHA-256");
        return 0;
    }
    if (!read_checkpoint(join_path(out_dir, "kd-final.nqk").c_str(), &payload, sha, error, cap)) return 0;
    if (std::memcmp(sha, expect, 32) != 0) {
        set_error(error, cap, "the distillation checkpoint's SHA-256 did not match the journal");
        return 0;
    }
    return unpack_kd_final(m, payload, error, cap);
}

/* ---------- factorize a dense linear into STE latents then harden ---------- */

static int factorize_linear(Linear *L, const pchie_nq_recon_config *cfg, char *error, size_t cap) {
    pchie_nq_config fc;
    pchie_nq_config_init(&fc);
    fc.bits = cfg->bits;
    fc.outer_iters = cfg->admm_outer_iters;
    fc.inner_iters = cfg->admm_inner_iters;
    fc.reg = cfg->admm_reg;
    fc.seed = cfg->seed;
    fc.use_cuda = cfg->use_cuda;
    const int rank = pchie_nq_choose_rank(L->rows, L->cols, cfg->bits);
    if (rank <= 0) {
        set_error(error, cap, "could not choose a rank for a layer");
        return 0;
    }
    L->rank = rank;
    L->U.assign((size_t)L->rows * (size_t)rank, 0);
    L->V.assign((size_t)L->cols * (size_t)rank, 0);
    L->s1.assign((size_t)L->rows, 0.0f);
    L->s2.assign((size_t)L->cols, 0.0f);
    pchie_nq_stats st;
    char err[PCHIE_NQ_ERROR_CAPACITY];
    const pchie_nq_status s =
        pchie_nq_factorize(L->W.data(), L->rows, L->cols, L->din.data(), L->dout.data(), &fc, L->U.data(), L->V.data(),
                           L->s1.data(), L->s2.data(), &st, err, sizeof err);
    if (s != PCHIE_NQ_OK) {
        if (error && cap) std::snprintf(error, cap, "factorize failed: %s", err);
        return 0;
    }
    L->Ulat.assign((size_t)L->rows * (size_t)rank, 0.0f);
    L->Vlat.assign((size_t)L->cols * (size_t)rank, 0.0f);
    for (size_t i = 0, n = L->Ulat.size(); i < n; ++i) L->Ulat[i] = (float)L->U[i];
    for (size_t i = 0, n = L->Vlat.size(); i < n; ++i) L->Vlat[i] = (float)L->V[i];
    L->factored = 1;
    L->finalized = 0;
    return 1;
}

static void finalize_linear(Linear *L) {
    if (!L->factored || L->finalized) return;
    const int r = L->rank;
    L->U.resize((size_t)L->rows * (size_t)r);
    L->V.resize((size_t)L->cols * (size_t)r);
    for (int i = 0; i < L->rows; ++i)
        for (int k = 0; k < r; ++k)
            L->U[(size_t)i * (size_t)r + (size_t)k] = (L->Ulat[(size_t)i * (size_t)r + (size_t)k] >= 0.0f) ? (int8_t)1
                                                                                                          : (int8_t)-1;
    for (int j = 0; j < L->cols; ++j)
        for (int k = 0; k < r; ++k)
            L->V[(size_t)j * (size_t)r + (size_t)k] = (L->Vlat[(size_t)j * (size_t)r + (size_t)k] >= 0.0f) ? (int8_t)1
                                                                                                          : (int8_t)-1;
    L->finalized = 1;
    L->Ulat.resize((size_t)L->rows * (size_t)r);
    L->Vlat.resize((size_t)L->cols * (size_t)r);
    for (size_t i = 0, n = L->Ulat.size(); i < n; ++i) L->Ulat[i] = (float)L->U[i];
    for (size_t i = 0, n = L->Vlat.size(); i < n; ++i) L->Vlat[i] = (float)L->V[i];
}

/* Weighted MSE over the residual output, importance[hidden]. */
static double sample_loss_and_dy(const float *y, const float *tgt, const float *imp, int T, int hidden, float *dY) {
    double loss = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < hidden; ++i) {
            const double e = (double)y[(size_t)t * (size_t)hidden + (size_t)i] -
                             (double)tgt[(size_t)t * (size_t)hidden + (size_t)i];
            const double w = (double)imp[i];
            loss += e * e * w;
            dY[(size_t)t * (size_t)hidden + (size_t)i] = (float)(2.0 * e * w);
        }
    }
    return loss;
}

static int slice_act(const float *all, int samples, int seq, int hidden, int idx, std::vector<float> *one) {
    if (idx < 0 || idx >= samples) return 0;
    one->assign(all + (size_t)idx * (size_t)seq * (size_t)hidden,
                all + (size_t)(idx + 1) * (size_t)seq * (size_t)hidden);
    return 1;
}

static void restore_teacher_block(Recon *m, int bi) {
    const size_t un = up_n(m), dn = down_n(m);
    linear_init_dense(&m->blocks[bi].up, m->mid, m->hidden);
    linear_init_dense(&m->blocks[bi].down, m->hidden, m->mid);
    std::memcpy(m->blocks[bi].up.W.data(), m->teacher_up.data() + (size_t)bi * un, un * sizeof(float));
    std::memcpy(m->blocks[bi].down.W.data(), m->teacher_down.data() + (size_t)bi * dn, dn * sizeof(float));
    shrink_diag(&m->blocks[bi].up.din, m->cfg.shrinkage);
    shrink_diag(&m->blocks[bi].up.dout, m->cfg.shrinkage);
    shrink_diag(&m->blocks[bi].down.din, m->cfg.shrinkage);
    shrink_diag(&m->blocks[bi].down.dout, m->cfg.shrinkage);
}

static int tune_nonfact(Recon *m, int bi, const float *inputs, const float *targets, char *error, size_t cap) {
    if (!m->cfg.tune_nonfact) return 1;
    Block *B = &m->blocks[bi];
    /* Only still-dense linears are trained, as in the published schedule. */
    Linear *train[2];
    int ntr = 0;
    if (!B->up.factored) train[ntr++] = &B->up;
    if (!B->down.factored) train[ntr++] = &B->down;
    if (ntr == 0) return 1;
    size_t nparam = 0;
    for (int i = 0; i < ntr; ++i) nparam += (size_t)train[i]->rows * (size_t)train[i]->cols;
    std::vector<float> packed(nparam), grad(nparam, 0.0f);
    size_t off = 0;
    for (int i = 0; i < ntr; ++i) {
        const size_t n = (size_t)train[i]->rows * (size_t)train[i]->cols;
        std::memcpy(packed.data() + off, train[i]->W.data(), n * sizeof(float));
        off += n;
    }
    Adam adam;
    adam_init(&adam, nparam);
    Rng rng;
    rng_seed(&rng, m->cfg.seed);
    const int steps = (int)std::ceil((double)m->samples / (double)(m->cfg.nonfact_batch > 0 ? m->cfg.nonfact_batch : 1)) *
                      m->cfg.nonfact_epochs;
    int step = 0, batch = m->cfg.nonfact_batch > 0 ? m->cfg.nonfact_batch : 1;
    const int T = m->seq, H = m->hidden, M = m->mid;
    const float *imp = B->down.dout.data();
    BlockCache cache;
    std::vector<int> perm((size_t)m->samples);
    std::vector<float> x, tgt, dY((size_t)T * (size_t)H);
    for (int ep = 0; ep < m->cfg.nonfact_epochs; ++ep) {
        rng_perm(&rng, m->samples, perm.data());
        std::fill(grad.begin(), grad.end(), 0.0f);
        int in_batch = 0;
        for (int s = 0; s < m->samples; ++s) {
            if (should_pause(&m->cfg, error, cap, "non-factored tune")) return -1;
            slice_act(inputs, m->samples, T, H, perm[s], &x);
            slice_act(targets, m->samples, T, H, perm[s], &tgt);
            block_forward(B, x.data(), T, H, M, &cache, 1);
            sample_loss_and_dy(cache.y.data(), tgt.data(), imp, T, H, dY.data());
            block_backward(B, &cache, dY.data(), T, H, M, 1, 0);
            /* pack grads */
            size_t go = 0;
            for (int i = 0; i < ntr; ++i) {
                const float *src = (train[i] == &B->up) ? cache.dW1.data() : cache.dW2.data();
                const size_t n = (size_t)train[i]->rows * (size_t)train[i]->cols;
                for (size_t k = 0; k < n; ++k) grad[go + k] += src[k] / (float)batch;
                go += n;
            }
            ++in_batch;
            if (in_batch == batch || s + 1 == m->samples) {
                const float lr = cosine_lr(m->cfg.nonfact_lr, m->cfg.nonfact_lr * 1e-4f, step, steps);
                adam_step(&adam, packed.data(), grad.data(), nparam, lr);
                off = 0;
                for (int i = 0; i < ntr; ++i) {
                    const size_t n = (size_t)train[i]->rows * (size_t)train[i]->cols;
                    std::memcpy(train[i]->W.data(), packed.data() + off, n * sizeof(float));
                    off += n;
                }
                std::fill(grad.begin(), grad.end(), 0.0f);
                in_batch = 0;
                ++step;
            }
        }
    }
    return 1;
}

static int tune_fact(Recon *m, int bi, const float *inputs, const float *targets, char *error, size_t cap) {
    if (!m->cfg.tune_fact) return 1;
    Block *B = &m->blocks[bi];
    Linear *tr[2];
    int ntr = 0;
    if (B->up.factored && !B->up.finalized) tr[ntr++] = &B->up;
    if (B->down.factored && !B->down.finalized) tr[ntr++] = &B->down;
    if (ntr == 0) return 1;
    /* Parameter vector: all Ulat, Vlat, s1, s2 for trained factors. Two LRs: binary vs scale. */
    struct Pg {
        Linear *L;
        int is_up;
        size_t u0, v0, s10, s20, nu, nv, ns1, ns2;
    };
    std::vector<Pg> pg;
    size_t nbin = 0, nsc = 0;
    for (int i = 0; i < ntr; ++i) {
        Pg p;
        p.L = tr[i];
        p.is_up = (tr[i] == &B->up);
        p.nu = (size_t)p.L->rows * (size_t)p.L->rank;
        p.nv = (size_t)p.L->cols * (size_t)p.L->rank;
        p.ns1 = (size_t)p.L->rows;
        p.ns2 = (size_t)p.L->cols;
        p.u0 = nbin;
        nbin += p.nu;
        p.v0 = nbin;
        nbin += p.nv;
        p.s10 = nsc;
        nsc += p.ns1;
        p.s20 = nsc;
        nsc += p.ns2;
        pg.push_back(p);
    }
    std::vector<float> pbin(nbin), gbin(nbin), psc(nsc), gsc(nsc);
    for (const Pg &p : pg) {
        std::memcpy(pbin.data() + p.u0, p.L->Ulat.data(), p.nu * sizeof(float));
        std::memcpy(pbin.data() + p.v0, p.L->Vlat.data(), p.nv * sizeof(float));
        std::memcpy(psc.data() + p.s10, p.L->s1.data(), p.ns1 * sizeof(float));
        std::memcpy(psc.data() + p.s20, p.L->s2.data(), p.ns2 * sizeof(float));
    }
    Adam abin, asc;
    adam_init(&abin, nbin);
    adam_init(&asc, nsc);
    Rng rng;
    rng_seed(&rng, m->cfg.seed);
    const int batch = m->cfg.fact_batch > 0 ? m->cfg.fact_batch : 1;
    const int steps = (int)std::ceil((double)m->samples / (double)batch) * m->cfg.fact_epochs;
    int step = 0;
    const int T = m->seq, H = m->hidden, M = m->mid;
    const float *imp = B->down.dout.data();
    BlockCache cache;
    std::vector<int> perm((size_t)m->samples);
    std::vector<float> x, tgt, dY((size_t)T * (size_t)H), gU, gV, gs1, gs2;
    for (int ep = 0; ep < m->cfg.fact_epochs; ++ep) {
        rng_perm(&rng, m->samples, perm.data());
        std::fill(gbin.begin(), gbin.end(), 0.0f);
        std::fill(gsc.begin(), gsc.end(), 0.0f);
        int in_batch = 0;
        for (int s = 0; s < m->samples; ++s) {
            if (should_pause(&m->cfg, error, cap, "factor tune")) return -1;
            slice_act(inputs, m->samples, T, H, perm[s], &x);
            slice_act(targets, m->samples, T, H, perm[s], &tgt);
            block_forward(B, x.data(), T, H, M, &cache, 1);
            sample_loss_and_dy(cache.y.data(), tgt.data(), imp, T, H, dY.data());
            block_backward(B, &cache, dY.data(), T, H, M, 0, 1);
            for (const Pg &p : pg) {
                const float *dW = p.is_up ? cache.dW1.data() : cache.dW2.data();
                ste_grads(p.L, dW, &gU, &gV, &gs1, &gs2);
                for (size_t k = 0; k < p.nu; ++k) gbin[p.u0 + k] += gU[k] / (float)batch;
                for (size_t k = 0; k < p.nv; ++k) gbin[p.v0 + k] += gV[k] / (float)batch;
                for (size_t k = 0; k < p.ns1; ++k) gsc[p.s10 + k] += gs1[k] / (float)batch;
                for (size_t k = 0; k < p.ns2; ++k) gsc[p.s20 + k] += gs2[k] / (float)batch;
            }
            ++in_batch;
            if (in_batch == batch || s + 1 == m->samples) {
                const float lrb = cosine_lr(m->cfg.fact_binary_lr, m->cfg.fact_scale_lr * 1e-4f, step, steps);
                const float lrs = cosine_lr(m->cfg.fact_scale_lr, m->cfg.fact_scale_lr * 1e-4f, step, steps);
                adam_step(&abin, pbin.data(), gbin.data(), nbin, lrb);
                adam_step(&asc, psc.data(), gsc.data(), nsc, lrs);
                for (const Pg &p : pg) {
                    std::memcpy(p.L->Ulat.data(), pbin.data() + p.u0, p.nu * sizeof(float));
                    std::memcpy(p.L->Vlat.data(), pbin.data() + p.v0, p.nv * sizeof(float));
                    std::memcpy(p.L->s1.data(), psc.data() + p.s10, p.ns1 * sizeof(float));
                    std::memcpy(p.L->s2.data(), psc.data() + p.s20, p.ns2 * sizeof(float));
                }
                std::fill(gbin.begin(), gbin.end(), 0.0f);
                std::fill(gsc.begin(), gsc.end(), 0.0f);
                in_batch = 0;
                ++step;
            }
        }
    }
    return 1;
}

static void teacher_block_forward(const Recon *m, int bi, const float *X, int T, std::vector<float> *Y) {
    Block tb;
    linear_init_dense(&tb.up, m->mid, m->hidden);
    linear_init_dense(&tb.down, m->hidden, m->mid);
    std::memcpy(tb.up.W.data(), m->teacher_up.data() + (size_t)bi * up_n(m), up_n(m) * sizeof(float));
    std::memcpy(tb.down.W.data(), m->teacher_down.data() + (size_t)bi * down_n(m), down_n(m) * sizeof(float));
    BlockCache c;
    block_forward(&tb, X, T, m->hidden, m->mid, &c, 0);
    *Y = c.y;
}

static void student_block_forward(const Block *B, const float *X, int T, int hidden, int mid, std::vector<float> *Y) {
    BlockCache c;
    block_forward(B, X, T, hidden, mid, &c, 0);
    *Y = c.y;
}

/* Scale-only KD on the tiny language head. Teacher logits are cached. */
static int tune_model_kd(Recon *m, char *error, size_t cap, double *out_loss) {
    if (!m->cfg.tune_model || m->vocab <= 0) {
        if (out_loss) *out_loss = 0.0;
        return 1;
    }
    std::vector<Linear *> fl;
    size_t nsc = 0;
    for (int b = 0; b < m->n_blocks; ++b) {
        for (Linear *L : {&m->blocks[b].up, &m->blocks[b].down}) {
            if (!L->factored) continue;
            fl.push_back(L);
            nsc += (size_t)L->rows + (size_t)L->cols;
        }
    }
    if (nsc == 0) {
        if (out_loss) *out_loss = 0.0;
        return 1;
    }
    std::vector<float> psc(nsc), gsc(nsc);
    size_t off = 0;
    for (Linear *L : fl) {
        std::memcpy(psc.data() + off, L->s1.data(), (size_t)L->rows * sizeof(float));
        off += (size_t)L->rows;
        std::memcpy(psc.data() + off, L->s2.data(), (size_t)L->cols * sizeof(float));
        off += (size_t)L->cols;
    }
    auto push_scales = [&]() {
        size_t o = 0;
        for (Linear *L : fl) {
            std::memcpy(L->s1.data(), psc.data() + o, (size_t)L->rows * sizeof(float));
            o += (size_t)L->rows;
            std::memcpy(L->s2.data(), psc.data() + o, (size_t)L->cols * sizeof(float));
            o += (size_t)L->cols;
        }
    };
    Adam adam;
    adam_init(&adam, nsc);
    const int T = m->seq, H = m->hidden, V = m->vocab, N = m->samples, Mid = m->mid;
    std::vector<float> tlog((size_t)N * (size_t)T * (size_t)V);
    for (int s = 0; s < N; ++s) {
        std::vector<float> x(m->calib.begin() + (size_t)s * T * H, m->calib.begin() + (size_t)(s + 1) * T * H);
        for (int b = 0; b < m->n_blocks; ++b) {
            std::vector<float> y;
            teacher_block_forward(m, b, x.data(), T, &y);
            x.swap(y);
        }
        gemm_xtwt(x.data(), m->head.data(), tlog.data() + (size_t)s * T * V, T, V, H);
    }
    Rng rng;
    rng_seed(&rng, m->cfg.seed);
    const int steps = N * m->cfg.model_epochs;
    int step = 0;
    double last = 0.0;
    std::vector<int> perm((size_t)N);
    for (int ep = 0; ep < m->cfg.model_epochs; ++ep) {
        rng_perm(&rng, N, perm.data());
        double eloss = 0.0;
        for (int si = 0; si < N; ++si) {
            if (should_pause(&m->cfg, error, cap, "scale-only distillation")) return -1;
            const int s = perm[si];
            std::vector<float> x(m->calib.begin() + (size_t)s * T * H, m->calib.begin() + (size_t)(s + 1) * T * H);
            std::vector<BlockCache> caches((size_t)m->n_blocks);
            for (int b = 0; b < m->n_blocks; ++b) {
                block_forward(&m->blocks[b], x.data(), T, H, Mid, &caches[b], 1);
                x = caches[b].y;
            }
            std::vector<float> slog((size_t)T * (size_t)V);
            gemm_xtwt(x.data(), m->head.data(), slog.data(), T, V, H);
            const float *tl = tlog.data() + (size_t)s * T * V;
            double loss = 0.0;
            std::vector<float> dlogits((size_t)T * (size_t)V, 0.0f);
            for (int t = 0; t < T; ++t) {
                double maxp = tl[t * V], maxq = slog[t * V];
                for (int v = 1; v < V; ++v) {
                    if (tl[t * V + v] > maxp) maxp = tl[t * V + v];
                    if (slog[t * V + v] > maxq) maxq = slog[t * V + v];
                }
                double zp = 0.0, zq = 0.0;
                for (int v = 0; v < V; ++v) {
                    zp += std::exp((double)tl[t * V + v] - maxp);
                    zq += std::exp((double)slog[t * V + v] - maxq);
                }
                for (int v = 0; v < V; ++v) {
                    const double p = std::exp((double)tl[t * V + v] - maxp) / zp;
                    const double q = std::exp((double)slog[t * V + v] - maxq) / zq;
                    loss += -p * std::log(q + 1e-12);
                    dlogits[t * V + v] = (float)(q - p);
                }
            }
            loss /= (double)T;
            eloss += loss;
            std::vector<float> dact((size_t)T * (size_t)H, 0.0f);
            gemm_dX(m->head.data(), dlogits.data(), dact.data(), T, V, H, 0);
            std::fill(gsc.begin(), gsc.end(), 0.0f);
            for (int b = m->n_blocks - 1; b >= 0; --b) {
                block_backward(&m->blocks[b], &caches[b], dact.data(), T, H, Mid, 0, 1);
                size_t cursor = 0;
                for (int bb = 0; bb < m->n_blocks; ++bb) {
                    for (int which = 0; which < 2; ++which) {
                        Linear *L = which == 0 ? &m->blocks[bb].up : &m->blocks[bb].down;
                        if (!L->factored) continue;
                        if (bb == b) {
                            const float *dW = which == 0 ? caches[b].dW1.data() : caches[b].dW2.data();
                            std::vector<float> gU, gV, gs1, gs2;
                            ste_grads(L, dW, &gU, &gV, &gs1, &gs2);
                            for (size_t k = 0; k < gs1.size(); ++k) gsc[cursor + k] += gs1[k];
                            cursor += gs1.size();
                            for (size_t k = 0; k < gs2.size(); ++k) gsc[cursor + k] += gs2[k];
                            cursor += gs2.size();
                        } else {
                            cursor += (size_t)L->rows + (size_t)L->cols;
                        }
                    }
                }
                dact = caches[b].dx;
            }
            const float lr = cosine_lr(m->cfg.model_kd_lr, m->cfg.model_kd_lr * 1e-4f, step, steps > 0 ? steps : 1);
            adam_step(&adam, psc.data(), gsc.data(), nsc, lr);
            push_scales();
            ++step;
        }
        last = eloss / (double)N;
    }
    if (out_loss) *out_loss = last;
    return 1;
}

static int init_workspace(const char *out_dir, char *error, size_t cap) {
    if (!out_dir || !out_dir[0]) {
        set_error(error, cap, "an output directory is required");
        return 0;
    }
    mkdir_p(out_dir);
    mkdir_p(join_path(out_dir, "blocks").c_str());
    return 1;
}

static int write_initial_snapshots(const char *out_dir, Recon *m, Journal *j, char *error, size_t cap) {
    std::vector<uint8_t> tbody, cbody;
    pack_teacher(m, &tbody);
    uint8_t tsha[32], csha[32];
    const std::string tp = join_path(out_dir, "teacher.nqt");
    if (!write_bytes_atomic(tp.c_str(), tbody, tsha, error, cap)) return 0;
    cbody.assign((const uint8_t *)m->calib.data(), (const uint8_t *)m->calib.data() + act_n(m) * sizeof(float));
    const std::string cp = join_path(out_dir, "calib.bin");
    if (!write_bytes_atomic(cp.c_str(), cbody, csha, error, cap)) return 0;
    hex32(tsha, j->teacher_sha);
    hex32(csha, j->calib_sha);
    j->n_blocks = m->n_blocks;
    j->hidden = m->hidden;
    j->mid = m->mid;
    j->vocab = m->vocab;
    j->samples = m->samples;
    j->seq = m->seq;
    j->seed = m->seed;
    j->completed = 0;
    j->kd_done = 0;
    j->block_sha.clear();
    return save_journal(out_dir, j, error, cap);
}

static int load_snapshots(const char *out_dir, Recon *m, const Journal *j, char *error, size_t cap) {
    std::vector<uint8_t> tpay, cpay;
    uint8_t tsha[32], csha[32], expect[32];
    if (!read_checkpoint(join_path(out_dir, "teacher.nqt").c_str(), &tpay, tsha, error, cap)) return 0;
    if (!parse_hex32(j->teacher_sha, expect) || std::memcmp(tsha, expect, 32) != 0) {
        set_error(error, cap, "the teacher snapshot's SHA-256 did not match the journal");
        return 0;
    }
    if (!unpack_teacher(m, tpay, error, cap)) return 0;
    if (!read_checkpoint(join_path(out_dir, "calib.bin").c_str(), &cpay, csha, error, cap)) return 0;
    if (!parse_hex32(j->calib_sha, expect) || std::memcmp(csha, expect, 32) != 0) {
        set_error(error, cap, "the calibration file's SHA-256 did not match the journal");
        return 0;
    }
    if (cpay.size() != act_n(m) * sizeof(float)) {
        set_error(error, cap, "the calibration file has the wrong size");
        return 0;
    }
    m->calib.resize(act_n(m));
    std::memcpy(m->calib.data(), cpay.data(), cpay.size());
    return 1;
}

static int load_completed_blocks(const char *out_dir, Recon *m, const Journal *j, char *error, size_t cap) {
    if (j->completed <= 0) {
        m->student_act = m->calib;
        m->teacher_act = m->calib;
        for (int b = 0; b < m->n_blocks; ++b) restore_teacher_block(m, b);
        return 1;
    }
    for (int i = 0; i < j->completed; ++i) {
        char name[64];
        std::snprintf(name, sizeof name, "blocks/block-%03d.nqb", i);
        std::vector<uint8_t> payload;
        uint8_t sha[32], expect[32];
        if (i >= (int)j->block_sha.size() || !parse_hex32(j->block_sha[i].c_str(), expect)) {
            set_error(error, cap, "the journal is missing a block SHA-256");
            return 0;
        }
        if (!read_checkpoint(join_path(out_dir, name).c_str(), &payload, sha, error, cap)) return 0;
        if (std::memcmp(sha, expect, 32) != 0) {
            set_error(error, cap, "a block checkpoint's SHA-256 did not match the journal");
            return 0;
        }
        if (!unpack_block_ckpt(m, payload, i, error, cap)) return 0;
    }
    for (int b = j->completed; b < m->n_blocks; ++b) restore_teacher_block(m, b);
    return 1;
}

static int commit_block(const char *out_dir, Recon *m, Journal *j, int bi, char *error, size_t cap) {
    std::vector<uint8_t> body;
    pack_block_ckpt(m, bi, &body);
    uint8_t sha[32];
    char name[64], hex[65];
    std::snprintf(name, sizeof name, "blocks/block-%03d.nqb", bi);
    if (!write_bytes_atomic(join_path(out_dir, name).c_str(), body, sha, error, cap)) return 0;
    hex32(sha, hex);
    if ((int)j->block_sha.size() <= bi) j->block_sha.resize((size_t)bi + 1);
    j->block_sha[bi] = hex;
    j->completed = bi + 1;
    return save_journal(out_dir, j, error, cap);
}

static pchie_nq_recon_status run_recon(const char *out_dir, Recon *m, Journal *j, const pchie_nq_recon_config *cfg,
                                       pchie_nq_recon_stats *stats, char *error, size_t cap) {
#ifdef _OPENMP
    /* One thread keeps a resume bit-identical to a full run. */
    omp_set_num_threads(1);
#endif
    if (stats) std::memset(stats, 0, sizeof(*stats));
    if (stats) {
        stats->blocks = m->n_blocks;
        stats->blocks_already = j->completed;
        stats->used_cuda = 0;
    }
    if (cfg->respect_gpu_busy && pchie_nq_gpu_is_busy(error, cap)) return PCHIE_NQ_RECON_BUSY;
    if (should_pause(cfg, error, cap, "start")) return PCHIE_NQ_RECON_PAUSED;

    const int T = m->seq, H = m->hidden;
    int newly = 0;
    for (int bi = j->completed; bi < m->n_blocks; ++bi) {
        if (should_pause(cfg, error, cap, "before a block")) return PCHIE_NQ_RECON_PAUSED;
        restore_teacher_block(m, bi);
        /* Teacher targets from the running FP activations. */
        std::vector<float> targets(act_n(m)), student_out(act_n(m));
        for (int s = 0; s < m->samples; ++s) {
            std::vector<float> x, y;
            slice_act(m->teacher_act.data(), m->samples, T, H, s, &x);
            teacher_block_forward(m, bi, x.data(), T, &y);
            std::memcpy(targets.data() + (size_t)s * T * H, y.data(), (size_t)T * H * sizeof(float));
        }
        Linear *order[2] = {&m->blocks[bi].up, &m->blocks[bi].down};
        for (int li = 0; li < 2; ++li) {
            const int tn = tune_nonfact(m, bi, m->student_act.data(), targets.data(), error, cap);
            if (tn < 0) return PCHIE_NQ_RECON_PAUSED;
            if (!tn) return PCHIE_NQ_RECON_BAD_ARGUMENT;
            if (!factorize_linear(order[li], cfg, error, cap)) return PCHIE_NQ_RECON_BAD_ARGUMENT;
            const int tf = tune_fact(m, bi, m->student_act.data(), targets.data(), error, cap);
            if (tf < 0) return PCHIE_NQ_RECON_PAUSED;
            if (!tf) return PCHIE_NQ_RECON_BAD_ARGUMENT;
            finalize_linear(order[li]);
        }
        /* Student activations for the next block. */
        for (int s = 0; s < m->samples; ++s) {
            std::vector<float> x, y;
            slice_act(m->student_act.data(), m->samples, T, H, s, &x);
            student_block_forward(&m->blocks[bi], x.data(), T, H, m->mid, &y);
            std::memcpy(student_out.data() + (size_t)s * T * H, y.data(), (size_t)T * H * sizeof(float));
        }
        m->teacher_act.swap(targets);
        m->student_act.swap(student_out);
        if (!commit_block(out_dir, m, j, bi, error, cap)) return PCHIE_NQ_RECON_IO;
        ++newly;
        if (stats) {
            stats->blocks_done_this_run = newly;
            stats->last_block_loss = 0.0;
        }
        if (cfg->stop_after_blocks > 0 && newly >= cfg->stop_after_blocks) {
            if (error && cap) std::snprintf(error, cap, "paused after %d block(s) as requested", newly);
            if (stats) stats->paused = 1;
            return PCHIE_NQ_RECON_PAUSED;
        }
    }
    if (!j->kd_done && cfg->tune_model) {
        if (should_pause(cfg, error, cap, "before distillation")) return PCHIE_NQ_RECON_PAUSED;
        double kd = 0.0;
        const int tk = tune_model_kd(m, error, cap, &kd);
        if (tk < 0) return PCHIE_NQ_RECON_PAUSED;
        if (!tk) return PCHIE_NQ_RECON_BAD_ARGUMENT;
        /* A dedicated kd-final.nqk is committed before the journal is marked
         * done, so a kill mid-write never leaves a half-applied distillation. */
        if (!commit_kd_final(out_dir, m, j, error, cap)) return PCHIE_NQ_RECON_IO;
        if (stats) stats->model_kd_loss = kd;
    }
    if (stats) stats->paused = 0;
    return PCHIE_NQ_RECON_OK;
}

void pchie_nq_recon_config_init(pchie_nq_recon_config *config) {
    if (!config) return;
    std::memset(config, 0, sizeof(*config));
    config->bits = 1.0f;
    config->admm_outer_iters = 400;
    config->admm_inner_iters = 5;
    config->admm_reg = 0.03f;
    config->seed = 0;
    config->samples = 128;
    config->seq = 2048;
    config->nonfact_epochs = 8;
    config->fact_epochs = 8;
    config->model_epochs = 8;
    config->nonfact_batch = 4;
    config->fact_batch = 1;
    config->nonfact_lr = 1e-4f;
    config->fact_binary_lr = 1e-5f;
    config->fact_scale_lr = 1e-5f;
    config->model_kd_lr = 1e-5f;
    config->shrinkage = 0.4f;
    config->use_cuda = 0;
    config->tune_nonfact = 1;
    config->tune_fact = 1;
    config->tune_model = 1;
    config->stop_after_blocks = 0;
    config->pause_if_input_ms = 0;
    config->respect_gpu_busy = 0;
}

pchie_nq_recon_status pchie_nq_recon_tiny(const char *out_dir, int blocks, int hidden, int mid, int vocab,
                                          const pchie_nq_recon_config *config, pchie_nq_recon_stats *stats,
                                          char *error, size_t error_capacity) {
    pchie_nq_recon_config local;
    if (!config) {
        pchie_nq_recon_config_init(&local);
        config = &local;
    }
    if (blocks <= 0 || hidden <= 0 || mid <= 0 || vocab <= 0 || config->samples <= 0 || config->seq <= 0) {
        set_error(error, error_capacity, "a positive block count, shape, sample count and sequence length are required");
        return PCHIE_NQ_RECON_BAD_ARGUMENT;
    }
    if (!init_workspace(out_dir, error, error_capacity)) return PCHIE_NQ_RECON_IO;
    /* Resume if a journal is already present. */
    const std::string jp = join_path(out_dir, "journal.nqj");
    FILE *jf = std::fopen(jp.c_str(), "rb");
    if (jf) {
        std::fclose(jf);
        return pchie_nq_recon_resume(out_dir, config, stats, error, error_capacity);
    }
    Recon m;
    make_tiny(&m, blocks, hidden, mid, vocab, config);
    Journal j;
    if (!write_initial_snapshots(out_dir, &m, &j, error, error_capacity)) return PCHIE_NQ_RECON_IO;
    return run_recon(out_dir, &m, &j, config, stats, error, error_capacity);
}

pchie_nq_recon_status pchie_nq_recon_resume(const char *out_dir, const pchie_nq_recon_config *config,
                                            pchie_nq_recon_stats *stats, char *error, size_t error_capacity) {
    pchie_nq_recon_config local;
    if (!config) {
        pchie_nq_recon_config_init(&local);
        config = &local;
    }
    Journal j;
    if (!load_journal(out_dir, &j, error, error_capacity)) return PCHIE_NQ_RECON_IO;
    Recon m;
    m.cfg = *config;
    m.seed = j.seed;
    m.samples = j.samples;
    m.seq = j.seq;
    m.n_blocks = j.n_blocks;
    m.hidden = j.hidden;
    m.mid = j.mid;
    m.vocab = j.vocab;
    if (!load_snapshots(out_dir, &m, &j, error, error_capacity)) return PCHIE_NQ_RECON_MISMATCH;
    m.blocks.resize((size_t)m.n_blocks);
    if (!load_completed_blocks(out_dir, &m, &j, error, error_capacity)) return PCHIE_NQ_RECON_MISMATCH;
    if (j.kd_done) {
        if (!load_kd_final(out_dir, &m, &j, error, error_capacity)) return PCHIE_NQ_RECON_MISMATCH;
        if (stats) {
            std::memset(stats, 0, sizeof(*stats));
            stats->blocks = m.n_blocks;
            stats->blocks_already = j.completed;
        }
        return PCHIE_NQ_RECON_OK;
    }
    return run_recon(out_dir, &m, &j, config, stats, error, error_capacity);
}
