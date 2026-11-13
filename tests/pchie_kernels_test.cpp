// Step 3 kernel gate: AVX2 == scalar bit-for-bit, integer dot == dequantized float dot
// (within quantization error), KV pool CoW and ring trimming, tanh error bound.
#include "pchie/pchie_kquants_det.h"
#include "pchie/pchie_kv_pools.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace pchie;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)
static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint16_t rand_half(std::mt19937& r, float lo, float hi) {   // small positive fp16 scale
    std::uniform_real_distribution<float> d(lo, hi);
    float f = d(r); uint32_t u; std::memcpy(&u, &f, 4);
    const int e = int((u >> 23) & 0xFF) - 127 + 15; const uint32_t m = (u >> 13) & 0x3FF;
    return uint16_t((e << 10) | m);
}
int main() {
    std::mt19937 rng(1234);
    const int n = 2560, nb = n / QK_K;
    std::vector<float> x(n); std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    std::vector<BlockQ8_K> xq(nb); quantize_row_q8_K_deterministic(x.data(), xq.data(), n);
    std::vector<float> xd(n);
    for (int b = 0; b < nb; ++b) for (int i = 0; i < QK_K; ++i) xd[b * QK_K + i] = xq[b].d_fp32 * xq[b].qs[i];
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<BlockQ4_K> w4(nb); std::vector<BlockQ6_K> w6(nb);
        for (auto& b : w4) { for (auto& c : b.scales) c = uint8_t(rng()); for (auto& c : b.qs) c = uint8_t(rng()); b.d_fp16 = rand_half(rng, 1e-4f, 1e-2f); b.dmin_fp16 = rand_half(rng, 1e-4f, 1e-2f); }
        for (auto& b : w6) { for (auto& c : b.ql) c = uint8_t(rng()); for (auto& c : b.qh) c = uint8_t(rng()); for (auto& c : b.scales) c = int8_t(rng()); b.d_fp16 = rand_half(rng, 1e-4f, 1e-2f); }
        const float a4 = dot_q4_K_q8_K<false>(w4.data(), xq.data(), nb), s4 = dot_q4_K_q8_K<true>(w4.data(), xq.data(), nb);
        const float a6 = dot_q6_K_q8_K<false>(w6.data(), xq.data(), nb), s6 = dot_q6_K_q8_K<true>(w6.data(), xq.data(), nb);
        CHECK(fbits(a4) == fbits(s4), "Q4_K simd != scalar trial %d", trial);
        CHECK(fbits(a6) == fbits(s6), "Q6_K simd != scalar trial %d", trial);
        std::vector<float> wd(n); double r4 = 0, r6 = 0, m4 = 0, m6 = 0;
        dequant_q4_K_row(w4.data(), wd.data(), n); for (int i = 0; i < n; ++i) { r4 += double(wd[i]) * xd[i]; m4 += std::fabs(double(wd[i]) * xd[i]); }
        dequant_q6_K_row(w6.data(), wd.data(), n); for (int i = 0; i < n; ++i) { r6 += double(wd[i]) * xd[i]; m6 += std::fabs(double(wd[i]) * xd[i]); }
        CHECK(std::fabs(a4 - r4) <= 1e-5 * m4 + 1e-6, "Q4_K int dot vs dequant ref %g vs %g", a4, r4);
        CHECK(std::fabs(a6 - r6) <= 1e-5 * m6 + 1e-6, "Q6_K int dot vs dequant ref %g vs %g", a6, r6);
        std::vector<BlockQ8_0W> w8(n / 32); for (auto& b : w8) { for (auto& c : b.qs) c = int8_t(int(rng() % 255) - 127); b.d_fp16 = rand_half(rng, 1e-4f, 1e-2f); }
        std::vector<BlockQ8_0> x8(n / 32); quantize_row_q8_0_deterministic(x.data(), x8.data(), n);
        CHECK(fbits(dot_q8_0_q8_0<false>(w8.data(), x8.data(), n / 32)) == fbits(dot_q8_0_q8_0<true>(w8.data(), x8.data(), n / 32)), "Q8_0 simd != scalar");
    }
    double maxerr = 0;
    for (float z = -12.f; z <= 12.f; z += 1e-3f) maxerr = std::max(maxerr, std::fabs(double(tanh_deterministic(z)) - std::tanh(double(z))));
    std::printf("tanh_deterministic max abs err on [-12,12]: %.3g\n", maxerr);
    CHECK(maxerr < 2e-4, "tanh error %g", maxerr);
    CHECK(tanh_deterministic(9.f) == 1.0f && tanh_deterministic(-50.f) == -1.0f, "tanh saturation");

    // KV pool: fork shares, first write copies, trim frees, release returns everything.
    KVPagePool pool(2, 2, 8, 64);
    KVPagePool::Table a;
    for (uint32_t p = 0; p < 40; ++p) { float* k = pool.k_write(a, 1, p); for (int i = 0; i < 16; ++i) k[i] = float(p * 100 + i); }
    CHECK(pool.live_pages() == 3, "pages after 40 tokens = %u", pool.live_pages());
    KVPagePool::Table b = pool.fork(a);
    CHECK(pool.live_pages() == 3 && pool.ref_count(a.pages[0]) == 2, "fork shares");
    pool.k_write(b, 1, 39)[0] = -1.f;   // CoW on last page
    CHECK(pool.live_pages() == 4 && pool.k_read(a, 1, 39)[0] == 3900.f && pool.k_read(b, 1, 39)[0] == -1.f, "CoW isolation");
    pool.trim_before(b, 33);           // pages 0,1 dead for b, still alive for a
    CHECK(pool.live_pages() == 4 && pool.ref_count(a.pages[0]) == 1, "trim drops b refs");
    pool.release(a); pool.release(b);
    CHECK(pool.live_pages() == 0, "release all");
    if (fails) std::printf("[PCHIE GATE] step3 kernels: %d FAIL\n", fails);
    else std::printf("[PCHIE GATE] step3 kernels: PASS\n");
    return fails ? 1 : 0;
}
