// Bolt PCHIE - ternary research path: AVX2 kernels (clean-room). Every kernel computes the same exact integer
// block sums as the shipping scalar reference and then runs the shared float tail, so the bits match.
#include "pchie_ternary_internal.h"
#include "pchie_ternary_research_internal.h"
#include <cstring>

#if PCHIE_TERNARY_X86
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#define PCHIE_TARGET(x)
#else
#define PCHIE_TARGET(x) __attribute__((target(x)))
#endif
#endif

namespace pchie {
namespace ternary {
namespace research {
namespace detail {
using ternary::detail::tail_block_step;
using ternary::detail::tail_tensor;

#if PCHIE_TERNARY_X86
PCHIE_TARGET("avx2")
static inline int32_t hsum256(__m256i acc) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// Signed alternative: w = c - 1 in {-1, 0, 1}; sign_epi8(x, w) = w * x exactly (x is never -128), then a
// multiply-add by ones. No per-block activation sum is needed.
PCHIE_TARGET("avx2")
static int32_t block_signed_avx2(const uint8_t* blk, const int8_t* x) {
    const __m256i mask = _mm256_set1_epi8(3), one8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk));
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < 4; ++k) {
        const __m128i sh = _mm_cvtsi32_si128(2 * k);
        const __m256i w = _mm256_sub_epi8(_mm256_and_si256(_mm256_srl_epi16(v, sh), mask), one8);
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k * 32));
        const __m256i p = _mm256_sign_epi8(xv, w);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(one8, p), ones16));
    }
    return hsum256(acc);
}
PCHIE_TARGET("avx2")
static int32_t block_biased_avx2(const uint8_t* blk, const int8_t* x) {
    const __m256i mask = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk));
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < 4; ++k) {
        const __m128i sh = _mm_cvtsi32_si128(2 * k);
        const __m256i c = _mm256_and_si256(_mm256_srl_epi16(v, sh), mask);
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k * 32));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(c, xv), ones));
    }
    return hsum256(acc);
}

PCHIE_TARGET("avx2")
float row_i2_signed_avx2(const View& w, uint64_t row, const ActRow& x) noexcept {
    const uint64_t nb = w.cols / kBlock;
    const uint8_t* rc = w.codes + row * w.row_bytes;
    const int8_t* xq = x.q.data();
    if (w.scale_mode == ScaleMode::PerTensor) {
        int32_t s = 0;
        for (uint64_t b = 0; b < nb; ++b) s += block_signed_avx2(rc + b * kBlockBytes, xq + b * kBlock);
        return tail_tensor(s, w.scale_at(0), x.scale);
    }
    float acc = 0.0f;
    for (uint64_t b = 0; b < nb; ++b)
        acc = tail_block_step(acc, block_signed_avx2(rc + b * kBlockBytes, xq + b * kBlock), w.scale_at(row * nb + b));
    return acc * x.scale;
}

// Zero-skip: a block whose 32 bytes are all 0x55 (every code 1, every weight 0) contributes exactly 0.
PCHIE_TARGET("avx2")
static inline int32_t block_zeroskip_avx2(const uint8_t* blk, const int8_t* x, int32_t bsum) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk));
    if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x55))) == -1) return 0;
    return block_biased_avx2(blk, x) - bsum;
}
PCHIE_TARGET("avx2")
float row_i2_zeroskip_avx2(const View& w, uint64_t row, const ActRow& x) noexcept {
    const uint64_t nb = w.cols / kBlock;
    const uint8_t* rc = w.codes + row * w.row_bytes;
    const int8_t* xq = x.q.data();
    if (w.scale_mode == ScaleMode::PerTensor) {
        int32_t s = 0;
        for (uint64_t b = 0; b < nb; ++b) s += block_zeroskip_avx2(rc + b * kBlockBytes, xq + b * kBlock, x.bsum[size_t(b)]);
        return tail_tensor(s, w.scale_at(0), x.scale);
    }
    float acc = 0.0f;
    for (uint64_t b = 0; b < nb; ++b)
        acc = tail_block_step(acc, block_zeroskip_avx2(rc + b * kBlockBytes, xq + b * kBlock, x.bsum[size_t(b)]),
                              w.scale_at(row * nb + b));
    return acc * x.scale;
}

// B3B: 32 code bytes cover the 26 of this block (the 6 extra lanes meet zero activations). The load never
// reads past the tensor: the last block goes through a zero-padded copy.
PCHIE_TARGET("avx2")
static inline __m256i load_b3b(const PackedTensor& w, uint64_t off) {
    if (off + 32 <= w.codes.size()) return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.codes.data() + off));
    alignas(32) uint8_t tmp[32] = {0};
    std::memcpy(tmp, w.codes.data() + off, size_t(w.codes.size() - off));
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
}
// floor(v / 3) for 0 <= v <= 255 is (v * 21846) >> 16 exactly (checked exhaustively by the test bench).
PCHIE_TARGET("avx2")
static int32_t block_b3b_avx2(__m256i v, const int8_t* xp) {
    const __m256i m3 = _mm256_set1_epi16(21846), three = _mm256_set1_epi16(3);
    __m256i lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v));        // j 0..15
    __m256i hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1));   // j 16..31
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < 5; ++k) {
        __m256i tl, th;
        if (k < 4) {
            const __m256i ql = _mm256_mulhi_epu16(lo, m3), qh = _mm256_mulhi_epu16(hi, m3);
            tl = _mm256_sub_epi16(lo, _mm256_mullo_epi16(ql, three));
            th = _mm256_sub_epi16(hi, _mm256_mullo_epi16(qh, three));
            lo = ql; hi = qh;
        } else { tl = lo; th = hi; }   // valid bytes <= 242, so the top trit is <= 2 (junk lanes meet zeros)
        const __m256i xl = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(xp + k * 32)));
        const __m256i xh = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(xp + k * 32 + 16)));
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_madd_epi16(tl, xl), _mm256_madd_epi16(th, xh)));
    }
    return hsum256(acc);
}
PCHIE_TARGET("avx2")
static inline int32_t blk_b3b(const PackedTensor& w, uint64_t off, const int8_t* xp, int32_t bsum) {
    return block_b3b_avx2(load_b3b(w, off), xp) - bsum;
}
PCHIE_TARGET("avx2")
float row_b3b_avx2(const PackedTensor& w, uint64_t row, const ActRow& x, const B3BAct& xa) noexcept {
    const uint64_t nb = w.cols / kBlock;
    if (w.scale_mode == ScaleMode::PerTensor) {
        int32_t s = 0;
        for (uint64_t b = 0; b < nb; ++b)
            s += blk_b3b(w, (row * nb + b) * kB3BBlockBytes, xa.xp.data() + b * kB3BActSlots, x.bsum[size_t(b)]);
        return tail_tensor(s, w.scales[0], x.scale);
    }
    float acc = 0.0f;
    for (uint64_t b = 0; b < nb; ++b)
        acc = tail_block_step(acc, blk_b3b(w, (row * nb + b) * kB3BBlockBytes, xa.xp.data() + b * kB3BActSlots, x.bsum[size_t(b)]),
                              w.scales[size_t(row * nb + b)]);
    return acc * x.scale;
}

// TL3: one pshufb looks up 32 rows' indices in the 16-entry table of one activation triple (low and high bytes
// of the int16 sums in two tables), then the sign bits negate. Per row and block |sum| <= 127 * 128 = 16256,
// and every partial sum is bounded by the same figure, so int16 lanes never overflow.
PCHIE_TARGET("avx2")
void tile_tl3_avx2(const PackedTensor& w, uint64_t tile, const ActRow& x, const TL3Act& xa, float* y) noexcept {
    const uint64_t nb = w.cols / kBlock;
    const __m256i nib = _mm256_set1_epi8(0x0F);
    const __m256i bitsel = _mm256_set1_epi64x(int64_t(0x8040201008040201ull));
    const __m256i bctrl = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    int32_t tot[kTL3TileRows] = {0};
    float acc[kTL3TileRows] = {0};
    alignas(32) int16_t s16[32];
    for (uint64_t b = 0; b < nb; ++b) {
        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
        for (int t = 0; t < kTL3Triples; ++t) {
            const uint8_t* p = w.codes.data() + tl3_offset(tile, nb, b, t);
            const __m128i ib = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            const __m256i idx = _mm256_and_si256(
                _mm256_set_m128i(_mm_srli_epi16(ib, 4), ib), nib);                  // rows 0..15 | 16..31
            const size_t e = size_t((b * kTL3Triples + uint64_t(t)) * 16);
            const __m256i tlo = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(xa.lo.data() + e)));
            const __m256i thi = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(xa.hi.data() + e)));
            const __m256i rl = _mm256_shuffle_epi8(tlo, idx), rh = _mm256_shuffle_epi8(thi, idx);
            __m256i v0 = _mm256_unpacklo_epi8(rl, rh), v1 = _mm256_unpackhi_epi8(rl, rh);
            uint32_t sb; std::memcpy(&sb, p + 16, 4);
            const __m256i sm = _mm256_shuffle_epi8(_mm256_set1_epi32(int32_t(sb)), bctrl);
            const __m256i m8 = _mm256_cmpeq_epi8(_mm256_and_si256(sm, bitsel), bitsel);   // 0xFF where sign set
            const __m256i m0 = _mm256_unpacklo_epi8(m8, m8), m1 = _mm256_unpackhi_epi8(m8, m8);
            v0 = _mm256_sub_epi16(_mm256_xor_si256(v0, m0), m0);
            v1 = _mm256_sub_epi16(_mm256_xor_si256(v1, m1), m1);
            a0 = _mm256_add_epi16(a0, v0); a1 = _mm256_add_epi16(a1, v1);
        }
        // a0 holds rows 0..7 | 16..23, a1 rows 8..15 | 24..31 (the unpack order).
        _mm256_store_si256(reinterpret_cast<__m256i*>(s16), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(s16 + 16), a1);
        int32_t rs[kTL3TileRows];
        for (int i = 0; i < 8; ++i) {
            rs[i] = s16[i]; rs[16 + i] = s16[8 + i]; rs[8 + i] = s16[16 + i]; rs[24 + i] = s16[24 + i];
        }
        const uint64_t row0 = tile * kTL3TileRows;
        if (w.scale_mode == ScaleMode::PerTensor) for (int r = 0; r < kTL3TileRows; ++r) tot[r] += rs[r];
        else for (int r = 0; r < kTL3TileRows; ++r)
            acc[r] = tail_block_step(acc[r], rs[r], w.scales[size_t((row0 + uint64_t(r)) * nb + b)]);
    }
    for (int r = 0; r < kTL3TileRows; ++r)
        y[r] = w.scale_mode == ScaleMode::PerTensor ? tail_tensor(tot[r], w.scales[0], x.scale) : acc[r] * x.scale;
}

PCHIE_TARGET("avx2")
void decode_i2_avx2(const uint8_t* rc, uint64_t nb, int8_t* q) noexcept {
    const __m256i mask = _mm256_set1_epi8(3), one8 = _mm256_set1_epi8(1);
    for (uint64_t b = 0; b < nb; ++b) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rc + b * kBlockBytes));
        for (int k = 0; k < 4; ++k) {
            const __m128i sh = _mm_cvtsi32_si128(2 * k);
            const __m256i w = _mm256_sub_epi8(_mm256_and_si256(_mm256_srl_epi16(v, sh), mask), one8);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + b * kBlock + uint64_t(k * 32)), w);
        }
    }
}

PCHIE_TARGET("avx2")
void decode_b3b_avx2(const PackedTensor& w, uint64_t row, int8_t* q) noexcept {
    const uint64_t nb = w.cols / kBlock;
    const __m256i m3 = _mm256_set1_epi16(21846), three = _mm256_set1_epi16(3), one = _mm256_set1_epi16(1);
    alignas(32) int8_t tmp[32];
    for (uint64_t b = 0; b < nb; ++b) {
        const __m256i v = load_b3b(w, (row * nb + b) * kB3BBlockBytes);
        __m256i lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v));
        __m256i hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1));
        int8_t* qb = q + b * kBlock;
        for (int k = 0; k < 5; ++k) {
            __m256i tl, th;
            if (k < 4) {
                const __m256i ql = _mm256_mulhi_epu16(lo, m3), qh = _mm256_mulhi_epu16(hi, m3);
                tl = _mm256_sub_epi16(lo, _mm256_mullo_epi16(ql, three));
                th = _mm256_sub_epi16(hi, _mm256_mullo_epi16(qh, three));
                lo = ql; hi = qh;
            } else { tl = lo; th = hi; }
            const __m256i packed = _mm256_permute4x64_epi64(
                _mm256_packs_epi16(_mm256_sub_epi16(tl, one), _mm256_sub_epi16(th, one)), 0xD8);
            if (k < 4) {
                // plane k is weights 26k .. 26k+25; the 6 extra bytes are overwritten by plane k+1 (or stay
                // inside the block for k = 3), so the 32-byte store never leaves this block.
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(qb + 26 * k), packed);
            } else {
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), packed);
                std::memcpy(qb + 104, tmp, 24);
            }
        }
    }
}
#else
float row_i2_signed_avx2(const View&, uint64_t, const ActRow&) noexcept { return 0.0f; }   // never dispatched
float row_i2_zeroskip_avx2(const View&, uint64_t, const ActRow&) noexcept { return 0.0f; }
float row_b3b_avx2(const PackedTensor&, uint64_t, const ActRow&, const B3BAct&) noexcept { return 0.0f; }
void tile_tl3_avx2(const PackedTensor&, uint64_t, const ActRow&, const TL3Act&, float* y) noexcept {
    for (int r = 0; r < kTL3TileRows; ++r) y[r] = 0.0f;
}
void decode_i2_avx2(const uint8_t*, uint64_t, int8_t*) noexcept {}
void decode_b3b_avx2(const PackedTensor&, uint64_t, int8_t*) noexcept {}
#endif

}  // namespace detail
}  // namespace research
}  // namespace ternary
}  // namespace pchie
