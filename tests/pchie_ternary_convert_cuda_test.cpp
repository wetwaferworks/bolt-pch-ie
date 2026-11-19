// Bolt PCHIE - ternary converter: the CUDA fit must give the CPU fit's codes and scales bit for bit.
// Exits 77 (ctest: skipped) when the build has no CUDA backend or the machine has no CUDA device.
#include "pchie/pchie_ternary_convert.h"
#include "pchie_ternary_convert_testutil.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace tconv_test;

static int g_fail = 0, g_pass = 0;
#define CHECK(c) do { if (c) ++g_pass; else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static uint64_t parity(const std::vector<float> &w, ternary_group_fit_kind fk, double *cpu_ms = nullptr,
                       double *gpu_ms = nullptr) {
    const uint64_t ng = w.size() / TERNARY_GROUP_WEIGHT_COUNT;
    std::vector<int8_t> qc(w.size()), qg(w.size(), 9);
    std::vector<float> gc(ng), gg(ng, -1.0f);
    auto t0 = std::chrono::steady_clock::now();
    if (ternary_fit_groups_cpu(w.data(), ng, fk, 0, qc.data(), gc.data()) != TERNARY_CONVERT_OK) return ~0ull;
    auto t1 = std::chrono::steady_clock::now();
    char err[TERNARY_CONVERT_ERROR_CAPACITY]{};
    if (ternary_fit_groups_cuda(w.data(), ng, fk, qg.data(), gg.data(), err, sizeof err) != TERNARY_CONVERT_OK) {
        std::printf("cuda: %s\n", err);
        return ~0ull;
    }
    auto t2 = std::chrono::steady_clock::now();
    if (cpu_ms) *cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (gpu_ms) *gpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    uint64_t diff = 0;
    for (uint64_t g = 0; g < ng; ++g)
        if (std::memcmp(qc.data() + g * TERNARY_GROUP_WEIGHT_COUNT, qg.data() + g * TERNARY_GROUP_WEIGHT_COUNT,
                        TERNARY_GROUP_WEIGHT_COUNT) != 0 ||
            std::memcmp(&gc[g], &gg[g], 4) != 0)
            ++diff;
    return diff;
}

int main() {
    const ternary_cuda_info info = ternary_query_cuda_info();
    if (!info.built || info.device_count == 0) {
        std::printf("pchie_ternary_convert_cuda_test: skipped (%s)\n", info.built ? "no CUDA device" : "built without CUDA");
        return 77;
    }
    std::printf("device: %s, %.0f MiB\n", info.device_name, double(info.total_bytes) / 1048576.0);

    uint64_t total = 0, different = 0;
    for (uint64_t ng : {1ull, 2ull, 31ull, 32ull, 33ull, 63ull, 64ull, 65ull, 1000ull, 4097ull}) {
        for (int kind = 0; kind < 6; ++kind)
            for (ternary_group_fit_kind fk : {TERNARY_GROUP_FIT_ABSMEAN, TERNARY_GROUP_FIT_LEAST_SQUARES}) {
                const auto w = weights(ng * TERNARY_GROUP_WEIGHT_COUNT, uint32_t(ng * 10 + uint64_t(kind)), kind);
                const uint64_t d = parity(w, fk);
                CHECK(d == 0);
                total += ng;
                different += (d == ~0ull ? ng : d);
            }
    }
    std::vector<float> e(8 * TERNARY_GROUP_WEIGHT_COUNT, 0.0f);
    for (int i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) {
        e[1 * TERNARY_GROUP_WEIGHT_COUNT + i] = 1e-40f;
        e[2 * TERNARY_GROUP_WEIGHT_COUNT + i] = (i < TERNARY_GROUP_WEIGHT_COUNT / 2) ? 3.0f : 1.0f;
        e[3 * TERNARY_GROUP_WEIGHT_COUNT + i] = (i % 2) ? 3.0e38f : -3.0e38f;
        e[4 * TERNARY_GROUP_WEIGHT_COUNT + i] = (i == 17) ? -50.0f : 0.001f;
        e[5 * TERNARY_GROUP_WEIGHT_COUNT + i] = (i % 2) ? 0.5f : -0.5f;
        e[6 * TERNARY_GROUP_WEIGHT_COUNT + i] = float(i) * 1e-3f;
        e[7 * TERNARY_GROUP_WEIGHT_COUNT + i] = (i % 3 == 0) ? 1.0f : ((i % 3 == 1) ? -1.0f : 0.5f);
    }
    for (ternary_group_fit_kind fk : {TERNARY_GROUP_FIT_ABSMEAN, TERNARY_GROUP_FIT_LEAST_SQUARES}) {
        const uint64_t d = parity(e, fk);
        CHECK(d == 0);
        total += 8;
        different += d;
    }
    std::printf("CUDA vs CPU: %llu groups compared, %llu differ\n", (unsigned long long)total,
                (unsigned long long)different);

    auto bad = weights(64 * TERNARY_GROUP_WEIGHT_COUNT, 5, 0);
    bad[300] = std::numeric_limits<float>::infinity();
    std::vector<int8_t> q(bad.size(), 1);
    std::vector<float> g(64, 1.0f);
    char err[TERNARY_CONVERT_ERROR_CAPACITY]{};
    CHECK(ternary_fit_groups_cuda(bad.data(), 64, TERNARY_GROUP_FIT_LEAST_SQUARES, q.data(), g.data(), err,
                                  sizeof err) == TERNARY_CONVERT_NON_FINITE);
    bool zeroed = true;
    for (int8_t v : q) zeroed = zeroed && v == 0;
    for (float v : g) zeroed = zeroed && v == 0.0f;
    CHECK(zeroed);

    const auto dir = temp_dir("cuda");
    const uint64_t rows = 40, cols = 1024;
    const auto w = weights(rows * cols, 99, 2);
    const auto path = dir / "m.safetensors";
    write_file(path, make_safetensors({{"layers.0.w", "BF16", {rows, cols}, bf16_bytes(w)}}));
    ternary_source_file f{};
    CHECK(ternary_open_source(path.string().c_str(), &f, err, sizeof err));
    std::FILE *fp = std::fopen(path.string().c_str(), "rb");
    if (fp && f.tensor_count) {
        for (ternary_fit_backend be : {TERNARY_FIT_BACKEND_CPU, TERNARY_FIT_BACKEND_CUDA}) {
            ternary_convert_options o;
            ternary_convert_options_init(&o);
            o.backend = be;
            o.check_parity = 1;
            o.chunk_weights = 7 * cols;
            ternary_convert_result r{};
            CHECK(ternary_convert_tensor(fp, &f, &f.tensors[0], &o, &r, err, sizeof err));
            CHECK(r.parity_groups_checked == rows * cols / TERNARY_GROUP_WEIGHT_COUNT && r.parity_groups_different == 0);
            ternary_convert_result_clear(&r);
        }
    }
    if (fp) std::fclose(fp);
    ternary_source_file_clear(&f);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    const auto big = weights(2560ull * 10240ull, 1234, 0);
    for (ternary_group_fit_kind fk : {TERNARY_GROUP_FIT_ABSMEAN, TERNARY_GROUP_FIT_LEAST_SQUARES}) {
        double cms = 0, gms = 0;
        const double k0 = ternary_query_cuda_counters()->kernel_milliseconds;
        const uint64_t d = parity(big, fk, &cms, &gms);
        CHECK(d == 0);
        std::printf("2560 x 10240 %s: CPU %.1f ms (all threads), CUDA %.1f ms with copies, kernel %.1f ms; %llu groups differ\n",
                    fk == TERNARY_GROUP_FIT_ABSMEAN ? "absmean" : "least squares", cms, gms,
                    ternary_query_cuda_counters()->kernel_milliseconds - k0, (unsigned long long)d);
    }
    const ternary_cuda_counters *c = ternary_query_cuda_counters();
    std::printf("CUDA: kernel %.1f ms, copies %.1f ms, own peak %.1f MiB, device peak %.1f MiB\n",
                c->kernel_milliseconds, c->copy_milliseconds, double(c->own_bytes_peak) / 1048576.0,
                double(c->device_used_peak) / 1048576.0);
    ternary_cuda_release();
    std::printf("pchie_ternary_convert_cuda_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
