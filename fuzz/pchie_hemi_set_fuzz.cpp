// Fuzz driver for MJHEMI2 sample sets (pchie::hemispheres::SampleSet::parse). Two ways:
//   - standalone (default): pchie_hemi_set_fuzz <iterations> <seed>. Mutates a valid set
//     (bit flips, bytes set to edge values, truncation, extension, and mutations resealed
//     so they get past the SHA-256) with a fixed-seed generator, so a run is reproducible.
//   - libFuzzer: build with -DPCHIE_LIBFUZZER and -fsanitize=fuzzer.
// Every input that parses must re-encode to the same bytes and give a sane tally.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#include "pchie/Sha256.h"
#include "pchie/hemispheres/SampleSet.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace pchie;
using namespace pchie::hemispheres;

static int exercise(const uint8_t* data, size_t len) {
    SampleSet s;
    std::string err;
    if (!SampleSet::parse(data, len, s, err)) {
        if (err.empty() || !s.records().empty()) std::abort();
        return 0;
    }
    const std::vector<uint8_t> again = s.encode();
    if (again.size() != len || std::memcmp(again.data(), data, len) != 0) std::abort();
    const Tally t = s.tally();
    if (t.steps != s.records().size() || t.cleared + t.not_yet > t.steps || t.judged() > t.steps || t.judge_bp() > 10000) std::abort();
    for (const Record& r : s.records()) {
        const uint32_t a = pick(r.watcher_probs.data(), s.header().n_actions, 0xFFFFu);
        if (a >= s.header().n_actions) std::abort();
    }
    return 1;
}

#if defined(PCHIE_LIBFUZZER)
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    exercise(data, size);
    return 0;
}
#else
static std::vector<uint8_t> seed_set(uint32_t n, uint8_t features, uint8_t actions, Purpose purpose = Purpose::Lesson) {
    Header h;
    h.round = 3; h.pass = 1; h.purpose = purpose; h.world = World::GridMaze; h.n_features = features; h.n_actions = actions; h.seed = 42;
    SampleSet s(h);
    std::string err;
    for (uint32_t i = 0; i < n; ++i) {
        Record r;
        r.episode = i / 4; r.step = uint16_t(i % 4);
        for (uint32_t f = 0; f < features; ++f) r.features[f] = int8_t(i * 3 + f);
        float lg[kMaxActions];
        for (uint32_t a = 0; a < actions; ++a) lg[a] = float((i * 7 + a * 5) % 11);
        to_q16(lg, actions, (1u << actions) - 1u, r.actor_probs.data());
        for (uint32_t a = 0; a < actions; ++a) lg[a] = float((i * 3 + a * 2) % 7);
        to_q16(lg, actions, (1u << actions) - 1u, r.watcher_probs.data());
        r.actor_move = uint8_t(pick(r.actor_probs.data(), actions, (1u << actions) - 1u));
        r.verdict = judge(r.watcher_probs.data(), actions, (1u << actions) - 1u, r.actor_move);
        r.grade = Grade(1 + i % 3);
        if (!s.add(r, err)) std::abort();
    }
    return s.encode();
}

int main(int argc, char** argv) {
    const long iterations = argc > 1 ? std::atol(argv[1]) : 100000;
    const unsigned seed = argc > 2 ? unsigned(std::atoi(argv[2])) : 1u;
    std::mt19937_64 rng(seed);
    const std::vector<std::vector<uint8_t>> seeds = {seed_set(12, 8, 6), seed_set(1, 1, 2), seed_set(5, 64, 16, Purpose::Holdout), seed_set(0, 8, 6)};
    long parsed = 0;
    for (const auto& s : seeds) if (!exercise(s.data(), s.size())) { std::printf("a seed does not parse\n"); return 1; }
    for (long it = 0; it < iterations; ++it) {
        std::vector<uint8_t> b = seeds[size_t(rng() % seeds.size())];
        const int kind = int(rng() % 6);
        const int edits = 1 + int(rng() % 4);
        for (int e = 0; e < edits && !b.empty(); ++e) {
            const size_t at = size_t(rng() % b.size());
            if (kind == 0) b[at] ^= uint8_t(1u << (rng() % 8));
            else if (kind == 1) { static const uint8_t edge[] = {0, 1, 2, 3, 4, 5, 16, 17, 64, 65, 0x7F, 0x80, 0xFF}; b[at] = edge[rng() % sizeof edge]; }
            else if (kind == 2) b.resize(size_t(rng() % (b.size() + 1)));
            else if (kind == 3) b.insert(b.end(), size_t(1 + rng() % 40), uint8_t(rng()));
            else b[at] = uint8_t(rng());
        }
        if (kind >= 4 && b.size() >= kTrailerBytes) {   // reseal, so the field checks are what refuses it
            Sha256 s;
            s.update(b.data(), b.size() - kTrailerBytes);
            s.final(b.data() + b.size() - kTrailerBytes);
        }
        parsed += exercise(b.data(), b.size());
    }
    std::printf("pchie_hemi_set_fuzz: %ld inputs, %ld parsed, no crash, no invariant broken\n", iterations, parsed);
    return 0;
}
#endif
