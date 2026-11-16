// CUDA backend guards on a real GPU with tiny hand-built models (no model download, a few seconds).
// A well-formed tiny model initialises; a model with a Q8_0 row wider than the matvec's shared partial-sum
// array is refused at init with an error instead of being launched.
#include "pchie/pchie_cuda.h"
#include "pchie/pchie_gemma4.h"
#include "pchie_tiny_gguf.h"

#include <cstdio>
#include <functional>
#include <string>

using namespace pchie;
using namespace pchie_tiny;
static int passes = 0, fails = 0;
static void check(bool ok, const std::string& name) {
    std::printf("[CUDAGUARD] %-90s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    ok ? ++passes : ++fails;
}

// Loads m, makes its cache, runs f(model). Returns false (and prints why) if the model does not load.
static bool with_model(const Model& m, const char* name, const std::function<void(Gemma4Model&)>& f) {
    const std::string p = std::string("cuda_guard_") + name + ".gguf";
    write_file(m, p);
    bool ok = false;
    {
        Gemma4Model g; std::string err;
        ok = g.load(p, err);
        if (!ok) std::printf("           %s did not load: %s\n", name, err.c_str());
        else { g.init_cache(64, 2, false); f(g); }
    }
    std::remove(p.c_str());
    return ok;
}

int main() {
    std::printf("[CUDAGUARD] CUDA backend guards (tiny models)\n");
    CudaInfo ci; std::string qerr;
    if (!cuda_query(ci, qerr)) { std::printf("[CUDAGUARD] no CUDA device (%s): skipped\n", qerr.c_str()); return 0; }

    bool good_init = false;
    check(with_model(good_model(), "good", [&](Gemma4Model& g) {
        CudaBackend cb; std::string err;
        good_init = cb.init(g, -1, 0, false, err);
        if (!good_init) std::printf("           init error: %s\n", err.c_str());
    }), "the tiny model loads");
    check(good_init, "the tiny model initialises on the GPU");

    {   // d_ff 8192 with Q8_0 FF weights: ffn_down has ne0 = 8192, 256 blocks per row.
        Model m = good_model();
        m.set("gemma4.feed_forward_length", 8192);
        m.find("blk.0.ffn_gate.weight")->dims = {32, 8192}; m.find("blk.0.ffn_gate.weight")->type = 8;
        m.find("blk.0.ffn_up.weight")->dims = {32, 8192};   m.find("blk.0.ffn_up.weight")->type = 8;
        m.find("blk.0.ffn_down.weight")->dims = {8192, 32}; m.find("blk.0.ffn_down.weight")->type = 8;
        bool refused = false; std::string err;
        check(with_model(m, "q8_wide", [&](Gemma4Model& g) {
            CudaBackend cb;
            refused = !cb.init(g, -1, 0, false, err);
        }), "a Q8_0 model with d_ff 8192 loads on the CPU path");
        std::printf("           q8_wide -> %s\n", err.c_str());
        check(refused && err.find("ffn_down") != std::string::npos, "CudaBackend::init refuses Q8_0 ffn_down with 256 blocks per row (would overflow shared memory)");
    }
    std::printf("[CUDAGUARD] %d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
