// CUDA backend guards on a real GPU with tiny hand-built models (no model download, a few seconds).
// A well-formed tiny model initialises; a model with a Q8_0 row wider than the matvec's shared partial-sum
// array is refused at init with an error instead of being launched. A model whose sliding head is wider than
// its global head runs on the GPU with the same logits as the CPU (run under compute-sanitizer to see any
// write past a scratch buffer). Destroying the backend unhooks its copy-on-write mirror from the model.
#include "pchie/pchie_cuda.h"
#include "pchie/pchie_gemma4.h"
#include "pchie_tiny_gguf.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <functional>
#include <string>

using namespace pchie;
using namespace pchie_tiny;
// True when the device finished every launched kernel without an error.
static void device_ok(bool& ok) { ok = cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess; }
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
    {   // A sliding head (64) wider than the global head (32): the GPU scratch must be sized from the wider one.
        // Run it under compute-sanitizer to see a write past q/k/v if the scratch is too small.
        Model m = good_model();
        m.swa_pattern = {1};
        m.set("gemma4.attention.key_length_swa", 64); m.set("gemma4.attention.value_length_swa", 64);
        m.set("gemma4.rope.dimension_count_swa", 64); m.set("gemma4.attention.head_count", 2);
        m.find("blk.0.attn_q.weight")->dims = {32, 128}; m.find("blk.0.attn_output.weight")->dims = {128, 32};
        m.find("blk.0.attn_k.weight")->dims = {32, 64}; m.find("blk.0.attn_v.weight")->dims = {32, 64};
        m.find("blk.0.attn_q_norm.weight")->dims = {64}; m.find("blk.0.attn_k_norm.weight")->dims = {64};
        bool ran = false, same = true; std::string err;
        check(with_model(m, "wide_swa", [&](Gemma4Model& g) {
            CudaBackend cb;
            if (!cb.init(g, -1, 0, false, err)) { std::printf("           init error: %s\n", err.c_str()); return; }
            ThreadPool pool(1);
            Gemma4Model::Branch gb = g.new_branch(), cbr = g.new_branch();
            std::vector<float> lg, lc;
            for (int32_t tok : {2, 3, 3}) {
                cb.forward(gb, tok, lg, pool);
                g.forward(cbr, tok, lc, pool);
                if (lg != lc) same = false;
            }
            device_ok(ran);
            g.release(gb); g.release(cbr);
        }), "a model whose sliding head is wider than its global head loads");
        check(ran, "the GPU runs three tokens on it with no CUDA error");
        check(same, "GPU logits equal the CPU logits on it");
    }
    {   // The KV pools belong to the model and outlive the backend. Once the backend is gone, a copy-on-write on
        // the CPU path must not call its mirror copy (that would cudaMemcpy into freed device memory).
        bool hooked = false, unhooked = false, cpu_ok = false;
        check(with_model(good_model(), "lifetime", [&](Gemma4Model& g) {
            {
                CudaBackend cb; std::string err;
                if (!cb.init(g, -1, 0, false, err)) { std::printf("           init error: %s\n", err.c_str()); return; }
                hooked = bool(g.pool_swa().on_copy) && bool(g.pool_glob().on_copy);
            }
            unhooked = !g.pool_swa().on_copy && !g.pool_glob().on_copy;
            ThreadPool pool(1);
            Gemma4Model::Branch a = g.new_branch(); std::vector<float> lg;
            g.forward(a, 2, lg, pool);
            Gemma4Model::Branch b = g.fork(a);       // shares a's page
            g.forward(b, 3, lg, pool);               // first write into the shared page: copy-on-write
            cpu_ok = g.pool_glob().cow_copies() + g.pool_swa().cow_copies() > 0 && lg.size() == 4;
            g.release(b); g.release(a);
        }), "the tiny model loads (backend lifetime)");
        check(hooked, "a live backend mirrors copy-on-write page copies to the GPU");
        check(unhooked, "destroying the backend removes its copy hooks from the model's KV pools");
        check(cpu_ok, "a CPU copy-on-write after the backend is gone runs without touching the GPU");
    }
    std::printf("[CUDAGUARD] %d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
