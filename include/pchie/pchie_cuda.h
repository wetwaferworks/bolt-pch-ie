// Bolt PCHIE - CUDA backend for the gemma4 forward pass (clean-room).
// Every GPU kernel mirrors a CPU routine op for op (same integer-exact dots, same
// sequential float order, FMA contraction off, shared pchie_det_math.h), so a CUDA
// run is meant to give the same bits as the CPU run.
// Layers 0..G-1 run on the GPU, G..41 on the CPU (G = what fits the VRAM budget).
// token_embd and per_layer_token_embd stay memory-mapped on the host.
#pragma once
#include "pchie/pchie_gemma4.h"
#include <memory>
#include <string>
#include <vector>

namespace pchie {

struct CudaInfo { std::string name; size_t free_bytes = 0, total_bytes = 0; int sm = 0; };
bool cuda_query(CudaInfo& out, std::string& err);

class CudaBackend {
public:
    CudaBackend();
    ~CudaBackend();
    // model.init_cache() must run first. gpu_layers < 0: as many as fit in
    // (free VRAM - reserve_mb). gpu_lm_head: also copy token_embd to VRAM for the LM head.
    bool init(Gemma4Model& model, int gpu_layers, size_t reserve_mb, bool gpu_lm_head, std::string& err, int max_batch = 1);
    void forward(Gemma4Model::Branch& b, int32_t token, std::vector<float>& logits, ThreadPool& pool);
    // B <= max_batch branches in one pass; per-branch bits equal B = 1.
    void forward_batch(Gemma4Model::Branch* const* bs, const int32_t* toks, int B, std::vector<float>* logits, ThreadPool& pool);
    int gpu_layers() const;
    size_t vram_bytes() const;   // bytes this backend allocated
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace pchie
