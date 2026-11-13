// LABELED SCRATCH CUDA IModel adapter for the visnav text model.
// Same Gemma4Model / tokenizer / CoW KV as NemotronModel; forward() goes through the existing Step-4 CudaBackend
// (bit-exact-by-design kernels). Never used by an official runner. Official NemotronModel / CPU path untouched.
#pragma once
#include "pchie/IModel.h"
#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_cuda.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/Sha256.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
namespace pchie {
class CudaNemotronModel final : public IModel {
 public:
  bool load(const std::string& path, std::string& err) override {
    if (!g_.load(path, err)) return false;
    weights_sha_ = GgufWeightsHash::sha256_file(path, err);
    if (weights_sha_.size() != Sha256::kDigestBytes) { if (err.empty()) err = "weights SHA-256 failed"; return false; }
    return true;
  }
  std::vector<int32_t> tokenize(const std::string& t) const override { return g_.tok().encode_with_bos(t); }
  std::vector<int32_t> tokenize_no_bos(const std::string& t) const override { return g_.tok().encode(t); }
  std::vector<int32_t> tokenize_untrusted(const std::string& t) const override { return g_.tok().encode(t, false); }
  std::string decode(int32_t id) const override { return g_.tok().decode(id); }
  void init_cache(uint32_t max_ctx, uint32_t branches) override {
    g_.init_cache(max_ctx, branches, /*keep_swa_history=*/false);
    const char* el = std::getenv("PCHIE_CUDA_LAYERS"); const char* er = std::getenv("PCHIE_CUDA_RESERVE_MB");
    const int gl = el ? std::atoi(el) : -1; const size_t rs = er ? size_t(std::atoll(er)) : 256;
    CudaInfo c0; std::string e; cuda_query(c0, e); free_before_mb_ = c0.free_bytes >> 20; total_mb_ = c0.total_bytes >> 20; gpu_name_ = c0.name;
    pool_ = std::make_unique<ThreadPool>(threads_);
    ok_ = cb_.init(g_, gl, rs, false, err_, 1);
    CudaInfo c1; cuda_query(c1, e); free_after_mb_ = c1.free_bytes >> 20;
  }
  ModelBranch new_branch() override { return from_g(g_.new_branch()); }
  ModelBranch fork(const ModelBranch& b) override { return from_g(g_.fork(to_g(b))); }
  void release(ModelBranch& b) override { Gemma4Model::Branch g = to_g(b); g_.release(g); b = from_g(g); }
  void forward(ModelBranch& b, int32_t token) override { Gemma4Model::Branch br = to_g(b); cb_.forward(br, token, logits_, *pool_); b = from_g(br); }
  const std::vector<float>& logits() const override { return logits_; }
  const std::vector<float>& hidden_state() const override { return hidden_; }  // not produced on the CUDA path (visnav does not read it)
  std::vector<uint8_t> weights_sha256() const override { return weights_sha_; }
  uint64_t directory_digest() const override { return g_.gguf().directory_digest(); }
  int vocab_size() const override { return g_.cfg().n_vocab; }
  int d_model() const override { return g_.cfg().d_model; }
  int n_layer() const override { return g_.cfg().n_layer; }
  KVPagePool& pool_swa() override { return g_.pool_swa(); }
  KVPagePool& pool_glob() override { return g_.pool_glob(); }
  // NemotronModel-compatible extras used by visnav
  void set_dispatcher(IJobDispatcher*) {}
  void set_threads(int t) { threads_ = t < 1 ? 1 : t; }
  int dispatcher_size() const noexcept { return threads_; }
  bool cuda_ok() const { return ok_; }
  const std::string& cuda_err() const { return err_; }
  int gpu_layers() const { return cb_.gpu_layers(); }
  double vram_mb() const { return cb_.vram_bytes() / 1048576.0; }
  size_t free_before_mb_ = 0, free_after_mb_ = 0, total_mb_ = 0; std::string gpu_name_;
 private:
  static Gemma4Model::Branch to_g(const ModelBranch& b) { Gemma4Model::Branch g; g.swa = b.swa; g.glob = b.glob; g.n_past = b.n_past; return g; }
  static ModelBranch from_g(const Gemma4Model::Branch& g) { ModelBranch b; b.swa = g.swa; b.glob = g.glob; b.n_past = g.n_past; return b; }
  Gemma4Model g_; CudaBackend cb_; std::unique_ptr<ThreadPool> pool_; int threads_ = 8; bool ok_ = false; std::string err_;
  std::vector<float> logits_, hidden_; std::vector<uint8_t> weights_sha_;
};
}  // namespace pchie
