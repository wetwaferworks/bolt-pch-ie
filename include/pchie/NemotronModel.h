// Bolt PCHIE - NemotronModel (Little Nemo / Nemotron 3.5 Bolt E4B, gemma4 GGUF).
// Wraps Gemma4Model + SerialJobDispatcher. Does not use the legacy ThreadPool path.
#pragma once

#include "pchie/IModel.h"
#include "pchie/SerialJobDispatcher.h"
#include "pchie/pchie_gemma4.h"

#include <string>
#include <vector>

namespace pchie {

class NemotronModel final : public IModel {
 public:
  bool load(const std::string& path, std::string& err) override;
  std::vector<int32_t> tokenize(const std::string& text) const override;         // honors add_bos_token
  std::vector<int32_t> tokenize_no_bos(const std::string& text) const override;  // vocab-id lookups
  std::vector<int32_t> tokenize_untrusted(const std::string& text) const override;  // observations
  std::string decode(int32_t id) const override;

  void init_cache(uint32_t max_ctx, uint32_t branches) override;
  ModelBranch new_branch() override;
  ModelBranch fork(const ModelBranch& b) override;
  void release(ModelBranch& b) override;

  void forward(ModelBranch& b, int32_t token) override;
  const std::vector<float>& logits() const override { return logits_; }
  const std::vector<float>& hidden_state() const override { return hidden_; }

  std::vector<uint8_t> weights_sha256() const override { return weights_sha_; }
  uint64_t directory_digest() const override;
  int vocab_size() const override;
  int d_model() const override;
  int n_layer() const override;

  // B4: route forward work through an external dispatcher (nullptr = internal serial).
  void set_dispatcher(IJobDispatcher* d) { ext_ = d; }
  int dispatcher_size() const noexcept { return ext_ ? ext_->size() : dispatch_.size(); }

  KVPagePool& pool_swa() override;
  KVPagePool& pool_glob() override;

 private:
  static Gemma4Model::Branch to_g(const ModelBranch& b);
  static ModelBranch from_g(const Gemma4Model::Branch& b);

  Gemma4Model g_;
  IJobDispatcher& pool() { return ext_ ? *ext_ : dispatch_; }

  SerialJobDispatcher dispatch_;
  IJobDispatcher* ext_ = nullptr;
  std::vector<float> logits_, hidden_, residual_, ple_;
  std::vector<uint8_t> weights_sha_;
  std::string path_;
};

}  // namespace pchie