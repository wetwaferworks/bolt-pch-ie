// Bolt PCHIE - NemotronModel wraps Gemma4 reader/tokenizer/forward/CoW KV.
// Forward uses SerialJobDispatcher by default, or a caller-set IJobDispatcher
// (JoinJobDispatcher in B4). No std::mutex, no legacy ThreadPool.
#include "pchie/NemotronModel.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/Sha256.h"

namespace pchie {

Gemma4Model::Branch NemotronModel::to_g(const ModelBranch& b) {
  Gemma4Model::Branch g;
  g.swa = b.swa;
  g.glob = b.glob;
  g.n_past = b.n_past;
  return g;
}

ModelBranch NemotronModel::from_g(const Gemma4Model::Branch& g) {
  ModelBranch b;
  b.swa = g.swa;
  b.glob = g.glob;
  b.n_past = g.n_past;
  return b;
}

bool NemotronModel::load(const std::string& path, std::string& err) {
  path_ = path;
  if (!g_.load(path, err)) return false;
  weights_sha_ = GgufWeightsHash::sha256_file(path, err);
  if (weights_sha_.size() != Sha256::kDigestBytes) {
    if (err.empty()) err = "weights SHA-256 failed";
    return false;
  }
  return true;
}

// BOS policy: prompts get one leading BOS when the GGUF sets add_bos_token.
std::vector<int32_t> NemotronModel::tokenize(const std::string& text) const {
  return g_.tok().encode_with_bos(text);
}

std::vector<int32_t> NemotronModel::tokenize_no_bos(const std::string& text) const {
  return g_.tok().encode(text);
}

// Observation contract: untrusted text never yields a special id; no BOS.
std::vector<int32_t> NemotronModel::tokenize_untrusted(const std::string& text) const {
  return g_.tok().encode(text, /*parse_special=*/false);
}

std::string NemotronModel::decode(int32_t id) const {
  return g_.tok().decode(id);
}

void NemotronModel::init_cache(uint32_t max_ctx, uint32_t branches) {
  g_.init_cache(max_ctx, branches, /*keep_swa_history=*/false);
}

ModelBranch NemotronModel::new_branch() { return from_g(g_.new_branch()); }

ModelBranch NemotronModel::fork(const ModelBranch& b) {
  return from_g(g_.fork(to_g(b)));
}

void NemotronModel::release(ModelBranch& b) {
  Gemma4Model::Branch g = to_g(b);
  g_.release(g);
  b = from_g(g);
}

void NemotronModel::forward(ModelBranch& b, int32_t token) {
  Gemma4Model::Branch br = to_g(b);
  const int D = g_.cfg().d_model;
  const int NP = g_.cfg().n_layer * g_.cfg().d_ple;
  residual_.assign(size_t(D), 0.0f);
  ple_.assign(size_t(NP), 0.0f);
  g_.begin_token(br);
  g_.embed(token, residual_.data(), ple_.data());
  g_.ple_finish(residual_.data(), ple_.data(), pool());
  for (int l = 0; l < g_.cfg().n_layer; ++l)
    g_.layer(l, br, residual_.data(), ple_.data(), pool());
  br.n_past++;
  hidden_.resize(size_t(D));
  g_.rmsnorm(residual_.data(), g_.out_norm(), hidden_.data(), D);
  g_.head(residual_.data(), logits_, pool());
  b = from_g(br);
}

uint64_t NemotronModel::directory_digest() const { return g_.gguf().directory_digest(); }
int NemotronModel::vocab_size() const { return g_.cfg().n_vocab; }
int NemotronModel::d_model() const { return g_.cfg().d_model; }
int NemotronModel::n_layer() const { return g_.cfg().n_layer; }

KVPagePool& NemotronModel::pool_swa() { return g_.pool_swa(); }
KVPagePool& NemotronModel::pool_glob() { return g_.pool_glob(); }

}  // namespace pchie