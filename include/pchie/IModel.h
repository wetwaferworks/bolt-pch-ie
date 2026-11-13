// Bolt PCHIE - model runtime boundary (virtual). Implementations are concrete finals.
// tokenize, forward on CoW KV pages, logits, hidden state for eval_leaf, weights hash.
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "pchie/ModelBranch.h"
#include "pchie/pchie_kv_pools.h"

namespace pchie {

class IModel {
 public:
  virtual ~IModel() = default;

  virtual bool load(const std::string& path, std::string& err) = 0;
  // BOS policy: tokenize() encodes a sequence prompt and honors the GGUF
  // add_bos_token flag (one leading BOS, never doubled). tokenize_no_bos() is the raw piece
  // encoding for vocabulary-id lookups of a single word (e.g. an action word's logit); it is
  // not for building a prompt that is forwarded from a fresh branch.
  virtual std::vector<int32_t> tokenize(const std::string& text) const = 0;
  virtual std::vector<int32_t> tokenize_no_bos(const std::string& text) const = 0;
  // Observation contract: raw / observation / failure-trace / any untrusted text.
  // parse_special=false and no BOS: literal "<bos>", "<|turn>", ... become ordinary pieces and the
  // result never holds a special id. tokenize() and tokenize_no_bos() are for trusted harness
  // text only; every caller that tokenizes text it did not author MUST use this.
  virtual std::vector<int32_t> tokenize_untrusted(const std::string& text) const = 0;
  virtual std::string decode(int32_t id) const = 0;

  virtual void init_cache(uint32_t max_ctx, uint32_t branches) = 0;
  virtual ModelBranch new_branch() = 0;
  virtual ModelBranch fork(const ModelBranch& b) = 0;
  virtual void release(ModelBranch& b) = 0;

  // One token at branch.n_past; updates last logits() and hidden_state().
  virtual void forward(ModelBranch& b, int32_t token) = 0;
  virtual const std::vector<float>& logits() const = 0;
  // Post-output-RMSNorm hidden vector, length d_model(), for eval_leaf.
  virtual const std::vector<float>& hidden_state() const = 0;

  virtual std::vector<uint8_t> weights_sha256() const = 0;
  virtual uint64_t directory_digest() const = 0;
  virtual int vocab_size() const = 0;
  virtual int d_model() const = 0;
  virtual int n_layer() const = 0;

  virtual KVPagePool& pool_swa() = 0;
  virtual KVPagePool& pool_glob() = 0;
};

}  // namespace pchie