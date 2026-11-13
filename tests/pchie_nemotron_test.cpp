// Bolt PCHIE - IModel + NemotronModel (Little Nemo) CPU tests.
// No CUDA. No legacy ThreadPool / mutex. Coordinator-thread only.
#include "pchie/SerialJobDispatcher.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/NemotronModel.h"
#include "pchie/DualPoolForkSnapshotStore.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/Sha256.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/ChainRecord.h"
#include "pchie/ModelManifest.h"
#include "pchie/pchie_tree.h"
#include "pchie/ModelPath.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

using namespace pchie;

static const std::string& model_path() {
  static const std::string p = [] {
    std::string path, err;
    if (!pchie::resolve_model_path(path, err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return std::string();
    }
    return path;
  }();
  return p;
}

static const char* kPrompt = "Hi";  // short: 42-layer Q4 CPU forward is expensive

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[NEMOTRON] %-72s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail; std::fflush(stdout);
}

static uint64_t fnv_bytes(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  uint64_t h = FNV_OFFSET;
  constexpr uint64_t P = 1099511628211ULL;
  for (size_t i = 0; i < n; ++i) { h ^= uint64_t(b[i]); h *= P; }
  return h;
}

static std::vector<uint8_t> sha_bytes(const void* p, size_t n) {
  return Sha256::hash(static_cast<const uint8_t*>(p), n);
}

static void test_serial_dispatcher() {
  SerialJobDispatcher d;
  check(d.size() == 1, "SerialJobDispatcher::size() == 1");
  std::vector<int> order;
  d.parallel_for(5, [&](int i) { order.push_back(i); });
  const bool in_order = order.size() == 5 && order[0] == 0 && order[1] == 1 &&
                        order[2] == 2 && order[3] == 3 && order[4] == 4;
  check(in_order, "SerialJobDispatcher visits 0..n-1 in increasing order");
}

static void test_weights_hash_small_file() {
  namespace fs = std::filesystem;
  const auto tmp = fs::temp_directory_path() / "pchie_p3_sha_abc.bin";
  {
    std::ofstream o(tmp, std::ios::binary);
    o.write("abc", 3);
  }
  std::string err;
  auto d = GgufWeightsHash::sha256_file(tmp.string(), err);
  check(err.empty() && d.size() == 32, "GgufWeightsHash sha256_file(\"abc\") returns 32 bytes");
  const char* abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  check(GgufWeightsHash::to_hex(d) == abc, "GgufWeightsHash(\"abc\") matches FIPS digest");
  std::filesystem::remove(tmp);
}

static void test_little_nemo_file_present() {
  check(std::filesystem::exists(model_path().c_str()),
        "Little Nemo GGUF exists at the configured path");
}

static void test_streamed_weights_hash_deterministic() {
  std::string err1, err2;
  const auto a = GgufWeightsHash::sha256_file(model_path().c_str(), err1);
  const auto b = GgufWeightsHash::sha256_file(model_path().c_str(), err2);
  check(err1.empty() && err2.empty() && a.size() == 32 && b.size() == 32,
        "streamed SHA-256 of Little Nemo GGUF succeeds twice");
  check(a != std::vector<uint8_t>(32, 0), "weights SHA-256 is not all zeros");
  check(a == b, "streamed SHA-256 is run-to-run deterministic");
  if (a.size() == 32) {
    std::printf("[NEMOTRON] Little Nemo weights SHA-256=%s\n", GgufWeightsHash::to_hex(a).c_str());
  }
}

static void test_load_tokenize_dims() {
  NemotronModel m;
  std::string err;
  const bool ok = m.load(model_path().c_str(), err);
  check(ok, "NemotronModel::load(Little Nemo) succeeds");
  if (!ok) {
    std::printf("[NEMOTRON] load err: %s\n", err.c_str());
    return;
  }
  check(m.n_layer() == 42, "n_layer == 42 (Gemma-4 E4B)");
  check(m.d_model() == 2560, "d_model == 2560");
  check(m.vocab_size() > 0, "vocab_size > 0");
  check(m.directory_digest() == 0x57726ACEDD4BE8B6ull,
        "directory_digest matches prior CPU probe 0x57726ACEDD4BE8B6");
  std::string herr;
  const auto file_sha = GgufWeightsHash::sha256_file(model_path().c_str(), herr);
  check(m.weights_sha256() == file_sha, "IModel::weights_sha256 matches streamed file SHA-256");

  const auto ids = m.tokenize(kPrompt);
  check(!ids.empty(), "tokenize(\"Hi\") yields at least one token");
  std::printf("[NEMOTRON] tokenize(\"%s\") -> %zu tokens, first=%d\n", kPrompt, ids.size(),
              ids.empty() ? -1 : int(ids[0]));

  // Hold the model for later tests via a static? We'll reload is expensive; tests below load again
  // would blow RAM/time. Combined in test_forward_and_kv below.
  (void)m;
}

static void test_forward_hidden_kv_and_replay() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "forward test: load succeeded");
    return;
  }
  const auto ids = m.tokenize(kPrompt);
  if (ids.empty()) {
    check(false, "forward test: tokenize nonempty");
    return;
  }
  // Keep context tiny: at most first 2 tokens + 1 rollback step.
  std::vector<int32_t> pref = ids;
  if (pref.size() > 2) pref.resize(2);

  m.init_cache(/*max_ctx*/32, /*branches*/4);
  ModelBranch a = m.new_branch();
  for (int32_t t : pref) m.forward(a, t);
  const std::vector<float> logits1 = m.logits();
  const std::vector<float> hidden1 = m.hidden_state();
  check(int(logits1.size()) == m.vocab_size() && !logits1.empty(), "logits size == vocab_size");
  check(int(hidden1.size()) == m.d_model() && !hidden1.empty(), "hidden_state size == d_model");

  // Intra-load replay: second branch, same prefix, bit-identical logits + hidden.
  ModelBranch b = m.new_branch();
  for (int32_t t : pref) m.forward(b, t);
  check(m.logits() == logits1, "run-to-run (second branch) logits bit-identical");
  check(m.hidden_state() == hidden1, "run-to-run (second branch) hidden_state bit-identical");

  const auto logits_sha = sha_bytes(logits1.data(), logits1.size() * sizeof(float));
  const uint64_t logits_fnv = fnv_bytes(logits1.data(), logits1.size() * sizeof(float));
  const auto hidden_sha = sha_bytes(hidden1.data(), hidden1.size() * sizeof(float));
  bool collide8 = logits_sha.size() >= 8 &&
                  std::memcmp(&logits_fnv, logits_sha.data(), 8) == 0;
  check(!collide8, "logits FNV-1a64 != first 8 bytes of SHA-256 (no accidental agreement)");
  if (collide8)
    std::printf("[NEMOTRON] ALERT: FNV vs SHA disagreement FAILED on logits — report it immediately\n");

  std::printf("[NEMOTRON] logits sha256=%s fnv=0x%016llX hidden sha256=%s\n",
              GgufWeightsHash::to_hex(logits_sha).c_str(),
              (unsigned long long)logits_fnv,
              GgufWeightsHash::to_hex(hidden_sha).c_str());

  // Real-weight KV snapshot / rollback (closes P2 gap).
  Fnv1a64HashChain chain;
  chain.reset();
  chain.append(ChainRecord{ChainRecordKind::Prompt, std::vector<uint8_t>(kPrompt, kPrompt + 2)});
  DualPoolForkSnapshotStore store(m.pool_swa(), m.pool_glob());
  store.snapshot(1, a, chain.head(), chain.length());

  const int32_t extra = pref.back();  // one more step (repeat last id is fine)
  m.forward(a, extra);
  const std::vector<float> after = m.logits();
  check(after != logits1, "forward after snapshot changes logits");
  chain.append(ChainRecord{ChainRecordKind::Commit, {1}});

  ModelBranch restored;
  std::vector<uint8_t> head;
  size_t len = 0;
  check(store.rollback(1, restored, head, len), "DualPool rollback(1) succeeds");
  chain.restore_head(head, len);
  m.release(a);
  a = m.fork(restored);
  m.release(restored);
  m.forward(a, extra);
  check(m.logits() == after, "rollback + same next token reproduces post-snapshot logits");

  m.release(a);
  m.release(b);

  // Provisional manifest (NOT a locked golden).
  ModelManifest man;
  man.path = model_path().c_str();
  man.architecture = "gemma4";
  man.name = "Nemotron 3.5 Bolt E4B";
  man.weights_sha256_hex = GgufWeightsHash::to_hex(m.weights_sha256());
  man.directory_digest = m.directory_digest();
  man.prompt = kPrompt;
  man.logits_sha256_hex = GgufWeightsHash::to_hex(logits_sha);
  man.hidden_sha256_hex = GgufWeightsHash::to_hex(hidden_sha);
  man.logits_fnv1a64 = logits_fnv;
  std::error_code ec;
  const auto man_dir = std::filesystem::temp_directory_path() / "pchie_p3";
  std::filesystem::create_directories(man_dir, ec);
  const auto man_path = (man_dir / "p3_little_nemo_manifest.txt").string();
  std::ofstream out(man_path, std::ios::binary);
  if (out) {
    out << "provisional=1\nlocked=0\n";
    out << "path=" << man.path << "\n";
    out << "architecture=" << man.architecture << "\n";
    out << "name=" << man.name << "\n";
    out << "weights_sha256=" << man.weights_sha256_hex << "\n";
    out << "directory_digest=0x" << std::hex << man.directory_digest << std::dec << "\n";
    out << "prompt=" << man.prompt << "\n";
    out << "logits_sha256=" << man.logits_sha256_hex << "\n";
    out << "hidden_sha256=" << man.hidden_sha256_hex << "\n";
    char fnv[32];
    std::snprintf(fnv, sizeof(fnv), "0x%016llX", (unsigned long long)man.logits_fnv1a64);
    out << "logits_fnv1a64=" << fnv << "\n";
  }
  check(bool(out), "wrote provisional model manifest (not a locked golden)");
}

int main() {
  std::printf("[NEMOTRON] IModel + NemotronModel (Little Nemo) CPU tests\n");
  test_serial_dispatcher();
  test_weights_hash_small_file();
  test_little_nemo_file_present();
  test_streamed_weights_hash_deterministic();
  test_load_tokenize_dims();
  test_forward_hidden_kv_and_replay();
  std::printf("[NEMOTRON] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
