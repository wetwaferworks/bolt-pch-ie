// Bolt PCHIE BOS policy: honor the GGUF
// tokenizer.ggml.add_bos_token flag. For Little Nemo it is True, so IModel::tokenize /
// Gemma4Tokenizer::encode_with_bos prepend <bos> (id 2) once, never doubled; an explicit
// "<bos>" or a chat-template prompt that already starts with BOS is left alone.
// IModel::tokenize_no_bos / Gemma4Tokenizer::encode stay the raw piece encoding, used only for
// vocabulary-id lookups of a single word (B1/Stage B action logits, L1 component names).
// Real Little Nemo GGUF on CPU (tokenizer only; no forward). No golden is locked.
#include "pchie/Dividor.h"
#include "pchie/Multiplexor.h"
#include "pchie/NemotronModel.h"
#include "pchie/PchGate.h"
#include "pchie/pchie_gemma4.h"
#include "pchie/ModelPath.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

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


static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[BOS-POLICY] %-76s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static void print_ids(const char* tag, const std::vector<int32_t>& v) {
  std::printf("[BOS-POLICY] %s ->", tag);
  for (int32_t t : v) std::printf(" %d", t);
  std::printf("\n");
}

static size_t count_of(const std::vector<int32_t>& v, int32_t x) {
  size_t n = 0;
  for (int32_t t : v) if (t == x) ++n;
  return n;
}

static void test_tokenizer_policy() {
  Gemma4Model g;
  std::string err;
  if (!g.load(model_path().c_str(), err)) {
    check(false, "Gemma4Model::load(Little Nemo) succeeds");
    std::printf("[BOS-POLICY] load err: %s\n", err.c_str());
    return;
  }
  const Gemma4Tokenizer& tk = g.tok();
  check(tk.add_bos && tk.bos == 2, "GGUF tokenizer.ggml.add_bos_token=True is read; bos id 2");
  const auto raw = tk.encode("Hi");
  const auto with = tk.encode_with_bos("Hi");
  print_ids("encode(\"Hi\")", raw);
  print_ids("encode_with_bos(\"Hi\")", with);
  check(raw == std::vector<int32_t>{10979}, "encode(\"Hi\") (raw) is unchanged: {10979}");
  check(with == std::vector<int32_t>{2, 10979}, "encode_with_bos(\"Hi\") is {2, 10979}");
  check(tk.encode_with_bos("<bos>Hi") == std::vector<int32_t>{2, 10979},
        "explicit \"<bos>Hi\" is not doubled: {2, 10979}");
  check(tk.encode_with_bos(tk.decode(2) + "Hi") == with, "decode(2) + \"Hi\" round-trips to one BOS");
  const std::vector<int32_t> chat_want = {2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107};
  const auto chat_explicit = tk.encode_with_bos("<bos><|turn>user\nHi<turn|>\n<|turn>model\n");
  const auto chat_implicit = tk.encode_with_bos("<|turn>user\nHi<turn|>\n<|turn>model\n");
  print_ids("chat template (explicit <bos>)", chat_explicit);
  check(chat_explicit == chat_want && count_of(chat_explicit, 2) == 1,
        "chat template with explicit <bos>: exactly one BOS, ids as decoded (case C)");
  check(chat_implicit == chat_want, "chat template without <bos>: BOS added once, same ids");
  check(tk.encode("<bos><|turn>user\nHi<turn|>\n<|turn>model\n") == chat_want,
        "legacy step3-6 prompts (literal <bos>, raw encode) are byte-identical");
  check(tk.encode_with_bos("").empty(), "empty text stays empty (no lone BOS)");
  check(tk.encode_with_bos("<bos>") == std::vector<int32_t>{2}, "\"<bos>\" alone is one BOS");
  Gemma4Tokenizer off = tk;
  off.add_bos = false;
  check(off.encode_with_bos("Hi") == raw && off.encode_with_bos("<bos>Hi") == with,
        "flag off: encode_with_bos == encode (policy follows the GGUF flag only)");
}

static void test_model_policy() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "NemotronModel::load(Little Nemo) succeeds");
    std::printf("[BOS-POLICY] load err: %s\n", err.c_str());
    return;
  }
  IModel& im = m;
  check(im.tokenize("Hi") == std::vector<int32_t>{2, 10979}, "IModel::tokenize(\"Hi\") is {2, 10979}");
  check(im.tokenize_no_bos("Hi") == std::vector<int32_t>{10979}, "IModel::tokenize_no_bos(\"Hi\") is {10979}");
  check(im.tokenize("<bos>Hi") == std::vector<int32_t>{2, 10979}, "IModel::tokenize(\"<bos>Hi\") is not doubled");
  // Action words and L1 component names are vocab-id lookups: first id must be the word, not BOS.
  const char* words[] = {"North", "South", "East", "West", "Pickup", "Use", Dividor::kName, Multiplexor::kName};
  bool words_ok = true;
  for (const char* w : words) {
    const auto a = im.tokenize_no_bos(w);
    const auto b = im.tokenize(w);
    if (a.empty() || a[0] == 2 || b.size() != a.size() + 1 || b[0] != 2 ||
        !std::equal(a.begin(), a.end(), b.begin() + 1))
      words_ok = false;
    std::printf("[BOS-POLICY] word %-12s no_bos first=%d n=%zu | tokenize first=%d n=%zu\n", w,
                a.empty() ? -1 : a[0], a.size(), b.empty() ? -1 : b[0], b.size());
  }
  check(words_ok, "vocab lookups (actions, components) use tokenize_no_bos: first id is the word");
  // Prompt prefill (B1 / Stage B task names) gets exactly one leading BOS.
  const char* names[] = {"GoToKeyA", "PickUpKeyA", "GoToDoorA", "UseKeyA", "GoToExit", "GoToDoor0", "GoToDoor1"};
  bool names_ok = true;
  for (const char* n : names) {
    const auto p = im.tokenize(n);
    if (p.size() < 2 || p[0] != 2 || count_of(p, 2) != 1) names_ok = false;
  }
  check(names_ok, "task-name prompts (prime_logits) start with exactly one BOS");
  // Pin the open item: the provisional stand-in reject set holds id 2, so a gated path that
  // included the BOS anchor would be rejected. Callers gate proposal paths without the anchor.
  PchGate gate;
  check(!gate.Pass(ProposalTokens::from_proposal(im.tokenize("Hi"))) && gate.LastRejectToken() == 2,
        "OPEN ITEM pinned: stand-in PchGate rejects a path containing BOS (id 2)");
  check(gate.Pass(ProposalTokens::from_proposal(im.tokenize_no_bos("Hi"))), "stand-in PchGate passes the BOS-free gated prefix {10979}");
}

int main() {
  std::printf("[BOS-POLICY] BOS policy (honor GGUF add_bos_token); real GGUF tokenizer, CPU\n");
  test_tokenizer_policy();
  test_model_policy();
  std::printf("[BOS-POLICY] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
