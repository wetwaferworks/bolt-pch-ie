// Generation gate: greedy generation from a fixed prompt with the GGUF tokenizer.
// Prints the text, tokens/s and a run hash over (token ids + every logit's bits).
// Tier 1 = the run hash is identical at 1, 2 and 16 threads.
//   pchie_generate <model.gguf> <threads> [max_new] [chat|raw] [prompt...]
#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_tree.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
using namespace pchie;
int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: pchie_generate <model.gguf> <threads> [max_new] [chat|raw] [prompt]\n"); return 2; }
    const int threads = std::atoi(argv[2]);
    const int max_new = argc > 3 ? std::atoi(argv[3]) : 48;
    const bool chat = !(argc > 4 && std::strcmp(argv[4], "raw") == 0);
    std::string user = argc > 5 ? argv[5] : "In two sentences, explain why the sky is blue.";
    Gemma4Model m; std::string err;
    if (!m.load(argv[1], err)) { std::printf("LOAD ERROR: %s\n", err.c_str()); return 1; }
    const std::string prompt = chat ? "<bos><|turn>user\n" + user + "<turn|>\n<|turn>model\n" : "<bos>" + user;
    std::vector<int32_t> ids = m.tok().encode(prompt);
    std::printf("prompt tokens (%zu):", ids.size());
    for (auto i : ids) std::printf(" %d", i);
    std::printf("\n");
    m.init_cache(4096, 2);
    ThreadPool pool(threads);
    auto br = m.new_branch();
    std::vector<float> logits;
    uint64_t h = FNV_OFFSET;
    auto fold = [&](const std::vector<float>& lg) { for (float f : lg) { uint32_t u; std::memcpy(&u, &f, 4); h = hash_mix64(h, u); } };
    const int32_t stop = m.tok().id_of("<turn|>");
    auto t0 = std::chrono::steady_clock::now();
    for (auto id : ids) m.forward(br, id, logits, pool);
    auto t1 = std::chrono::steady_clock::now();
    fold(logits);
    std::string out; int produced = 0;
    for (int s = 0; s < max_new; ++s) {
        int32_t best = 0;
        for (int32_t v = 1; v < int32_t(logits.size()); ++v) if (logits[size_t(v)] > logits[size_t(best)]) best = v; // lowest id wins ties
        h = hash_mix64(h, uint64_t(best));
        if (best == m.tok().eos || best == stop) break;
        out += m.tok().decode(best); ++produced;
        std::printf("%s", m.tok().decode(best).c_str()); std::fflush(stdout);
        m.forward(br, best, logits, pool);
        fold(logits);
    }
    auto t2 = std::chrono::steady_clock::now();
    const double pp = std::chrono::duration<double>(t1 - t0).count(), tg = std::chrono::duration<double>(t2 - t1).count();
    std::printf("\n---\nthreads=%d prompt=%zu tok in %.2fs (%.2f tok/s)  generated=%d tok in %.2fs (%.2f tok/s)\n",
                threads, ids.size(), pp, ids.size() / pp, produced, tg, produced / (tg > 0 ? tg : 1));
    std::printf("[PCHIE GATE] step3 run hash=0x%016llX\n", (unsigned long long)h);
    m.release(br);
    return 0;
}
