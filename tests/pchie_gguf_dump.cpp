// Read-only metadata + tensor directory dump (gemma4 spec verification).
#include "pchie/pchie_gguf.h"
#include <cstdio>
int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: pchie_gguf_dump <model.gguf> [all]\n"); return 2; }
    pchie::GgufFile g; std::string err;
    if (!g.open(argv[1], err)) { std::printf("GGUF ERROR: %s\n", err.c_str()); return 1; }
    for (auto& kv : g.meta_str) if (kv.first.find("chat_template") == std::string::npos) std::printf("S %s = %s\n", kv.first.c_str(), kv.second.c_str());
    for (auto& kv : g.meta_num) std::printf("N %s = %.9g\n", kv.first.c_str(), kv.second);
    for (auto& kv : g.meta_num_arr) { std::printf("NA %s [%zu] =", kv.first.c_str(), kv.second.size()); for (size_t i = 0; i < kv.second.size() && i < 64; ++i) std::printf(" %g", kv.second[i]); std::printf("\n"); }
    for (auto& kv : g.meta_str_arr) { std::printf("SA %s [%zu] first:", kv.first.c_str(), kv.second.size()); for (size_t i = 0; i < kv.second.size() && i < 8; ++i) std::printf(" '%s'", kv.second[i].c_str()); std::printf("\n"); }
    bool all = argc > 2;
    for (auto& t : g.tensors) {
        if (!all && t.name.rfind("blk.", 0) == 0 && t.name.rfind("blk.0.", 0) != 0 && t.name.rfind("blk.5.", 0) != 0 && t.name.rfind("blk.24.", 0) != 0 && t.name.rfind("blk.29.", 0) != 0) continue;
        std::printf("T %-40s %-5s", t.name.c_str(), pchie::ggml_type_name(t.type));
        for (auto d : t.dims) std::printf(" %llu", (unsigned long long)d);
        std::printf("\n");
    }
    return 0;
}
