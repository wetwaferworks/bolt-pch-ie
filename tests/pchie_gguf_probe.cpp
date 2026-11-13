// Read-only probe: prints architecture, tensor-type histogram and directory digest.
#include "pchie/pchie_gguf.h"
#include <cstdio>
#include <map>
int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: pchie_gguf_probe <model.gguf>\n"); return 2; }
    pchie::GgufFile g; std::string err;
    if (!g.open(argv[1], err)) { std::printf("GGUF ERROR: %s\n", err.c_str()); return 1; }
    std::printf("arch=%s name=%s tensors=%zu data_offset=%llu\n", g.meta_str["general.architecture"].c_str(),
                g.meta_str["general.name"].c_str(), g.tensors.size(), (unsigned long long)g.data_offset);
    std::map<std::string, int> hist;
    for (auto& t : g.tensors) hist[pchie::ggml_type_name(t.type)]++;
    for (auto& kv : hist) std::printf("  %-5s %d\n", kv.first.c_str(), kv.second);
    std::printf("[PCHIE GATE] gguf directory digest=0x%016llX\n", (unsigned long long)g.directory_digest());
    return 0;
}
