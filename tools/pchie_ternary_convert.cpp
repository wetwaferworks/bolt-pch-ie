// Bolt PCHIE - ternary converter command line.
//
//   pchie_ternary_convert MODEL --list
//   pchie_ternary_convert MODEL --out DIR [--fit lsq|absmean] [--backend cpu|cuda] [--check-parity]
//                         [--match TEXT]... [--skip TEXT]... [--no-default-skips] [--limit N] [--threads N]
//                         [--chunk-mweights N]
//
// MODEL is a GGUF or safetensors file. It is opened read-only. DIR must be new or empty and may not sit inside the
// model's folder; nothing in it is ever overwritten. For each converted matrix the tool writes NNNN_name.ptn (a
// PTN1 ternary tensor, I2 packing, one scale per 128 weights), and at the end manifest.tsv (one line per tensor:
// shape, source type, error statistics, timings, SHA-256 of the .ptn), layers.tsv (error per layer) and
// summary.txt. By default tensors whose names contain embd, embed, vision or audio are skipped.
//
// The error statistics compare each weight with gamma * q: relative error ||W - R|| / ||W||, cosine, RMSE,
// largest absolute error and the share of zero codes. They say how close the ternary matrix is to the source
// matrix; they say nothing about model quality.
#include "pchie/Sha256.h"
#include "pchie/pchie_ternary_convert.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace pchie::ternary;
using namespace pchie::ternary::convert;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: pchie_ternary_convert MODEL --list\n"
                 "       pchie_ternary_convert MODEL --out DIR [--fit lsq|absmean] [--backend cpu|cuda] [--check-parity]\n"
                 "                             [--match TEXT]... [--skip TEXT]... [--no-default-skips] [--limit N]\n"
                 "                             [--threads N] [--chunk-mweights N]\n");
    return 2;
}

std::string hex(const std::vector<uint8_t>& d) {
    static const char* k = "0123456789abcdef";
    std::string s;
    for (uint8_t b : d) { s += k[b >> 4]; s += k[b & 15]; }
    return s;
}

std::string safe_name(const std::string& n) {
    std::string s;
    for (char c : n) s += (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '_') ? c : '_';
    if (s.size() > 160) s.resize(160);
    return s;
}

// "blk.12.ffn_up.weight" or "model.language_model.layers.12.mlp.up_proj.weight" -> "12"; anything else -> "other".
std::string layer_of(const std::string& n) {
    for (const char* key : {"blk.", "layers."}) {
        size_t p = n.find(key);
        while (p != std::string::npos) {
            size_t i = p + std::strlen(key), j = i;
            while (j < n.size() && std::isdigit(static_cast<unsigned char>(n[j]))) ++j;
            if (j > i && j < n.size() && n[j] == '.') return n.substr(i, j - i);
            p = n.find(key, p + 1);
        }
    }
    return "other";
}

bool write_new_file(const fs::path& p, const std::vector<uint8_t>& b, std::string& err) {
    if (fs::exists(p)) { err = "refusing to overwrite " + p.string(); return false; }
    const fs::path tmp = p.string() + ".partial";
    if (fs::exists(tmp)) { err = "refusing to overwrite " + tmp.string(); return false; }
    std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
    if (!f) { err = "cannot create " + tmp.string(); return false; }
    const bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed) { err = "write failed: " + tmp.string(); return false; }
    std::error_code ec;
    fs::rename(tmp, p, ec);
    if (ec) { err = "rename failed: " + p.string() + ": " + ec.message(); return false; }
    return true;
}
bool write_text(const fs::path& p, const std::string& s, std::string& err) {
    return write_new_file(p, std::vector<uint8_t>(s.begin(), s.end()), err);
}

bool inside(const fs::path& child, const fs::path& parent) {
    auto c = child.begin(), p = parent.begin();
    for (; p != parent.end(); ++p, ++c) {
        if (c == child.end()) return false;
        std::string a = c->string(), b = p->string();
#if defined(_WIN32)
        for (auto& ch : a) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        for (auto& ch : b) ch = char(std::tolower(static_cast<unsigned char>(ch)));
#endif
        if (a != b) return false;
    }
    return true;
}

std::string fmt(const char* f, double v) { char b[64]; std::snprintf(b, sizeof b, f, v); return b; }

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const std::string model = argv[1];
    std::string out_dir, fit_s = "lsq", backend_s = "cpu";
    bool list = false, parity = false, default_skips = true;
    std::vector<std::string> match, skip;
    long limit = -1;
    ConvertOptions opt;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string& v) { if (i + 1 >= argc) return false; v = argv[++i]; return true; };
        std::string v;
        if (a == "--list") list = true;
        else if (a == "--check-parity") parity = true;
        else if (a == "--no-default-skips") default_skips = false;
        else if (a == "--out") { if (!next(out_dir)) return usage(); }
        else if (a == "--fit") { if (!next(fit_s)) return usage(); }
        else if (a == "--backend") { if (!next(backend_s)) return usage(); }
        else if (a == "--match") { if (!next(v)) return usage(); match.push_back(v); }
        else if (a == "--skip") { if (!next(v)) return usage(); skip.push_back(v); }
        else if (a == "--limit") { if (!next(v)) return usage(); limit = std::atol(v.c_str()); if (limit < 0) return usage(); }
        else if (a == "--threads") { if (!next(v)) return usage(); opt.threads = std::atoi(v.c_str()); if (opt.threads < 0 || opt.threads > 64) return usage(); }
        else if (a == "--chunk-mweights") {
            if (!next(v)) return usage();
            const long m = std::atol(v.c_str());
            if (m < 1 || m > 1024) return usage();
            opt.chunk_weights = uint64_t(m) << 20;
        } else return usage();
    }
    if (fit_s == "lsq") opt.fit = FitKind::LeastSquares;
    else if (fit_s == "absmean") opt.fit = FitKind::Absmean;
    else return usage();
    if (backend_s == "cpu") opt.backend = Backend::Cpu;
    else if (backend_s == "cuda") opt.backend = Backend::Cuda;
    else return usage();
    opt.check_parity = parity;
    if (default_skips) for (const char* s : {"embd", "embed", "vision", "audio"}) skip.push_back(s);

    SourceFile src;
    std::string err;
    if (!open_source(model, src, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    std::vector<size_t> picked;
    for (size_t i = 0; i < src.tensors.size(); ++i) {
        const SourceTensor& t = src.tensors[i];
        if (!convertible(t, nullptr)) continue;
        bool m = match.empty();
        for (const auto& s : match) m = m || t.name.find(s) != std::string::npos;
        for (const auto& s : skip) m = m && t.name.find(s) == std::string::npos;
        if (m) picked.push_back(i);
    }
    if (limit >= 0 && picked.size() > size_t(limit)) picked.resize(size_t(limit));

    if (list) {
        std::printf("%s: %s, %zu tensors, %zu selected\n", model.c_str(),
                    src.format == SourceFile::Format::Gguf ? "GGUF" : "safetensors", src.tensors.size(), picked.size());
        std::vector<bool> sel(src.tensors.size(), false);
        for (size_t i : picked) sel[i] = true;
        for (size_t i = 0; i < src.tensors.size(); ++i) {
            const SourceTensor& t = src.tensors[i];
            std::string why = "ok";
            convertible(t, &why);
            std::printf("%c %-60s %-6s rows %8llu cols %6llu  %s\n", sel[i] ? '*' : ' ', t.name.c_str(), t.type_name.c_str(),
                        (unsigned long long)t.rows, (unsigned long long)t.cols, sel[i] ? "" : why.c_str());
        }
        return 0;
    }
    if (out_dir.empty()) return usage();
    if (picked.empty()) { std::fprintf(stderr, "error: no tensor selected\n"); return 1; }

    // Output folder: new or empty, and never inside the model's own folder.
    std::error_code ec;
    const fs::path model_dir = fs::weakly_canonical(fs::absolute(fs::path(model)).parent_path(), ec);
    const fs::path out = fs::weakly_canonical(fs::absolute(fs::path(out_dir)), ec);
    if (inside(out, model_dir)) { std::fprintf(stderr, "error: the output folder may not be inside the model's folder\n"); return 1; }
    if (fs::exists(out) && !(fs::is_directory(out) && fs::is_empty(out))) {
        std::fprintf(stderr, "error: %s exists and is not an empty folder\n", out.string().c_str()); return 1;
    }
    fs::create_directories(out, ec);
    if (ec) { std::fprintf(stderr, "error: cannot create %s\n", out.string().c_str()); return 1; }

    if (opt.backend == Backend::Cuda || opt.check_parity) {
        const CudaInfo ci = cuda_info();
        if (!ci.built || ci.devices == 0) { std::fprintf(stderr, "error: CUDA %s\n", ci.built ? "device not found" : "backend not built"); return 1; }
        std::printf("CUDA device: %s (%.0f MiB)\n", ci.device_name.c_str(), double(ci.total_bytes) / 1048576.0);
    }

    std::unique_ptr<std::FILE, int (*)(std::FILE*)> fp(std::fopen(model.c_str(), "rb"), &std::fclose);
    if (!fp) { std::fprintf(stderr, "error: cannot open %s\n", model.c_str()); return 1; }

    const auto t_start = std::chrono::steady_clock::now();
    std::string manifest = "index\tname\tsource_type\trows\tcols\tfit\trel_error\tcosine\trmse\tmax_abs_error\tzero_share\t"
                           "zero_groups\tread_ms\tfit_ms\tother_fit_ms\tstats_ms\tparity_groups\tparity_different\tptn_bytes\tsha256\tfile\n";
    std::map<std::string, ErrorStats> per_layer;
    std::map<std::string, ErrorStats> per_kind;   // by the tensor's last name parts, e.g. "ffn_up.weight"
    ErrorStats all;
    double read_ms = 0, fit_ms = 0, other_ms = 0, stats_ms = 0, write_ms = 0;
    uint64_t parity_checked = 0, parity_diff = 0, bytes_out = 0, weights_in = 0;
    for (size_t k = 0; k < picked.size(); ++k) {
        const SourceTensor& t = src.tensors[picked[k]];
        TensorResult r;
        if (!convert_tensor(fp.get(), src, t, opt, r, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        const auto tw = std::chrono::steady_clock::now();
        const std::vector<uint8_t> blob = serialize(r.out);
        View check;
        Limits lim; lim.max_rows = kHardMaxRows; lim.max_cols = kHardMaxCols; lim.max_bytes = uint64_t(1) << 31;
        if (parse(blob.data(), blob.size(), lim, check) != Status::Ok) { std::fprintf(stderr, "error: %s: output did not re-parse\n", t.name.c_str()); return 1; }
        char idx[16]; std::snprintf(idx, sizeof idx, "%04zu", k);
        const std::string fname = std::string(idx) + "_" + safe_name(t.name) + ".ptn";
        if (!write_new_file(out / fname, blob, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        const std::string sha = hex(pchie::Sha256::hash(blob.data(), blob.size()));
        write_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
        read_ms += r.read_ms; fit_ms += r.fit_ms; other_ms += r.other_fit_ms; stats_ms += r.stats_ms;
        parity_checked += r.parity_groups_checked; parity_diff += r.parity_groups_different;
        bytes_out += blob.size(); weights_in += t.rows * t.cols;
        all.add(r.stats);
        per_layer[layer_of(t.name)].add(r.stats);
        std::string kind = t.name;
        const size_t lp = layer_of(t.name) == "other" ? std::string::npos : t.name.find("." + layer_of(t.name) + ".");
        if (lp != std::string::npos) kind = t.name.substr(lp + layer_of(t.name).size() + 2);
        per_kind[kind].add(r.stats);
        const ErrorStats& s = r.stats;
        manifest += std::string(idx) + "\t" + t.name + "\t" + t.type_name + "\t" + std::to_string(t.rows) + "\t" + std::to_string(t.cols) +
                    "\t" + fit_s + "\t" + fmt("%.6f", s.rel_error()) + "\t" + fmt("%.6f", s.cosine()) + "\t" + fmt("%.6g", s.rmse()) +
                    "\t" + fmt("%.6g", s.max_abs_err) + "\t" + fmt("%.4f", double(s.zeros) / double(s.n)) + "\t" +
                    std::to_string(s.zero_groups) + "\t" + fmt("%.1f", r.read_ms) + "\t" + fmt("%.1f", r.fit_ms) + "\t" +
                    fmt("%.1f", r.other_fit_ms) + "\t" + fmt("%.1f", r.stats_ms) + "\t" + std::to_string(r.parity_groups_checked) +
                    "\t" + std::to_string(r.parity_groups_different) + "\t" + std::to_string(blob.size()) + "\t" + sha + "\t" + fname + "\n";
        std::printf("[%zu/%zu] %-56s %-5s %6llux%-6llu rel %.4f cos %.4f zeros %.3f fit %.0f ms%s\n", k + 1, picked.size(),
                    t.name.c_str(), t.type_name.c_str(), (unsigned long long)t.rows, (unsigned long long)t.cols, s.rel_error(),
                    s.cosine(), double(s.zeros) / double(s.n), r.fit_ms,
                    opt.check_parity ? (r.parity_groups_different ? "  PARITY MISMATCH" : "  parity ok") : "");
        std::fflush(stdout);
    }
    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();

    auto line = [](const std::string& key, const ErrorStats& s) {
        return key + "\t" + std::to_string(s.n) + "\t" + fmt("%.6f", s.rel_error()) + "\t" + fmt("%.6f", s.cosine()) + "\t" +
               fmt("%.6g", s.rmse()) + "\t" + fmt("%.6g", s.max_abs_err) + "\t" + fmt("%.4f", double(s.zeros) / double(s.n)) + "\n";
    };
    std::string layers = "layer\tweights\trel_error\tcosine\trmse\tmax_abs_error\tzero_share\n";
    std::vector<std::pair<long, std::string>> order;
    for (const auto& kv : per_layer) order.push_back({kv.first == "other" ? 1L << 30 : std::atol(kv.first.c_str()), kv.first});
    std::sort(order.begin(), order.end());
    for (const auto& o : order) layers += line(o.second, per_layer[o.second]);
    std::string kinds = "tensor\tweights\trel_error\tcosine\trmse\tmax_abs_error\tzero_share\n";
    for (const auto& kv : per_kind) kinds += line(kv.first, kv.second);

    const CudaCounters& cc = cuda_counters();
    std::string sum;
    sum += "Ternary conversion summary\n";
    sum += "source: " + model + " (" + (src.format == SourceFile::Format::Gguf ? "GGUF" : "safetensors") + ", " +
           std::to_string(src.file_size) + " bytes, opened read-only)\n";
    sum += "fit: " + fit_s + ", backend: " + backend_s + (opt.check_parity ? ", parity checked against the other backend" : "") +
           ", threads: " + (opt.threads ? std::to_string(opt.threads) : std::string("all")) +
           ", chunk: " + std::to_string(opt.chunk_weights >> 20) + " Mi weights\n";
    sum += "tensors: " + std::to_string(picked.size()) + ", weights: " + std::to_string(weights_in) + ", output bytes: " +
           std::to_string(bytes_out) + "\n";
    sum += "overall: " + line("all", all);
    sum += "time: total " + fmt("%.1f", total_ms / 1000.0) + " s; read+decode " + fmt("%.1f", read_ms / 1000.0) + " s; fit " +
           fmt("%.1f", fit_ms / 1000.0) + " s; other-backend fit " + fmt("%.1f", other_ms / 1000.0) + " s; error stats " +
           fmt("%.1f", stats_ms / 1000.0) + " s; serialize+write " + fmt("%.1f", write_ms / 1000.0) + " s\n";
    if (opt.backend == Backend::Cuda || opt.check_parity)
        sum += "cuda: kernel " + fmt("%.1f", cc.kernel_ms) + " ms, copies " + fmt("%.1f", cc.copy_ms) + " ms, converter's own peak " +
               fmt("%.1f", double(cc.own_bytes_peak) / 1048576.0) + " MiB, whole-device peak seen " +
               fmt("%.1f", double(cc.device_used_peak) / 1048576.0) + " MiB (includes other programs)\n";
    if (opt.check_parity)
        sum += "parity: " + std::to_string(parity_checked) + " groups compared, " + std::to_string(parity_diff) + " different\n";
    sum += "These numbers compare each ternary matrix with its source matrix only; they are not a measure of model quality.\n";
    if (!write_text(out / "manifest.tsv", manifest, err) || !write_text(out / "layers.tsv", layers, err) ||
        !write_text(out / "tensors_by_kind.tsv", kinds, err) || !write_text(out / "summary.txt", sum, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
    }
    cuda_release();
    std::printf("%s", sum.c_str());
    return (opt.check_parity && parity_diff) ? 3 : 0;
}
