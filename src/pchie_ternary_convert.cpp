// Bolt PCHIE - ternary converter: source readers, CPU fit, error statistics, streaming (clean-room).
// See include/pchie/pchie_ternary_convert.h and docs/ternary/README.md, "Converter".
#include "pchie/pchie_ternary_convert.h"
#include "pchie/MiniJson.h"
#include "pchie/pchie_gguf.h"
#include "pchie/pchie_kquants_det.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace pchie {
namespace ternary {
namespace convert {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

bool mul_ok(uint64_t a, uint64_t b, uint64_t& out) noexcept {
    if (a != 0 && b > ~uint64_t(0) / a) return false;
    out = a * b;
    return true;
}

int seek64(std::FILE* fp, uint64_t off) noexcept {
#if defined(_WIN32)
    return _fseeki64(fp, static_cast<long long>(off), SEEK_SET);
#else
    return fseeko(fp, static_cast<off_t>(off), SEEK_SET);
#endif
}

uint64_t file_length(std::FILE* fp) noexcept {
#if defined(_WIN32)
    if (_fseeki64(fp, 0, SEEK_END) != 0) return 0;
    const long long n = _ftelli64(fp);
#else
    if (fseeko(fp, 0, SEEK_END) != 0) return 0;
    const long long n = static_cast<long long>(ftello(fp));
#endif
    return n < 0 ? 0 : uint64_t(n);
}

int pick_threads(int threads, uint64_t work_items) {
    int t = threads > 0 ? threads : int(std::thread::hardware_concurrency());
    if (t < 1) t = 1;
    if (t > 64) t = 64;
    if (uint64_t(t) > work_items) t = int(work_items ? work_items : 1);
    return t;
}

// Runs fn(begin, end, worker) over [0, n) cut into one contiguous range per worker; joins before returning.
// Workers never share an output, so there is nothing to lock.
template <class Fn>
void split_run(uint64_t n, int threads, Fn fn) {
    const int t = pick_threads(threads, n);
    if (t <= 1) { fn(uint64_t(0), n, 0); return; }
    std::vector<std::thread> pool;
    pool.reserve(size_t(t));
    for (int i = 0; i < t; ++i) {
        const uint64_t b = n * uint64_t(i) / uint64_t(t), e = n * uint64_t(i + 1) / uint64_t(t);
        pool.emplace_back([=, &fn] { fn(b, e, i); });
    }
    for (auto& th : pool) th.join();
}

struct DtypeInfo { const char* name; uint64_t size; SrcType type; };
const DtypeInfo kSafetensorsDtypes[] = {
    {"F64", 8, SrcType::Other},  {"F32", 4, SrcType::F32},    {"F16", 2, SrcType::F16},
    {"BF16", 2, SrcType::BF16},  {"I64", 8, SrcType::Other},  {"I32", 4, SrcType::Other},
    {"I16", 2, SrcType::Other},  {"I8", 1, SrcType::Other},   {"U64", 8, SrcType::Other},
    {"U32", 4, SrcType::Other},  {"U16", 2, SrcType::Other},  {"U8", 1, SrcType::Other},
    {"BOOL", 1, SrcType::Other}, {"F8_E4M3", 1, SrcType::Other}, {"F8_E5M2", 1, SrcType::Other},
};

// A JSON number that is a whole number in [0, 2^53] (every such double is exact).
bool json_whole(const MiniJson& v, uint64_t& out) noexcept {
    if (v.type != MiniJson::Type::Number) return false;
    const double d = v.number;
    if (!(d >= 0.0 && d <= 9007199254740992.0) || d != std::floor(d)) return false;
    out = uint64_t(d);
    return true;
}

bool fail(SourceFile& out, std::string& err, const std::string& why) {
    out = SourceFile();
    err = why;
    return false;
}

} // namespace

const char* src_type_name(SrcType t) noexcept {
    switch (t) {
        case SrcType::F32: return "F32";
        case SrcType::F16: return "F16";
        case SrcType::BF16: return "BF16";
        case SrcType::Q8_0: return "Q8_0";
        case SrcType::Q4_K: return "Q4_K";
        case SrcType::Q6_K: return "Q6_K";
        default: return "other";
    }
}

// ------------------------------------------------------------------------------------------------ layout
bool check_layout(const SourceFile& f, std::string& err) {
    if (f.data_begin > f.data_end || f.data_end > f.file_size) { err = "data region outside the file"; return false; }
    std::vector<size_t> order;
    order.reserve(f.tensors.size());
    for (size_t i = 0; i < f.tensors.size(); ++i) {
        const SourceTensor& t = f.tensors[i];
        if (t.offset < f.data_begin || t.offset > f.data_end || t.nbytes > f.data_end - t.offset) {
            err = "tensor " + t.name + ": extent outside the data region"; return false;
        }
        if (t.nbytes > 0) order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return f.tensors[a].offset != f.tensors[b].offset ? f.tensors[a].offset < f.tensors[b].offset : a < b;
    });
    for (size_t k = 1; k < order.size(); ++k) {
        const SourceTensor& p = f.tensors[order[k - 1]];
        const SourceTensor& c = f.tensors[order[k]];
        if (p.offset + p.nbytes > c.offset) { err = "tensors " + p.name + " and " + c.name + " overlap"; return false; }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ safetensors
bool parse_safetensors_header(const char* json, uint64_t header_len, uint64_t file_size, const std::string& path,
                              SourceFile& out, std::string& err) {
    out = SourceFile();
    if (!json) return fail(out, err, "safetensors: no header");
    if (file_size < 8 || header_len > file_size - 8) return fail(out, err, "safetensors: header longer than the file");
    if (header_len < 2 || header_len > kMaxSafetensorsHeader) return fail(out, err, "safetensors: header length out of range");
    if (json[0] != '{') return fail(out, err, "safetensors: header is not a JSON object");
    // The format allows trailing spaces as padding; nothing else may follow the object.
    std::string text(json, size_t(header_len));
    while (!text.empty() && text.back() == ' ') text.pop_back();
    MiniJson root;
    MiniJson::Limits lim;
    lim.max_bytes = size_t(kMaxSafetensorsHeader);
    lim.max_depth = 4;
    lim.reject_duplicate_keys = true;
    std::string jerr;
    if (!MiniJson::parse(text, root, jerr, lim)) return fail(out, err, "safetensors: " + jerr);
    if (root.type != MiniJson::Type::Object) return fail(out, err, "safetensors: header is not a JSON object");
    if (root.obj.size() > kMaxTensors + 1) return fail(out, err, "safetensors: too many tensors");

    SourceFile f;
    f.format = SourceFile::Format::Safetensors;
    f.path = path;
    f.file_size = file_size;
    f.data_begin = 8 + header_len;
    f.data_end = file_size;
    const uint64_t data_size = f.data_end - f.data_begin;
    f.tensors.reserve(root.obj.size());
    for (const auto& kv : root.obj) {
        const std::string& name = kv.first;
        const MiniJson& v = kv.second;
        if (name == "__metadata__") {
            if (v.type != MiniJson::Type::Object) return fail(out, err, "safetensors: __metadata__ is not an object");
            for (const auto& m : v.obj)
                if (m.second.type != MiniJson::Type::String) return fail(out, err, "safetensors: __metadata__ values must be strings");
            continue;
        }
        if (name.empty() || name.size() > kMaxNameBytes || name.find('\0') != std::string::npos)
            return fail(out, err, "safetensors: bad tensor name");
        if (v.type != MiniJson::Type::Object || v.obj.size() != 3) return fail(out, err, "safetensors: tensor " + name + ": expected dtype, shape and data_offsets");
        const MiniJson* dt = v.get("dtype");
        const MiniJson* sh = v.get("shape");
        const MiniJson* off = v.get("data_offsets");
        if (!dt || !sh || !off) return fail(out, err, "safetensors: tensor " + name + ": expected dtype, shape and data_offsets");
        if (dt->type != MiniJson::Type::String) return fail(out, err, "safetensors: tensor " + name + ": dtype is not a string");
        const DtypeInfo* info = nullptr;
        for (const auto& d : kSafetensorsDtypes) if (dt->str == d.name) info = &d;
        if (!info) return fail(out, err, "safetensors: tensor " + name + ": unknown dtype " + dt->str.substr(0, 32));
        if (sh->type != MiniJson::Type::Array || sh->arr.size() > kMaxDims) return fail(out, err, "safetensors: tensor " + name + ": bad shape");
        SourceTensor t;
        t.name = name;
        t.type_name = info->name;
        t.type = info->type;
        uint64_t count = 1;
        for (const auto& d : sh->arr) {
            uint64_t dim = 0;
            if (!json_whole(d, dim)) return fail(out, err, "safetensors: tensor " + name + ": a dimension is not a whole number >= 0");
            if (!mul_ok(count, dim, count)) return fail(out, err, "safetensors: tensor " + name + ": element count overflows");
            t.shape.push_back(dim);
        }
        uint64_t bytes = 0;
        if (!mul_ok(count, info->size, bytes)) return fail(out, err, "safetensors: tensor " + name + ": size overflows");
        if (off->type != MiniJson::Type::Array || off->arr.size() != 2) return fail(out, err, "safetensors: tensor " + name + ": bad data_offsets");
        uint64_t b = 0, e = 0;
        if (!json_whole(off->arr[0], b) || !json_whole(off->arr[1], e)) return fail(out, err, "safetensors: tensor " + name + ": data_offsets are not whole numbers >= 0");
        if (b > e) return fail(out, err, "safetensors: tensor " + name + ": data_offsets begin after end");
        if (e > data_size) return fail(out, err, "safetensors: tensor " + name + ": data past the end of the file");
        if (e - b != bytes) return fail(out, err, "safetensors: tensor " + name + ": data_offsets do not match dtype and shape");
        t.offset = f.data_begin + b;
        t.nbytes = bytes;
        t.cols = t.shape.empty() ? 1 : t.shape.back();
        t.rows = t.cols ? count / t.cols : 0;
        t.row_bytes = t.cols * info->size;   // cols * size <= bytes, no overflow
        f.tensors.push_back(std::move(t));
    }
    if (!check_layout(f, err)) { out = SourceFile(); err = "safetensors: " + err; return false; }
    out = std::move(f);
    return true;
}

bool open_safetensors(const std::string& path, SourceFile& out, std::string& err) {
    out = SourceFile();
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> fp(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!fp) { err = "cannot open " + path; return false; }
    const uint64_t size = file_length(fp.get());
    if (size < 8 || seek64(fp.get(), 0) != 0) { err = "safetensors: file too short"; return false; }
    uint8_t lb[8];
    if (std::fread(lb, 1, 8, fp.get()) != 8) { err = "safetensors: cannot read the header length"; return false; }
    uint64_t hl = 0;
    for (int i = 7; i >= 0; --i) hl = (hl << 8) | lb[i];
    if (hl > size - 8 || hl < 2 || hl > kMaxSafetensorsHeader) { err = "safetensors: header length out of range"; return false; }
    std::vector<char> hdr(static_cast<size_t>(hl));
    if (std::fread(hdr.data(), 1, hdr.size(), fp.get()) != hdr.size()) { err = "safetensors: short header"; return false; }
    return parse_safetensors_header(hdr.data(), hl, size, path, out, err);
}

// ------------------------------------------------------------------------------------------------ GGUF
bool open_gguf(const std::string& path, SourceFile& out, std::string& err) {
    out = SourceFile();
    GgufFile g;
    if (!g.open(path, err)) { err = "GGUF: " + err; return false; }
    SourceFile f;
    f.format = SourceFile::Format::Gguf;
    f.path = path;
    f.file_size = g.file_size;
    f.data_begin = g.data_offset;
    f.data_end = g.file_size;
    f.tensors.reserve(g.tensors.size());
    for (const auto& ti : g.tensors) {
        if (!ti.size_known) return fail(out, err, "GGUF: tensor " + ti.name + " has a type of unknown size (" + ggml_type_name(ti.type) + "); refusing the file");
        if (ti.dims.empty() || ti.dims.size() > kMaxDims) return fail(out, err, "GGUF: tensor " + ti.name + ": bad dimension count");
        SourceTensor t;
        t.name = ti.name;
        t.type_name = ggml_type_name(ti.type);
        switch (ti.type) {
            case 0: t.type = SrcType::F32; break;
            case 1: t.type = SrcType::F16; break;
            case 8: t.type = SrcType::Q8_0; break;
            case 12: t.type = SrcType::Q4_K; break;
            case 14: t.type = SrcType::Q6_K; break;
            case 30: t.type = SrcType::BF16; break;
            default: t.type = SrcType::Other; break;
        }
        for (size_t i = ti.dims.size(); i-- > 0;) t.shape.push_back(ti.dims[i]);
        t.cols = ti.dims[0];
        t.rows = t.cols ? ti.n_elements / t.cols : 0;
        t.offset = g.data_offset + ti.offset;   // the reader checked offset + nbytes against the file
        t.nbytes = ti.nbytes;
        uint64_t be = 0, bb = 0;
        if (ggml_type_block(ti.type, be, bb) && be != 0 && t.cols % be == 0) t.row_bytes = t.cols / be * bb;
        if (t.rows != 0 && t.row_bytes * t.rows != t.nbytes) return fail(out, err, "GGUF: tensor " + ti.name + ": size does not match its shape");
        f.tensors.push_back(std::move(t));
    }
    if (!check_layout(f, err)) { out = SourceFile(); err = "GGUF: " + err; return false; }
    out = std::move(f);
    return true;
}

bool open_source(const std::string& path, SourceFile& out, std::string& err) {
    out = SourceFile();
    uint8_t magic[4] = {0, 0, 0, 0};
    {
        std::unique_ptr<std::FILE, int (*)(std::FILE*)> fp(std::fopen(path.c_str(), "rb"), &std::fclose);
        if (!fp) { err = "cannot open " + path; return false; }
        if (std::fread(magic, 1, 4, fp.get()) != 4) { err = "file too short: " + path; return false; }
    }
    if (std::memcmp(magic, "GGUF", 4) == 0) return open_gguf(path, out, err);
    return open_safetensors(path, out, err);
}

bool convertible(const SourceTensor& t, std::string* why) {
    auto no = [&](const char* r) { if (why) *why = r; return false; };
    if (t.type == SrcType::Other) return no("type not decodable here");
    if (t.shape.size() < 2) return no("not a matrix");
    if (t.rows == 0 || t.cols == 0) return no("empty");
    if (t.cols % uint64_t(kGroup) != 0) return no("row length is not a multiple of 128");
    if (t.rows > kHardMaxRows || t.cols > kHardMaxCols) return no("over the ternary size ceiling");
    if (t.row_bytes == 0 || t.row_bytes * t.rows != t.nbytes) return no("size does not match the shape");
    return true;
}

// ------------------------------------------------------------------------------------------------ decode
Status decode_rows(const SourceTensor& t, const uint8_t* raw, uint64_t n, float* out) noexcept {
    if (!raw || !out || t.cols == 0 || t.cols > kHardMaxCols) return Status::BadArgument;
    const uint64_t cols = t.cols;
    const int ci = int(cols);
    for (uint64_t r = 0; r < n; ++r) {
        const uint8_t* src = raw + r * t.row_bytes;
        float* dst = out + r * cols;
        switch (t.type) {
            case SrcType::F32: std::memcpy(dst, src, size_t(cols) * 4); break;
            case SrcType::F16:
                for (uint64_t c = 0; c < cols; ++c) dst[c] = det_fp16_to_f32(uint16_t(src[2 * c] | (src[2 * c + 1] << 8)));
                break;
            case SrcType::BF16:
                for (uint64_t c = 0; c < cols; ++c) dst[c] = det_bf16_to_f32(uint16_t(src[2 * c] | (src[2 * c + 1] << 8)));
                break;
            case SrcType::Q8_0:
                if (cols % 32) return Status::BadArgument;
                dequant_q8_0_row(reinterpret_cast<const BlockQ8_0W*>(src), dst, ci); break;
            case SrcType::Q4_K:
                if (cols % QK_K) return Status::BadArgument;
                dequant_q4_K_row(reinterpret_cast<const BlockQ4_K*>(src), dst, ci); break;
            case SrcType::Q6_K:
                if (cols % QK_K) return Status::BadArgument;
                dequant_q6_K_row(reinterpret_cast<const BlockQ6_K*>(src), dst, ci); break;
            default: return Status::BadArgument;
        }
        for (uint64_t c = 0; c < cols; ++c) if (!std::isfinite(dst[c])) return Status::NonFinite;
    }
    return Status::Ok;
}

bool read_rows(std::FILE* fp, const SourceFile& f, const SourceTensor& t, uint64_t r0, uint64_t n,
               std::vector<uint8_t>& raw, float* out, std::string& err) {
    std::string why;
    if (!fp || !out) { err = "read_rows: bad argument"; return false; }
    if (!convertible(t, &why)) { err = t.name + ": " + why; return false; }
    if (r0 > t.rows || n > t.rows - r0) { err = t.name + ": rows out of range"; return false; }
    const uint64_t off = t.offset + r0 * t.row_bytes, len = n * t.row_bytes;
    if (off < f.data_begin || off > f.data_end || len > f.data_end - off) { err = t.name + ": read outside the data region"; return false; }
    raw.resize(static_cast<size_t>(len));
    if (seek64(fp, off) != 0 || std::fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
        err = t.name + ": short read (file changed?)"; return false;
    }
    const Status st = decode_rows(t, raw.data(), n, out);
    if (st != Status::Ok) { err = t.name + ": decode failed: " + status_name(st); return false; }
    return true;
}

// ------------------------------------------------------------------------------------------------ fit
Status fit_groups_cpu(const float* w, uint64_t n_groups, FitKind fit, int threads, int8_t* q, float* gamma) {
    if (!w || !q || !gamma || n_groups == 0 || uint8_t(fit) > 1) return Status::BadArgument;
    const int t = pick_threads(threads, n_groups);
    std::vector<uint8_t> bad(size_t(t), 0);   // one flag per worker, read after the join
    split_run(n_groups, t, [&](uint64_t b, uint64_t e, int wi) {
        float sorted[kGroup];
        for (uint64_t g = b; g < e; ++g) {
            const float* wg = w + g * kGroup;
            bool ok = true;
            for (int i = 0; i < kGroup; ++i) ok = ok && std::isfinite(wg[i]);
            if (!ok) { bad[size_t(wi)] = 1; continue; }
            gamma[g] = fit_group(wg, fit, sorted, q + g * kGroup);
        }
    });
    for (uint8_t f : bad) if (f) {
        std::memset(q, 0, size_t(n_groups * kGroup));
        std::memset(gamma, 0, size_t(n_groups) * sizeof(float));
        return Status::NonFinite;
    }
    return Status::Ok;
}

// ------------------------------------------------------------------------------------------------ stats
double ErrorStats::rel_error() const noexcept { return sum_w2 > 0 ? std::sqrt(sum_e2 / sum_w2) : 0.0; }
double ErrorStats::cosine() const noexcept {
    return (sum_w2 > 0 && sum_r2 > 0) ? sum_wr / std::sqrt(sum_w2 * sum_r2) : 0.0;
}
double ErrorStats::rmse() const noexcept { return n ? std::sqrt(sum_e2 / double(n)) : 0.0; }
void ErrorStats::add(const ErrorStats& o) noexcept {
    n += o.n; zeros += o.zeros; zero_groups += o.zero_groups; groups += o.groups;
    sum_w2 += o.sum_w2; sum_e2 += o.sum_e2; sum_wr += o.sum_wr; sum_r2 += o.sum_r2;
    max_abs_err = std::max(max_abs_err, o.max_abs_err);
}

void accumulate_stats(const float* w, const int8_t* q, const float* gamma, uint64_t rows, uint64_t cols,
                      int threads, ErrorStats& s) {
    if (!w || !q || !gamma || rows == 0 || cols == 0 || cols % kGroup) return;
    std::vector<ErrorStats> per_row(static_cast<size_t>(rows));
    const uint64_t nb = cols / kGroup;
    split_run(rows, threads, [&](uint64_t b, uint64_t e, int) {
        for (uint64_t r = b; r < e; ++r) {
            ErrorStats& rs = per_row[size_t(r)];
            for (uint64_t blk = 0; blk < nb; ++blk) {
                const uint64_t g = r * nb + blk;
                const double gd = double(gamma[g]);
                if (gamma[g] == 0.0f) ++rs.zero_groups;
                ++rs.groups;
                for (int i = 0; i < kGroup; ++i) {
                    const uint64_t k = g * kGroup + uint64_t(i);
                    const double wv = double(w[k]), rv = gd * double(q[k]), ev = wv - rv;
                    rs.sum_w2 += wv * wv; rs.sum_e2 += ev * ev; rs.sum_wr += wv * rv; rs.sum_r2 += rv * rv;
                    if (std::fabs(ev) > rs.max_abs_err) rs.max_abs_err = std::fabs(ev);
                    if (q[k] == 0) ++rs.zeros;
                }
                rs.n += kGroup;
            }
        }
    });
    for (const auto& rs : per_row) s.add(rs);   // fixed row order: the same sums for any thread count
}

// ------------------------------------------------------------------------------------------------ convert
bool convert_tensor(std::FILE* fp, const SourceFile& f, const SourceTensor& t, const ConvertOptions& opt,
                    TensorResult& res, std::string& err) {
    res = TensorResult();
    std::string why;
    if (!convertible(t, &why)) { err = t.name + ": " + why; return false; }
    const uint64_t cols = t.cols, rows = t.rows, nb = cols / kGroup;
    const uint64_t rb = row_bytes_for(Packing::I2, cols);
    uint64_t chunk_rows = std::max<uint64_t>(1, opt.chunk_weights / cols);
    if (chunk_rows > rows) chunk_rows = rows;

    Tensor& out = res.out;
    out.packing = Packing::I2;
    out.scale_mode = ScaleMode::PerBlock;
    out.rows = rows;
    out.cols = cols;
    out.codes.assign(static_cast<size_t>(rows * rb), 0);
    out.scales.assign(static_cast<size_t>(rows * nb), 0.0f);

    std::vector<uint8_t> raw;
    std::vector<float> w(static_cast<size_t>(chunk_rows * cols));
    std::vector<int8_t> q(w.size()), q2;
    std::vector<float> g2;
    if (opt.check_parity) { q2.resize(w.size()); g2.resize(static_cast<size_t>(chunk_rows * nb)); }

    auto run_fit = [&](Backend b, const float* wp, uint64_t ng, int8_t* qp, float* gp, std::string& e) -> Status {
        if (b == Backend::Cuda) return fit_groups_cuda(wp, ng, opt.fit, qp, gp, e);
        return fit_groups_cpu(wp, ng, opt.fit, opt.threads, qp, gp);
    };
    for (uint64_t r0 = 0; r0 < rows; r0 += chunk_rows) {
        const uint64_t n = std::min(chunk_rows, rows - r0);
        const uint64_t ng = n * nb;
        auto t0 = Clock::now();
        if (!read_rows(fp, f, t, r0, n, raw, w.data(), err)) return false;
        res.read_ms += ms_since(t0);

        float* gp = out.scales.data() + r0 * nb;
        t0 = Clock::now();
        std::string ferr;
        Status st = run_fit(opt.backend, w.data(), ng, q.data(), gp, ferr);
        res.fit_ms += ms_since(t0);
        if (st != Status::Ok) { err = t.name + ": fit failed: " + status_name(st) + (ferr.empty() ? "" : " (" + ferr + ")"); return false; }

        if (opt.check_parity) {
            const Backend other = opt.backend == Backend::Cuda ? Backend::Cpu : Backend::Cuda;
            t0 = Clock::now();
            st = run_fit(other, w.data(), ng, q2.data(), g2.data(), ferr);
            res.other_fit_ms += ms_since(t0);
            if (st != Status::Ok) { err = t.name + ": parity fit failed: " + status_name(st) + (ferr.empty() ? "" : " (" + ferr + ")"); return false; }
            for (uint64_t g = 0; g < ng; ++g) {
                const bool same = std::memcmp(q.data() + g * kGroup, q2.data() + g * kGroup, kGroup) == 0 &&
                                  std::memcmp(gp + g, g2.data() + g, sizeof(float)) == 0;
                if (!same) ++res.parity_groups_different;
            }
            res.parity_groups_checked += ng;
        }

        t0 = Clock::now();
        accumulate_stats(w.data(), q.data(), gp, n, cols, opt.threads, res.stats);
        res.stats_ms += ms_since(t0);
        for (uint64_t r = 0; r < n; ++r) {
            st = pack_row(q.data() + r * cols, cols, Packing::I2, out.codes.data() + (r0 + r) * rb);
            if (st != Status::Ok) { err = t.name + ": pack failed: " + status_name(st); return false; }
        }
    }
    View v;
    const Status vs = view_of(out, v);
    if (vs != Status::Ok) { err = t.name + ": packed tensor did not validate: " + status_name(vs); return false; }
    return true;
}

} // namespace convert
} // namespace ternary
} // namespace pchie
