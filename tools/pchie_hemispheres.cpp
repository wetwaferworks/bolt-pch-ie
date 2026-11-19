// pchie_hemispheres: the left and right hemispheres on a real model (Bolt E4B), with
// PCHIE as the world and the answer key and Mjolnir as the pilot.
//
//   pchie_hemispheres probe --gguf FILE --ternary DIR [--tail-from 38] [--threads 8]
//       Speed of the trunk and of the tail, the tail's agreement with the full forward
//       pass (bit for bit), and how close the ternary and Q4 tail weights are.
//   pchie_hemispheres run --gguf FILE --ternary DIR --data DIR --mjolnir MJ_HEMI_EXE
//       [--tail-from 38] [--train-mazes 6] [--holdout-mazes 10] [--size 7] [--seed 1]
//       [--classroom-rounds 2] [--swap-rounds 2] [--threads 8] [--chunk 8] [--save-model]
//       Stage 1, the classroom, then stage 2, the hemisphere swap.
//
// The two hemispheres are the same network at two precisions in its last layers
// (tail-from .. 41): the left runs the ternary weights the converter made from the
// original BF16 model, the right runs the Q4_K weights of the GGUF. Every model the
// pipeline can make lies on the line between them: W(t) = (1 - t) * ternary + t * Q4,
// with t = i / 8 for i = 0 .. 8. That one-parameter blend is the only weight
// edit this tool makes; each lesson picks the point that best fits its targets, and
// Mjolnir's gate decides on worlds neither side has seen.
//
// Mjolnir is not told any score. For every decision it gets the lesson and holdout
// sample sets, each naming the SHA-256 of the tail that acted and the tail that
// watched, and works the scores out itself (see mj_hemi_gate.h in Mjolnir). It is
// started as a program with an argument list, never through a shell. Every ternary
// file is checked against the SHA-256 in the converter's manifest before it is used.
//
// Nothing here writes the model file or a converter output. The kept model is written
// as a checksummed overlay (the tail weights in BF16 plus a manifest) in the data folder.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_cuda.h"
#include "pchie/pchie_det_math.h"
#include "pchie/pchie_ternary_convert.h"
#include "pchie/Sha256.h"
#include "pchie/hemispheres/GridMazeWorld.h"
#include "pchie/hemispheres/Hemisphere.h"
#include "pchie/hemispheres/SampleSet.h"
#include "pchie/hemispheres/Spawn.h"
#include "pchie/hemispheres/TernaryFolder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace pchie;
using namespace pchie::hemispheres;

namespace {

constexpr int kGrid = 8;   // t = i / 8, i = 0 (ternary) .. 8 (Q4)
constexpr uint32_t kA = GridMazeWorld::kActions;

struct Options {
    std::string cmd, gguf, ternary, data, mjolnir;
    int tail_from = 38, train_mazes = 6, holdout_mazes = 10, size = 7, classroom_rounds = 2, swap_rounds = 2;
    int threads = 8, chunk = 8, step_cap = 3;
    uint64_t seed = 1;
    bool save_model = false;
};

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- the tail weights: ternary and Q4, as BF16 --------------------------------------
struct TailTensor {
    std::string hf, gguf;
    uint64_t rows = 0, cols = 0;
    std::vector<uint16_t> ternary, q4, blend;
};

const char* kSlotsHf[] = {"self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                          "per_layer_input_gate", "per_layer_projection"};
const char* kSlotsGguf[] = {"attn_q", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "inp_gate", "proj"};

// A ternary tensor, read whole and checked against the manifest's SHA-256 before use.
bool load_ternary(const TernaryFolder& folder, const std::string& name, TailTensor& t, std::string& err) {
    std::vector<uint8_t> raw;
    if (!folder.read_verified(name, 1ull << 31, raw, err)) return false;
    const std::string& file = folder.by_name.at(name).file;
    ternary_ptn1_view v{};
    if (ternary_parse_ptn1(raw.data(), raw.size(), raw.size(), &v) != TERNARY_CONVERT_OK) { err = file + ": not a valid PTN1 file"; return false; }
    if (v.rows != t.rows || v.columns != t.cols) { err = file + ": shape differs from the model's"; return false; }
    if (v.columns % 128 != 0 || v.scale_count != v.rows * (v.columns / 128)) { err = file + ": scale count does not match"; return false; }
    t.ternary.assign(size_t(t.rows * t.cols), 0);
    std::vector<int8_t> codes(size_t(t.cols));
    for (uint64_t r = 0; r < t.rows; ++r) {
        if (ternary_unpack_row_i2(v.packed_codes + r * v.bytes_per_row, t.cols, codes.data()) != TERNARY_CONVERT_OK) {
            err = file + ": bad row";
            return false;
        }
        for (uint64_t c = 0; c < t.cols; ++c) {
            float scale;
            std::memcpy(&scale, v.scales_raw + 4 * (r * (t.cols / 128) + c / 128), 4);
            t.ternary[size_t(r * t.cols + c)] = to_bf16(float(codes[size_t(c)]) * scale);
        }
    }
    return true;
}

bool load_q4(std::FILE* f, const ternary_source_file& src, TailTensor& t, std::string& err) {
    const ternary_source_tensor* st = nullptr;
    for (size_t i = 0; i < src.tensor_count; ++i)
        if (src.tensors[i].name && t.gguf == src.tensors[i].name) st = &src.tensors[i];
    if (!st) { err = "GGUF has no " + t.gguf; return false; }
    if (st->rows != t.rows || st->columns != t.cols) { err = t.gguf + ": shape differs"; return false; }
    t.q4.assign(size_t(t.rows * t.cols), 0);
    std::vector<float> rowbuf(size_t(t.cols) * 64);
    uint8_t* raw = nullptr;
    size_t cap = 0;
    char e[256] = {0};
    for (uint64_t r = 0; r < t.rows; r += 64) {
        const uint64_t n = std::min<uint64_t>(64, t.rows - r);
        if (!ternary_read_rows(f, &src, st, r, n, &raw, &cap, rowbuf.data(), e, sizeof e)) {
            std::free(raw);
            err = t.gguf + ": " + e;
            return false;
        }
        for (uint64_t i = 0; i < n * t.cols; ++i) t.q4[size_t(r * t.cols + i)] = to_bf16(rowbuf[size_t(i)]);
    }
    std::free(raw);
    return true;
}

// ---- the model and its two parts ----------------------------------------------------
struct Captured {
    std::vector<float> x, ple;
    Gemma4Model::Branch br;
    std::array<float, kA> full_logits{};   // the full forward pass with the weights loaded at the time
};

struct Runner {
    Options o;
    Gemma4Model m;
    CudaBackend cb;
    std::unique_ptr<ThreadPool> pool;
    std::unique_ptr<Gemma4OverlayTarget> overlay;
    std::array<int32_t, kA> move_ids{};
    Gemma4Model::Branch prefix;
    std::vector<TailTensor> tail;
    std::string manifest_sha, gguf_digest;
    double trunk_tokens = 0, trunk_s = 0, tail_s = 0;
    uint64_t tails = 0;

    bool init(std::string& err) {
        if (!m.load(o.gguf, err)) return false;
        char d[32];
        std::snprintf(d, sizeof d, "%016llx", (unsigned long long)m.gguf().directory_digest());
        gguf_digest = d;
        const int NL = m.cfg().n_layer;
        if (o.tail_from < m.cfg().n_layer_kv || o.tail_from >= NL) {
            err = "--tail-from must name a layer that shares its KV (" + std::to_string(m.cfg().n_layer_kv) + " .. " +
                  std::to_string(NL - 1) + ")";
            return false;
        }
        m.init_cache(256, uint32_t(o.chunk + 2));
        pool = std::make_unique<ThreadPool>(o.threads);
        if (!cb.init(m, o.tail_from, 384, false, err, 1)) return false;
        if (cb.gpu_layers() > o.tail_from) { err = "the GPU took layers past --tail-from"; return false; }
        overlay = std::make_unique<Gemma4OverlayTarget>(m, cb.gpu_layers());
        for (uint32_t a = 0; a < kA; ++a) {
            const std::vector<int32_t> ids = m.tok().encode(GridMazeWorld::action_name(a), false);
            if (ids.empty()) { err = "cannot tokenize a move name"; return false; }
            move_ids[a] = ids[0];
            for (uint32_t b = 0; b < a; ++b)
                if (move_ids[b] == move_ids[a]) { err = "two move names start with the same token"; return false; }
        }
        const std::string pre =
            "<|turn>user\nYou walk a grid maze: pick up the key, open the door with it, then reach the exit. "
            "North is up. Answer with one word: North, South, East, West, Pickup or Use.\n";
        const std::vector<int32_t> ids = m.tok().encode_with_bos(pre);
        prefix = m.new_branch();
        std::vector<float> lg;
        const double t0 = now_s();
        for (int32_t id : ids) cb.forward(prefix, id, lg, *pool);
        trunk_s += now_s() - t0;
        trunk_tokens += double(ids.size());
        return true;
    }

    bool load_tail(std::string& err) {
        TernaryFolder folder;
        if (!folder.open(o.ternary, err)) return false;
        manifest_sha = folder.manifest_sha256;
        ternary_source_file src{};
        char e[256] = {0};
        if (!ternary_open_gguf(o.gguf.c_str(), &src, e, sizeof e)) { err = std::string("GGUF: ") + e; return false; }
        std::FILE* f = std::fopen(o.gguf.c_str(), "rb");
        if (!f) { ternary_source_file_clear(&src); err = "cannot open the GGUF"; return false; }
        bool ok = true;
        for (int l = o.tail_from; ok && l < m.cfg().n_layer; ++l) {
            for (size_t s = 0; ok && s < sizeof kSlotsHf / sizeof kSlotsHf[0]; ++s) {
                TailTensor t;
                t.hf = "model.language_model.layers." + std::to_string(l) + "." + kSlotsHf[s] + ".weight";
                t.gguf = "blk." + std::to_string(l) + "." + kSlotsGguf[s] + ".weight";
                if (!overlay->shape(t.hf, t.rows, t.cols)) { ok = false; err = "model has no " + t.hf; break; }
                if (!overlay->editable(t.hf)) { ok = false; err = t.hf + " is not editable here"; break; }
                ok = load_ternary(folder, t.hf, t, err) && load_q4(f, src, t, err);
                if (ok) { t.blend.assign(t.q4.size(), 0); tail.push_back(std::move(t)); }
            }
        }
        std::fclose(f);
        ternary_source_file_clear(&src);
        return ok;
    }

    // Point the tail at W(i / 8). sha (optional) gets the SHA-256 of every BF16 byte, in order.
    bool use_blend(int i, std::array<uint8_t, 32>* sha, std::string& err) {
        const float t = float(i) / float(kGrid);
        Sha256 s;
        for (TailTensor& w : tail) {
            if (i == 0) w.blend = w.ternary;
            else if (i == kGrid) w.blend = w.q4;
            else
                for (size_t k = 0; k < w.blend.size(); ++k)
                    w.blend[k] = to_bf16((1.0f - t) * from_bf16(w.ternary[k]) + t * from_bf16(w.q4[k]));
            if (sha) s.update(reinterpret_cast<const uint8_t*>(w.blend.data()), w.blend.size() * 2);
            if (!overlay->set_bf16(w.hf, w.blend.data(), err)) return false;
        }
        if (sha) s.final(sha->data());
        return true;
    }

    // Run the trunk over one state; capture the residual entering --tail-from at the last token.
    void trunk(const MazeGrid& g, Captured& c) {
        const std::string text = GridMazeWorld::describe(g) + "<turn|>\n<|turn>model\n";
        const std::vector<int32_t> ids = m.tok().encode(text);
        const int D = m.cfg().d_model, NP = m.cfg().n_layer * m.cfg().d_ple;
        c.x.assign(size_t(D), 0.0f);
        c.ple.assign(size_t(NP), 0.0f);
        c.br = m.fork(prefix);
        std::vector<float> lg;
        const double t0 = now_s();
        for (size_t k = 0; k < ids.size(); ++k) {
            if (k + 1 == ids.size()) m.set_capture(o.tail_from, c.x.data(), c.ple.data());
            cb.forward(c.br, ids[k], lg, *pool);
        }
        m.set_capture(-1, nullptr, nullptr);
        trunk_s += now_s() - t0;
        trunk_tokens += double(ids.size());
        for (uint32_t a = 0; a < kA; ++a) c.full_logits[a] = lg[size_t(move_ids[a])];
    }

    // The tail and the six head rows for the weights in place now.
    void tail_logits(const Captured& c, std::array<float, kA>& out) {
        const double t0 = now_s();
        const int D = m.cfg().d_model;
        std::vector<float> x = c.x;
        Gemma4Model::Branch b = c.br;   // read-only view: the shared-KV layers only read
        b.n_past = c.br.n_past - 1;
        for (int l = o.tail_from; l < m.cfg().n_layer; ++l) m.layer(l, b, x.data(), c.ple.data(), *pool);
        std::vector<float> u(static_cast<size_t>(D));
        m.rmsnorm(x.data(), m.out_norm(), u.data(), D);
        const size_t row_bytes = size_t(D / 32) * 34;   // Q8_0: a 2-byte scale and 32 int8 per block
        const float cap = m.cfg().softcap, icap = 1.0f / m.cfg().softcap;
        for (uint32_t a = 0; a < kA; ++a) {
            QTensor row;
            row.data = static_cast<const uint8_t*>(m.tok_embd().data) + size_t(move_ids[a]) * row_bytes;
            row.type = 8;
            row.ne0 = uint64_t(D);
            row.ne1 = 1;
            float y = 0.0f;
            m.matvec_batch(row, u.data(), 1, &y, *pool);
            out[a] = cap * tanh_deterministic(y * icap);
        }
        tail_s += now_s() - t0;
        ++tails;
    }
};

// ---- the world's states and every model's distribution on them ----------------------
struct State {
    MazeGrid g;
    uint32_t episode = 0, step = 0, legal = 0;
    int8_t f[GridMazeWorld::kFeatures] = {};
    Grade grade[kA] = {};
};

std::vector<State> make_states(uint64_t seed, int mazes, int size) {
    std::vector<State> out;
    uint64_t s = seed;
    int made = 0;
    for (int tries = 0; made < mazes && tries < mazes * 20; ++tries) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const MazeGrid g(s, size, size);
        const std::vector<MazeGrid> path = GridMazeWorld::guided_states(g);
        if (path.size() < 3) continue;
        for (size_t k = 0; k < path.size(); ++k) {
            State st;
            st.g = path[k];
            st.episode = uint32_t(made);
            st.step = uint32_t(k);
            st.legal = GridMazeWorld::legal_mask(st.g);
            GridMazeWorld::features(st.g, st.f);
            for (uint32_t a = 0; a < kA; ++a) st.grade[a] = GridMazeWorld::grade(st.g, a);
            out.push_back(st);
        }
        ++made;
    }
    return out;
}

using Table = std::vector<std::array<uint16_t, kMaxActions>>;   // one distribution per state
struct Tables { std::vector<Table> train, hold; };              // [grid point][state]

bool fill_tables(Runner& R, const std::vector<State>& states, std::vector<Table>& out,
                 std::vector<std::array<uint8_t, 32>>& shas, std::string& err) {
    out.assign(kGrid + 1, Table(states.size()));
    const bool want_sha = shas.empty();
    if (want_sha) shas.resize(kGrid + 1);
    for (size_t c0 = 0; c0 < states.size(); c0 += size_t(R.o.chunk)) {
        const size_t c1 = std::min(states.size(), c0 + size_t(R.o.chunk));
        std::vector<Captured> cap(c1 - c0);
        R.overlay->restore_all();
        for (size_t k = c0; k < c1; ++k) R.trunk(states[k].g, cap[k - c0]);
        for (int i = 0; i <= kGrid; ++i) {
            if (!R.use_blend(i, want_sha && c0 == 0 ? &shas[size_t(i)] : nullptr, err)) return false;
            for (size_t k = c0; k < c1; ++k) {
                std::array<float, kA> lg;
                R.tail_logits(cap[k - c0], lg);
                auto& p = out[size_t(i)][k];
                if (!to_q16(lg.data(), kA, states[k].legal, p.data())) { err = "a logit was not finite"; return false; }
            }
        }
        for (auto& c : cap) R.m.release(c.br);
        std::printf("  states %zu/%zu  (trunk %.1f tok/s, tail %.0f ms each)\n", c1, states.size(),
                    R.trunk_tokens / std::max(1e-9, R.trunk_s), 1000.0 * R.tail_s / double(std::max<uint64_t>(1, R.tails)));
        std::fflush(stdout);
    }
    R.overlay->restore_all();
    return true;
}

uint32_t play_bp(const std::vector<State>& st, const Table& t) {
    uint64_t ok = 0;
    for (size_t k = 0; k < st.size(); ++k) {
        const uint32_t a = pick(t[k].data(), kA, st[k].legal);
        ok += a < kA && st[k].grade[a] == Grade::Cleared;
    }
    return st.empty() ? 0 : uint32_t(ok * 10000 / st.size());
}

SampleSet make_set(const std::vector<State>& st, const Table& actor, const Table& watcher, uint32_t round, uint8_t pass,
                   Side side, uint64_t seed, const std::array<uint8_t, 32>& a_sha, const std::array<uint8_t, 32>& w_sha,
                   Purpose purpose = Purpose::Lesson) {
    Header h;
    h.round = round; h.pass = pass; h.purpose = purpose; h.actor_side = side; h.world = World::GridMaze;
    h.n_features = GridMazeWorld::kFeatures; h.n_actions = kA; h.seed = seed;
    std::memcpy(h.actor.data(), a_sha.data(), kFingerprint);
    std::memcpy(h.observer.data(), w_sha.data(), kFingerprint);
    SampleSet s(h);
    std::string err;
    for (size_t k = 0; k < st.size(); ++k) {
        Record r;
        r.episode = st[k].episode;
        r.step = uint16_t(st[k].step);
        const uint32_t a = pick(actor[k].data(), kA, st[k].legal);
        r.actor_move = uint8_t(a < kA ? a : 0);
        r.grade = st[k].grade[r.actor_move];
        r.verdict = judge(watcher[k].data(), kA, st[k].legal, r.actor_move);
        std::memcpy(r.features.data(), st[k].f, GridMazeWorld::kFeatures);
        std::copy(actor[k].begin(), actor[k].end(), r.actor_probs.begin());
        std::copy(watcher[k].begin(), watcher[k].end(), r.watcher_probs.begin());
        if (!s.add(r, err)) { std::fprintf(stderr, "internal: %s\n", err.c_str()); std::exit(1); }
    }
    return s;
}

struct Target { size_t state; uint32_t move; double weight; };

// Mjolnir's combiner: the key decides the sign; a watcher's move counts only where the key
// failed the actor, the watcher objected with confidence, and as far as it earned trust.
void actor_targets(const SampleSet& s, std::vector<Target>& out, size_t base) {
    const Tally t = s.tally();
    double trust = t.judged() ? 2.0 * double(t.judged_right()) / double(t.judged()) - 1.0 : 0.0;
    if (trust < 0) trust = 0;
    for (size_t k = 0; k < s.records().size(); ++k) {
        const Record& r = s.records()[k];
        if (r.grade == Grade::Cleared) out.push_back({base + k, r.actor_move, 1.0});
        else if (r.grade == Grade::NotYet) {
            out.push_back({base + k, r.actor_move, -0.5});
            if (r.verdict.reason == Reason::PreferOther && r.verdict.choice != r.actor_move)
                out.push_back({base + k, r.verdict.choice, 0.75 * trust * r.verdict.confidence_bp / 10000.0});
        }
    }
}

// Mjolnir's watcher lessons (mj_classroom_watcher_targets).
void watcher_targets(const SampleSet& s, std::vector<Target>& out) {
    for (size_t k = 0; k < s.records().size(); ++k) {
        const Record& r = s.records()[k];
        const bool cleared = r.grade == Grade::Cleared, failed = r.grade == Grade::NotYet;
        const bool agreed = r.verdict.reason == Reason::Agree;
        const bool objected = r.verdict.reason == Reason::PreferOther || r.verdict.reason == Reason::Illegal;
        double w = 0;
        if (objected && cleared) w = 1.0;
        else if (agreed && failed) w = -1.0;
        else if (agreed && cleared) w = 0.25;
        else if (objected && failed) w = -0.25;
        if (w != 0) out.push_back({k, r.actor_move, w});
    }
}

// How well a table fits a set of targets: weighted mean log-probability.
double fit(const std::vector<Target>& tg, const Table& t) {
    double sum = 0, wsum = 0;
    for (const Target& x : tg) {
        const double p = std::max(1.0, double(t[x.state][x.move])) / double(kProbOne);
        sum += x.weight * std::log(p);
        wsum += std::fabs(x.weight);
    }
    return wsum > 0 ? sum / wsum : 0;
}

int best_point(const std::vector<Target>& tg, const std::vector<Table>& tables, int from, int lo, int hi) {
    int best = -1;
    double best_fit = -1e300;
    for (int i = lo; i <= hi; ++i) {
        if (i == from) continue;
        const double f = fit(tg, tables[size_t(i)]);
        if (f > best_fit) { best_fit = f; best = i; }
    }
    return best;
}

// Mjolnir's gate, started without a shell (see Spawn.h). 0 keep, 3 roll back, anything else refused.
int ask_mjolnir(const Options& o, std::vector<std::string> args) {
    args.insert(args.begin(), {o.mjolnir, "gate", "--data", o.data});
    return run_program(args);
}

bool make_dir(const std::string& d, std::string& err) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::u8path(d), ec);
    if (ec || !std::filesystem::is_directory(std::filesystem::u8path(d), ec)) { err = "cannot make the folder " + d; return false; }
    return true;
}

std::string point_name(int i) { return "t=" + std::to_string(i) + "/8"; }

bool save_set(const Options& o, const std::string& name, const SampleSet& s, std::string& path, std::string& err) {
    path = o.data + "/" + name;
    std::array<uint8_t, 32> sha;
    return s.save(path, sha, err);
}

bool save_model(Runner& R, int i, const std::string& dir, std::string& err) {
    std::array<uint8_t, 32> all;
    if (!R.use_blend(i, &all, err)) return false;
    const std::string bin = dir + "/tail.bf16", man = dir + "/manifest.tsv";
    std::FILE* f = std::fopen((bin + ".new").c_str(), "wb");
    if (!f) { err = "cannot write " + bin; return false; }
    std::ostringstream m;
    m << "name\trows\tcols\toffset\tbytes\tsha256\n";
    uint64_t off = 0;
    bool ok = true;
    for (const TailTensor& t : R.tail) {
        const size_t n = t.blend.size() * 2;
        uint8_t sha[32];
        Sha256 s;
        s.update(reinterpret_cast<const uint8_t*>(t.blend.data()), n);
        s.final(sha);
        ok = ok && std::fwrite(t.blend.data(), 1, n, f) == n;
        m << t.hf << '\t' << t.rows << '\t' << t.cols << '\t' << off << '\t' << n << '\t' << hex(sha, 32) << '\n';
        off += n;
    }
    ok = (std::fclose(f) == 0) && ok;
    R.overlay->restore_all();
    if (!ok || std::rename((bin + ".new").c_str(), bin.c_str()) != 0) { err = "cannot write " + bin; return false; }
    std::ofstream mf(man, std::ios::binary);
    mf << m.str();
    std::ofstream rf(dir + "/RECIPE.txt", std::ios::binary);
    rf << "The new model: the GGUF below with layers " << R.o.tail_from << ".." << (R.m.cfg().n_layer - 1)
       << " replaced by tail.bf16.\n"
       << "gguf\t" << R.o.gguf << "\ngguf_directory_digest\t" << R.gguf_digest << "\nternary_manifest_sha256\t"
       << R.manifest_sha << "\nblend\t" << point_name(i) << " (W = (1 - t) * ternary + t * Q4)\ntail_sha256\t"
       << hex(all.data(), 32) << "\n";
    return mf.good() && rf.good();
}

int cmd_probe(Runner& R) {
    std::string err;
    if (!R.load_tail(err)) { std::fprintf(stderr, "pchie_hemispheres: %s\n", err.c_str()); return 1; }
    std::printf("GPU layers 0..%d, CPU layers %d..%d; tail = layers %d..%d (%zu tensors)\n", R.cb.gpu_layers() - 1,
                R.cb.gpu_layers(), R.m.cfg().n_layer - 1, R.o.tail_from, R.m.cfg().n_layer - 1, R.tail.size());
    for (const TailTensor& t : R.tail) {
        double ab = 0, aa = 0, bb = 0;
        for (size_t k = 0; k < t.q4.size(); ++k) {
            const double a = from_bf16(t.ternary[k]), b = from_bf16(t.q4[k]);
            ab += a * b; aa += a * a; bb += b * b;
        }
        std::printf("  %-55s %6llu x %-6llu cosine(ternary, Q4) %.4f\n", t.hf.c_str(), (unsigned long long)t.rows,
                    (unsigned long long)t.cols, ab / std::sqrt(aa * bb));
    }
    const std::vector<State> st = make_states(R.o.seed, 2, R.o.size);
    std::printf("%zu states from 2 mazes; first: %s\n", st.size(), st.empty() ? "-" : GridMazeWorld::describe(st[0].g).c_str());
    uint32_t same = 0, n = 0;
    for (size_t k = 0; k < st.size() && k < 4; ++k) {
        Captured c;
        R.trunk(st[k].g, c);
        std::array<float, kA> lg;
        R.tail_logits(c, lg);   // the loaded Q4 weights: must equal the full pass
        for (uint32_t a = 0; a < kA; ++a) { same += std::memcmp(&lg[a], &c.full_logits[a], 4) == 0; ++n; }
        if (!R.use_blend(0, nullptr, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::array<float, kA> tl;
        R.tail_logits(c, tl);
        R.overlay->restore_all();
        std::printf("  state %zu: Q4 logits", k);
        for (uint32_t a = 0; a < kA; ++a) std::printf(" %s %.3f", GridMazeWorld::action_name(a), lg[a]);
        std::printf("\n           ternary tail");
        for (uint32_t a = 0; a < kA; ++a) std::printf(" %s %.3f", GridMazeWorld::action_name(a), tl[a]);
        std::printf("\n");
        R.m.release(c.br);
    }
    std::printf("tail against the full forward pass: %u of %u logits identical to the bit\n", same, n);
    std::printf("trunk %.2f tok/s (%.0f tokens), tail %.0f ms each\n", R.trunk_tokens / R.trunk_s, R.trunk_tokens,
                1000.0 * R.tail_s / double(std::max<uint64_t>(1, R.tails)));
    return same == n ? 0 : 1;
}

int cmd_run(Runner& R) {
    const Options& o = R.o;
    std::string err;
    if (!R.load_tail(err)) { std::fprintf(stderr, "pchie_hemispheres: %s\n", err.c_str()); return 1; }
    if (!make_dir(o.data, err)) { std::fprintf(stderr, "pchie_hemispheres: %s\n", err.c_str()); return 1; }
    const std::vector<State> train = make_states(o.seed * 1000003ull + 11, o.train_mazes, o.size);
    const std::vector<State> hold = make_states(o.seed * 1000003ull + 977, o.holdout_mazes, o.size);
    std::printf("%zu lesson states (%d mazes), %zu holdout states (%d mazes), %dx%d\n", train.size(), o.train_mazes,
                hold.size(), o.holdout_mazes, o.size, o.size);
    std::vector<std::array<uint8_t, 32>> sha;
    Tables T;
    const double t0 = now_s();
    std::printf("lesson states, 9 points on the line from ternary (t=0) to Q4 (t=8/8):\n");
    if (!fill_tables(R, train, T.train, sha, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::printf("holdout states:\n");
    if (!fill_tables(R, hold, T.hold, sha, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::printf("model work: %.0f s\n", now_s() - t0);
    std::printf("holdout play score along the line:");
    for (int i = 0; i <= kGrid; ++i) std::printf(" %s:%u", point_name(i).c_str(), play_bp(hold, T.hold[size_t(i)]));
    std::printf("\n");

    int L = 0, Rt = kGrid;   // the left starts ternary, the right Q4
    auto sh = [&](int i) { return hex(sha[size_t(i)].data(), 32); };
    const uint64_t hold_seed = o.seed * 1000003ull + 977;
    // Holdout evidence for Mjolnir: the held-out states played by model `a` and watched by `w`,
    // every move graded by the answer key. Mjolnir works the scores out from these itself.
    auto holdout = [&](int a, int w, std::string& path) {
        const SampleSet s = make_set(hold, T.hold[size_t(a)], T.hold[size_t(w)], 0, 1, Side::Left, hold_seed,
                                     sha[size_t(a)], sha[size_t(w)], Purpose::Holdout);
        const std::string name = a == w ? "holdout-" + std::to_string(a) + ".mjhemi"
                                        : "holdout-" + std::to_string(a) + "-watched-by-" + std::to_string(w) + ".mjhemi";
        return save_set(o, name, s, path, err);
    };

    // ---- stage 1: the classroom
    std::printf("== stage 1: the classroom (left acts, right watches)\n");
    for (int r = 1; r <= o.classroom_rounds; ++r) {
        const SampleSet lesson = make_set(train, T.train[size_t(L)], T.train[size_t(Rt)], uint32_t(r), 1, Side::Left,
                                          o.seed, sha[size_t(L)], sha[size_t(Rt)]);
        std::string lp;
        if (!save_set(o, "classroom-" + std::to_string(r) + "-lesson.mjhemi", lesson, lp, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const Tally ty = lesson.tally();
        std::vector<Target> at, wt;
        actor_targets(lesson, at, 0);
        watcher_targets(lesson, wt);
        const int La = best_point(at, T.train, L, std::max(0, L - o.step_cap), std::min(kGrid, L + o.step_cap));
        const int Rw = best_point(wt, T.train, Rt, std::max(0, Rt - o.step_cap), std::min(kGrid, Rt + o.step_cap));
        std::printf("classroom %d: lesson %u steps, cleared %u; watcher agreed-right %u, shared mistakes %u, objected-right %u, "
                    "objected-wrong %u, unsure %u; %zu actor targets, %zu watcher targets\n",
                    r, ty.steps, ty.cleared, ty.agree_right, ty.agree_wrong, ty.object_right, ty.object_wrong, ty.unsure,
                    at.size(), wt.size());
        std::string hL, hLa, hRt, hRw, jRt, jRw;
        if (!holdout(L, L, hL) || !holdout(La, La, hLa) || !holdout(Rt, Rt, hRt) || !holdout(Rw, Rw, hRw) ||
            !holdout(L, Rt, jRt) || !holdout(L, Rw, jRw)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::printf("  actor candidate %s -> %s\n", point_name(L).c_str(), point_name(La).c_str());
        const int ka = ask_mjolnir(o, {"--kind", "actor", "--round", std::to_string(r), "--lesson", lp, "--holdout", hL,
                                       "--holdout", hLa, "--parent", sh(L), "--candidate", sh(La), "--note",
                                       "left " + point_name(L) + " to " + point_name(La)});
        std::printf("  watcher candidate %s -> %s\n", point_name(Rt).c_str(), point_name(Rw).c_str());
        const int kw = ask_mjolnir(o, {"--kind", "watcher", "--round", std::to_string(r), "--lesson", lp, "--holdout", hRt,
                                       "--holdout", hRw, "--holdout", jRt, "--holdout", jRw, "--parent", sh(Rt),
                                       "--candidate", sh(Rw), "--note", "right " + point_name(Rt) + " to " + point_name(Rw)});
        if (ka != 0 && ka != 3) { std::fprintf(stderr, "Mjolnir refused the actor decision (%d)\n", ka); return 1; }
        if (kw != 0 && kw != 3) { std::fprintf(stderr, "Mjolnir refused the watcher decision (%d)\n", kw); return 1; }
        if (ka == 0) L = La;
        if (kw == 0) Rt = Rw;
        if (ka != 0 && kw != 0) { std::printf("  nothing kept; the classroom stops\n"); break; }
    }
    std::printf("after the classroom: left %s (%u bp), right %s (%u bp)\n", point_name(L).c_str(),
                play_bp(hold, T.hold[size_t(L)]), point_name(Rt).c_str(), play_bp(hold, T.hold[size_t(Rt)]));

    // ---- stage 2: the hemisphere swap
    std::printf("== stage 2: the hemisphere swap\n");
    int child_kept = -1;
    for (int r = 1; r <= o.swap_rounds; ++r) {
        const SampleSet p1 = make_set(train, T.train[size_t(L)], T.train[size_t(Rt)], uint32_t(r), 1, Side::Left, o.seed, sha[size_t(L)], sha[size_t(Rt)]);
        const SampleSet p2 = make_set(train, T.train[size_t(Rt)], T.train[size_t(L)], uint32_t(r), 2, Side::Right, o.seed, sha[size_t(Rt)], sha[size_t(L)]);
        std::string f1, f2;
        if (!save_set(o, "swap-" + std::to_string(r) + "-pass1.mjhemi", p1, f1, err) ||
            !save_set(o, "swap-" + std::to_string(r) + "-pass2.mjhemi", p2, f2, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::vector<Target> tg;
        actor_targets(p1, tg, 0);
        actor_targets(p2, tg, 0);   // the same states in the same order, so the indices line up
        const int lo = std::min(L, Rt) + 1, hi = std::max(L, Rt) - 1;
        if (lo > hi) { std::printf("swap %d: the parents are neighbours on the line; no child lies between them\n", r); break; }
        const int C = best_point(tg, T.train, -1, lo, hi);
        // which parent is the stronger one only decides which side the child takes; Mjolnir scores all three
        const uint32_t sl = play_bp(hold, T.hold[size_t(L)]), sr = play_bp(hold, T.hold[size_t(Rt)]);
        const Tally t1 = p1.tally(), t2 = p2.tally();
        std::printf("swap %d: pass 1 cleared %u of %u, pass 2 cleared %u of %u; child %s from %zu targets\n", r, t1.cleared,
                    t1.steps, t2.cleared, t2.steps, point_name(C).c_str(), tg.size());
        std::string hl, hr, hc;
        if (!holdout(L, L, hl) || !holdout(Rt, Rt, hr) || !holdout(C, C, hc)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const int k = ask_mjolnir(o, {"--kind", "swap", "--round", std::to_string(r), "--lesson", f1, "--lesson", f2,
                                      "--holdout", hl, "--holdout", hr, "--holdout", hc, "--parent", sh(sl >= sr ? L : Rt),
                                      "--other", sh(sl >= sr ? Rt : L), "--candidate", sh(C), "--note",
                                      "child " + point_name(C)});
        if (k != 0 && k != 3) { std::fprintf(stderr, "Mjolnir refused the swap decision (%d)\n", k); return 1; }
        if (k == 3) break;
        child_kept = C;
        // the stronger parent moves to the other side, the child takes its old side
        if (sl >= sr) { Rt = L; L = C; } else { L = Rt; Rt = C; }
    }
    int final_point = child_kept;
    if (final_point < 0) final_point = play_bp(hold, T.hold[size_t(L)]) >= play_bp(hold, T.hold[size_t(Rt)]) ? L : Rt;
    std::printf("new model: %s, holdout %u bp (%s)\n", point_name(final_point).c_str(), play_bp(hold, T.hold[size_t(final_point)]),
                child_kept >= 0 ? "a swap child that beat both parents" : "no swap child was kept; the stronger classroom side");
    std::printf("tail sha256 %s\n", sh(final_point).c_str());
    if (o.save_model) {
        const std::string dir = o.data + "/new-model";
        if (!make_dir(dir, err)) { std::fprintf(stderr, "pchie_hemispheres: %s\n", err.c_str()); return 1; }
        if (!save_model(R, final_point, dir, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::printf("saved the new model's tail to %s (tail.bf16, manifest.tsv, RECIPE.txt)\n", dir.c_str());
    }
    std::printf("trunk %.2f tok/s over %.0f tokens; %llu tails at %.0f ms\n", R.trunk_tokens / R.trunk_s, R.trunk_tokens,
                (unsigned long long)R.tails, 1000.0 * R.tail_s / double(std::max<uint64_t>(1, R.tails)));
    return 0;
}

int usage() {
    std::fprintf(stderr,
                 "usage: pchie_hemispheres probe --gguf FILE --ternary DIR [--tail-from 38] [--threads 8]\n"
                 "       pchie_hemispheres run --gguf FILE --ternary DIR --data DIR --mjolnir MJ_HEMI\n"
                 "                [--tail-from 38] [--train-mazes 6] [--holdout-mazes 10] [--size 7] [--seed 1]\n"
                 "                [--classroom-rounds 2] [--swap-rounds 2] [--threads 8] [--chunk 8] [--step-cap 3]\n"
                 "                [--save-model]\n");
    return 2;
}

bool parse_int(const char* s, int lo, int hi, int& out) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!s[0] || *end || v < lo || v > hi) return false;
    out = int(v);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    if (argc < 2) return usage();
    o.cmd = argv[1];
    if (o.cmd != "probe" && o.cmd != "run") return usage();
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--save-model") { o.save_model = true; continue; }
        if (i + 1 >= argc) return usage();
        const char* v = argv[++i];
        int n = 0;
        if (a == "--gguf") o.gguf = v;
        else if (a == "--ternary") o.ternary = v;
        else if (a == "--data") o.data = v;
        else if (a == "--mjolnir") o.mjolnir = v;
        else if (a == "--tail-from" && parse_int(v, 1, 255, n)) o.tail_from = n;
        else if (a == "--train-mazes" && parse_int(v, 1, 200, n)) o.train_mazes = n;
        else if (a == "--holdout-mazes" && parse_int(v, 1, 400, n)) o.holdout_mazes = n;
        else if (a == "--size" && parse_int(v, 5, 16, n)) o.size = n;
        else if (a == "--seed" && parse_int(v, 0, 1000000, n)) o.seed = uint64_t(n);
        else if (a == "--classroom-rounds" && parse_int(v, 0, 8, n)) o.classroom_rounds = n;
        else if (a == "--swap-rounds" && parse_int(v, 0, 8, n)) o.swap_rounds = n;
        else if (a == "--threads" && parse_int(v, 1, 64, n)) o.threads = n;
        else if (a == "--chunk" && parse_int(v, 1, 64, n)) o.chunk = n;
        else if (a == "--step-cap" && parse_int(v, 1, kGrid, n)) o.step_cap = n;
        else return usage();
    }
    if (o.gguf.empty() || o.ternary.empty()) return usage();
    if (o.cmd == "run" && (o.data.empty() || o.mjolnir.empty())) return usage();
    {   // every path is checked before use; none of them ever reaches a shell
        const std::pair<const char*, const std::string*> paths[] = {{"--gguf", &o.gguf}, {"--ternary", &o.ternary},
                                                                    {"--data", &o.data}, {"--mjolnir", &o.mjolnir}};
        for (const auto& p : paths) {
            std::string why;
            if (p.second->empty() && (o.cmd == "probe")) continue;
            if (!path_ok(*p.second, why)) { std::fprintf(stderr, "pchie_hemispheres: %s: %s\n", p.first, why.c_str()); return 2; }
        }
        std::error_code ec;
        if (o.cmd == "run" && !std::filesystem::is_regular_file(std::filesystem::u8path(o.mjolnir), ec)) {
            std::fprintf(stderr, "pchie_hemispheres: --mjolnir must name the mj_hemi program file\n");
            return 2;
        }
    }
    Runner R;
    R.o = o;
    std::string err;
    try {
        if (!R.init(err)) { std::fprintf(stderr, "pchie_hemispheres: %s\n", err.c_str()); return 1; }
        return o.cmd == "probe" ? cmd_probe(R) : cmd_run(R);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pchie_hemispheres: %s\n", e.what());
        return 1;
    }
}
