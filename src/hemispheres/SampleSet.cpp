// Bolt PCHIE - MJHEMI2 sample sets and verdicts. See include/pchie/hemispheres/SampleSet.h.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#include "pchie/hemispheres/SampleSet.h"
#include "pchie/Sha256.h"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace pchie {
namespace hemispheres {

static const uint8_t kMagic[8] = {'M', 'J', 'H', 'E', 'M', 'I', '2', 0};
static constexpr uint32_t kVersion = 2;

static int32_t to_bp(int32_t diff) { return int32_t((int64_t(diff) * 10000) / int64_t(kProbOne)); }

uint32_t pick(const uint16_t* probs, uint32_t n_actions, uint32_t legal) {
    uint32_t best = kMaxActions;
    if (n_actions > kMaxActions) return kMaxActions;
    for (uint32_t a = 0; a < n_actions; ++a) {
        if (!(legal & (1u << a))) continue;
        if (best == kMaxActions || probs[a] > probs[best]) best = a;
    }
    return best;
}

Verdict judge(const uint16_t* p, uint32_t n_actions, uint32_t legal, uint32_t actor_move) {
    Verdict v;
    uint32_t top = kMaxActions, second = kMaxActions;
    int32_t p_top = 0, p_second = 0, p_other = 0;
    bool any_other = false;
    if (n_actions > kMaxActions) n_actions = kMaxActions;
    for (uint32_t a = 0; a < n_actions; ++a) {
        if (!(legal & (1u << a))) continue;
        const int32_t q = p[a];
        if (top == kMaxActions || q > p_top) { second = top; p_second = p_top; top = a; p_top = q; }
        else if (second == kMaxActions || q > p_second) { second = a; p_second = q; }
    }
    if (second == kMaxActions) p_second = 0;
    v.choice = uint8_t(top == kMaxActions ? 0 : top);
    v.confidence_bp = uint16_t(to_bp(p_top - p_second));
    if (actor_move >= n_actions || !(legal & (1u << actor_move))) {
        v.reason = Reason::Illegal;
        v.score_bp = -10000;
        return v;
    }
    const int32_t p_actor = p[actor_move];
    for (uint32_t a = 0; a < n_actions; ++a) {
        if (a == actor_move || !(legal & (1u << a))) continue;
        if (!any_other || int32_t(p[a]) > p_other) p_other = p[a];
        any_other = true;
    }
    v.score_bp = int16_t(to_bp(p_actor - p_other));
    if (top == actor_move) v.reason = Reason::Agree;
    else if (v.confidence_bp < kUnsureBp) v.reason = Reason::Unsure;
    else v.reason = Reason::PreferOther;
    return v;
}

bool to_q16(const float* logits, uint32_t n, uint32_t legal, uint16_t* out) {
    if (n == 0 || n > kMaxActions) return false;
    double mx = -INFINITY;
    uint32_t best = kMaxActions;
    for (uint32_t a = 0; a < kMaxActions; ++a) out[a] = 0;
    for (uint32_t a = 0; a < n; ++a) {
        if (!(legal & (1u << a))) continue;
        if (!std::isfinite(logits[a])) return false;
        if (best == kMaxActions || double(logits[a]) > mx) { mx = logits[a]; best = a; }
    }
    if (best == kMaxActions) return false;
    double e[kMaxActions] = {}, sum = 0.0;
    for (uint32_t a = 0; a < n; ++a)
        if (legal & (1u << a)) { e[a] = std::exp(double(logits[a]) - mx); sum += e[a]; }
    uint32_t total = 0;
    for (uint32_t a = 0; a < n; ++a)
        if (legal & (1u << a)) { out[a] = uint16_t(std::floor(e[a] / sum * double(kProbOne))); total += out[a]; }
    if (total > kProbOne) return false;   // cannot happen with a finite softmax; refuse rather than wrap
    out[best] = uint16_t(out[best] + (kProbOne - total));
    return true;
}

bool SampleSet::header_ok(const Header& h) {
    if (h.pass != 1 && h.pass != 2) return false;
    if (uint8_t(h.actor_side) > 1) return false;
    if (uint8_t(h.world) < 1 || uint8_t(h.world) > 3) return false;
    if (h.n_features < 1 || h.n_features > kMaxFeatures) return false;
    if (h.n_actions < 2 || h.n_actions > kMaxActions) return false;
    if (uint8_t(h.purpose) != 1 && uint8_t(h.purpose) != 2) return false;
    return true;
}

static bool probs_ok(const std::array<uint16_t, kMaxActions>& p, uint32_t na) {
    uint32_t sum = 0;
    for (uint32_t a = 0; a < kMaxActions; ++a) {
        if (a >= na && p[a] != 0) return false;
        sum += p[a];
    }
    return sum + na >= kProbOne && sum <= kProbOne + na;
}

bool SampleSet::record_ok(const Header& h, const Record& r) {
    if (!header_ok(h)) return false;
    if (r.actor_move >= h.n_actions || r.verdict.choice >= h.n_actions) return false;
    if (uint8_t(r.grade) < 1 || uint8_t(r.grade) > 3) return false;
    if (uint8_t(r.verdict.reason) < 1 || uint8_t(r.verdict.reason) > 4) return false;
    if (r.verdict.score_bp < -10000 || r.verdict.score_bp > 10000) return false;
    if (r.verdict.confidence_bp > 10000) return false;
    for (uint32_t f = h.n_features; f < kMaxFeatures; ++f)
        if (r.features[f] != 0) return false;
    return probs_ok(r.actor_probs, h.n_actions) && probs_ok(r.watcher_probs, h.n_actions);
}

bool SampleSet::add(const Record& r, std::string& err) {
    if (!record_ok(h_, r)) { err = "record does not fit the header or is out of range"; return false; }
    if (r_.size() >= kMaxRecords) { err = "too many records"; return false; }
    if (!r_.empty()) {
        const Record& p = r_.back();
        if (r.episode < p.episode || (r.episode == p.episode && r.step <= p.step)) {
            err = "records must be strictly increasing by (episode, step)";
            return false;
        }
    }
    r_.push_back(r);
    return true;
}

Tally SampleSet::tally() const {
    Tally t;
    for (const Record& r : r_) {
        const bool cleared = r.grade == Grade::Cleared, failed = r.grade == Grade::NotYet;
        ++t.steps;
        t.cleared += cleared;
        t.not_yet += failed;
        if (r.verdict.reason == Reason::Unsure) ++t.unsure;
        if (r.verdict.reason == Reason::Agree) {
            t.agree_right += cleared;
            t.agree_wrong += failed;
        } else if (r.verdict.reason == Reason::PreferOther || r.verdict.reason == Reason::Illegal) {
            t.object_right += failed;
            t.object_wrong += cleared;
        }
    }
    return t;
}

static void put16(uint8_t* p, uint32_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
static void put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i)); }
static void put64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i)); }
static uint32_t get16(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8); }
static uint32_t get32(const uint8_t* p) { uint32_t v = 0; for (int i = 3; i >= 0; --i) v = (v << 8) | p[i]; return v; }
static uint64_t get64(const uint8_t* p) { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = (v << 8) | p[i]; return v; }

std::vector<uint8_t> SampleSet::encode() const {
    const uint32_t rb = kRecordFixed + h_.n_features + 4u * h_.n_actions;
    std::vector<uint8_t> out(size_t(kHeaderBytes) + r_.size() * rb + kTrailerBytes, 0);
    uint8_t* p = out.data();
    std::memcpy(p, kMagic, 8);
    put16(p + 8, kVersion);
    put16(p + 10, kHeaderBytes);
    put32(p + 12, h_.round);
    p[16] = h_.pass;
    p[17] = uint8_t(h_.actor_side);
    p[18] = uint8_t(h_.world);
    p[19] = h_.n_features;
    p[20] = h_.n_actions;
    p[21] = uint8_t(h_.purpose);
    put64(p + 24, h_.seed);
    std::memcpy(p + 32, h_.actor.data(), kFingerprint);
    std::memcpy(p + 64, h_.observer.data(), kFingerprint);
    put32(p + 96, uint32_t(r_.size()));
    put32(p + 100, rb);
    uint8_t* q = p + kHeaderBytes;
    for (const Record& r : r_) {
        put32(q, r.episode);
        put16(q + 4, r.step);
        q[6] = r.actor_move;
        q[7] = uint8_t(r.grade);
        q[8] = r.verdict.choice;
        q[9] = uint8_t(r.verdict.reason);
        put16(q + 10, uint32_t(uint16_t(r.verdict.score_bp)));
        put16(q + 12, r.verdict.confidence_bp);
        q += kRecordFixed;
        std::memcpy(q, r.features.data(), h_.n_features);
        q += h_.n_features;
        for (uint32_t a = 0; a < h_.n_actions; ++a, q += 2) put16(q, r.actor_probs[a]);
        for (uint32_t a = 0; a < h_.n_actions; ++a, q += 2) put16(q, r.watcher_probs[a]);
    }
    Sha256 s;
    s.update(out.data(), out.size() - kTrailerBytes);
    s.final(out.data() + out.size() - kTrailerBytes);
    return out;
}

bool SampleSet::parse(const uint8_t* buf, size_t len, SampleSet& out, std::string& err) {
    out = SampleSet();
    if (buf == nullptr || len < size_t(kHeaderBytes) + kTrailerBytes) { err = "too short"; return false; }
    if (std::memcmp(buf, kMagic, 8) != 0) { err = "not MJHEMI2"; return false; }
    if (get16(buf + 8) != kVersion || get16(buf + 10) != kHeaderBytes) { err = "unknown version"; return false; }
    uint8_t sha[32];
    Sha256 s;
    s.update(buf, len - kTrailerBytes);
    s.final(sha);
    if (std::memcmp(sha, buf + len - kTrailerBytes, 32) != 0) { err = "the seal does not match"; return false; }
    if (buf[22] || buf[23]) { err = "reserved bytes are not zero"; return false; }
    for (size_t k = 104; k < kHeaderBytes; ++k)
        if (buf[k]) { err = "reserved bytes are not zero"; return false; }
    Header h;
    h.round = get32(buf + 12);
    h.pass = buf[16];
    h.actor_side = Side(buf[17]);
    h.world = World(buf[18]);
    h.n_features = buf[19];
    h.n_actions = buf[20];
    h.purpose = Purpose(buf[21]);
    h.seed = get64(buf + 24);
    std::memcpy(h.actor.data(), buf + 32, kFingerprint);
    std::memcpy(h.observer.data(), buf + 64, kFingerprint);
    if (!header_ok(h)) { err = "header out of range"; return false; }
    const uint32_t n = get32(buf + 96), rb = get32(buf + 100);
    if (rb != kRecordFixed + h.n_features + 4u * h.n_actions || n > kMaxRecords) { err = "bad record size or count"; return false; }
    if (uint64_t(len) != uint64_t(kHeaderBytes) + uint64_t(n) * rb + kTrailerBytes) { err = "length does not match"; return false; }
    SampleSet set(h);
    set.r_.reserve(n);
    const uint8_t* q = buf + kHeaderBytes;
    for (uint32_t i = 0; i < n; ++i) {
        Record r;
        r.episode = get32(q);
        r.step = uint16_t(get16(q + 4));
        r.actor_move = q[6];
        r.grade = Grade(q[7]);
        r.verdict.choice = q[8];
        r.verdict.reason = Reason(q[9]);
        r.verdict.score_bp = int16_t(uint16_t(get16(q + 10)));
        r.verdict.confidence_bp = uint16_t(get16(q + 12));
        q += kRecordFixed;
        std::memcpy(r.features.data(), q, h.n_features);
        q += h.n_features;
        for (uint32_t a = 0; a < h.n_actions; ++a, q += 2) r.actor_probs[a] = uint16_t(get16(q));
        for (uint32_t a = 0; a < h.n_actions; ++a, q += 2) r.watcher_probs[a] = uint16_t(get16(q));
        if (!set.add(r, err)) return false;
    }
    out = std::move(set);
    return true;
}

bool SampleSet::save(const std::string& path, std::array<uint8_t, 32>& sha, std::string& err) const {
    const std::vector<uint8_t> b = encode();
    Sha256 s;
    s.update(b.data(), b.size());
    s.final(sha.data());
    const std::string tmp = path + ".new";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) { err = "cannot write " + tmp; return false; }
    const bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
    if (std::fclose(f) != 0 || !ok) { std::remove(tmp.c_str()); err = "cannot write " + tmp; return false; }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); err = "cannot rename to " + path; return false; }
    return true;
}

std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s(2 * n, '0');
    for (size_t i = 0; i < n; ++i) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
    return s;
}

} // namespace hemispheres
} // namespace pchie
