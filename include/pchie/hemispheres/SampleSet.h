// Bolt PCHIE - the left and right hemispheres: verdicts and MJHEMI2 sample sets.
//
// The same rules and the same bytes as Mjolnir's mj_hemi (include/mjolnir/mj_hemi.h
// there; the format is in Mjolnir's docs/FORMATS.md). PCHIE writes the sets while a
// real model plays a world; Mjolnir reads them, decides what is kept and journals it.
// Integer verdicts, so both sides agree to the bit. Every parse is bounds-checked.
// The idea is Jaydee Wetwork's. MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace hemispheres {

constexpr uint32_t kMaxActions = 16;
constexpr uint32_t kMaxFeatures = 64;
constexpr uint32_t kMaxRecords = 1u << 20;
constexpr uint32_t kHeaderBytes = 128;
constexpr uint32_t kTrailerBytes = 32;   // SHA-256 of everything before it
constexpr uint32_t kRecordFixed = 14;
constexpr uint32_t kProbOne = 65535;     // probabilities are stored as p * 65535
constexpr uint32_t kFingerprint = 32;    // the SHA-256 of a model (or of its working copy)
constexpr uint32_t kUnsureBp = 1000;     // below this margin a disagreement is "unsure"

enum class Side : uint8_t { Left = 0, Right = 1 };
enum class World : uint8_t { ToyMaze = 1, GridMaze = 2, Circuit = 3 };
enum class Reason : uint8_t { Agree = 1, PreferOther = 2, Unsure = 3, Illegal = 4 };
enum class Grade : uint8_t { Cleared = 1, NotYet = 2, Ungraded = 3 };
// A lesson is learned from; a holdout set is only scored, on states never in a lesson.
enum class Purpose : uint8_t { Lesson = 1, Holdout = 2 };

struct Verdict {
    uint8_t choice = 0;            // the watcher's own move
    Reason reason = Reason::Agree;
    int16_t score_bp = 0;          // p(actor's move) - p(best other move)
    uint16_t confidence_bp = 0;    // p(top) - p(second)
};

struct Header {
    uint32_t round = 0;
    uint8_t pass = 1;              // 1 or 2 (the classroom's lesson is pass 1)
    Side actor_side = Side::Left;
    World world = World::GridMaze;
    uint8_t n_features = 0;        // 1..64
    uint8_t n_actions = 0;         // 2..16
    Purpose purpose = Purpose::Lesson;
    uint64_t seed = 0;
    std::array<uint8_t, kFingerprint> actor{}, observer{};
};

struct Record {
    uint32_t episode = 0;
    uint16_t step = 0;
    uint8_t actor_move = 0;
    Grade grade = Grade::Ungraded;
    Verdict verdict;
    std::array<int8_t, kMaxFeatures> features{};
    std::array<uint16_t, kMaxActions> actor_probs{}, watcher_probs{};
};

struct Tally {
    uint32_t steps = 0, cleared = 0, not_yet = 0;
    uint32_t agree_right = 0, agree_wrong = 0, object_right = 0, object_wrong = 0, unsure = 0;
    uint32_t judged() const { return agree_right + agree_wrong + object_right + object_wrong; }
    uint32_t judged_right() const { return agree_right + object_right; }
    // judgments the answer key confirms, basis points (Mjolnir's mj_classroom_judge_bp)
    uint32_t judge_bp() const { return judged() ? uint32_t(uint64_t(judged_right()) * 10000u / judged()) : 0u; }
};

// The legal move with the highest probability (ties to the lower move); kMaxActions if none.
uint32_t pick(const uint16_t* probs, uint32_t n_actions, uint32_t legal);
// The watcher's verdict on the actor's move (Mjolnir's mj_hemi_judge, bit for bit).
Verdict judge(const uint16_t* watcher_probs, uint32_t n_actions, uint32_t legal, uint32_t actor_move);
// Softmax over the legal moves, stored as p * 65535; the rounding remainder goes to the
// most likely move so the legal entries sum to exactly 65535. Illegal moves get 0.
// Refuses (returns false) if nothing is legal or a legal logit is not finite.
bool to_q16(const float* logits, uint32_t n_actions, uint32_t legal, uint16_t* out);

class SampleSet {
public:
    explicit SampleSet(const Header& h = Header{}) : h_(h) {}
    const Header& header() const { return h_; }
    const std::vector<Record>& records() const { return r_; }
    // Checks the record against the header and the order (episode, step strictly increasing).
    bool add(const Record& r, std::string& err);
    Tally tally() const;
    // MJHEMI1 bytes, sealed with SHA-256.
    std::vector<uint8_t> encode() const;
    // Refuses anything Mjolnir's reader refuses.
    static bool parse(const uint8_t* data, size_t len, SampleSet& out, std::string& err);
    // Write to path (a new file next to it, then renamed over it); sha gets the file's SHA-256.
    bool save(const std::string& path, std::array<uint8_t, 32>& sha, std::string& err) const;
    static bool header_ok(const Header& h);
    static bool record_ok(const Header& h, const Record& r);
private:
    Header h_;
    std::vector<Record> r_;
};

std::string hex(const uint8_t* p, size_t n);

} // namespace hemispheres
} // namespace pchie
