// Bolt PCHIE - TEST STAND-IN gate for pre-hash-chain baselines (NOT the PchGate safety / PCH gate).
//
// LABEL (required): StrictStandInGate is a TEST STAND-IN for pre-hash-chain hash separation
// and reject_count plumbing only. AcceptAllGate proves determinism, not safety.
// The real PCH / safety gate is PchGate. Do not treat this class as a safety control.
//
// Hot-path shape matches AcceptAllGate: concrete final, Pass(const vector<int32_t>&).
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include "pchie/pchie_tree.h"  // hash_mix64 / FNV_OFFSET (reject_set_hash)

namespace pchie {

// Committed reject set A (control) + B (content markers) from the pre-hash-chain proposal.
inline std::vector<int32_t> DefaultStrictStandInRejectIds() {
    return {0, 2, 3, 4, 46, 48, 50, 100, 105, 818, 63411, 63937};
}

class StrictStandInGate final {
public:
    // Sorted unique; negatives dropped (same shape as TurnBoundaryTerminalPolicy).
    explicit StrictStandInGate(std::vector<int32_t> reject_ids)
        : reject_count_(0), last_reject_token_(-1), reject_set_hash_(0) {
        for (int32_t t : reject_ids) if (t >= 0) reject_ids_.push_back(t);
        std::sort(reject_ids_.begin(), reject_ids_.end());
        reject_ids_.erase(std::unique(reject_ids_.begin(), reject_ids_.end()), reject_ids_.end());
        // reject_set_hash: FNV-1a over sorted IDs only (never RejectCount).
        // Caveat: uint64 FNV stand-in for manifest reject_set_hash; full SHA-256 hex
        // can replace/supplement this when harness/manifest wiring lands.
        uint64_t h = FNV_OFFSET;
        for (int32_t id : reject_ids_) h = hash_mix64(h, static_cast<uint64_t>(static_cast<uint32_t>(id)));
        // Ensure nonempty sets never hash to 0 (FNV_OFFSET alone is nonzero; mix keeps it so in practice).
        reject_set_hash_ = h ? h : 1ull;
    }

    // Return false iff any path token is in the reject set (first hit wins for LastRejectToken).
    // Empty path -> true (matches root Pass({})).
    bool Pass(const std::vector<int32_t>& path) const noexcept {
        for (int32_t t : path) {
            if (std::binary_search(reject_ids_.begin(), reject_ids_.end(), t)) {
                reject_count_.fetch_add(1, std::memory_order_relaxed);
                last_reject_token_.store(t, std::memory_order_relaxed);
                return false;
            }
        }
        return true;
    }

    // Diagnostic only — must NOT feed PuctSearch::digest, tree/logit digests, or run_hash.
    uint64_t RejectCount() const noexcept {
        return reject_count_.load(std::memory_order_relaxed);
    }

    // First offending token id from the most recent rejecting Pass(), or -1.
    int32_t LastRejectToken() const noexcept {
        return last_reject_token_.load(std::memory_order_relaxed);
    }

    const std::vector<int32_t>& RejectIds() const noexcept { return reject_ids_; }

    // Hash of the sorted reject ID list only (never RejectCount). Manifest field reject_set_hash.
    uint64_t RejectSetHash() const noexcept { return reject_set_hash_; }

    // Empty reject set = bypass of the strict stand-in. Harness must FAIL the run
    // (case name: strict-stand-in-bypass) when this is true for a "strict" slot.
    bool IsBypass() const noexcept { return reject_ids_.empty(); }

private:
    std::vector<int32_t> reject_ids_;
    mutable std::atomic<uint64_t> reject_count_;
    mutable std::atomic<int32_t> last_reject_token_;
    uint64_t reject_set_hash_;
};

} // namespace pchie