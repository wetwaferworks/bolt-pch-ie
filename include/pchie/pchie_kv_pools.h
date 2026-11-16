// Bolt PCHIE - split copy-on-write paged KV pools for gemma4 (clean-room).
//
// Two pools, per the file's metadata (shared_kv_layers = 18, sliding_window = 512):
//   sliding pool: the 20 owned sliding layers, H_kv = 2, d_h = 256, ring-capped per branch
//   global pool : the 4 owned global layers,  H_kv = 2, d_h = 512, unbounded
// A page holds 16 tokens of K and V (fp32) for every layer in its pool, so one page
// table per branch per pool is enough. Forking shares pages (ref count +1); the first
// write into a shared page copies it (CoW). The sliding pool keeps
// ceil(W/16)+1 = 33 logical pages (a 512-token window that does not start on a page
// boundary touches 33 pages); older pages are released as the window slides.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace pchie {

constexpr uint32_t KV_PAGE_TOKENS = 16;

class KVPagePool {
public:
    KVPagePool(uint32_t layers, uint32_t kv_heads, uint32_t head_dim, uint32_t max_pages);
    struct Table {                          // one branch's view of the pool
        std::vector<int32_t> pages;         // logical page -> physical page (-1 = released)
    };
    Table fork(const Table& parent);        // share every live page
    void  release(Table& t);
    // Pointers for writing token `pos` of `layer` (CoW on first write to a shared page).
    float* k_write(Table& t, uint32_t layer, uint32_t pos);
    float* v_write(Table& t, uint32_t layer, uint32_t pos);
    const float* k_read(const Table& t, uint32_t layer, uint32_t pos) const;
    const float* v_read(const Table& t, uint32_t layer, uint32_t pos) const;
    // Drop pages that hold only positions < first_needed (sliding window ring cap).
    void trim_before(Table& t, uint32_t first_needed);
    // Keep only the pages holding positions < n_tokens (step-back / fork at an earlier node).
    void truncate(Table& t, uint32_t n_tokens);
    uint32_t live_pages() const noexcept { return max_pages_ - static_cast<uint32_t>(free_.size()); }
    uint32_t ref_count(int32_t p) const noexcept { return refs_[size_t(p)]; }
    size_t   page_floats() const noexcept { return page_floats_; }
    uint32_t max_pages() const noexcept { return max_pages_; }
    // Search audit helpers (never part of any run hash: physical page ids stay out of hashes).
    uint64_t page_checksum(int32_t p) const noexcept;          // FNV over the page's float bits
    uint32_t table_live(const Table& t) const noexcept;        // live (non-released) logical pages
    uint64_t cow_copies() const noexcept { return cow_copies_; }
    uint64_t overwrite_violations() const noexcept { return overwrite_violations_; }
    // Float offset of (layer, token slot inside its page, K or V) within one page.
    size_t   elem_offset(uint32_t layer, uint32_t pos, bool v) const noexcept { return offset(layer, pos % KV_PAGE_TOKENS, v); }
    int32_t  page_of(const Table& t, uint32_t pos) const { return t.pages[pos / KV_PAGE_TOKENS]; }
    // Called after a CoW copy (src page -> dst page) so a mirror arena (GPU) can copy too.
    std::function<void(int32_t, int32_t)> on_copy;
private:
    int32_t alloc();
    void    ensure_private(Table& t, uint32_t pos);
    size_t  offset(uint32_t layer, uint32_t slot, bool v) const noexcept;
    int32_t checked_page(const Table& t, uint32_t layer, uint32_t pos) const;
    void checked_ref(int32_t p) const;
    uint32_t layers_, heads_, dim_, max_pages_;
    size_t page_floats_;
    std::vector<float>    arena_;
    std::vector<uint32_t> refs_;
    std::vector<int32_t>  free_;            // descending; back() is the lowest id
    uint64_t cow_copies_ = 0, overwrite_violations_ = 0;
};

} // namespace pchie
