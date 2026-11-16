#include "pchie/pchie_kv_pools.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace pchie {

KVPagePool::KVPagePool(uint32_t layers, uint32_t kv_heads, uint32_t head_dim, uint32_t max_pages)
    : layers_(layers), heads_(kv_heads), dim_(head_dim), max_pages_(max_pages) {
    page_floats_ = size_t(layers) * 2 * KV_PAGE_TOKENS * kv_heads * head_dim;
    arena_.assign(page_floats_ * max_pages, 0.0f);
    refs_.assign(max_pages, 0);
    for (int32_t i = int32_t(max_pages) - 1; i >= 0; --i) free_.push_back(i);
}

int32_t KVPagePool::alloc() {
    if (free_.empty()) throw std::runtime_error("KV pool exhausted");
    const int32_t p = free_.back(); free_.pop_back();
    refs_[size_t(p)] = 1;
    return p;
}

namespace {
// Freed pages are pushed during a loop and the free list is sorted once when the loop ends (also on a
// throw), which leaves exactly the order the old sort-after-every-push gave.
struct FreeListResort {
    std::vector<int32_t>& free_list;
    bool dirty = false;
    ~FreeListResort() { if (dirty) std::sort(free_list.begin(), free_list.end(), std::greater<int32_t>()); }
};
}  // namespace

KVPagePool::Table KVPagePool::fork(const Table& parent) {
    Table t = parent;
    for (int32_t p : t.pages) if (p >= 0) { checked_ref(p); if (refs_[size_t(p)] == UINT32_MAX) throw std::overflow_error("KV ref count overflow"); refs_[size_t(p)]++; }
    return t;
}

void KVPagePool::release(Table& t) {
    FreeListResort resort{free_};
    for (int32_t& p : t.pages) {
        if (p < 0) continue;
        checked_ref(p);
        if (--refs_[size_t(p)] == 0) {
            free_.push_back(p);
            resort.dirty = true;
        }
        p = -1;
    }
    t.pages.clear();
}

void KVPagePool::trim_before(Table& t, uint32_t first_needed) {
    FreeListResort resort{free_};
    const uint32_t last_dead = first_needed / KV_PAGE_TOKENS; // pages < last_dead are fully dead
    for (uint32_t lp = 0; lp < last_dead && lp < t.pages.size(); ++lp) {
        int32_t& p = t.pages[lp];
        if (p < 0) continue;
        checked_ref(p);
        if (--refs_[size_t(p)] == 0) {
            free_.push_back(p);
            resort.dirty = true;
        }
        p = -1;
    }
}

void KVPagePool::truncate(Table& t, uint32_t n_tokens) {
    FreeListResort resort{free_};
    const size_t keep = (size_t(n_tokens) + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS;
    for (size_t lp = keep; lp < t.pages.size(); ++lp) {
        int32_t& p = t.pages[lp];
        if (p < 0) continue;
        checked_ref(p);
        if (--refs_[size_t(p)] == 0) {
            free_.push_back(p);
            resort.dirty = true;
        }
        p = -1;
    }
    if (t.pages.size() > keep) t.pages.resize(keep);
}

void KVPagePool::ensure_private(Table& t, uint32_t pos) {
    const uint32_t lp = pos / KV_PAGE_TOKENS;
    // Logical pages grow with position (the sliding pool trims old ones), so this is a hard ceiling on the
    // page table itself: 2^20 pages = 16M tokens, far beyond any context, and refuses a wild position.
    if (lp >= (1u << 20)) throw std::out_of_range("KV position exceeds the page-table ceiling");
    if (t.pages.size() <= lp) t.pages.resize(lp + 1, -1);
    int32_t& p = t.pages[lp];
    if (p < 0) { p = alloc(); return; }
    checked_ref(p);
    if (refs_[size_t(p)] > 1) {               // copy-on-write
        const int32_t np = alloc();
        std::memcpy(&arena_[size_t(np) * page_floats_], &arena_[size_t(p) * page_floats_], page_floats_ * sizeof(float));
        try { if (on_copy) on_copy(p, np); }
        catch (...) { refs_[size_t(np)] = 0; free_.push_back(np); std::sort(free_.begin(), free_.end(), std::greater<int32_t>()); throw; }
        refs_[size_t(p)]--;
        ++cow_copies_;
        p = np;
    }
    if (refs_[size_t(p)] != 1) ++overwrite_violations_;      // a write must only ever hit a private page
}

size_t KVPagePool::offset(uint32_t layer, uint32_t slot, bool v) const noexcept {
    return ((size_t(layer) * 2 + (v ? 1 : 0)) * KV_PAGE_TOKENS + slot) * heads_ * dim_;
}

float* KVPagePool::k_write(Table& t, uint32_t layer, uint32_t pos) {
    ensure_private(t, pos);
    return &arena_[size_t(t.pages[pos / KV_PAGE_TOKENS]) * page_floats_ + offset(layer, pos % KV_PAGE_TOKENS, false)];
}
float* KVPagePool::v_write(Table& t, uint32_t layer, uint32_t pos) {
    ensure_private(t, pos);
    return &arena_[size_t(t.pages[pos / KV_PAGE_TOKENS]) * page_floats_ + offset(layer, pos % KV_PAGE_TOKENS, true)];
}
const float* KVPagePool::k_read(const Table& t, uint32_t layer, uint32_t pos) const {
    const int32_t p = checked_page(t, layer, pos);
    return arena_.data() + size_t(p) * page_floats_ + offset(layer, pos % KV_PAGE_TOKENS, false);
}
const float* KVPagePool::v_read(const Table& t, uint32_t layer, uint32_t pos) const {
    const int32_t p = checked_page(t, layer, pos);
    return arena_.data() + size_t(p) * page_floats_ + offset(layer, pos % KV_PAGE_TOKENS, true);
}

void KVPagePool::checked_ref(int32_t p) const {
    if (p < 0 || size_t(p) >= refs_.size() || refs_[size_t(p)] == 0) throw std::out_of_range("KV page is outside the pool or has no live reference");
}
int32_t KVPagePool::checked_page(const Table& t, uint32_t layer, uint32_t pos) const {
    if (layer >= layers_ || pos / KV_PAGE_TOKENS >= t.pages.size()) throw std::out_of_range("KV read outside the layer or page table");
    const int32_t p = t.pages[pos / KV_PAGE_TOKENS];
    checked_ref(p);
    return p;
}

uint64_t KVPagePool::page_checksum(int32_t p) const noexcept {
    uint64_t h = 14695981039346656037ULL;
    const uint32_t* u = reinterpret_cast<const uint32_t*>(&arena_[size_t(p) * page_floats_]);
    for (size_t i = 0; i < page_floats_; ++i) { h ^= u[i]; h *= 1099511628211ULL; }
    return h;
}

uint32_t KVPagePool::table_live(const Table& t) const noexcept {
    uint32_t n = 0; for (int32_t p : t.pages) if (p >= 0) ++n; return n;
}

} // namespace pchie
