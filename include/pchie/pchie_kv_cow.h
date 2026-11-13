// Bolt PCHIE - copy-on-write paged KV cache (clean-room, after the paged-attention idea).
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace pchie {

struct PhysicalKVBlock {
    uint32_t block_id  = 0;
    uint32_t ref_count = 0;
    uint8_t* data_ptr  = nullptr;
};

class PagedKVCache {
public:
    PagedKVCache(uint32_t layers, uint32_t kv_heads, uint32_t head_dim, uint32_t max_blocks);

    // Share a parent's pages with a new node (ref counts +1, no tensor copy).
    std::vector<uint32_t> clone_block_table(const std::vector<uint32_t>& parent_table);
    // Ensure the page for seq_pos is privately owned (CoW if shared); returns write pointer.
    uint8_t* prepare_write_slot(std::vector<uint32_t>& block_table, uint32_t seq_pos);
    // Drop one reference on every page; pages reaching 0 go back to the free list,
    // which stays sorted so the next allocation is always the lowest free id.
    void release_block_table(std::vector<uint32_t>& block_table);

    uint32_t free_blocks() const noexcept { return static_cast<uint32_t>(free_list_.size()); }
    uint32_t ref_count(uint32_t blk) const noexcept { return blocks_[blk].ref_count; }
    size_t   bytes_per_token() const noexcept { return bytes_per_token_; }

private:
    uint32_t alloc_block();
    size_t bytes_per_token_ = 0;
    size_t bytes_per_block_ = 0;
    std::vector<uint8_t>         arena_;
    std::vector<PhysicalKVBlock> blocks_;
    std::vector<uint32_t>        free_list_; // sorted DESCENDING; back() is the lowest id
};

} // namespace pchie
