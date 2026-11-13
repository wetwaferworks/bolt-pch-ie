#include "pchie/pchie_kv_cow.h"
#include "pchie/pchie_tree.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace pchie {

PagedKVCache::PagedKVCache(uint32_t layers, uint32_t kv_heads, uint32_t head_dim, uint32_t max_blocks) {
    bytes_per_token_ = static_cast<size_t>(layers) * 2u * kv_heads * head_dim * sizeof(uint16_t);
    bytes_per_block_ = bytes_per_token_ * KV_BLOCK_SIZE;
    arena_.assign(bytes_per_block_ * max_blocks, 0);
    blocks_.resize(max_blocks);
    free_list_.reserve(max_blocks);
    for (uint32_t i = 0; i < max_blocks; ++i) {
        blocks_[i].block_id = i;
        blocks_[i].data_ptr = arena_.data() + static_cast<size_t>(i) * bytes_per_block_;
        free_list_.push_back(max_blocks - 1 - i);
    }
}

std::vector<uint32_t> PagedKVCache::clone_block_table(const std::vector<uint32_t>& t) {
    for (uint32_t b : t) blocks_[b].ref_count++;
    return t;
}

uint8_t* PagedKVCache::prepare_write_slot(std::vector<uint32_t>& table, uint32_t seq_pos) {
    const uint32_t lb = seq_pos / KV_BLOCK_SIZE;
    const uint32_t off = seq_pos % KV_BLOCK_SIZE;
    if (lb > table.size()) throw std::logic_error("KV write skips a logical block");
    if (lb == table.size()) {
        table.push_back(alloc_block());
    } else {
        const uint32_t pb = table[lb];
        if (blocks_[pb].ref_count > 1) {
            const uint32_t cow = alloc_block();
            std::memcpy(blocks_[cow].data_ptr, blocks_[pb].data_ptr, off * bytes_per_token_);
            blocks_[pb].ref_count--;
            table[lb] = cow;
        }
    }
    return blocks_[table[lb]].data_ptr + off * bytes_per_token_;
}

void PagedKVCache::release_block_table(std::vector<uint32_t>& table) {
    for (uint32_t b : table) {
        if (blocks_[b].ref_count == 0) throw std::logic_error("KV double release");
        if (--blocks_[b].ref_count == 0) {
            auto it = std::lower_bound(free_list_.begin(), free_list_.end(), b, std::greater<uint32_t>());
            free_list_.insert(it, b);
        }
    }
    table.clear();
}

uint32_t PagedKVCache::alloc_block() {
    if (free_list_.empty()) throw std::runtime_error("PagedKVCache out of physical blocks");
    const uint32_t id = free_list_.back();
    free_list_.pop_back();
    blocks_[id].ref_count = 1;
    return id;
}

} // namespace pchie
