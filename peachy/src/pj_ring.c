/* Peachy SPSC message ring. MIT license. */
#include "pj_ring.h"

#include <string.h>

#include "pj_base.h"

#define PAD_MARK 0xFFFFFFFFu

static uint64_t rec_size(size_t len) { return 8u + (((uint64_t)len + 7u) & ~(uint64_t)7u); }

void pj_ring_init(pj_ring* r, uint8_t* buf, uint64_t cap) {
  memset(r, 0, sizeof(*r));
  r->buf = buf;
  r->cap = cap;
}

uint64_t pj_ring_max_message(const pj_ring* r) { return r->cap / 2u - 16u; }

int pj_ring_push(pj_ring* r, const void* data, size_t len) {
  uint64_t h = r->head; /* only this thread writes head */
  uint64_t t = pj_load_u64(&r->tail);
  uint64_t need = rec_size(len), pos = h & (r->cap - 1u), contig = r->cap - pos, total = need;
  uint32_t l32;
  if (len > pj_ring_max_message(r)) return -1;
  if (contig < need) total = contig + need;
  if (r->cap - (h - t) < total) return 1;
  if (contig < need) {
    l32 = PAD_MARK;
    memcpy(r->buf + pos, &l32, 4);
    h += contig;
    pos = 0;
  }
  l32 = (uint32_t)len;
  memcpy(r->buf + pos, &l32, 4);
  if (len) memcpy(r->buf + pos + 8, data, len);
  pj_store_u64(&r->head, h + need);
  return 0;
}

int pj_ring_pop(pj_ring* r, void* out, size_t cap, size_t* len) {
  uint64_t t = r->tail; /* only this thread writes tail */
  for (;;) {
    uint64_t h = pj_load_u64(&r->head), pos;
    uint32_t l32;
    if (t == h) {
      *len = 0;
      return 1;
    }
    pos = t & (r->cap - 1u);
    memcpy(&l32, r->buf + pos, 4);
    if (l32 == PAD_MARK) {
      t += r->cap - pos;
      pj_store_u64(&r->tail, t);
      continue;
    }
    if (l32 > cap) {
      *len = l32;
      return -1;
    }
    if (l32) memcpy(out, r->buf + pos + 8, l32);
    *len = l32;
    pj_store_u64(&r->tail, t + rec_size(l32));
    return 0;
  }
}
