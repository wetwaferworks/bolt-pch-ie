/* Peachy single-producer, single-consumer message ring. Atomics only, no locks. MIT license.
 * One thread pushes, one other thread pops. Head and tail are byte counters published with release stores and
 * read with acquire loads. Records are 8-byte aligned: a 4-byte length, the payload, padding. A record that
 * would wrap is preceded by a pad marker that sends the reader back to the start. */
#ifndef PJ_RING_H
#define PJ_RING_H
#include <stddef.h>
#include <stdint.h>

typedef struct pj_ring {
  uint8_t* buf;
  uint64_t cap; /* power of two */
  volatile uint64_t head; /* written by the producer only */
  uint8_t pad1[56];
  volatile uint64_t tail; /* written by the consumer only */
  uint8_t pad2[56];
} pj_ring;

void pj_ring_init(pj_ring* r, uint8_t* buf, uint64_t cap);
uint64_t pj_ring_max_message(const pj_ring* r);
int pj_ring_push(pj_ring* r, const void* data, size_t len);                 /* 0 ok, 1 full, -1 too big */
int pj_ring_pop(pj_ring* r, void* out, size_t cap, size_t* len);            /* 0 ok, 1 empty, -1 out too small */
#endif
