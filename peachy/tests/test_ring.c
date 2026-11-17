/* SPSC ring: edges in one thread, then a two-thread stress run (run it under ThreadSanitizer too). MIT license. */
#include "pt.h"
#include "pj_base.h"
#include "pj_platform.h"
#include "pj_ring.h"

static void single_thread(void) {
  static uint8_t buf[1024];
  pj_ring r;
  char out[600];
  size_t n;
  int k, ok = 1;
  pj_ring_init(&r, buf, sizeof(buf));
  CHECK(pj_ring_pop(&r, out, sizeof(out), &n) == 1 && n == 0, "empty pop");
  CHECK(pj_ring_push(&r, out, (size_t)pj_ring_max_message(&r) + 1) == -1, "too big refused");
  CHECK(pj_ring_push(&r, "hello", 5) == 0, "push");
  CHECK(pj_ring_pop(&r, out, 3, &n) == -1 && n == 5, "small out buffer refused, message kept");
  CHECK(pj_ring_pop(&r, out, sizeof(out), &n) == 0 && n == 5 && memcmp(out, "hello", 5) == 0, "pop");
  /* fill until full */
  for (k = 0; k < 100; ++k)
    if (pj_ring_push(&r, "0123456789abcdef", 16) != 0) break;
  CHECK(k > 10 && k < 100, "ring fills and then says full");
  while (pj_ring_pop(&r, out, sizeof(out), &n) == 0) {
  }
  /* many wraps with sizes that force pad markers */
  for (k = 0; k < 5000; ++k) {
    char msg[300];
    size_t len = (size_t)((k * 37) % 290) + 1, i;
    for (i = 0; i < len; ++i) msg[i] = (char)(k + (int)i);
    if (pj_ring_push(&r, msg, len) != 0) {
      ok = 0;
      break;
    }
    if (pj_ring_pop(&r, out, sizeof(out), &n) != 0 || n != len || memcmp(out, msg, len) != 0) {
      ok = 0;
      break;
    }
  }
  CHECK(ok, "5000 wrapping round trips");
}

typedef struct stress {
  pj_ring r;
  uint64_t count;
  volatile uint32_t producer_done;
  uint64_t produced_sum;
} stress;

static uint8_t g_buf[1 << 16];

static void producer(void* arg) {
  stress* s = (stress*)arg;
  uint64_t k;
  uint64_t sum = 0;
  for (k = 0; k < s->count;) {
    char msg[200];
    size_t len = (size_t)(k % 190) + 9, i;
    memcpy(msg, &k, 8);
    for (i = 8; i < len; ++i) msg[i] = (char)(k * 31 + i);
    if (pj_ring_push(&s->r, msg, len) == 0) {
      sum += len;
      ++k;
    }
  }
  s->produced_sum = sum;
  pj_store_u32(&s->producer_done, 1);
}

static void stress_run(void) {
  stress s;
  pj_thread th;
  uint64_t expect = 0, sum = 0;
  int ok = 1;
  memset(&s, 0, sizeof(s));
  pj_ring_init(&s.r, g_buf, sizeof(g_buf));
  s.count = 300000;
  CHECK(pj_thread_start(&th, producer, &s) == 0, "producer thread");
  while (expect < s.count) {
    char out[256];
    size_t n, i;
    int r = pj_ring_pop(&s.r, out, sizeof(out), &n);
    uint64_t k;
    if (r == 1) continue;
    if (r != 0 || n < 9) {
      ok = 0;
      break;
    }
    memcpy(&k, out, 8);
    if (k != expect || n != (size_t)(k % 190) + 9) {
      ok = 0;
      break;
    }
    for (i = 8; i < n; ++i)
      if (out[i] != (char)(k * 31 + i)) ok = 0;
    sum += n;
    ++expect;
  }
  pj_thread_join(&th);
  CHECK(ok && expect == s.count, "300000 messages in order, intact");
  CHECK(pj_load_u32(&s.producer_done) && sum == s.produced_sum, "byte totals agree");
}

int main(int argc, char** argv) {
  pt_init(argc, argv);
  single_thread();
  stress_run();
  return pt_done("peachy_ring");
}
