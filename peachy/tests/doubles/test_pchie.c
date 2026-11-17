/* A stand-in PCHIE library for Peachy's tests. It implements the draft pchie_api table and decides with a
 * trivial rule (the first legal direction, north, east, south, west). It is not PCHIE and makes no claim to
 * plan well; it exists so the planner path, both direct and through Mjolnir, can be tested before pchie.dll
 * exists. Test knobs are extra exports. MIT license. */
#include <stdlib.h>
#include <string.h>

#include "peachy/planner_abi.h"

#if defined(_WIN32)
#include <windows.h>
#define TP_EXPORT __declspec(dllexport)
static void tp_sleep(unsigned ms) { Sleep(ms); }
#else
#include <time.h>
#define TP_EXPORT __attribute__((visibility("default")))
static void tp_sleep(unsigned ms) {
  struct timespec ts;
  ts.tv_sec = ms / 1000u;
  ts.tv_nsec = (long)(ms % 1000u) * 1000000L;
  nanosleep(&ts, NULL);
}
#endif

struct pchie_runtime { int magic; };
struct pchie_session { int magic; uint32_t width; uint64_t decided; };

/* Test state. Test doubles may keep it; the real library keeps none. */
static char g_seen[65536];
static size_t g_seen_len;
static unsigned g_delay_ms;
static int g_behavior; /* 0 normal, 1 always east, 2 out-of-range action, 3 decide fails */
static uint32_t g_flags_seen = 0xFFFFFFFFu;
static int g_live;

TP_EXPORT size_t peachy_test_pchie_seen(char* buf, size_t cap) {
  size_t n = g_seen_len < cap ? g_seen_len : cap;
  if (buf && n) memcpy(buf, g_seen, n);
  return g_seen_len;
}
TP_EXPORT void peachy_test_pchie_reset(void) {
  g_seen_len = 0;
  g_delay_ms = 0;
  g_behavior = 0;
  g_flags_seen = 0xFFFFFFFFu;
}
TP_EXPORT void peachy_test_pchie_set_delay_ms(unsigned ms) { g_delay_ms = ms; }
TP_EXPORT void peachy_test_pchie_set_behavior(int b) { g_behavior = b; }
TP_EXPORT uint32_t peachy_test_pchie_flags_seen(void) { return g_flags_seen; }
TP_EXPORT int peachy_test_pchie_live(void) { return g_live; }

static pchie_status PCHIE_CALL get_version(pchie_version_info* out) {
  if (!out || out->struct_size < sizeof(*out)) return PCHIE_E_INVALID_ARG;
  memset((char*)out + 4, 0, sizeof(*out) - 4);
  out->abi_major = PCHIE_ABI_MAJOR;
  out->abi_minor = PCHIE_ABI_MINOR;
  strcpy(out->build_id, "peachy test double");
  return PCHIE_OK;
}
static const char* PCHIE_CALL status_name(pchie_status s) { return s == PCHIE_OK ? "ok" : "error"; }
static pchie_status PCHIE_CALL runtime_create(const pchie_runtime_config* cfg, pchie_runtime** out) {
  if (!cfg || !out || cfg->struct_size < sizeof(*cfg)) return PCHIE_E_INVALID_ARG;
  *out = (pchie_runtime*)calloc(1, sizeof(pchie_runtime));
  if (!*out) return PCHIE_E_INVALID_ARG;
  (*out)->magic = 1;
  ++g_live;
  return PCHIE_OK;
}
static void PCHIE_CALL runtime_destroy(pchie_runtime* rt) {
  if (rt) {
    --g_live;
    free(rt);
  }
}
static pchie_status PCHIE_CALL session_create(pchie_runtime* rt, pchie_model* m, const pchie_session_config* cfg,
                                              pchie_session** out) {
  (void)m;
  if (!rt || !cfg || !out || cfg->struct_size < sizeof(*cfg)) return PCHIE_E_INVALID_ARG;
  g_flags_seen = cfg->flags;
  *out = (pchie_session*)calloc(1, sizeof(pchie_session));
  if (!*out) return PCHIE_E_INVALID_ARG;
  (*out)->magic = 2;
  (*out)->width = cfg->width;
  ++g_live;
  return PCHIE_OK;
}
static void PCHIE_CALL session_destroy(pchie_session* s) {
  if (s) {
    --g_live;
    free(s);
  }
}
static pchie_status PCHIE_CALL session_decide(pchie_session* s, const pchie_observation* o, pchie_decision* d) {
  uint32_t a = PCHIE_ACT_HOLD, k;
  if (!s || !o || !d || o->struct_size < sizeof(*o) || d->struct_size < sizeof(*d) || !o->text) return PCHIE_E_INVALID_ARG;
  if (g_seen_len + o->text_len + 1 < sizeof(g_seen)) {
    memcpy(g_seen + g_seen_len, o->text, o->text_len);
    g_seen_len += o->text_len;
    g_seen[g_seen_len++] = '\x1e';
  }
  if (g_delay_ms) tp_sleep(g_delay_ms);
  if (g_behavior == 3) return PCHIE_E_INVALID_ARG;
  for (k = 1; k <= 4; ++k)
    if (o->legal_mask & (1u << k)) {
      a = k;
      break;
    }
  if (g_behavior == 1) a = PCHIE_ACT_EAST;
  if (g_behavior == 2) a = 99;
  d->action = a;
  d->obs_tick = o->tick;
  d->seq = ++s->decided;
  d->valid_for_ticks = 1;
  return PCHIE_OK;
}
static void PCHIE_CALL cancel(pchie_session* s) { (void)s; }

static const pchie_api k_api = {
    sizeof(pchie_api),
#if defined(PEACHY_TEST_WRONG_ABI)
    PCHIE_ABI_MAJOR + 7u,
#else
    PCHIE_ABI_MAJOR,
#endif
    PCHIE_ABI_MINOR, get_version, status_name, runtime_create, runtime_destroy, NULL, NULL, NULL, session_create,
    session_destroy, NULL, NULL, session_decide, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, cancel};

TP_EXPORT pchie_status PCHIE_CALL pchie_get_api(uint32_t abi_major, const pchie_api** out) {
  if (!out) return PCHIE_E_INVALID_ARG;
  *out = NULL;
#if !defined(PEACHY_TEST_WRONG_ABI)
  if (abi_major != PCHIE_ABI_MAJOR) return PCHIE_E_ABI_MISMATCH;
#else
  (void)abi_major;
#endif
  *out = &k_api;
  return PCHIE_OK;
}
