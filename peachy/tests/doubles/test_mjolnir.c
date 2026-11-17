/* A stand-in Mjolnir library for Peachy's tests: it loads the PCHIE library from the path it is given and hands
 * the PCHIE table to Peachy, the way the real Mjolnir library is meant to once it exists. MIT license. */
#include <string.h>

#include "peachy/planner_abi.h"
#include "pj_platform.h"

#if defined(_WIN32)
#define TM_EXPORT __declspec(dllexport)
#else
#define TM_EXPORT __attribute__((visibility("default")))
#endif

static void* g_lib;
static const pchie_api* g_api;

static int32_t PCHIE_CALL attach_pchie(const char* path, size_t len) {
  char p[1024];
  pchie_get_api_fn get;
  if (g_lib || !path || len == 0 || len >= sizeof(p)) return -1;
  memcpy(p, path, len);
  p[len] = 0;
  g_lib = pj_dl_open(p);
  if (!g_lib) return -1;
  get = (pchie_get_api_fn)pj_dl_fn(g_lib, "pchie_get_api");
  if (!get || get(PCHIE_ABI_MAJOR, &g_api) != PCHIE_OK || !g_api) {
    pj_dl_close(g_lib);
    g_lib = NULL;
    g_api = NULL;
    return -1;
  }
  return 0;
}
static int32_t PCHIE_CALL get_pchie_api(const pchie_api** out) {
  if (!out || !g_api) return -1;
  *out = g_api;
  return 0;
}
static void PCHIE_CALL detach_pchie(void) {
  g_api = NULL;
  if (g_lib) pj_dl_close(g_lib);
  g_lib = NULL;
}

static const mjolnir_api k_api = {sizeof(mjolnir_api), MJOLNIR_ABI_MAJOR, MJOLNIR_ABI_MINOR, attach_pchie, get_pchie_api,
                                  detach_pchie};

TM_EXPORT int32_t PCHIE_CALL mjolnir_get_api(uint32_t abi_major, const mjolnir_api** out) {
  if (!out) return -1;
  *out = NULL;
  if (abi_major != MJOLNIR_ABI_MAJOR) return -1;
  *out = &k_api;
  return 0;
}
