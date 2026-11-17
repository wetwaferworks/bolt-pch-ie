/* The shared library across its boundary: one export, version checks, struct sizes, 1000 load/unload cycles
 * with sessions inside, and memory that comes back. MIT license. */
#include "pt.h"
#include "pj_platform.h"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
static uint64_t rss_bytes(void) {
  PROCESS_MEMORY_COUNTERS pmc;
  memset(&pmc, 0, sizeof(pmc));
  if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
  return (uint64_t)pmc.PagefileUsage;
}
#else
static uint64_t rss_bytes(void) {
  FILE* f = fopen("/proc/self/statm", "r");
  unsigned long size = 0, res = 0;
  if (!f) return 0;
  if (fscanf(f, "%lu %lu", &size, &res) != 2) res = 0;
  fclose(f);
  return (uint64_t)res * 4096u;
}
#endif

static char g_dir[600], g_allow[700];
static const char* T = "grok-test-token-0123456789abcdef";

static int32_t PEACHY_CALL sink(void* ctx, const char* m, size_t n) {
  (void)m;
  (void)n;
  ++*(int*)ctx;
  return 0;
}

static int session_round(const peachy_api* A, int own_thread) {
  static int counter;
  char name[33];
  peachy_session_config c;
  PeachySession* s = NULL;
  char err[128];
  int outs = 0, ok;
  const char* ping = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n";
  memset(&c, 0, sizeof(c));
  c.struct_size = sizeof(c);
  c.write = sink;
  c.write_ctx = &outs;
  c.allowlist_path = g_allow;
  c.audit_dir = g_dir;
  snprintf(name, sizeof(name), "dll-%d", ++counter);
  c.session_name = name;
  c.client_name = "grok";
  c.client_token = (const uint8_t*)T;
  c.client_token_len = strlen(T);
  if (A->session_open(&c, &s, err, sizeof(err)) != PEACHY_OK) return 0;
  if (!own_thread) {
    A->feed(s, ping, strlen(ping), NULL);
    A->pump(s, 1000000000ull, NULL);
    ok = outs == 1;
  } else {
    char out[512];
    size_t n = 0;
    uint64_t t_end = pj_mono_ns() + 10000000000ull;
    A->thread_start(s);
    A->post_input(s, ping, strlen(ping));
    while (n == 0 && pj_mono_ns() < t_end) A->take_output(s, out, sizeof(out), &n);
    ok = n > 0;
    /* close with the thread still running: close stops it */
  }
  A->session_close(s);
  return ok;
}

int main(int argc, char** argv) {
  void* lib;
  peachy_get_api_fn get;
  const peachy_api* A = NULL;
  peachy_version_info vi;
  int k, ok = 1, sessions_ok = 1;
  uint64_t rss0 = 0, rss1;
  char h[65], doc[512];
  int n;
  pt_init(argc, argv);
  if (argc < 3) return 2;
  pt_newdir(g_dir, sizeof(g_dir), "dll");
  pt_token_sha_hex(T, h);
  n = snprintf(doc, sizeof(doc), "{\"version\":1,\"clients\":[{\"name\":\"grok\",\"transports\":[\"stdio\"],\"token_sha256\":\"%s\"}]}", h);
  snprintf(g_allow, sizeof(g_allow), "%s/allow.json", g_dir);
  pt_write_file(g_allow, doc, (size_t)n);

  lib = pj_dl_open(argv[2]);
  CHECK(lib != NULL, "library loads");
  if (!lib) return pt_done("peachy_dll");
  get = (peachy_get_api_fn)pj_dl_fn(lib, "peachy_get_api");
  CHECK(get != NULL, "peachy_get_api resolves");
  CHECK(pj_dl_fn(lib, "pj_json_parse") == NULL && pj_dl_fn(lib, "pj_test_fail_audit_after") == NULL &&
            pj_dl_fn(lib, "pj_sha256_once") == NULL,
        "internal symbols are not exported");
  if (!get) return pt_done("peachy_dll");
  CHECK(get(PEACHY_ABI_MAJOR + 1, &A) == PEACHY_E_ABI_MISMATCH && A == NULL, "other ABI major refused");
  CHECK(get(PEACHY_ABI_MAJOR, &A) == PEACHY_OK && A && A->struct_size == sizeof(peachy_api) &&
            A->abi_major == PEACHY_ABI_MAJOR,
        "API table");
  memset(&vi, 0, sizeof(vi));
  vi.struct_size = sizeof(vi) - 4;
  CHECK(A->get_version(&vi) == PEACHY_E_INVALID_ARG, "short version struct refused");
  vi.struct_size = 70000;
  CHECK(A->get_version(&vi) == PEACHY_E_INVALID_ARG, "huge struct_size refused");
  vi.struct_size = sizeof(vi);
  CHECK(A->get_version(&vi) == PEACHY_OK && vi.alpha == 1 && strstr(vi.build_id, "peachy") != NULL, "version info");
  printf("library %s, abi %u.%u, checking build %u\n", vi.build_id, vi.abi_major, vi.abi_minor, vi.checking_build);
  CHECK(session_round(A, 0), "pump session through the boundary");
  CHECK(session_round(A, 1), "own-thread session through the boundary, closed while running");
  pj_dl_close(lib);

  for (k = 0; k < 1000; ++k) {
    lib = pj_dl_open(argv[2]);
    if (!lib) {
      ok = 0;
      break;
    }
    get = (peachy_get_api_fn)pj_dl_fn(lib, "peachy_get_api");
    if (!get || get(PEACHY_ABI_MAJOR, &A) != PEACHY_OK) {
      ok = 0;
      pj_dl_close(lib);
      break;
    }
    if (k % 10 == 0 && !session_round(A, (k / 10) % 2)) sessions_ok = 0;
    pj_dl_close(lib);
    if (k == 99) rss0 = rss_bytes();
  }
  rss1 = rss_bytes();
  CHECK(ok, "1000 load/unload cycles");
  CHECK(sessions_ok, "100 sessions across the cycles");
  printf("memory after cycle 100: %llu KiB, after 1000: %llu KiB\n", (unsigned long long)(rss0 / 1024),
         (unsigned long long)(rss1 / 1024));
  /* Sanitizer builds hold freed memory in quarantine on purpose, so the size check only means something in a
   * normal build; leaks in checking builds are LeakSanitizer's job. */
  if (vi.checking_build) printf("memory growth check skipped in a checking build\n");
  else CHECK(rss0 == 0 || rss1 < rss0 + (16u << 20), "memory does not grow across cycles (under 16 MiB)");
  return pt_done("peachy_dll");
}
