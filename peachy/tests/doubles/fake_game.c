/* A fake game MCP server for Peachy's tests: stdio JSON-RPC, a maze file for a world, and switches for every way
 * a real game link can go wrong (permission errors, rate limits, crashes, garbage, huge lines, slowness, hangs).
 * World: x = column, z = row, floor at y = 63, feet at 64, head at 65. Outside the maze everything is stone.
 * Maze: '#' wall, '.' floor, '~' no floor, 'S' start (floor). MIT license. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pj_json.h"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
static void fg_sleep(unsigned ms) { Sleep(ms); }
#else
#include <time.h>
static void fg_sleep(unsigned ms) {
  struct timespec ts;
  ts.tv_sec = ms / 1000u;
  ts.tv_nsec = (long)(ms % 1000u) * 1000000L;
  nanosleep(&ts, NULL);
}
#endif

static char maze[64][65];
static int rows, cols, sx, sz;
static long deny_first, rate_every, crash_after, garbage_after, long_after, hang_after, error_after;
static unsigned slow_ms;
static int boot_lines, notify, no_init_reply, bad_state;
static const char* sign_text;
static const char* stats_path;
static long calls;

static int block_at(long x, long y, long z) {
  char c;
  if (x < 0 || z < 0 || z >= rows || x >= cols) return 1;
  c = maze[z][x];
  if (c == 0 || c == '\n' || c == '\r') return 1;
  if (y < 63) return 1;
  if (y > 65) return 0;
  if (c == '#') return 1;
  if (y == 63) return c == '~' ? 0 : 2;
  return 0;
}

static void put_escaped(char* out, size_t cap, size_t* n, const char* s) {
  for (; *s && *n + 8 < cap; ++s) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') {
      out[(*n)++] = '\\';
      out[(*n)++] = (char)c;
    } else if (c < 0x20) {
      *n += (size_t)snprintf(out + *n, cap - *n, "\\u%04x", c);
    } else {
      out[(*n)++] = (char)c;
    }
  }
  out[*n] = 0;
}

static void reply_text(long long id, const char* payload, int is_error) {
  static char buf[1 << 16];
  size_t n = 0;
  n += (size_t)snprintf(buf, sizeof(buf), "{\"jsonrpc\":\"2.0\",\"id\":%lld,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"", id);
  put_escaped(buf, sizeof(buf), &n, payload);
  n += (size_t)snprintf(buf + n, sizeof(buf) - n, "\"}],\"isError\":%s}}\n", is_error ? "true" : "false");
  if (notify) fputs("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\"}}\n", stdout);
  fwrite(buf, 1, n, stdout);
  fflush(stdout);
}

static void stat_line(const char* tool) {
  FILE* f;
  if (!stats_path) return;
  f = fopen(stats_path, "ab");
  if (!f) return;
  fprintf(f, "%s\n", tool);
  fclose(f);
}

static void load_maze(const char* path) {
  FILE* f = fopen(path, "rb");
  char line[256];
  if (!f) exit(2);
  while (rows < 64 && fgets(line, sizeof(line), f)) {
    int k;
    size_t L = strcspn(line, "\r\n");
    line[L] = 0;
    if (L > 64) L = 64;
    memcpy(maze[rows], line, L);
    maze[rows][L] = 0;
    if ((int)L > cols) cols = (int)L;
    for (k = 0; k < (int)L; ++k)
      if (line[k] == 'S') {
        sx = k;
        sz = rows;
      }
    ++rows;
  }
  fclose(f);
}

int main(int argc, char** argv) {
  static char line[1 << 20];
  static unsigned char scratch[1 << 22];
  pj_limits lim = {1u << 20, 32, 256, 4096, 1u << 16, 16384};
  pj_doc d;
  int i;
#if defined(_WIN32)
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  for (i = 1; i + 1 < argc; i += 2) {
    const char *k = argv[i], *v = argv[i + 1];
    if (!strcmp(k, "--maze")) load_maze(v);
    else if (!strcmp(k, "--deny-first")) deny_first = atol(v);
    else if (!strcmp(k, "--rate-every")) rate_every = atol(v);
    else if (!strcmp(k, "--crash-after")) crash_after = atol(v);
    else if (!strcmp(k, "--garbage-after")) garbage_after = atol(v);
    else if (!strcmp(k, "--long-after")) long_after = atol(v);
    else if (!strcmp(k, "--hang-after")) hang_after = atol(v);
    else if (!strcmp(k, "--error-after")) error_after = atol(v);
    else if (!strcmp(k, "--slow-ms")) slow_ms = (unsigned)atol(v);
    else if (!strcmp(k, "--boot-lines")) boot_lines = atoi(v);
    else if (!strcmp(k, "--notify")) notify = atoi(v);
    else if (!strcmp(k, "--no-init-reply")) no_init_reply = atoi(v);
    else if (!strcmp(k, "--bad-state")) bad_state = atoi(v);
    else if (!strcmp(k, "--sign")) sign_text = v;
    else if (!strcmp(k, "--stats")) stats_path = v;
  }
  for (i = 0; i < boot_lines; ++i) printf("[boot] loading chunk %d of the fake world\n", i);
  fflush(stdout);
  while (fgets(line, sizeof(line), stdin)) {
    size_t n = strcspn(line, "\r\n");
    int32_t vid, vm, vp, vn, va;
    long long id;
    if (n == 0) continue;
    if (pj_json_parse(line, n, &lim, scratch, pj_json_scratch_bytes(&lim), &d) != 0) continue;
    vid = pj_get(&d, 0, "id");
    vm = pj_get(&d, 0, "method");
    if (vid < 0 || !pj_is(&d, vid, PJ_NUM)) continue; /* notification */
    id = (long long)d.t[vid].i;
    if (pj_str_eq(&d, vm, "initialize")) {
      if (no_init_reply) continue;
      printf("{\"jsonrpc\":\"2.0\",\"id\":%lld,\"result\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{\"tools\":{}},"
             "\"serverInfo\":{\"name\":\"fake-lithokraft\",\"version\":\"0\"}}}\n",
             id);
      fflush(stdout);
      continue;
    }
    if (!pj_str_eq(&d, vm, "tools/call")) {
      printf("{\"jsonrpc\":\"2.0\",\"id\":%lld,\"error\":{\"code\":-32601,\"message\":\"method not found\"}}\n", id);
      fflush(stdout);
      continue;
    }
    vp = pj_get(&d, 0, "params");
    vn = pj_get(&d, vp, "name");
    va = pj_get(&d, vp, "arguments");
    ++calls;
    stat_line(pj_is(&d, vn, PJ_STR) ? d.t[vn].s : "?");
    if (crash_after && calls >= crash_after) exit(3);
    if (hang_after && calls >= hang_after) {
      fg_sleep(30000);
      exit(4);
    }
    if (slow_ms) fg_sleep(slow_ms);
    if (garbage_after && calls >= garbage_after) {
      printf("{garbage\n");
      fflush(stdout);
      continue;
    }
    if (long_after && calls >= long_after) {
      long k;
      for (k = 0; k < 300000; ++k) fputc('a', stdout);
      fputc('\n', stdout);
      fflush(stdout);
      continue;
    }
    if (error_after && calls >= error_after) {
      printf("{\"jsonrpc\":\"2.0\",\"id\":%lld,\"error\":{\"code\":-32603,\"message\":\"internal\"}}\n", id);
      fflush(stdout);
      continue;
    }
    if (calls <= deny_first) {
      reply_text(id, "permission denied: this client has not been granted world_read", 1);
      continue;
    }
    if (rate_every && calls % rate_every == 0) {
      reply_text(id, "rate limit exceeded, slow down", 1);
      continue;
    }
    if (pj_str_eq(&d, vn, "player_get_state")) {
      char p[512];
      if (bad_state)
        snprintf(p, sizeof(p), "{\"x\":%d.5,\"y\":64.0,\"z\":%d.5}", sx, sz);
      else
        snprintf(p, sizeof(p), "{\"x\":%d.5,\"y\":64.0,\"z\":%d.5,\"yaw\":90.0,\"pitch\":0.0,\"note\":\"%s\"}", sx, sz,
                 sign_text ? sign_text : "");
      reply_text(id, p, 0);
    } else if (pj_str_eq(&d, vn, "world_get_block")) {
      char p[512];
      int64_t x = 0, y = 0, z = 0;
      pj_get_int(&d, pj_get(&d, va, "x"), -100000, 100000, &x);
      pj_get_int(&d, pj_get(&d, va, "y"), -100000, 100000, &y);
      pj_get_int(&d, pj_get(&d, va, "z"), -100000, 100000, &z);
      snprintf(p, sizeof(p), "{\"blockId\":%d,\"name\":\"block\",\"sign\":\"%s\"}", block_at((long)x, (long)y, (long)z),
               sign_text ? sign_text : "");
      reply_text(id, p, 0);
    } else if (pj_str_eq(&d, vn, "player_release")) {
      reply_text(id, "{\"released\":true}", 0);
    } else {
      reply_text(id, "unknown tool", 1);
    }
  }
  return 0;
}
