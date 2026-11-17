/* SHA-256 vectors and the hash-chained audit log: append, verify, and every kind of tampering. MIT license. */
#include "pt.h"
#include "pj_audit.h"
#include "pj_base.h"
#include "pj_platform.h"

static int sha_is(const void* data, size_t n, const char* hex) {
  uint8_t h[32];
  char got[65];
  pj_sha256_once(data, n, h);
  pt_hex(h, 32, got);
  return strcmp(got, hex) == 0;
}

static void sha_vectors(void) {
  static char million[1000000];
  pj_sha256 c;
  uint8_t h1[32], h2[32];
  size_t i;
  CHECK(sha_is("", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), "sha256 empty");
  CHECK(sha_is("abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "sha256 abc");
  CHECK(sha_is("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56,
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
        "sha256 448-bit");
  CHECK(sha_is("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
               112, "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"),
        "sha256 896-bit");
  memset(million, 'a', sizeof(million));
  CHECK(sha_is(million, sizeof(million), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"), "sha256 1M a");
  /* chunked updates give the same answer for every split */
  pj_sha256_once(million, 1000, h1);
  for (i = 0; i <= 1000; i += 7) {
    pj_sha256_init(&c);
    pj_sha256_update(&c, million, i);
    pj_sha256_update(&c, million + i, 1000 - i);
    pj_sha256_final(&c, h2);
    if (memcmp(h1, h2, 32) != 0) break;
  }
  CHECK(i > 1000, "split updates agree");
}

static void write_log(const char* dir, char* path_out, size_t cap, int records, uint8_t head[32]) {
  pj_audit a;
  int k, ok = pj_audit_open(&a, dir, "t") == 0;
  for (k = 0; ok && k < records; ++k) {
    char body[64];
    int n = snprintf(body, sizeof(body), "\"what\":\"r%d\"", k);
    ok = pj_audit_append(&a, (uint64_t)k * 1000u, body, (size_t)n) == 0;
  }
  CHECK(ok, "log written");
  memcpy(head, a.head, 32);
  snprintf(path_out, cap, "%s", a.path);
  pj_audit_close(&a);
}

/* Splits a file into lines (each keeps its LF). Returns count. */
static int split_lines(char* text, char** lines, int max) {
  int n = 0;
  char* p = text;
  while (*p && n < max) {
    char* nl = strchr(p, '\n');
    lines[n++] = p;
    if (!nl) break;
    p = nl + 1;
  }
  return n;
}

static int verify_rc(const char* path, pj_audit_report* r) { return pj_audit_verify(path, r); }

static void write_lines(const char* path, char** lines, const int* order, int n, char* all) {
  FILE* f = fopen(path, "wb");
  int k;
  (void)all;
  for (k = 0; k < n; ++k) {
    char* l = lines[order[k]];
    char* nl = strchr(l, '\n');
    fwrite(l, 1, (size_t)(nl - l) + 1, f);
  }
  fclose(f);
}

static void audit_chain(void) {
  char dir[600], path[600], tpath[700];
  uint8_t head[32];
  pj_audit_report r;
  char *text, *copy, *lines[64];
  size_t n;
  int nl, order[64], k;
  pt_newdir(dir, sizeof(dir), "audit");
  write_log(dir, path, sizeof(path), 10, head);
  CHECK(verify_rc(path, &r) == 0 && r.records == 10 && memcmp(r.head, head, 32) == 0, "clean log verifies, head matches");
  text = pt_read_file(path, &n);
  CHECK(text != NULL, "read log");
  if (!text) return;
  copy = (char*)malloc(n + 1);

  /* exclusive create: a second log with the same name in the same second gets a new file */
  {
    pj_audit a, b;
    int oa = pj_audit_open(&a, dir, "same"), ob = pj_audit_open(&b, dir, "same");
    CHECK(oa == 0 && ob == 0 && strcmp(a.path, b.path) != 0, "two logs never share a file");
    pj_audit_close(&a);
    pj_audit_close(&b);
  }

  /* flip one byte at many positions: every flip is caught */
  {
    int caught = 0, tried = 0;
    size_t pos;
    for (pos = 0; pos < n; pos += 13) {
      if (text[pos] == '\n') continue;
      memcpy(copy, text, n + 1);
      copy[pos] = (char)(copy[pos] ^ 0x01);
      if (copy[pos] == '\n') continue;
      snprintf(tpath, sizeof(tpath), "%s/flip.log", dir);
      pt_write_file(tpath, copy, n);
      ++tried;
      if (verify_rc(tpath, &r) == 1) ++caught;
    }
    CHECK(tried > 50 && caught == tried, "every flipped byte is caught");
    printf("byte flips caught: %d of %d\n", caught, tried);
  }
  memcpy(copy, text, n + 1);
  nl = split_lines(copy, lines, 64);
  CHECK(nl == 10, "ten lines");
  snprintf(tpath, sizeof(tpath), "%s/edit.log", dir);
  /* drop a middle record */
  for (k = 0, n = 0; k < nl; ++k)
    if (k != 4) order[n++] = k;
  write_lines(tpath, lines, order, (int)n, copy);
  CHECK(verify_rc(tpath, &r) == 1 && r.first_bad == 5, "dropped record caught at the gap");
  /* reorder two records */
  for (k = 0; k < nl; ++k) order[k] = k;
  order[3] = 4;
  order[4] = 3;
  write_lines(tpath, lines, order, nl, copy);
  CHECK(verify_rc(tpath, &r) == 1 && r.first_bad == 4, "reordered records caught");
  /* duplicated record */
  for (k = 0; k < 5; ++k) order[k] = k;
  order[5] = 4;
  write_lines(tpath, lines, order, 6, copy);
  CHECK(verify_rc(tpath, &r) == 1 && r.first_bad == 6, "duplicated record caught");
  /* first record removed */
  for (k = 1; k < nl; ++k) order[k - 1] = k;
  write_lines(tpath, lines, order, nl - 1, copy);
  CHECK(verify_rc(tpath, &r) == 1 && r.first_bad == 1, "missing first record caught");
  /* tail truncated by whole records: the chain is still whole, but the head no longer matches the one the
   * client was given by audit_head. That comparison is how a cut tail is found. */
  for (k = 0; k < 7; ++k) order[k] = k;
  write_lines(tpath, lines, order, 7, copy);
  CHECK(verify_rc(tpath, &r) == 0 && r.records == 7 && memcmp(r.head, head, 32) != 0, "cut tail: head differs");
  /* last line cut mid-record */
  memcpy(copy, text, strlen(text) + 1);
  pt_write_file(tpath, copy, strlen(text) - 10);
  CHECK(verify_rc(tpath, &r) == 1 && r.first_bad == 10, "cut last line caught");
  /* a forged record appended with a correct-looking hash of the wrong prev */
  {
    char forged[400];
    size_t L = strlen(text);
    char* buf = (char*)malloc(L + sizeof(forged));
    int m = snprintf(forged, sizeof(forged),
                     "{\"v\":1,\"seq\":11,\"mono_ns\":1,\"local\":\"x\",\"prev\":\"%064d\"} %064d\n", 0, 0);
    memcpy(buf, text, L);
    memcpy(buf + L, forged, (size_t)m);
    pt_write_file(tpath, buf, L + (size_t)m);
    CHECK(verify_rc(tpath, &r) == 1 && r.first_bad == 11, "forged record caught");
    free(buf);
  }
  /* not a log at all, and a missing file */
  pt_write_file(tpath, "hello\n", 6);
  CHECK(verify_rc(tpath, &r) == 1 && r.first_bad == 1, "garbage caught");
  snprintf(tpath, sizeof(tpath), "%s/none.log", dir);
  CHECK(verify_rc(tpath, &r) == -1, "missing file is unreadable");
  free(copy);
  free(text);
}

static void audit_failures(void) {
  char dir[600];
  pj_audit a;
  pt_newdir(dir, sizeof(dir), "auditfail");
  CHECK(pj_audit_open(&a, dir, "Bad Name") != 0, "bad session name refused");
  CHECK(pj_audit_open(&a, "/definitely/not/here/peachy", "t") != 0, "missing dir refused");
  CHECK(pj_audit_open(&a, dir, "f") == 0, "open");
  a.fail_after = 3;
  CHECK(pj_audit_append(&a, 1, "", 0) == 0 && pj_audit_append(&a, 2, "", 0) == 0, "first two writes");
  CHECK(pj_audit_append(&a, 3, "", 0) != 0 && a.failed, "third write fails");
  CHECK(pj_audit_append(&a, 4, "", 0) != 0, "and every later write fails too");
  pj_audit_close(&a);
  {
    static char big[9000];
    CHECK(pj_audit_open(&a, dir, "g") == 0, "open g");
    memset(big, 'a', sizeof(big));
    CHECK(pj_audit_append(&a, 1, big, sizeof(big)) != 0 && a.failed, "oversized record fails safe");
    pj_audit_close(&a);
  }
}

int main(int argc, char** argv) {
  pt_init(argc, argv);
  sha_vectors();
  audit_chain();
  audit_failures();
  return pt_done("peachy_audit");
}
