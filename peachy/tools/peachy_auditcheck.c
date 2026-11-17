/* peachy_auditcheck: re-derives a Peachy audit log's hash chain, read-only. MIT license.
 *   peachy_auditcheck <log> [--expect-head <hex>] [--expect-length <n>]
 * Exit codes: 0 whole (and matching, if asked), 1 broken, 2 unreadable or bad usage, 3 whole but the head or
 * length differs from what was expected (for example, records cut from the end). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pj_audit.h"

int main(int argc, char** argv) {
  pj_audit_report r;
  const char *want_head = NULL, *want_len = NULL;
  char head[65];
  int i, rc;
  static const char* digits = "0123456789abcdef";
  if (argc < 2) {
    fprintf(stderr, "usage: peachy_auditcheck <log> [--expect-head <hex>] [--expect-length <n>]\n");
    return 2;
  }
  for (i = 2; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--expect-head")) want_head = argv[i + 1];
    else if (!strcmp(argv[i], "--expect-length")) want_len = argv[i + 1];
    else return 2;
  }
  rc = pj_audit_verify(argv[1], &r);
  if (rc < 0) {
    fprintf(stderr, "peachy_auditcheck: cannot read the log\n");
    return 2;
  }
  for (i = 0; i < 32; ++i) {
    head[2 * i] = digits[r.head[i] >> 4];
    head[2 * i + 1] = digits[r.head[i] & 15];
  }
  head[64] = 0;
  if (rc == 1) {
    printf("BROKEN at record %llu: %s (records before it are whole)\n", (unsigned long long)r.first_bad, r.why);
    return 1;
  }
  printf("OK %llu records, head %s\n", (unsigned long long)r.records, head);
  if (want_head && strcmp(want_head, head) != 0) {
    printf("MISMATCH: expected head %s\n", want_head);
    return 3;
  }
  if (want_len && strtoull(want_len, NULL, 10) != r.records) {
    printf("MISMATCH: expected %s records\n", want_len);
    return 3;
  }
  return 0;
}
