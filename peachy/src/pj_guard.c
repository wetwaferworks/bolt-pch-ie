/* Peachy file guard: the verdict. The facts come from the platform layer. MIT license. */
#include "pj_guard.h"

#include "pj_base.h"

static int refuse(char* why, size_t cap, const char* s) {
  if (why && cap) pj_copy_cstr(why, cap, s);
  return -1;
}

int pj_guard_check(const pj_file_facts* f, int kind, char* why, size_t why_cap) {
  if (why && why_cap) why[0] = 0;
  if (!f || !f->known) return refuse(why, why_cap, "its permissions could not be read");
  if (kind != PJ_GUARD_ALLOWLIST && kind != PJ_GUARD_TOKEN && kind != PJ_GUARD_CODE)
    return refuse(why, why_cap, "unknown kind of file");
  if (f->link_in_path) return refuse(why, why_cap, "a link or junction is in its path");
  if (!f->regular) return refuse(why, why_cap, "it is not an ordinary file");
  if (kind == PJ_GUARD_TOKEN) {
    if (f->others_can_read) return refuse(why, why_cap, "other accounts can read it");
    return 0;
  }
  if (kind == PJ_GUARD_ALLOWLIST && f->elevated)
    return refuse(why, why_cap, "this process is elevated, so it could change its own grants");
  if (f->changeable) return refuse(why, why_cap, "this account can change it");
  if (f->path_changeable) return refuse(why, why_cap, "this account can replace it through a folder above it");
  return 0;
}
