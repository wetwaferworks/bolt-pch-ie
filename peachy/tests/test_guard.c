/* The file guard: the verdict for every kind of fact, and the real facts for files this test can make (review
 * 2026-11-16, H1). A protected allowlist can't be made without elevation, so its accepted path is covered by the
 * verdict table here and checked by hand on a protected folder. MIT license. */
#include "pt.h"
#include "pj_guard.h"
#include "pj_platform.h"

#if defined(_WIN32)
#include <windows.h>
#include <aclapi.h>
/* Adds an allow entry for Everyone to read path. 0 on success. */
static int open_to_everyone(const char* path) {
  BYTE world[SECURITY_MAX_SID_SIZE];
  DWORD sz = sizeof(world);
  PSECURITY_DESCRIPTOR sd = NULL;
  PACL old = NULL, acl = NULL;
  EXPLICIT_ACCESS_A ea;
  int rc = -1;
  if (!CreateWellKnownSid(WinWorldSid, NULL, world, &sz)) return -1;
  if (GetNamedSecurityInfoA(path, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, NULL, NULL, &old, NULL, &sd) != ERROR_SUCCESS)
    return -1;
  memset(&ea, 0, sizeof(ea));
  ea.grfAccessPermissions = GENERIC_READ;
  ea.grfAccessMode = GRANT_ACCESS;
  ea.grfInheritance = NO_INHERITANCE;
  ea.Trustee.TrusteeForm = TRUSTEE_IS_SID;
  ea.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
  ea.Trustee.ptstrName = (LPSTR)world;
  if (SetEntriesInAclA(1, &ea, old, &acl) == ERROR_SUCCESS &&
      SetNamedSecurityInfoA((LPSTR)path, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, NULL, NULL, acl, NULL) == ERROR_SUCCESS)
    rc = 0;
  if (acl) LocalFree(acl);
  LocalFree(sd);
  return rc;
}
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

static int verdict(const pj_file_facts* f, int kind, const char* want) {
  char why[96];
  int r = pj_guard_check(f, kind, why, sizeof(why));
  if (!want) return r == 0 && why[0] == 0;
  return r != 0 && strstr(why, want) != NULL;
}

static void table(void) {
  pj_file_facts ok, f;
  memset(&ok, 0, sizeof(ok));
  ok.known = 1;
  ok.regular = 1;
  CHECK(verdict(&ok, PJ_GUARD_ALLOWLIST, NULL), "a protected allowlist is accepted");
  CHECK(verdict(&ok, PJ_GUARD_TOKEN, NULL), "a private token is accepted");
  CHECK(verdict(NULL, PJ_GUARD_ALLOWLIST, "could not be read"), "no facts: refused");
  f = ok;
  f.known = 0;
  CHECK(verdict(&f, PJ_GUARD_ALLOWLIST, "could not be read") && verdict(&f, PJ_GUARD_TOKEN, "could not be read"),
        "facts that could not be read: refused, both kinds");
  CHECK(verdict(&ok, 0, "unknown kind") && verdict(&ok, 4, "unknown kind"), "an unknown kind is refused");
  /* code serve loads (review 2026-11-19, H-1): no link, ordinary, and neither it nor a folder above changeable */
  CHECK(verdict(&ok, PJ_GUARD_CODE, NULL), "protected code is accepted");
  f = ok;
  f.changeable = 1;
  CHECK(verdict(&f, PJ_GUARD_CODE, "can change it"), "a library this account can change: refused");
  f = ok;
  f.path_changeable = 1;
  CHECK(verdict(&f, PJ_GUARD_CODE, "folder above"), "a library that can be swapped through a folder: refused");
  f = ok;
  f.link_in_path = 1;
  CHECK(verdict(&f, PJ_GUARD_CODE, "link"), "a library behind a link: refused");
  f = ok;
  f.regular = 0;
  CHECK(verdict(&f, PJ_GUARD_CODE, "ordinary"), "a library that is not an ordinary file: refused");
  f = ok;
  f.known = 0;
  CHECK(verdict(&f, PJ_GUARD_CODE, "could not be read"), "a library whose permissions cannot be read: refused");
  f = ok;
  f.elevated = 1;
  CHECK(verdict(&f, PJ_GUARD_CODE, NULL), "the installer may check protected code while elevated");
  f = ok;
  f.link_in_path = 1;
  CHECK(verdict(&f, PJ_GUARD_ALLOWLIST, "link") && verdict(&f, PJ_GUARD_TOKEN, "link"), "a link in the path: refused");
  f = ok;
  f.regular = 0;
  CHECK(verdict(&f, PJ_GUARD_ALLOWLIST, "ordinary") && verdict(&f, PJ_GUARD_TOKEN, "ordinary"),
        "not an ordinary file: refused");
  f = ok;
  f.changeable = 1;
  CHECK(verdict(&f, PJ_GUARD_ALLOWLIST, "can change it"), "an allowlist this account can change: refused");
  CHECK(verdict(&f, PJ_GUARD_TOKEN, NULL), "a token this account can change is fine (it is this account's own)");
  f = ok;
  f.path_changeable = 1;
  CHECK(verdict(&f, PJ_GUARD_ALLOWLIST, "folder above"), "an allowlist that can be swapped through a folder: refused");
  f = ok;
  f.elevated = 1;
  CHECK(verdict(&f, PJ_GUARD_ALLOWLIST, "elevated"), "an elevated process may not load an allowlist");
  f = ok;
  f.others_can_read = 1;
  CHECK(verdict(&f, PJ_GUARD_TOKEN, "other accounts"), "a token other accounts can read: refused");
  CHECK(verdict(&f, PJ_GUARD_ALLOWLIST, NULL), "an allowlist others can read is fine (it holds only hashes)");
}

static void real_files(void) {
  char dir[600], file[700], sub[700], bad[800];
  pj_file_facts f;
  int elevated = pj_process_elevated();
  pt_newdir(dir, sizeof(dir), "guard");
  snprintf(file, sizeof(file), "%s/allow.json", dir);
  pt_write_file(file, "{\"version\":1,\"clients\":[]}", 27);
  pj_file_facts_get(file, &f);
  CHECK(f.known && f.regular && !f.link_in_path, "a scratch file: facts read");
  CHECK(f.changeable, "a scratch file is changeable by the account that made it");
  CHECK(verdict(&f, PJ_GUARD_ALLOWLIST, elevated ? "elevated" : "can change it"),
        "a scratch allowlist is refused: the outside models' account could rewrite it");
  if (!elevated)
    CHECK(verdict(&f, PJ_GUARD_CODE, "can change it"), "a library in a scratch folder is refused: it could be swapped");
  pj_file_facts_get(dir, &f);
  CHECK(f.known && !f.regular && verdict(&f, PJ_GUARD_TOKEN, "ordinary"), "a folder is not a token file");
  snprintf(bad, sizeof(bad), "%s/missing.json", dir);
  pj_file_facts_get(bad, &f);
  CHECK(!f.known && verdict(&f, PJ_GUARD_ALLOWLIST, "could not be read"), "a missing file is refused");
  pj_file_facts_get(NULL, &f);
  CHECK(!f.known, "no path: refused");
  snprintf(sub, sizeof(sub), "%s/sub", dir);
  pt_mkdir(sub);
  snprintf(bad, sizeof(bad), "%s/sub/../allow.json", dir);
  pj_file_facts_get(bad, &f);
#if defined(_WIN32)
  CHECK(f.known && f.changeable, "Windows: .. is resolved before the walk, so the real file is judged");
  pj_file_facts_get("\\\\server\\share\\allow.json", &f);
  CHECK(!f.known, "Windows: a UNC path is refused");
  /* a token made private to this account (what token-new does) is accepted; opened to Everyone, it is refused */
  snprintf(bad, sizeof(bad), "%s/grok.token", dir);
  pt_write_file(bad, "0123456789abcdef0123456789abcdef", 32);
  CHECK(pj_protect_path(bad, PJ_GUARD_TOKEN, 0) == 0, "Windows: a token file can be made private without elevation");
  pj_file_facts_get(bad, &f);
  CHECK(f.known && !f.others_can_read && verdict(&f, PJ_GUARD_TOKEN, NULL), "Windows: a private token is accepted");
  CHECK(open_to_everyone(bad) == 0, "Windows: open the token to Everyone");
  pj_file_facts_get(bad, &f);
  CHECK(f.known && f.others_can_read && verdict(&f, PJ_GUARD_TOKEN, "other accounts"),
        "Windows: a token Everyone can read is refused");
  CHECK(pj_protect_path(file, PJ_GUARD_ALLOWLIST, 0) != 0 || elevated,
        "Windows: an unelevated process cannot hand an allowlist to Administrators");
#else
  CHECK(!f.known, "POSIX: a path with .. is refused outright");
  snprintf(bad, sizeof(bad), "%s//allow.json", dir);
  pj_file_facts_get(bad, &f);
  CHECK(!f.known, "POSIX: a doubled slash is refused outright");
  /* a link to the file */
  snprintf(bad, sizeof(bad), "%s/link.json", dir);
  CHECK(symlink(file, bad) == 0, "make a symlink");
  pj_file_facts_get(bad, &f);
  CHECK(f.link_in_path && verdict(&f, PJ_GUARD_ALLOWLIST, "link"), "POSIX: an allowlist behind a symlink is refused");
  /* a token: private is accepted, group or world readable is refused */
  snprintf(bad, sizeof(bad), "%s/grok.token", dir);
  pt_write_file(bad, "0123456789abcdef0123456789abcdef", 32);
  CHECK(pj_protect_path(bad, PJ_GUARD_TOKEN, 0) == 0, "POSIX: a token file can be made 0600");
  pj_file_facts_get(bad, &f);
  CHECK(f.known && !f.others_can_read && verdict(&f, PJ_GUARD_TOKEN, NULL), "POSIX: a 0600 token is accepted");
  chmod(bad, 0640);
  pj_file_facts_get(bad, &f);
  CHECK(f.others_can_read && verdict(&f, PJ_GUARD_TOKEN, "other accounts"), "POSIX: a group-readable token is refused");
  chmod(bad, 0604);
  pj_file_facts_get(bad, &f);
  CHECK(f.others_can_read, "POSIX: a world-readable token is refused");
  CHECK(pj_protect_path(file, PJ_GUARD_ALLOWLIST, 0) != 0 || elevated,
        "POSIX: only root can hand an allowlist to root");
  /* relative paths are judged from the current folder */
  pj_file_facts_get("definitely-not-here.json", &f);
  CHECK(!f.known, "POSIX: a missing relative path is refused");
#endif
}

int main(int argc, char** argv) {
  pt_init(argc, argv);
  table();
  real_files();
  if (pj_process_elevated()) printf("note: running elevated; every allowlist is refused, as designed\n");
  return pt_done("peachy_guard");
}
