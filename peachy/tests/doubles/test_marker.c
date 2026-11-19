/* A library that leaves a file behind the moment it is loaded, before any of its functions is called: the test
 * for "nothing is loaded until it is checked" (review 2026-11-19, H-1). MIT license. */
#include <stdio.h>

#ifndef PEACHY_TEST_MARKER_PATH
#error "PEACHY_TEST_MARKER_PATH must name the file to leave behind"
#endif

static void leave_marker(void) {
  FILE* f = fopen(PEACHY_TEST_MARKER_PATH, "ab");
  if (f) {
    fputs("loaded\n", f);
    fclose(f);
  }
}

#if defined(_WIN32)
#include <windows.h>
BOOL WINAPI DllMain(HINSTANCE h, DWORD why, LPVOID r) {
  (void)h;
  (void)r;
  if (why == DLL_PROCESS_ATTACH) leave_marker();
  return TRUE;
}
#else
__attribute__((constructor)) static void on_load(void) { leave_marker(); }
#endif

#if defined(_WIN32)
__declspec(dllexport)
#endif
int peachy_test_marker_present(void) { return 1; }
