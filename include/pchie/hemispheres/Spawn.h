// Bolt PCHIE - starting a helper program without a shell.
//
// The hemispheres tool asks Mjolnir's mj_hemi for every keep-or-roll-back decision.
// The program and its arguments are passed as a list, never as one command line for a
// shell, so a path or a note cannot run anything: no quoting, no %VAR%, no $(...), no
// ;, & or |. On Windows the program is started with CreateProcessW by its full path (no
// search), and each argument is quoted the way the C runtime reads it back. Elsewhere
// it is posix_spawn with an argv array.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#pragma once
#include <string>
#include <vector>

namespace pchie {
namespace hemispheres {

// Runs argv[0] with the arguments argv[1..], waits, and returns its exit code; 127 when
// it could not be started (or argv is empty, or an argument is not valid UTF-8).
int run_program(const std::vector<std::string>& argv);

// The Windows command line run_program builds (exposed for the tests).
std::wstring windows_command_line(const std::vector<std::string>& argv, bool& ok);

// A path given on the command line: not empty, at most 1024 bytes, no control
// characters, and none of the characters Windows refuses in a path (" < > | ? *).
bool path_ok(const std::string& p, std::string& why);

} // namespace hemispheres
} // namespace pchie
