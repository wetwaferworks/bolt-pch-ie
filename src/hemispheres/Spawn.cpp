// Bolt PCHIE - starting a helper program without a shell. See include/pchie/hemispheres/Spawn.h.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#include "pchie/hemispheres/Spawn.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace pchie {
namespace hemispheres {

// UTF-8 to UTF-16, strict: a malformed sequence or a NUL fails.
static bool widen(const std::string& s, std::wstring& w) {
    w.clear();
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp, need;
        if (c == 0) return false;
        if (c < 0x80) { cp = c; need = 0; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1Fu; need = 1; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0Fu; need = 2; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07u; need = 3; }
        else return false;
        if (need && i + need >= s.size()) return false;   // the sequence runs past the end
        for (uint32_t k = 1; k <= need; ++k) {
            const unsigned char d = (unsigned char)s[i + k];
            if ((d & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (d & 0x3Fu);
        }
        if ((need == 1 && cp < 0x80) || (need == 2 && cp < 0x800) || (need == 3 && cp < 0x10000) || cp > 0x10FFFF ||
            (cp >= 0xD800 && cp <= 0xDFFF))
            return false;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            w += wchar_t(0xD800 + (cp >> 10));
            w += wchar_t(0xDC00 + (cp & 0x3FF));
        } else {
            w += wchar_t(cp);
        }
        i += need + 1;
    }
    return true;
}

// One argument, always quoted, with backslashes doubled only where the C runtime
// would otherwise read them as escaping a quote.
static void append_arg(std::wstring& cl, const std::wstring& a) {
    if (!cl.empty()) cl += L' ';
    cl += L'"';
    size_t bs = 0;
    for (wchar_t c : a) {
        if (c == L'\\') { ++bs; continue; }
        if (c == L'"') { cl.append(bs * 2 + 1, L'\\'); cl += L'"'; bs = 0; continue; }
        cl.append(bs, L'\\');
        bs = 0;
        cl += c;
    }
    cl.append(bs * 2, L'\\');
    cl += L'"';
}

std::wstring windows_command_line(const std::vector<std::string>& argv, bool& ok) {
    std::wstring cl, w;
    ok = !argv.empty();
    for (const std::string& a : argv) {
        if (!widen(a, w)) { ok = false; return std::wstring(); }
        append_arg(cl, w);
    }
    return cl;
}

int run_program(const std::vector<std::string>& argv) {
    if (argv.empty() || argv[0].empty()) return 127;
    std::fflush(stdout);
    std::fflush(stderr);
#ifdef _WIN32
    bool ok = false;
    std::wstring exe;
    const std::wstring cl = windows_command_line(argv, ok);
    if (!ok || !widen(argv[0], exe)) return 127;
    std::vector<wchar_t> buf(cl.begin(), cl.end());
    buf.push_back(0);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return 127;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 127;
    if (!GetExitCodeProcess(pi.hProcess, &code)) code = 127;
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return int(code);
#else
    std::vector<char*> av;
    for (const std::string& a : argv) av.push_back(const_cast<char*>(a.c_str()));
    av.push_back(nullptr);
    pid_t pid;
    if (posix_spawn(&pid, av[0], nullptr, nullptr, av.data(), environ) != 0) return 127;
    int st = 0;
    while (waitpid(pid, &st, 0) < 0)
        if (errno != EINTR) return 127;
    return WIFEXITED(st) ? WEXITSTATUS(st) : 127;
#endif
}

bool path_ok(const std::string& p, std::string& why) {
    if (p.empty()) { why = "the path is empty"; return false; }
    if (p.size() > 1024) { why = "the path is longer than 1024 bytes"; return false; }
    for (unsigned char c : p) {
        if (c < 0x20 || c == 0x7F) { why = "the path has a control character"; return false; }
        if (c == '"' || c == '<' || c == '>' || c == '|' || c == '?' || c == '*') {
            why = std::string("the path has a character Windows does not allow in paths: ") + char(c);
            return false;
        }
    }
    std::wstring w;
    if (!widen(p, w)) { why = "the path is not valid UTF-8"; return false; }
    return true;
}

} // namespace hemispheres
} // namespace pchie
