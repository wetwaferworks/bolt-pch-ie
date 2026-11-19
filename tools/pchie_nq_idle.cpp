/* Bolt PCHIE - idle runner for a NanoQuant reconstruct job.
 *
 * Watches a directory for a job file. If the machine is idle, the GPU is free,
 * and a reconstruct journal is present, it continues that job. An empty watch
 * directory is a no-op: this runner will not invent or start a new model run.
 * Original model files are never opened for write.
 *
 *   pchie_nq_idle --watch DIR [--exe PATH] [--idle-ms 2000] [--poll-ms 500]
 *
 * job.txt in DIR is a single line: the reconstruct output directory (which must
 * already contain journal.nqj). Optional job.args holds extra flags, one per line.
 */
#include "pchie/pchie_nanoquant_recon.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#endif

static void usage() {
    std::fprintf(stderr,
                 "usage: pchie_nq_idle --watch DIR [--exe PATH] [--idle-ms 2000] [--poll-ms 500]\n"
                 "  Continues a reconstruct job when the machine is idle and the GPU is free.\n"
                 "  An empty watch directory, or one without a journal, is a no-op.\n");
}

static int file_exists(const std::string &p) {
#ifdef _WIN32
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

static std::string join(const std::string &a, const char *b) {
    std::string s = a;
    if (!s.empty() && s.back() != '/' && s.back() != '\\') s.push_back('/');
    s += b;
    return s;
}

static std::string dirname_of(const std::string &p) {
    const size_t s = p.find_last_of("/\\");
    if (s == std::string::npos) return std::string(".");
    return p.substr(0, s);
}

#ifdef _WIN32
static std::string default_exe() {
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string self(buf, n);
    const std::string dir = dirname_of(self);
    const std::string cand = dir + "\\pchie_nanoquant.exe";
    if (file_exists(cand)) return cand;
    return "pchie_nanoquant.exe";
}
#else
static std::string default_exe() { return "pchie_nanoquant"; }
#endif

static std::string trim(std::string s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
    return s.substr(a, b - a);
}

static int read_job(const std::string &watch, std::string *out_dir, std::vector<std::string> *extra) {
    const std::string job = join(watch, "job.txt");
    if (!file_exists(job)) return 0;
    std::ifstream in(job.c_str());
    if (!in) return 0;
    std::string line, chosen;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        chosen = line;
        break;
    }
    if (chosen.empty()) return 0;
    *out_dir = chosen;
    const std::string args = join(watch, "job.args");
    if (file_exists(args)) {
        std::ifstream af(args.c_str());
        while (std::getline(af, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            extra->push_back(line);
        }
    }
    return 1;
}

#ifdef _WIN32
static int spawn_and_watch(const std::string &exe, const std::vector<std::string> &args, int idle_ms, int poll_ms,
                           char *error, size_t cap) {
    std::string cmd = "\"" + exe + "\"";
    for (const std::string &a : args) {
        cmd += " ";
        if (a.find(' ') != std::string::npos) cmd += "\"" + a + "\"";
        else cmd += a;
    }
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    std::memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    std::memset(&pi, 0, sizeof pi);
    std::vector<char> buf(cmd.begin(), cmd.end());
    buf.push_back(0);
    if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, CREATE_NEW_PROCESS_GROUP, nullptr, nullptr, &si,
                        &pi)) {
        std::snprintf(error, cap, "could not start %s", exe.c_str());
        return 1;
    }
    CloseHandle(pi.hThread);
    int rc = 0;
    for (;;) {
        const DWORD w = WaitForSingleObject(pi.hProcess, poll_ms > 0 ? (DWORD)poll_ms : 500);
        if (w == WAIT_OBJECT_0) {
            DWORD code = 0;
            GetExitCodeProcess(pi.hProcess, &code);
            rc = (int)code;
            break;
        }
        const uint32_t age = pchie_nq_last_input_age_ms();
        if ((int)age < idle_ms) {
            /* The child also pauses itself; if it has not exited shortly, stop it.
             * Checkpoints are atomic, so a mid-block stop is safe to resume. */
            const DWORD grace = WaitForSingleObject(pi.hProcess, 3000);
            if (grace != WAIT_OBJECT_0) {
                TerminateProcess(pi.hProcess, 4); /* PAUSED-like */
                WaitForSingleObject(pi.hProcess, 5000);
            }
            GetExitCodeProcess(pi.hProcess, (LPDWORD)&rc);
            std::snprintf(error, cap, "paused: user input %u ms ago", (unsigned)age);
            rc = 0; /* idle runner itself succeeds; the job will resume later */
            break;
        }
    }
    CloseHandle(pi.hProcess);
    return rc;
}
#else
static int spawn_and_watch(const std::string &, const std::vector<std::string> &, int, int, char *error, size_t cap) {
    std::snprintf(error, cap, "the idle runner is implemented for Windows Task Scheduler");
    return 2;
}
#endif

int main(int argc, char **argv) {
    const char *watch = nullptr, *exe_arg = nullptr;
    int idle_ms = 2000, poll_ms = 500;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--watch") == 0 && i + 1 < argc) watch = argv[++i];
        else if (std::strcmp(argv[i], "--exe") == 0 && i + 1 < argc) exe_arg = argv[++i];
        else if (std::strcmp(argv[i], "--idle-ms") == 0 && i + 1 < argc) idle_ms = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--poll-ms") == 0 && i + 1 < argc) poll_ms = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        } else {
            usage();
            return 2;
        }
    }
    if (!watch || !watch[0]) {
        usage();
        return 2;
    }
    if (idle_ms < 500) idle_ms = 500;
    if (poll_ms < 100) poll_ms = 100;

    const uint32_t age = pchie_nq_last_input_age_ms();
    if ((int)age < idle_ms) {
        std::printf("idle runner: user input %u ms ago, nothing to do\n", (unsigned)age);
        return 0;
    }
    char why[PCHIE_NQ_ERROR_CAPACITY];
    if (pchie_nq_gpu_is_busy(why, sizeof why)) {
        std::printf("idle runner: GPU or another job is busy (%s)\n", why);
        return 0;
    }
    std::string out_dir;
    std::vector<std::string> extra;
    if (!read_job(watch, &out_dir, &extra)) {
        std::printf("idle runner: no job in %s, nothing to do\n", watch);
        return 0;
    }
    const std::string journal = join(out_dir, "journal.nqj");
    if (!file_exists(journal)) {
        std::printf("idle runner: %s has no reconstruct journal; will not start a new job\n", out_dir.c_str());
        return 0;
    }
    const std::string exe = exe_arg ? std::string(exe_arg) : default_exe();
    std::vector<std::string> args;
    args.push_back("reconstruct");
    args.push_back("--out");
    args.push_back(out_dir);
    args.push_back("--pause-input-ms");
    {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%d", idle_ms);
        args.push_back(buf);
    }
    args.push_back("--respect-gpu-busy");
    args.insert(args.end(), extra.begin(), extra.end());

    std::printf("idle runner: continuing %s\n", out_dir.c_str());
    char error[PCHIE_NQ_ERROR_CAPACITY] = {0};
    const int rc = spawn_and_watch(exe, args, idle_ms, poll_ms, error, sizeof error);
    if (error[0]) std::printf("%s\n", error);
    return rc;
}
