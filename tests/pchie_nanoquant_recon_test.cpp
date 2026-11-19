// Bolt PCHIE - NanoQuant block-reconstruction resume tests (tiny model, CPU).
#include "pchie/pchie_nanoquant_recon.h"
#include <cstdio>
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
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#endif

static int g_fail = 0, g_pass = 0;
#define CHECK(c) do { if (c) ++g_pass; else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static std::string join(const std::string &a, const char *b) {
    std::string s = a;
    if (!s.empty() && s.back() != '/' && s.back() != '\\') s.push_back('/');
    s += b;
    return s;
}

static int read_all(const std::string &path, std::vector<unsigned char> *out) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < 0) { std::fclose(f); return 0; }
    out->resize((size_t)n);
    const int ok = std::fread(out->data(), 1, (size_t)n, f) == (size_t)n;
    std::fclose(f);
    return ok;
}

static void make_dir(const std::string &p) {
#ifdef _WIN32
    _mkdir(p.c_str());
#else
    mkdir(p.c_str(), 0755);
#endif
}

#ifdef _WIN32
static std::string test_root() {
    char buf[MAX_PATH];
    const DWORD n = GetTempPathA(MAX_PATH, buf);
    std::string s(buf, n);
    s += "pchie-nq-recon-test";
    _mkdir(s.c_str());
    return s;
}
#else
#include <cstdlib>
static std::string test_root() {
    std::string s = "/tmp/pchie-nq-recon-test";
    mkdir(s.c_str(), 0755);
    return s;
}
#endif

static void rm_tree(const std::string &path);

static void rm_tree(const std::string &path) {
#ifdef _WIN32
    std::string spec = path + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(spec.c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (std::strcmp(fd.cFileName, ".") == 0 || std::strcmp(fd.cFileName, "..") == 0) continue;
            const std::string c = path + "\\" + fd.cFileName;
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) rm_tree(c);
            else DeleteFileA(c.c_str());
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
    RemoveDirectoryA(path.c_str());
#else
    DIR *d = opendir(path.c_str());
    if (d) {
        while (dirent *e = readdir(d)) {
            if (std::strcmp(e->d_name, ".") == 0 || std::strcmp(e->d_name, "..") == 0) continue;
            const std::string c = path + "/" + e->d_name;
            rm_tree(c);
            unlink(c.c_str());
        }
        closedir(d);
    }
    rmdir(path.c_str());
#endif
}

static void tiny_cfg(pchie_nq_recon_config *c) {
    pchie_nq_recon_config_init(c);
    c->bits = 1.0f;
    c->admm_outer_iters = 16;
    c->admm_inner_iters = 3;
    c->seed = 7;
    c->samples = 4;
    c->seq = 4;
    c->nonfact_epochs = 1;
    c->fact_epochs = 1;
    c->model_epochs = 1;
    c->nonfact_batch = 2;
    c->fact_batch = 1;
    c->use_cuda = 0;
    c->pause_if_input_ms = 0;
    c->respect_gpu_busy = 0;
    c->stop_after_blocks = 0;
}

static int journal_completed(const std::string &dir) {
    std::ifstream in(join(dir, "journal.nqj"));
    if (!in) return -1;
    std::string line;
    int completed = -1, kd = -1;
    while (std::getline(in, line)) {
        if (line.rfind("completed ", 0) == 0) completed = std::atoi(line.c_str() + 10);
        if (line.rfind("kd ", 0) == 0) kd = std::atoi(line.c_str() + 3);
    }
    return (kd == 1) ? 100 + completed : completed;
}

static void test_kill_resume() {
    const std::string root = test_root();
    const std::string a = join(root, "full");
    const std::string b = join(root, "killed");
    rm_tree(a);
    rm_tree(b);
    make_dir(root);
    make_dir(a);
    make_dir(b);

    pchie_nq_recon_config cfg;
    tiny_cfg(&cfg);
    pchie_nq_recon_stats st;
    char err[PCHIE_NQ_ERROR_CAPACITY] = {0};

    const pchie_nq_recon_status fa =
        pchie_nq_recon_tiny(a.c_str(), 2, 16, 32, 16, &cfg, &st, err, sizeof err);
    CHECK(fa == PCHIE_NQ_RECON_OK);
    CHECK(st.blocks == 2);
    CHECK(st.blocks_done_this_run == 2);
    CHECK(st.paused == 0);
    CHECK(journal_completed(a) == 102);

    cfg.stop_after_blocks = 1;
    const pchie_nq_recon_status fb =
        pchie_nq_recon_tiny(b.c_str(), 2, 16, 32, 16, &cfg, &st, err, sizeof err);
    CHECK(fb == PCHIE_NQ_RECON_PAUSED);
    CHECK(st.blocks_done_this_run == 1);
    CHECK(journal_completed(b) == 1);

    cfg.stop_after_blocks = 0;
    const pchie_nq_recon_status fr = pchie_nq_recon_resume(b.c_str(), &cfg, &st, err, sizeof err);
    if (fr != PCHIE_NQ_RECON_OK) std::printf("resume err: %s\n", err);
    CHECK(fr == PCHIE_NQ_RECON_OK);
    CHECK(st.blocks_already == 1);
    CHECK(st.blocks_done_this_run == 1);
    CHECK(journal_completed(b) == 102);

    /* Bit-identical journals and block checkpoints. */
    for (const char *rel : {"journal.nqj", "teacher.nqt", "calib.bin", "kd-final.nqk",
                            "blocks/block-000.nqb", "blocks/block-001.nqb"}) {
        std::vector<unsigned char> xa, xb;
        CHECK(read_all(join(a, rel), &xa));
        CHECK(read_all(join(b, rel), &xb));
        CHECK(xa == xb);
    }

    /* A second resume is a no-op and does not change the bits. */
    std::vector<unsigned char> before, after;
    read_all(join(b, "journal.nqj"), &before);
    CHECK(pchie_nq_recon_resume(b.c_str(), &cfg, &st, err, sizeof err) == PCHIE_NQ_RECON_OK);
    read_all(join(b, "journal.nqj"), &after);
    CHECK(before == after);

    /* Last-input and busy queries must not crash. */
    (void)pchie_nq_last_input_age_ms();
    char why[256];
    (void)pchie_nq_gpu_is_busy(why, sizeof why);
}

int main() {
    test_kill_resume();
    std::printf("pchie_nanoquant_recon_test: %d pass, %d fail\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
