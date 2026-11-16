// Bolt PCHIE - MCP stdio client: launches LithoKraft with --mcp-stdio and speaks JSON-RPC,
// one message per line (engine docs/MCP-GAME-SEAM.md section 5), on the child's stdin/stdout.
// Concrete final, Windows only. Serial and single-threaded: no mutex, no atomic, no reader
// thread (stdout is polled with PeekNamedPipe; the game's stderr log goes straight to a file so
// no pipe can fill). The library core never links this; only the pchie_lk_pilot tool does.
// Stop order: close stdin (the game's MCP thread sees EOF), WM_CLOSE to the child's own windows,
// wait; TerminateProcess only on the child's own handle, only after the grace period.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "pchie/MiniJson.h"

#include <string>
#include <vector>

namespace pchie {

class McpStdioClient final {
 public:
  McpStdioClient() = default;
  McpStdioClient(const McpStdioClient&) = delete;
  McpStdioClient& operator=(const McpStdioClient&) = delete;
  ~McpStdioClient() { stop(3000); }

  // cmdline: full command line (exe quoted). stderr_log: file the game's stderr is written to.
  bool start(const std::wstring& cmdline, const std::wstring& workdir, const std::wstring& stderr_log,
             std::string& err) {
    if (proc_ || in_w_ || out_r_) {  // never overwrite live handles; call stop() first
      err = "already started";
      return false;
    }
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE in_r = nullptr, out_w = nullptr;
    auto close_all = [&](HANDLE log_h) {
      for (HANDLE* h : {&in_r, &out_w, &in_w_, &out_r_}) if (*h) { CloseHandle(*h); *h = nullptr; }
      if (log_h && log_h != INVALID_HANDLE_VALUE) CloseHandle(log_h);
    };
    if (!CreatePipe(&in_r, &in_w_, &sa, 1 << 16) || !CreatePipe(&out_r_, &out_w, &sa, 1 << 20)) {
      err = "CreatePipe failed";
      close_all(nullptr);
      return false;
    }
    SetHandleInformation(in_w_, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r_, HANDLE_FLAG_INHERIT, 0);
    HANDLE log = CreateFileW(stderr_log.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE) {
      err = "cannot create stderr log";
      close_all(nullptr);
      return false;
    }
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = log;
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> cl(cmdline.begin(), cmdline.end());
    cl.push_back(0);
    const BOOL ok = CreateProcessW(nullptr, cl.data(), nullptr, nullptr, TRUE, 0, nullptr,
                                   workdir.empty() ? nullptr : workdir.c_str(), &si, &pi);
    CloseHandle(in_r);
    CloseHandle(out_w);
    CloseHandle(log);
    if (!ok) {
      err = "CreateProcess failed (" + std::to_string(GetLastError()) + ")";
      for (HANDLE* h : {&in_w_, &out_r_}) if (*h) { CloseHandle(*h); *h = nullptr; }
      return false;
    }
    proc_ = pi.hProcess;
    pid_ = pi.dwProcessId;
    CloseHandle(pi.hThread);
    return true;
  }

  DWORD pid() const noexcept { return pid_; }
  int skipped_lines() const noexcept { return skipped_lines_; }

  bool alive() const {
    if (!proc_) return false;
    DWORD code = 0;
    return GetExitCodeProcess(proc_, &code) && code == STILL_ACTIVE;
  }

  bool send_line(const std::string& line, std::string& err) {
    if (!in_w_) {
      err = "stdin closed";
      return false;
    }
    std::string l = line + "\n";
    DWORD n = 0;
    if (!WriteFile(in_w_, l.data(), DWORD(l.size()), &n, nullptr) || n != l.size()) {
      err = "write to game stdin failed";
      return false;
    }
    return true;
  }

  bool notify(const std::string& method, const std::string& params_json, std::string& err) {
    return send_line("{\"jsonrpc\":\"2.0\",\"method\":" + MiniJson::quote(method) + ",\"params\":" + params_json + "}",
                     err);
  }

  // Sends a request and waits for the response with the same id (notifications are skipped).
  bool request(const std::string& method, const std::string& params_json, MiniJson& result, std::string& err,
               DWORD timeout_ms = 15000) {
    const int id = next_id_++;
    if (!send_line("{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(id) + ",\"method\":" + MiniJson::quote(method) +
                       ",\"params\":" + params_json + "}",
                   err))
      return false;
    const ULONGLONG t0 = GetTickCount64();
    for (;;) {
      const ULONGLONG el = GetTickCount64() - t0;
      if (el >= timeout_ms) {
        err = "timeout waiting for '" + method + "'";
        return false;
      }
      std::string line;
      if (!read_line(line, DWORD(timeout_ms - el), err)) return false;
      // The game prints its boot log lines to stdout before --mcp-stdio moves its logging to
      // stderr; those are not JSON-RPC. Skip any line that is not a JSON object (counted).
      if (line[0] != '{') {
        ++skipped_lines_;
        continue;
      }
      MiniJson msg;
      if (!MiniJson::parse(line, msg, err)) {
        err = "bad JSON from game: " + err + " (line length " + std::to_string(line.size()) + ") in: " + line.substr(0, 400) + " ... TAIL: " + (line.size() > 400 ? line.substr(line.size() > 1200 ? line.size() - 800 : 400) : std::string());  // lkbed diagnostic 2026-11-10: show the tail too
        return false;
      }
      const MiniJson* mid = msg.get("id");
      if (!mid || mid->type != MiniJson::Type::Number || int(mid->number) != id) continue;
      if (const MiniJson* e = msg.get("error")) {
        err = "rpc error " + std::to_string(int(e->num_or("code", 0))) + ": " + e->str_or("message", "");
        return false;
      }
      const MiniJson* r = msg.get("result");
      if (!r) {
        err = "response without result";
        return false;
      }
      result = *r;
      return true;
    }
  }

  void stop(DWORD grace_ms) {
    if (in_w_) {
      CloseHandle(in_w_);
      in_w_ = nullptr;
    }
    if (proc_) {
      if (alive()) {
        EnumWindows(&close_own_windows, LPARAM(pid_));
        if (WaitForSingleObject(proc_, grace_ms) == WAIT_TIMEOUT) TerminateProcess(proc_, 1);
        WaitForSingleObject(proc_, 5000);
      }
      CloseHandle(proc_);
      proc_ = nullptr;
    }
    if (out_r_) {
      CloseHandle(out_r_);
      out_r_ = nullptr;
    }
  }

  // Brings the child's own top-level window forward (look input needs a focused, captured window).
  void focus_window() const { EnumWindows(&focus_own_window, LPARAM(pid_)); }

 private:
  static BOOL CALLBACK close_own_windows(HWND h, LPARAM pid) {
    DWORD p = 0;
    GetWindowThreadProcessId(h, &p);
    if (p == DWORD(pid)) PostMessageW(h, WM_CLOSE, 0, 0);
    return TRUE;
  }
  static BOOL CALLBACK focus_own_window(HWND h, LPARAM pid) {
    DWORD p = 0;
    GetWindowThreadProcessId(h, &p);
    if (p == DWORD(pid) && IsWindowVisible(h)) {
      SetForegroundWindow(h);
      return FALSE;
    }
    return TRUE;
  }

  bool read_line(std::string& line, DWORD timeout_ms, std::string& err) {
    const ULONGLONG t0 = GetTickCount64();
    for (;;) {
      const size_t nl = buf_.find('\n');
      if (nl != std::string::npos) {
        line = buf_.substr(0, nl);
        buf_.erase(0, nl + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        return true;
      }
      DWORD avail = 0;
      if (!PeekNamedPipe(out_r_, nullptr, 0, nullptr, &avail, nullptr)) {
        err = "game stdout closed (game exited?)";
        return false;
      }
      if (avail) {
        std::string chunk(avail, '\0');
        DWORD n = 0;
        if (!ReadFile(out_r_, chunk.data(), avail, &n, nullptr)) {
          err = "read from game stdout failed";
          return false;
        }
        buf_.append(chunk.data(), n);
        continue;
      }
      if (GetTickCount64() - t0 >= timeout_ms) {
        err = "timeout reading game stdout";
        return false;
      }
      Sleep(2);
    }
  }

  HANDLE proc_ = nullptr, in_w_ = nullptr, out_r_ = nullptr;
  DWORD pid_ = 0;
  std::string buf_;
  int next_id_ = 1;
  int skipped_lines_ = 0;
};

}  // namespace pchie
