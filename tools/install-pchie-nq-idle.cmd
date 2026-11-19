@echo off
rem One-line install (no administrator). Also creates an empty watch directory.
if not exist "D:\scratch\nanoquant-jobs" mkdir "D:\scratch\nanoquant-jobs"
schtasks /Create /TN "PCHIE NanoQuant reconstruct (idle)" /XML "D:\scratch\pchie-2026-11-17\tools\pchie_nq_idle.xml" /F
