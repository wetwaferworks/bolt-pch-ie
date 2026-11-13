# Bolt PCHIE search-consistency gate. Passes only if:
#   * gates 1-5 still pass,
#   * the original tool passes in-process (KV ring, ledger, distinct picks, solo==batched, step-back, CPU==CUDA)
#     at 1, 2, 8 and 16 threads, and every [SEARCH-HASH] line is identical across thread counts and CUDA.
# Usage: powershell -File tools\run-search-gate.ps1 [-Model path] [-Out dir] [-Repeat 100]
param([string]$Model = $PCHIE_MODEL,
      [string]$Out = "$PSScriptRoot\..\build\s6b", [int]$Repeat = 0, [int]$Budget = 32)

# Lab roots come from the environment; there is no machine-specific default.
function Require-PchieEnv([string]$Name) {
  $v = [Environment]::GetEnvironmentVariable($Name)
  if ([string]::IsNullOrWhiteSpace($v)) {
    throw "Set environment variable $Name to the corresponding local path before running this lab script. There is no default path."
  }
  return $v.TrimEnd('\','/')
}
function Resolve-PchieModel {
  $f = [Environment]::GetEnvironmentVariable('PCHIE_MODEL')
  $d = [Environment]::GetEnvironmentVariable('PCHIE_MODELS')
  if (-not [string]::IsNullOrWhiteSpace($f)) { return $f }
  if (-not [string]::IsNullOrWhiteSpace($d)) { return (Join-Path $d 'Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf') }
  throw "Set PCHIE_MODEL to your GGUF weights, or PCHIE_MODELS to the directory that holds Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf."
}
$PCHIE_MODEL = Resolve-PchieModel
$ErrorActionPreference = 'Continue'
$bin = Resolve-Path "$PSScriptRoot\..\build"
New-Item -ItemType Directory -Force $Out | Out-Null
$fail = @()
function Run($name, $exe, [string[]]$a) {
    $log = Join-Path $Out "$name.txt"
    & (Join-Path $bin $exe) @a *> $log
    $rc = $LASTEXITCODE
    Add-Content -Path (Join-Path $Out 'log.txt') -Value "$name exit $rc"; Write-Host "$name exit $rc"
    if ($rc -ne 0) { $script:fail += $name }
    return $log
}
# gates 1-5
Run 'g1' 'pchie_spine_test.exe' @() | Out-Null
Run 'g2' 'pchie_timing_handshake_test.exe' @() | Out-Null
Run 'g3k' 'pchie_kernels_test.exe' @() | Out-Null
$g3 = @{}
foreach ($t in 1,2,8,16) { $l = Run "g3_t$t" 'pchie_generate.exe' @($Model, "$t"); $g3[$t] = (Select-String $l -Pattern 'run hash=(0x[0-9A-F]+)').Matches[0].Groups[1].Value }
if (($g3.Values | Select-Object -Unique).Count -ne 1) { $fail += 'g3-thread-hash' }
Run 'g4' 'pchie_cuda_gate.exe' @($Model, '8') | Out-Null
Run 'g5' 'pchie_fork_batching.exe' @($Model, '8') | Out-Null
# search gate per thread count
$hashes = @{}
foreach ($t in 1,2,8,16) {
    $l = Run "s6_t$t" 'pchie_search_gate.exe' @($Model, "$t", '-1', '256', "$Budget")
    $cpu = (Select-String $l -Pattern '^\[SEARCH-HASH\] cpu (.*)$' | % { $_.Matches[0].Groups[1].Value }) -join ';'
    $gpu = (Select-String $l -Pattern '^\[SEARCH-HASH\] cuda (.*)$' | % { $_.Matches[0].Groups[1].Value }) -join ';'
    if (-not $cpu -or $cpu -ne $gpu) { $fail += "s6_t$t-cpu-vs-cuda" }
    $hashes[$t] = $cpu
}
if (($hashes.Values | Select-Object -Unique).Count -ne 1) { $fail += 's6-thread-hash' }
"search gate hashes: $($hashes[1])" | Tee-Object -FilePath (Join-Path $Out 'log.txt') -Append
if ($fail.Count) { "[PCHIE GATE] search gate: FAIL ($($fail -join ', '))" } else { '[PCHIE GATE] search gate: PASS' }
if ($Repeat -gt 0) {
    foreach ($s in @(@('cuda','8'), @('cpu','16'), @('cpu','8'), @('cpu','2'), @('cpu','1'))) {
        Run "rep_$($s[0])_t$($s[1])" 'pchie_search_gate.exe' @($Model, $s[1], '-1', '256', "$Budget", 'repeat', "$Repeat", $s[0]) | Out-Null
    }
    if ($fail.Count) { "[PCHIE GATE] search gate + repeat: FAIL ($($fail -join ', '))" } else { '[PCHIE GATE] search gate + repeat: PASS' }
}
'ALL DONE' | Tee-Object -FilePath (Join-Path $Out 'log.txt') -Append
