# Shared helpers for the experiment launchers. CPU only, no GPU, no game window.
# Official build and records are left untouched.
# Set PCHIE_MODEL to the GGUF weights file, or PCHIE_MODELS to the directory that
# contains Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf.
$ErrorActionPreference = 'Stop'
$Repo  = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if ($env:PCHIE_MODEL) {
  $Model = $env:PCHIE_MODEL
} elseif ($env:PCHIE_MODELS) {
  $Model = Join-Path $env:PCHIE_MODELS 'Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf'
} else {
  $Model = $null
}
$RlExe = Join-Path $Repo 'build\Release\pchie_rl_l1l2_test.exe'
$SbExe = Join-Path $Repo 'build\Release\pchie_stageb_door_test.exe'
# Thread policy: every run uses all available hardware threads by default (detected, never hard-coded).
# When runs go in parallel the threads are shared out so the total stays at the machine's count.
# Results do not depend on the thread count (join dispatcher), so this only changes speed.
$Threads = [Math]::Max(1, [Environment]::ProcessorCount)
$MaxWorkers = 16   # JoinJobDispatcher::kMaxWorkers
function Get-Workers([int]$Requested, [int]$Parallel = 1) {
  if ($Requested -gt 0) { return [Math]::Min($Requested, $MaxWorkers) }
  return [Math]::Max(1, [Math]::Min($MaxWorkers, [Math]::Floor($Threads / [Math]::Max(1, $Parallel))))
}
function Assert-LoopGuard([string]$LoopGuard) {
  if ($LoopGuard -ne 'on' -and $LoopGuard -ne 'off') { throw "-LoopGuard takes on or off (on is the default; off reproduces the old walks)" }
}

function Write-Status([string]$File, [string]$Msg) {
  $line = '{0} ET  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Msg
  Add-Content -Path $File -Value $line; Write-Host $line
}
function Assert-Ready([string[]]$Files) {
  if (-not $Model) {
    throw "Set PCHIE_MODEL to your GGUF weights, or set PCHIE_MODELS to the directory that holds Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf."
  }
  foreach ($f in $Files) { if (-not $f -or -not (Test-Path $f)) { throw "missing: $f (build with: cmake --build build --config Release -j $Threads)" } }
}
function Assert-NewLabel([string]$Log) {
  if (Test-Path $Log) { throw "a log for this label already exists: $Log. Use a fresh -Root, or move the old one aside (VOID_<reason>_<file>)." }
}
# One harness run in the background, output to <Root>\logs\<label>.log. Returns the process.
function Start-RlRun([string]$Root, [string]$Label, [string[]]$RunArgs, [int]$Workers, [switch]$DryRun) {
  $log = Join-Path $Root "logs\$Label.log"
  $all = @('--root', ($Root -replace '\\', '/'), '--label', $Label, '--workers', "$Workers") + $RunArgs
  Write-Host ("  {0} {1}  > {2}" -f $RlExe, ($all -join ' '), $log)
  if ($DryRun) { return $null }
  Assert-NewLabel $log
  $p = Start-Process -FilePath $RlExe -ArgumentList $all -WorkingDirectory $Root -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden -PassThru
  $null = $p.Handle   # keep the handle so ExitCode is readable after the wait
  $p
}
function New-RunRoot([string]$Root) {
  foreach ($d in @('logs', 'scratch\rl\mem')) { New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null }
}
