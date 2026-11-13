# Stage B standard pair: one run of the standard Stage B harness with real Little Nemo, which is itself the pair
# (real run1 + real run2, checked for run-to-run determinism), after the unit section. Headless.
# -Watch opens a log-tail console that shows without taking focus and is never topmost.
# Uses all available threads by default (-Workers N to choose; results do not depend on it). Loop guard v2 is on by
# default; -LoopGuard off gives the old walk.
# CPU only. The official Stage B stall record is not touched. Expect about an hour.
# Requires PCHIE_MODEL or PCHIE_MODELS (see common.ps1 and experiments/README.md).
param([string]$OutDir = '', [int]$Workers = 0, [string]$LoopGuard = 'on', [switch]$Watch, [switch]$DryRun)
. (Join-Path $PSScriptRoot 'common.ps1')
if (-not $OutDir) { $OutDir = Join-Path $Repo 'runs\stageb' }
Assert-Ready @($SbExe, $Model)
$log = Join-Path $OutDir 'stageb_std.log'; $st = Join-Path $OutDir '_status_stageb.txt'
Assert-LoopGuard $LoopGuard
$sbArgs = @('--workers', "$(Get-Workers $Workers 1)", '--loop-guard', $LoopGuard)
Write-Host ("Stage B pair:`n  {0} {1}  > {2}" -f $SbExe, ($sbArgs -join ' '), $log)
if ($Watch) { Write-Host ("Log view:`n  wscript //B //Nologo {0} {1}" -f (Join-Path $PSScriptRoot 'view_log.vbs'), $log) }
if ($DryRun) { Write-Host 'Dry run: nothing started.'; exit 0 }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Assert-NewLabel $log
Remove-Item "$log.done" -ErrorAction SilentlyContinue
Write-Status $st "Stage B pair START $($sbArgs -join ' ')"
$p = Start-Process -FilePath $SbExe -ArgumentList $sbArgs -WorkingDirectory $OutDir -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden -PassThru
$null = $p.Handle
if ($Watch) { Start-Process wscript.exe -ArgumentList @('//B', '//Nologo', (Join-Path $PSScriptRoot 'view_log.vbs'), $log) }
$p.WaitForExit(); Set-Content "$log.done" "exit=$($p.ExitCode)"
Write-Status $st "Stage B pair END exit=$($p.ExitCode)"
Select-String -Path $log -Pattern 'real Little Nemo run[12]|workers=|loop guard:|honest result|run-to-run deterministic|\[STAGE-B\] (PASS|FAIL) \(' | % { $_.Line }
exit $p.ExitCode
