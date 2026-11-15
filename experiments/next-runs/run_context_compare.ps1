# Context size comparison: the same B1 run at ctx 1024 and ctx 2048, side by side.
# This is the test path for the 2048 context. Today it measures the text observation at both sizes
# (memory, speed, walk); once the ASCII observation lands, the same launcher compares the two view
# layouts in pchie/research/ContextBudget.h. Both runs start in parallel and share the threads.
# CPU only. Requires PCHIE_MODEL or PCHIE_MODELS (see common.ps1 and experiments/README.md).
# -Exe points at another build's pchie_rl_l1l2_test.exe (default: this clone's Release build).
param([string]$Root = '', [int]$Steps = 25, [int]$Budget = 8, [int]$Workers = 0, [string]$LoopGuard = 'on',
      [string]$Exe = '', [string[]]$Extra = @(), [switch]$DryRun)
. (Join-Path $PSScriptRoot 'common.ps1')
if ($Exe) { $RlExe = $Exe }
if (-not $Root) { $Root = Join-Path $Repo 'runs\ctx' }
Assert-Ready @($RlExe, $Model)
Assert-LoopGuard $LoopGuard
$tag = "s${Steps}b${Budget}"
$common = @('--maze', 'b1', '--steps', "$Steps", '--budget', "$Budget", '--loop-guard', $LoopGuard) + $Extra
$w = Get-Workers $Workers 2
if (-not $DryRun) { New-RunRoot $Root }
$st = Join-Path $Root 'logs\_status_ctx.txt'
Write-Host 'Context comparison runs:'
$runs = @()
foreach ($ctx in @(1024, 2048)) {
  $label = "b1_ctx${ctx}_$tag"
  $runs += , @($label, (Start-RlRun $Root $label ($common + @('--ctx', "$ctx")) $w -DryRun:$DryRun))
}
if ($DryRun) { Write-Host 'Dry run: nothing started.'; exit 0 }
Write-Status $st "context comparison START $tag workers=$w root=$Root"
$runs | % { $_[1].WaitForExit() }
Write-Status $st ('context comparison END exits=' + (($runs | % { $_[1].ExitCode }) -join ','))
foreach ($r in $runs) {
  $log = Join-Path $Root "logs\$($r[0]).log"
  Write-Host "== $($r[0]) (exit $($r[1].ExitCode))"
  Get-Content $log -Tail 6 | % { Write-Host "   $_" }
}
