# Planner budget comparison: L1 model-driven PUCT at budget 16 against null and oracle at the same budget, B1, 60 steps.
# Oracle runs first (seconds), then L1 and null in parallel. Then the comparer applies the bar: L1 must beat null.
# If it does not, the result is STOP AND REPORT (exit 10): no higher-budget run until the maintainer has seen it.
# Loop guard v2 is on by default (the 12 November L1/null 2-cell bounce should now break); -LoopGuard off gives the old walks.
# Threads: the oracle gets all of them; L1 and null share them (half each). -Workers N sets a per-run count instead.
# CPU only. Expect about Query-before-plan to 2.5 hours on an idle machine.
# Requires PCHIE_MODEL or PCHIE_MODELS (see common.ps1 and experiments/README.md).
param([string]$Root = '', [int]$Steps = 60, [int]$Budget = 16, [int]$Workers = 0, [string]$LoopGuard = 'on', [switch]$DryRun)
. (Join-Path $PSScriptRoot 'common.ps1')
if (-not $Root) { $Root = Join-Path $Repo 'runs\rl' }
Assert-Ready @($RlExe, $Model)
$tag = "s${Steps}b${Budget}"; $L1 = "b1_L1_$tag"; $NU = "b1_null_$tag"; $OR = "b1_oracle_$tag"
Assert-LoopGuard $LoopGuard
$common = @('--maze', 'b1', '--steps', "$Steps", '--budget', "$Budget", '--loop-guard', $LoopGuard)
$wAll = Get-Workers $Workers 1; $wPair = Get-Workers $Workers 2
if (-not $DryRun) { New-RunRoot $Root }
$st = Join-Path $Root 'logs\_status_planner.txt'
if (-not $DryRun) { Write-Status $st "planner comparison START $tag threads=$Threads oracle_workers=$wAll pair_workers=$wPair loop_guard=$LoopGuard root=$Root" }
Write-Host 'Planner comparison runs:'
$o = Start-RlRun $Root $OR $common $wAll -DryRun:$DryRun
if ($o) { $o.WaitForExit(); Write-Status $st "oracle END exit=$($o.ExitCode)" }
$a = Start-RlRun $Root $L1 ($common + @('--rl-puct-model')) $wPair -DryRun:$DryRun
$b = Start-RlRun $Root $NU ($common + @('--rl-puct-prior=model', '--rl-puct-value=null')) $wPair -DryRun:$DryRun
$logs = @($L1, $NU, $OR) | % { Join-Path $Root "logs\$_.log" }
Write-Host ("Comparer:`n  {0} --compare-l1 {1}" -f $RlExe, ($logs -join ' '))
if ($DryRun) { Write-Host 'Dry run: nothing started.'; exit 0 }
Write-Status $st "L1 pid=$($a.Id) null pid=$($b.Id) running"
$a.WaitForExit(); Write-Status $st "L1 END exit=$($a.ExitCode)"
$b.WaitForExit(); Write-Status $st "null END exit=$($b.ExitCode)"
$verdict = Join-Path $Root "logs\planner_verdict_$tag.txt"
& $RlExe --compare-l1 @logs | Tee-Object -FilePath $verdict
$code = $LASTEXITCODE
Write-Status $st "planner comparison verdict exit=$code (0 separated, 10 stop and report, 2 invalid); see $verdict"
exit $code
