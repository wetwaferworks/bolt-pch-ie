# Query-before-plan on the local side chain only. B1, 25 steps, budget 8 (the reference setting).
# Seeds the store from the official oracle B1 walk (under <Root>\scratch\rl\mem\ only), then runs L1 with the seeded store,
# null with the seeded store, and L1 with an empty store, in parallel. Hits are proposals through the gate, never a bypass,
# and the store is read-only during runs (qbp_write=0). The comparer reports engagement and whether hits changed the walk.
# Loop guard v2 is on by default; -LoopGuard off gives the old walks. Threads: the seed walk gets all of them, the
# three runs share them (a third each). -Workers N sets a per-run count instead.
# CPU only. Expect about 30 minutes on an idle machine.
# Requires PCHIE_MODEL or PCHIE_MODELS (see common.ps1 and experiments/README.md).
param([string]$Root = '', [int]$Steps = 25, [int]$Budget = 8, [int]$Workers = 0, [string]$LoopGuard = 'on', [switch]$DryRun)
. (Join-Path $PSScriptRoot 'common.ps1')
if (-not $Root) { $Root = Join-Path $Repo 'runs\rl' }
Assert-Ready @($RlExe, $Model)
$mem = ($Root -replace '\\', '/') + '/scratch/rl/mem'; $seed = "$mem/b1_seed_official.jsonl"; $empty = "$mem/b1_empty_control.jsonl"
$tag = "s${Steps}b${Budget}"; $S = "b1_L1_qbp_seeded_$tag"; $N = "b1_null_qbp_seeded_$tag"; $E = "b1_L1_qbp_empty_$tag"
Assert-LoopGuard $LoopGuard
$common = @('--maze', 'b1', '--steps', "$Steps", '--budget', "$Budget", '--rl-qbp', '--loop-guard', $LoopGuard)
$wAll = Get-Workers $Workers 1; $wTrio = Get-Workers $Workers 3
if (-not $DryRun) { New-RunRoot $Root }
$st = Join-Path $Root 'logs\_status_qbp.txt'
Write-Host 'Query-before-plan seed:'
$seedLog = Join-Path $Root 'logs\qbp_seed.log'
$seedArgs = @('--root', ($Root -replace '\\', '/'), '--seed-mem', $seed, '--label', 'qbp_seed', '--workers', "$wAll")
Write-Host ("  {0} {1}  > {2}" -f $RlExe, ($seedArgs -join ' '), $seedLog)
if (-not $DryRun) {
  Write-Status $st "query-before-plan START $tag threads=$Threads seed_workers=$wAll run_workers=$wTrio loop_guard=$LoopGuard root=$Root"
  Assert-NewLabel $seedLog
  $p = Start-Process -FilePath $RlExe -ArgumentList $seedArgs -WorkingDirectory $Root -RedirectStandardOutput $seedLog -RedirectStandardError "$seedLog.err" -WindowStyle Hidden -PassThru
  $null = $p.Handle
  $p.WaitForExit(); Write-Status $st "seed END exit=$($p.ExitCode)"
  if ($p.ExitCode -ne 0) { Write-Status $st 'query-before-plan STOPPED: the seed walk did not produce a verified store'; exit 2 }
  Remove-Item ($empty -replace '/', '\') -ErrorAction SilentlyContinue   # the control store must start empty
}
Write-Host 'Query-before-plan runs:'
$q = @( (Start-RlRun $Root $S ($common + @('--rl-puct-model', '--mem', $seed)) $wTrio -DryRun:$DryRun),
        (Start-RlRun $Root $N ($common + @('--rl-puct-prior=model', '--rl-puct-value=null', '--mem', $seed)) $wTrio -DryRun:$DryRun),
        (Start-RlRun $Root $E ($common + @('--rl-puct-model', '--mem', $empty)) $wTrio -DryRun:$DryRun) )
$logs = @($S, $E, $N) | % { Join-Path $Root "logs\$_.log" }
Write-Host ("Comparer:`n  {0} --compare-l2 {1}" -f $RlExe, ($logs -join ' '))
if ($DryRun) { Write-Host 'Dry run: nothing started.'; exit 0 }
$q | % { $_.WaitForExit() }; Write-Status $st ('runs END exits=' + (($q | % { $_.ExitCode }) -join ','))
$verdict = Join-Path $Root "logs\qbp_verdict_$tag.txt"
& $RlExe --compare-l2 @logs | Tee-Object -FilePath $verdict
$code = $LASTEXITCODE
Write-Status $st "query-before-plan verdict exit=$code (0 comparable, 2 invalid); see $verdict"
exit $code
