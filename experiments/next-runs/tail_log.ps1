# Tails one log until <log>.done exists. Read-only; used by view_log.vbs.
param([Parameter(Mandatory = $true)][string]$Log)
$host.UI.RawUI.WindowTitle = "PCHIE log view: $Log"
$n = 0
while ($true) {
  if (Test-Path $Log) { $l = @(Get-Content $Log); if ($l.Count -gt $n) { $l[$n..($l.Count - 1)]; $n = $l.Count } }
  if (Test-Path "$Log.done") { "--- finished: $(Get-Content "$Log.done")"; break }
  Start-Sleep -Seconds 2
}
