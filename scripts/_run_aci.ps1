Set-Location D:\dipcatcher
$py = "D:\dipcatcher\.venv\Scripts\python.exe"
$ev = "D:\dipcatcher\.dsh-24x7\eval-full"
$out = "D:\dipcatcher\.dsh-24x7\lane-aci"
New-Item -ItemType Directory -Force $out | Out-Null
$shards = @(
  "d1_bnbusdt_1d_deep.v2aug.npz",
  "d1_btcusdt_1d_deep.v2aug.npz",
  "d1_ethusdt_1d_deep.v2aug.npz",
  "d1_solusdt_1d_deep.v2aug.npz",
  "d1_xrpusdt_1d_deep.v2aug.npz",
  "h4f_bnbusdt_4h_deep.v2aug.npz",
  "h4f_btcusdt_4h_deep.v2aug.npz",
  "h4f_ethusdt_4h_deep.v2aug.npz",
  "h4f_solusdt_4h_deep.v2aug.npz",
  "h4f_xrpusdt_4h_deep.v2aug.npz"
)
foreach ($s in $shards) {
  $col = Join-Path $out ($s -replace '\.v2aug\.npz$', '.acicol.npz')
  if (Test-Path $col) { Write-Host "skip $s"; continue }
  $t = Get-Date
  & $py scripts\_aci_col.py --shard (Join-Path $ev $s) --bars-root data\raw\sources --out $col 2>&1
  Write-Host ("  took {0:n1}s" -f ((Get-Date) - $t).TotalSeconds)
}
Write-Host "ACI DONE"
