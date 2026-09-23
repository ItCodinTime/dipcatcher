$ErrorActionPreference = 'Stop'
Set-Location D:\dipcatcher
$py = '.venv\Scripts\python.exe'
$outdir = '.dsh-24x7\lane-stack2\cols'
$shards = @(
  'h4f_bnbusdt_4h_deep.v2aug.npz',
  'h4f_btcusdt_4h_deep.v2aug.npz',
  'h4f_ethusdt_4h_deep.v2aug.npz',
  'h4f_solusdt_4h_deep.v2aug.npz',
  'h4f_xrpusdt_4h_deep.v2aug.npz'
)
foreach ($s in $shards) {
  $out = Join-Path $outdir ("stack2_" + $s)
  Write-Host ("RUN " + $s + " " + (Get-Date -Format o))
  & $py scripts\_stack2_col.py --shard (Join-Path '.dsh-24x7\eval-full' $s) --bars-root data\raw\sources --out $out
  if ($LASTEXITCODE -ne 0) { Write-Host ("FAIL " + $s); exit 1 }
}
Write-Host 'H4F DONE'
