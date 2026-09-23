$ErrorActionPreference = 'Stop'
Set-Location D:\dipcatcher
$py = '.venv\Scripts\python.exe'
$outdir = '.dsh-24x7\lane-stack2\cols'
$shards = @(
  'd1_bnbusdt_1d_deep.v2aug.npz',
  'd1_ethusdt_1d_deep.v2aug.npz',
  'd1_solusdt_1d_deep.v2aug.npz',
  'd1_xrpusdt_1d_deep.v2aug.npz'
)
foreach ($s in $shards) {
  $out = Join-Path $outdir ("stack2_" + $s)
  Write-Host ("RUN " + $s + " " + (Get-Date -Format o))
  & $py scripts\_stack2_col.py --shard (Join-Path '.dsh-24x7\eval-full' $s) --bars-root data\raw\sources --out $out
  if ($LASTEXITCODE -ne 0) { Write-Host ("FAIL " + $s); exit 1 }
}
Write-Host 'D1REST DONE'
