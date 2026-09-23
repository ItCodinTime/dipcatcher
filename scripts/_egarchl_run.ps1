$ErrorActionPreference = 'Stop'
Set-Location D:\dipcatcher
$py = '.venv\Scripts\python.exe'
$outdir = '.dsh-24x7\lane-egarchl'
New-Item -ItemType Directory -Force -Path $outdir | Out-Null
$shards = @(
  'd1_bnbusdt_1d_deep.v2aug.npz',
  'd1_btcusdt_1d_deep.v2aug.npz',
  'd1_ethusdt_1d_deep.v2aug.npz',
  'd1_solusdt_1d_deep.v2aug.npz',
  'd1_xrpusdt_1d_deep.v2aug.npz',
  'h4f_bnbusdt_4h_deep.v2aug.npz',
  'h4f_btcusdt_4h_deep.v2aug.npz',
  'h4f_ethusdt_4h_deep.v2aug.npz',
  'h4f_solusdt_4h_deep.v2aug.npz',
  'h4f_xrpusdt_4h_deep.v2aug.npz'
)
foreach ($s in $shards) {
  $out = Join-Path $outdir ("egarchl_" + $s)
  Write-Host ("RUN " + $s)
  & $py scripts\_egarchl_col.py --shard (Join-Path '.dsh-24x7\eval-full' $s) --bars-root data\raw\sources --out $out
  if ($LASTEXITCODE -ne 0) { Write-Host ("FAIL " + $s); exit 1 }
}
Write-Host 'ALL DONE'
