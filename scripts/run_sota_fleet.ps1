# SOTA evaluation fleet launcher — runs detached on D:\dipcatcher.
# Wave 1: 11 daily assets x all targets x all challengers (400 origins).
# Wave 2: 5 4h assets + Kronos seed-robustness (seeds 11, 23) on btc/eth/sol.
$ErrorActionPreference = "Continue"
Set-Location D:\dipcatcher
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$py = "D:\dipcatcher\.venv\Scripts\python.exe"
$script = "scripts\sota_eval_kronos.py"
$outdir = ".dsh-24x7\eval-full"
New-Item -ItemType Directory -Force $outdir | Out-Null

$targets = "--kronos-repo third_party\kronos " +
  "--kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-2k " +
  "--kronos kronos_base=data\models\Kronos-base,data\models\Kronos-Tokenizer-base " +
  "--chronos2 chronos2=data\models\chronos-2 " +
  "--bolt bolt_small=data\models\chronos-bolt-small " +
  "--timesfm timesfm=data\models\timesfm-2.5-200m-pytorch"
$kronos_only = "--kronos-repo third_party\kronos " +
  "--kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-2k"
$common = "--origins 400 --samples 16 --seed 7 --n-boot 2000 --torch-threads 1 --checkpoint-every 50"

$daily = @(
  "data\raw\sources\btcusdt_1d_deep.parquet",
  "data\raw\sources\ethusdt_1d_deep.parquet",
  "data\raw\sources\solusdt_1d_deep.parquet",
  "data\raw\sources\bnbusdt_1d_deep.parquet",
  "data\raw\sources\xrpusdt_1d_deep.parquet",
  "data\raw\sources\adausdt_1d.parquet",
  "data\raw\sources\avaxusdt_1d.parquet",
  "data\raw\sources\dogeusdt_1d.parquet",
  "data\raw\sources\linkusdt_1d.parquet",
  "data\raw\sources\ltcusdt_1d.parquet",
  "data\raw\sources\trxusdt_1d.parquet"
)
$intraday = @(
  "data\raw\sources\btcusdt_4h_deep.parquet",
  "data\raw\sources\ethusdt_4h_deep.parquet",
  "data\raw\sources\solusdt_4h_deep.parquet",
  "data\raw\sources\bnbusdt_4h.parquet",
  "data\raw\sources\xrpusdt_4h.parquet"
)

function Start-Eval($barfile, $tag, $extra, $tgt) {
  $name = ($barfile -split '\\')[-1] -replace '\.parquet$',''
  $out = "$outdir\${tag}_${name}.json"
  $log = "$outdir\${tag}_${name}.stdout.log"
  $elog = "$outdir\${tag}_${name}.stderr.log"
  $argstr = "$script --bars $barfile $tgt $extra --out $out"
  Write-Output "LAUNCH $tag $name"
  Start-Process -FilePath $py -ArgumentList $argstr -WorkingDirectory "D:\dipcatcher" `
    -RedirectStandardOutput $log -RedirectStandardError $elog `
    -WindowStyle Hidden -PassThru
}

Write-Output "=== WAVE 1: daily fleet ($(Get-Date -Format o)) ==="
$w1 = @()
foreach ($f in $daily) { $w1 += Start-Eval $f "d1" $common $targets }
Write-Output "wave1 pids: $($w1.Id -join ',')"
$w1 | Wait-Process
Write-Output "=== WAVE 1 done ($(Get-Date -Format o)) ==="

Write-Output "=== WAVE 2: 4h + kronos seed robustness ($(Get-Date -Format o)) ==="
$w2 = @()
foreach ($f in $intraday) { $w2 += Start-Eval $f "h4" $common $targets }
foreach ($seed in @(11, 23)) {
  foreach ($f in @("data\raw\sources\btcusdt_1d_deep.parquet",
                   "data\raw\sources\ethusdt_1d_deep.parquet",
                   "data\raw\sources\solusdt_1d_deep.parquet")) {
    $seedargs = "--origins 200 --samples 16 --seed $seed --n-boot 2000 --torch-threads 1 --checkpoint-every 50"
    $w2 += Start-Eval $f "seed${seed}" $seedargs $kronos_only
  }
}
Write-Output "wave2 pids: $($w2.Id -join ',')"
$w2 | Wait-Process
Write-Output "=== WAVE 2 done ($(Get-Date -Format o)) ==="
Write-Output "FLEET COMPLETE"
