# SOTA eval wave 2: 4h assets + Kronos seed robustness. Waits for wave 1 to drain.
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
  Write-Host "LAUNCH $tag $name -> $out"
  $p = Start-Process -FilePath $py -ArgumentList $argstr -WorkingDirectory "D:\dipcatcher" `
    -RedirectStandardOutput $log -RedirectStandardError $elog `
    -WindowStyle Hidden -PassThru
  return ,$p
}

Write-Output "wave2 waiting for wave1 evals to drain ($(Get-Date -Format o))"
do {
  Start-Sleep -Seconds 120
  $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'sota_eval_kronos' })
  Write-Output ("  " + (Get-Date -Format HH:mm:ss) + " evals still running: " + $running.Count)
} while ($running.Count -gt 0)

Write-Output "=== WAVE 2 launch ($(Get-Date -Format o)) ==="
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
$w2 | Where-Object { $_ -is [System.Diagnostics.Process] } | Wait-Process
Write-Output "=== WAVE 2 done ($(Get-Date -Format o)) ==="
Write-Output "FLEET COMPLETE"
