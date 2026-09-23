# Respawn daily + 4h eval jobs WITHOUT kronos_base (it doubles per-origin cost and
# is already covered by the scoped proven eval). 300 origins, all other targets.
$ErrorActionPreference = "Continue"
$outdir = "D:\dipcatcher\.dsh-24x7\eval-full"
New-Item -ItemType Directory -Force $outdir | Out-Null

$targets = "--kronos-repo third_party\kronos " +
  "--kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-2k " +
  "--chronos2 chronos2=data\models\chronos-2 " +
  "--bolt bolt_small=data\models\chronos-bolt-small " +
  "--timesfm timesfm=data\models\timesfm-2.5-200m-pytorch"
$common = "--origins 300 --samples 16 --seed 7 --n-boot 2000 --torch-threads 1 --checkpoint-every 50"

$jobs = @()
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
foreach ($f in $daily) { $jobs += ,@($f, "d1", $common, $targets) }

$intraday = @(
  "data\raw\sources\btcusdt_4h_deep.parquet",
  "data\raw\sources\ethusdt_4h_deep.parquet",
  "data\raw\sources\solusdt_4h_deep.parquet",
  "data\raw\sources\bnbusdt_4h.parquet",
  "data\raw\sources\xrpusdt_4h.parquet"
)
foreach ($f in $intraday) { $jobs += ,@($f, "h4", $common, $targets) }

foreach ($j in $jobs) {
  $barfile = $j[0]; $tag = $j[1]; $extra = $j[2]; $tgt = $j[3]
  $name = ($barfile -split '\\')[-1] -replace '\.parquet$',''
  $out = "$outdir\${tag}_${name}.json"
  $log = "$outdir\${tag}_${name}.stdout.log"
  $elog = "$outdir\${tag}_${name}.stderr.log"
  $inner = "cd /d D:\dipcatcher && set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set TOKENIZERS_PARALLELISM=false&& .venv\Scripts\python.exe scripts\sota_eval_kronos.py --bars $barfile $tgt $extra --out $out 1> `"$log`" 2> `"$elog`""
  $cmdline = "cmd /c `"$inner`""
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmdline
    CurrentDirectory = 'D:\dipcatcher'
  }
  Write-Output ("{0} {1} -> rc={2} pid={3}" -f $tag, $name, $r.ReturnValue, $r.ProcessId)
}
Write-Output "SPAWNED $($jobs.Count) jobs"
