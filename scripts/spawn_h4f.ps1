# h4f fleet: full-coverage 4h rerun with hardened _fit_student_t (Phase A1).
# Replaces the complete-case h4_* shards: 5 deep 4h assets x 300 origins,
# canonical Kronos-Tokenizer-base pairing, seed 7.
$ErrorActionPreference = "Continue"
$outdir = "D:\dipcatcher\.dsh-24x7\eval-full"
New-Item -ItemType Directory -Force $outdir | Out-Null

$targets = "--kronos-repo third_party\kronos " +
  "--kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-base " +
  "--chronos2 chronos2=data\models\chronos-2 " +
  "--bolt bolt_small=data\models\chronos-bolt-small " +
  "--timesfm timesfm=data\models\timesfm-2.5-200m-pytorch"
$common = "--origins 300 --samples 16 --seed 7 --n-boot 2000 --torch-threads 1 --checkpoint-every 50"

$bars = @(
  "data\raw\sources\btcusdt_4h_deep.parquet",
  "data\raw\sources\ethusdt_4h_deep.parquet",
  "data\raw\sources\solusdt_4h_deep.parquet",
  "data\raw\sources\bnbusdt_4h_deep.parquet",
  "data\raw\sources\xrpusdt_4h_deep.parquet"
)

$i = 0
foreach ($barfile in $bars) {
  $name = ($barfile -split '\\')[-1] -replace '\.parquet$',''
  $out = "$outdir\h4f_${name}.json"
  $log = "$outdir\h4f_${name}.stdout.log"
  $elog = "$outdir\h4f_${name}.stderr.log"
  $inner = "cd /d D:\dipcatcher && set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set TOKENIZERS_PARALLELISM=false&& D:\evalenv\Scripts\pythonw.exe -u scripts\sota_eval_kronos.py --bars $barfile $targets $common --out $out 1> `"$log`" 2> `"$elog`""
  $cmdline = "cmd /c `"$inner`""
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmdline
    CurrentDirectory = 'D:\dipcatcher'
  }
  Write-Output ("h4f {0} -> rc={1} pid={2} ({3})" -f $name, $r.ReturnValue, $r.ProcessId, (Get-Date -Format HH:mm:ss))
  $i++
  if ($i -lt $bars.Count) { Start-Sleep -Seconds 90 }
}
Write-Output "SPAWNED $i jobs"
