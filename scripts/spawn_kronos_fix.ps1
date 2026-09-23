$ErrorActionPreference = "Continue"
$outdir = "D:\dipcatcher\.dsh-24x7\eval-full"
$kronos_tok = "--kronos-repo third_party\kronos --kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-base"
$common = "--origins 300 --samples 16 --seed 7 --n-boot 2000 --torch-threads 1 --checkpoint-every 50"
$files = @(
  @("data\raw\sources\btcusdt_1d_deep.parquet","d1fix"),
  @("data\raw\sources\ethusdt_1d_deep.parquet","d1fix"),
  @("data\raw\sources\solusdt_1d_deep.parquet","d1fix"),
  @("data\raw\sources\bnbusdt_1d_deep.parquet","d1fix"),
  @("data\raw\sources\xrpusdt_1d_deep.parquet","d1fix"),
  @("data\raw\sources\adausdt_1d.parquet","d1fix"),
  @("data\raw\sources\avaxusdt_1d.parquet","d1fix"),
  @("data\raw\sources\dogeusdt_1d.parquet","d1fix"),
  @("data\raw\sources\linkusdt_1d.parquet","d1fix"),
  @("data\raw\sources\ltcusdt_1d.parquet","d1fix"),
  @("data\raw\sources\trxusdt_1d.parquet","d1fix"),
  @("data\raw\sources\btcusdt_4h_deep.parquet","h4fix"),
  @("data\raw\sources\ethusdt_4h_deep.parquet","h4fix"),
  @("data\raw\sources\solusdt_4h_deep.parquet","h4fix"),
  @("data\raw\sources\bnbusdt_4h.parquet","h4fix"),
  @("data\raw\sources\xrpusdt_4h.parquet","h4fix")
)
foreach ($j in $files) {
  $barfile = $j[0]; $tag = $j[1]
  $name = ($barfile -split '\\')[-1] -replace '\.parquet$',''
  $out = "$outdir\${tag}_${name}.json"
  $log = "$outdir\${tag}_${name}.stdout.log"
  $elog = "$outdir\${tag}_${name}.stderr.log"
  $inner = "cd /d D:\dipcatcher && set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\Scripts\python.exe -u scripts\sota_eval_kronos.py --bars $barfile $kronos_tok $common --out $out 1> `"$log`" 2> `"$elog`""
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd /c `"$inner`""; CurrentDirectory = 'D:\dipcatcher' }
  Write-Output ("{0} {1} -> rc={2} pid={3}" -f $tag, $name, $r.ReturnValue, $r.ProcessId)
  Start-Sleep -Seconds 15
}
Write-Output "SPAWNED $($files.Count) fix jobs"
