$outdir = "D:\dipcatcher\.dsh-24x7\native"
$targets = "--kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-base " +
  "--chronos2 chronos2=data\models\chronos-2 " +
  "--bolt bolt_small=data\models\chronos-bolt-small " +
  "--timesfm timesfm=data\models\timesfm-2.5-200m-pytorch"
$common = "--origins 300 --samples 8 --seed 7 --torch-threads 1 --checkpoint-every 50"
$jobs = @(
  @("nd_btcusdt_1d_deep", "data\raw\sources\btcusdt_1d_deep.parquet", 40, 12),
  @("nd_ethusdt_1d_deep", "data\raw\sources\ethusdt_1d_deep.parquet", 40, 12),
  @("nd_solusdt_1d_deep", "data\raw\sources\solusdt_1d_deep.parquet", 40, 12),
  @("nd_bnbusdt_1d_deep", "data\raw\sources\bnbusdt_1d_deep.parquet", 40, 12),
  @("nd_xrpusdt_1d_deep", "data\raw\sources\xrpusdt_1d_deep.parquet", 40, 12),
  @("nd_adausdt_1d", "data\raw\sources\adausdt_1d.parquet", 40, 12),
  @("nd_avaxusdt_1d", "data\raw\sources\avaxusdt_1d.parquet", 40, 12),
  @("nd_dogeusdt_1d", "data\raw\sources\dogeusdt_1d.parquet", 40, 12),
  @("nd_linkusdt_1d", "data\raw\sources\linkusdt_1d.parquet", 40, 12),
  @("nd_ltcusdt_1d", "data\raw\sources\ltcusdt_1d.parquet", 40, 12),
  @("nd_trxusdt_1d", "data\raw\sources\trxusdt_1d.parquet", 40, 12),
  @("nh_btcusdt_4h_deep", "data\raw\sources\btcusdt_4h_deep.parquet", 90, 18),
  @("nh_ethusdt_4h_deep", "data\raw\sources\ethusdt_4h_deep.parquet", 90, 18),
  @("nh_solusdt_4h_deep", "data\raw\sources\solusdt_4h_deep.parquet", 90, 18),
  @("nh_bnbusdt_4h_deep", "data\raw\sources\bnbusdt_4h_deep.parquet", 90, 18),
  @("nh_xrpusdt_4h_deep", "data\raw\sources\xrpusdt_4h_deep.parquet", 90, 18)
)
foreach ($j in $jobs) {
  $name = $j[0]; $bars = $j[1]; $lb = $j[2]; $hz = $j[3]
  $inner = "cd /d D:\dipcatcher && set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set TOKENIZERS_PARALLELISM=false&& D:\evalenv\Scripts\pythonw.exe -u scripts\sota_eval_native.py --bars $bars --lookback $lb --horizon $hz --kronos-repo third_party\kronos $targets $common --out $outdir\$name.json 1> `"$outdir\$name.stdout.log`" 2> `"$outdir\$name.stderr.log`""
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd /c `"$inner`""
    CurrentDirectory = 'D:\dipcatcher'
  }
  $t = Get-Date -Format 'HH:mm:ss'
  Add-Content C:\Users\me\spawn_native4.out.log "$name -> rc=$($r.ReturnValue) pid=$($r.ProcessId) ($t)"
  Start-Sleep -Seconds 75
}
Add-Content C:\Users\me\spawn_native4.out.log "spawner done"
