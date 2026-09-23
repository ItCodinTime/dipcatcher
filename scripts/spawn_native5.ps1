$outdir = "D:\dipcatcher\.dsh-24x7\native"
# Challenger-only supplement for the 4 models added after the main fleet
# spawned. Same origins are reproduced deterministically (lookback/window/
# test-start driven, seed-independent), so these merge onto the main shards.
$common = "--no-targets --challengers dip_skt,dip_qar,dip_conf_t,dip_regime --no-lgbm " +
  "--origins 300 --samples 8 --seed 7 --torch-threads 1 --checkpoint-every 50"
$jobs = @(
  @("ns_btcusdt_1d_deep", "data\raw\sources\btcusdt_1d_deep.parquet", 40, 12),
  @("ns_ethusdt_1d_deep", "data\raw\sources\ethusdt_1d_deep.parquet", 40, 12),
  @("ns_solusdt_1d_deep", "data\raw\sources\solusdt_1d_deep.parquet", 40, 12),
  @("ns_bnbusdt_1d_deep", "data\raw\sources\bnbusdt_1d_deep.parquet", 40, 12),
  @("ns_xrpusdt_1d_deep", "data\raw\sources\xrpusdt_1d_deep.parquet", 40, 12),
  @("ns_adausdt_1d", "data\raw\sources\adausdt_1d.parquet", 40, 12),
  @("ns_avaxusdt_1d", "data\raw\sources\avaxusdt_1d.parquet", 40, 12),
  @("ns_dogeusdt_1d", "data\raw\sources\dogeusdt_1d.parquet", 40, 12),
  @("ns_linkusdt_1d", "data\raw\sources\linkusdt_1d.parquet", 40, 12),
  @("ns_ltcusdt_1d", "data\raw\sources\ltcusdt_1d.parquet", 40, 12),
  @("ns_trxusdt_1d", "data\raw\sources\trxusdt_1d.parquet", 40, 12),
  @("ns4_btcusdt_4h_deep", "data\raw\sources\btcusdt_4h_deep.parquet", 90, 18),
  @("ns4_ethusdt_4h_deep", "data\raw\sources\ethusdt_4h_deep.parquet", 90, 18),
  @("ns4_solusdt_4h_deep", "data\raw\sources\solusdt_4h_deep.parquet", 90, 18),
  @("ns4_bnbusdt_4h_deep", "data\raw\sources\bnbusdt_4h_deep.parquet", 90, 18),
  @("ns4_xrpusdt_4h_deep", "data\raw\sources\xrpusdt_4h_deep.parquet", 90, 18)
)
foreach ($j in $jobs) {
  $name = $j[0]; $bars = $j[1]; $lb = $j[2]; $hz = $j[3]
  $inner = "cd /d D:\dipcatcher && set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set TOKENIZERS_PARALLELISM=false&& D:\evalenv\Scripts\pythonw.exe -u scripts\sota_eval_native.py --bars $bars --lookback $lb --horizon $hz $common --out $outdir\$name.json 1> `"$outdir\$name.stdout.log`" 2> `"$outdir\$name.stderr.log`""
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd /c `"$inner`""
    CurrentDirectory = 'D:\dipcatcher'
  }
  $t = Get-Date -Format 'HH:mm:ss'
  Add-Content C:\Users\me\spawn_native5.out.log "$name -> rc=$($r.ReturnValue) pid=$($r.ProcessId) ($t)"
  Start-Sleep -Seconds 10
}
Add-Content C:\Users\me\spawn_native5.out.log "spawner done"
