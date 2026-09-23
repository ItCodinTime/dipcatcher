# Detached spawner: corrected TimesFM (contract v2) shards for every legacy cell.
# Each job runs timesfm-only under the cell's original protocol so the column
# can be spliced into the legacy matrices (see splice_timesfm_fix.py).
$outdir = "D:\dipcatcher\.dsh-24x7\eval-full"
$tfm = "--timesfm timesfm=data\models\timesfm-2.5-200m-pytorch"
$jobs = @(
  # v3 daily cells: 150 origins, standard bars
  @("tfmfix_d1_adausdt_1d", "data\raw\sources\adausdt_1d.parquet", 150),
  @("tfmfix_d1_avaxusdt_1d", "data\raw\sources\avaxusdt_1d.parquet", 150),
  @("tfmfix_d1_bnbusdt_1d", "data\raw\sources\bnbusdt_1d.parquet", 150),
  @("tfmfix_d1_btcusdt_1d", "data\raw\sources\btcusdt_1d.parquet", 150),
  @("tfmfix_d1_dogeusdt_1d", "data\raw\sources\dogeusdt_1d.parquet", 150),
  @("tfmfix_d1_ethusdt_1d", "data\raw\sources\ethusdt_1d.parquet", 150),
  @("tfmfix_d1_linkusdt_1d", "data\raw\sources\linkusdt_1d.parquet", 150),
  @("tfmfix_d1_ltcusdt_1d", "data\raw\sources\ltcusdt_1d.parquet", 150),
  @("tfmfix_d1_solusdt_1d", "data\raw\sources\solusdt_1d.parquet", 150),
  @("tfmfix_d1_trxusdt_1d", "data\raw\sources\trxusdt_1d.parquet", 150),
  @("tfmfix_d1_xrpusdt_1d", "data\raw\sources\xrpusdt_1d.parquet", 150),
  # v3 h4 cells: 150 origins, standard 4h bars
  @("tfmfix_h4_bnbusdt_4h", "data\raw\sources\bnbusdt_4h.parquet", 150),
  @("tfmfix_h4_btcusdt_4h", "data\raw\sources\btcusdt_4h.parquet", 150),
  @("tfmfix_h4_ethusdt_4h", "data\raw\sources\ethusdt_4h.parquet", 150),
  @("tfmfix_h4_solusdt_4h", "data\raw\sources\solusdt_4h.parquet", 150),
  @("tfmfix_h4_xrpusdt_4h", "data\raw\sources\xrpusdt_4h.parquet", 150),
  # v4 deep daily cells: 300 origins
  @("tfmfix_d1_btcusdt_1d_deep", "data\raw\sources\btcusdt_1d_deep.parquet", 300),
  @("tfmfix_d1_ethusdt_1d_deep", "data\raw\sources\ethusdt_1d_deep.parquet", 300),
  @("tfmfix_d1_solusdt_1d_deep", "data\raw\sources\solusdt_1d_deep.parquet", 300),
  @("tfmfix_d1_bnbusdt_1d_deep", "data\raw\sources\bnbusdt_1d_deep.parquet", 300),
  @("tfmfix_d1_xrpusdt_1d_deep", "data\raw\sources\xrpusdt_1d_deep.parquet", 300),
  # v4 deep h4 cells: 300 origins
  @("tfmfix_h4_btcusdt_4h_deep", "data\raw\sources\btcusdt_4h_deep.parquet", 300),
  @("tfmfix_h4_ethusdt_4h_deep", "data\raw\sources\ethusdt_4h_deep.parquet", 300),
  @("tfmfix_h4_solusdt_4h_deep", "data\raw\sources\solusdt_4h_deep.parquet", 300),
  # h4f full-coverage cells: 300 origins, deep 4h bars
  @("tfmfix_h4f_btcusdt_4h_deep", "data\raw\sources\btcusdt_4h_deep.parquet", 300),
  @("tfmfix_h4f_ethusdt_4h_deep", "data\raw\sources\ethusdt_4h_deep.parquet", 300),
  @("tfmfix_h4f_solusdt_4h_deep", "data\raw\sources\solusdt_4h_deep.parquet", 300),
  @("tfmfix_h4f_bnbusdt_4h_deep", "data\raw\sources\bnbusdt_4h_deep.parquet", 300),
  @("tfmfix_h4f_xrpusdt_4h_deep", "data\raw\sources\xrpusdt_4h_deep.parquet", 300)
)
foreach ($j in $jobs) {
  $name = $j[0]; $bars = $j[1]; $origins = $j[2]
  $common = "--origins $origins --samples 16 --seed 7 --n-boot 2000 --torch-threads 1 --checkpoint-every 50"
  $inner = "cd /d D:\dipcatcher && set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set TOKENIZERS_PARALLELISM=false&& D:\evalenv\Scripts\pythonw.exe -u scripts\sota_eval_kronos.py --bars $bars $tfm $common --out $outdir\$name.json 1> `"$outdir\$name.stdout.log`" 2> `"$outdir\$name.stderr.log`""
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd /c `"$inner`""
    CurrentDirectory = 'D:\dipcatcher'
  }
  $t = Get-Date -Format 'HH:mm:ss'
  Add-Content C:\Users\me\spawn_tfmfix.out.log "$name -> rc=$($r.ReturnValue) pid=$($r.ProcessId) ($t)"
  Start-Sleep -Seconds 45
}
Add-Content C:\Users\me\spawn_tfmfix.out.log "spawner done"
