# Detached spawner: tfmfix4 = contract-v2 TimesFM reruns for the 6 non-deep
# daily assets at 300 origins, matching the d1fix_* base shards so the
# corrected column can be spliced in for an 11-asset verified daily arena.
$outdir = "D:\dipcatcher\.dsh-24x7\eval-full"
$tfm = "--timesfm timesfm=data\models\timesfm-2.5-200m-pytorch"
$jobs = @(
  @("tfmfix4_d1_adausdt_1d", "data\raw\sources\adausdt_1d.parquet"),
  @("tfmfix4_d1_avaxusdt_1d", "data\raw\sources\avaxusdt_1d.parquet"),
  @("tfmfix4_d1_dogeusdt_1d", "data\raw\sources\dogeusdt_1d.parquet"),
  @("tfmfix4_d1_linkusdt_1d", "data\raw\sources\linkusdt_1d.parquet"),
  @("tfmfix4_d1_ltcusdt_1d", "data\raw\sources\ltcusdt_1d.parquet"),
  @("tfmfix4_d1_trxusdt_1d", "data\raw\sources\trxusdt_1d.parquet")
)
foreach ($j in $jobs) {
  $name = $j[0]; $bars = $j[1]
  $common = "--origins 300 --samples 16 --seed 7 --n-boot 2000 --torch-threads 1 --checkpoint-every 50"
  $inner = "cd /d D:\dipcatcher && set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set TOKENIZERS_PARALLELISM=false&& D:\evalenv\Scripts\pythonw.exe -u scripts\sota_eval_kronos.py --bars $bars $tfm $common --out $outdir\$name.json 1> `"$outdir\$name.stdout.log`" 2> `"$outdir\$name.stderr.log`""
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd /c `"$inner`""
    CurrentDirectory = 'D:\dipcatcher'
  }
  $t = Get-Date -Format 'HH:mm:ss'
  Add-Content C:\Users\me\spawn_tfmfix4.out.log "$name -> rc=$($r.ReturnValue) pid=$($r.ProcessId) ($t)"
  Start-Sleep -Seconds 45
}
Add-Content C:\Users\me\spawn_tfmfix4.out.log "spawner done"
