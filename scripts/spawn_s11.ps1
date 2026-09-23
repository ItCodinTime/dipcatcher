# Detached spawner: corrected-pairing seed-11 daily-deep replication fleet.
# Kronos-small + Kronos-Tokenizer-base (canonical), seed 11, 300 origins,
# 5 deep daily assets. Challenger slate matches v4/d1fix for mergeability.
$outdir = "D:\dipcatcher\.dsh-24x7\eval-full"
$targets = "--kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-base " +
  "--chronos2 chronos2=data\models\chronos-2 " +
  "--bolt bolt_small=data\models\chronos-bolt-small " +
  "--timesfm timesfm=data\models\timesfm-2.5-200m-pytorch"
$common = "--origins 300 --samples 16 --seed 11 --n-boot 2000 --torch-threads 1 --checkpoint-every 50"
$jobs = @(
  @("btcusdt_1d_deep", "data\raw\sources\btcusdt_1d_deep.parquet"),
  @("ethusdt_1d_deep", "data\raw\sources\ethusdt_1d_deep.parquet"),
  @("solusdt_1d_deep", "data\raw\sources\solusdt_1d_deep.parquet"),
  @("bnbusdt_1d_deep", "data\raw\sources\bnbusdt_1d_deep.parquet"),
  @("xrpusdt_1d_deep", "data\raw\sources\xrpusdt_1d_deep.parquet")
)
foreach ($j in $jobs) {
  $name = "s11_" + $j[0]
  $bars = $j[1]
  $inner = "cd /d D:\dipcatcher && set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set TOKENIZERS_PARALLELISM=false&& D:\evalenv\Scripts\pythonw.exe -u scripts\sota_eval_kronos.py --bars $bars --kronos-repo third_party\kronos $targets $common --out $outdir\$name.json 1> `"$outdir\$name.stdout.log`" 2> `"$outdir\$name.stderr.log`""
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd /c `"$inner`""
    CurrentDirectory = 'D:\dipcatcher'
  }
  $t = Get-Date -Format 'HH:mm:ss'
  Add-Content C:\Users\me\spawn_s11.out.log "$name -> rc=$($r.ReturnValue) pid=$($r.ProcessId) ($t)"
  Start-Sleep -Seconds 90
}
Add-Content C:\Users\me\spawn_s11.out.log "spawner done"
