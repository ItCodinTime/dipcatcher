$inner = 'cd /d D:\dipcatcher && .venv\Scripts\python.exe -u scripts\sota_eval_kronos.py --bars data\raw\sources\adausdt_1d.parquet --kronos-repo third_party\kronos --kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-2k --chronos2 chronos2=data\models\chronos-2 --bolt bolt_small=data\models\chronos-bolt-small --timesfm timesfm=data\models\timesfm-2.5-200m-pytorch --origins 6 --samples 4 --seed 5 --torch-threads 1 --out .dsh-24x7\eval-full\probe_one.json 1> .dsh-24x7\eval-full\probe_one.log 2> .dsh-24x7\eval-full\probe_one.err.log'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = "cmd /c `"$inner`""
  CurrentDirectory = 'D:\dipcatcher'
}
"rc=$($r.ReturnValue) pid=$($r.ProcessId)"
