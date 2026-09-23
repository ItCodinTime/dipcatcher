$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'powershell -NoProfile -ExecutionPolicy Bypass -File D:\dipcatcher\scripts\run_sota_fleet_wave2.ps1 > D:\dipcatcher\.dsh-24x7\eval-full\fleet_wave2.log 2> D:\dipcatcher\.dsh-24x7\eval-full\fleet_wave2.err.log'
  CurrentDirectory = 'D:\dipcatcher'
}
"spawn rc=$($r.ReturnValue) pid=$($r.ProcessId)"
