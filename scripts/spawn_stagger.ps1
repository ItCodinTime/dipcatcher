$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File D:\dipcatcher\scripts\spawn_staggered.ps1 > D:\dipcatcher\.dsh-24x7\eval-full\spawner.log 2>&1"'
  CurrentDirectory = 'D:\dipcatcher'
}
"rc=$($r.ReturnValue) pid=$($r.ProcessId)"
