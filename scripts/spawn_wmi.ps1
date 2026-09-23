$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'powershell -NoProfile -ExecutionPolicy Bypass -File D:\dipcatcher\scripts\mini2.ps1'
  CurrentDirectory = 'D:\dipcatcher'
}
"spawn rc=$($r.ReturnValue) pid=$($r.ProcessId)"
