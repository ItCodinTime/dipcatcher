$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd /c "cd /d D:\dipcatcher && .venv\Scripts\python.exe -c "import time,math; x=0.0; [x:=x+math.sqrt(i) for i in range(3000000)]; open(r".dsh-24x7\eval-full\wmi_python_ok.txt","w").write(str(x))" > .dsh-24x7\eval-full\wmi_py.log 2>&1"'
  CurrentDirectory = 'D:\dipcatcher'
}
"rc=$($r.ReturnValue) pid=$($r.ProcessId)"
