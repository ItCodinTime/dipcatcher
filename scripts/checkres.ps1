Get-ChildItem D:\dipcatcher\.dsh-24x7\eval-full\wmi_* -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
Get-Content D:\dipcatcher\.dsh-24x7\eval-full\wmi_py.log -ErrorAction SilentlyContinue
Get-Content D:\dipcatcher\.dsh-24x7\eval-full\wmi_python_ok.txt -ErrorAction SilentlyContinue
