Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object ProcessId, CreationDate, @{n="Cmd";e={$_.CommandLine.Substring(0,[Math]::Min(160,$_.CommandLine.Length))}} |
  Format-List
Write-Output "---eval-full---"
Get-ChildItem D:\dipcatcher\.dsh-24x7\eval-full | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
