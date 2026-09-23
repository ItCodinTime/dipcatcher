Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'sota_eval' } |
  ForEach-Object { Write-Output ("PID " + $_.ProcessId + ": " + $_.CommandLine); Write-Output "---" }
