$victims = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'sota_eval_kronos' }
foreach ($p in $victims) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output "killed py $($p.ProcessId)" }
$cmds = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -match 'sota_eval_kronos' }
foreach ($p in $cmds) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output "killed cmd $($p.ProcessId)" }
$pwsh = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'sota_fleet|sota_eval' }
foreach ($p in $pwsh) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output "killed ps $($p.ProcessId)" }
