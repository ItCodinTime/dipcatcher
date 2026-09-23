$victims = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'sota_eval_kronos' -and $_.CommandLine -match 'eval-full\\(d1|h4)_' }
foreach ($p in $victims) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output "killed $($p.ProcessId)" }
# also kill their cmd parents so nothing respawns
$cmds = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -match 'eval-full\\(d1|h4)_' }
foreach ($p in $cmds) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output "killed cmd $($p.ProcessId)" }
