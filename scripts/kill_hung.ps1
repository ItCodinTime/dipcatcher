# Kill hung sota_eval workers (0-CPU session-0 children) and the wave-2 orchestrator.
$victims = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'sota_eval_kronos' }
foreach ($p in $victims) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output "killed py $($p.ProcessId)" }
$orch = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'run_sota_fleet' }
foreach ($p in $orch) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output "killed orch $($p.ProcessId)" }
