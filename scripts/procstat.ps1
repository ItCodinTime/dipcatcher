$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'sota_eval' }
foreach ($p in $procs) {
  $tot = ($p.KernelModeTime + $p.UserModeTime)/1e7
  $age = (Get-Date) - $p.CreationDate
  $threads = (Get-Process -Id $p.ProcessId).Threads.Count
  Write-Output ("pid {0} age {1:N1}min cpu_total {2:N1}s threads {3}" -f $p.ProcessId, $age.TotalMinutes, $tot, $threads)
}
