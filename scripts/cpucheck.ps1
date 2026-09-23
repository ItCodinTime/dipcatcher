$p1 = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'sota_eval' }
$m1 = $p1 | Select-Object ProcessId, KernelModeTime, UserModeTime
Start-Sleep -Seconds 30
$p2 = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'sota_eval' }
foreach ($a in $m1) {
  $b = $p2 | Where-Object { $_.ProcessId -eq $a.ProcessId }
  if ($b) {
    $dt = ($b.KernelModeTime + $b.UserModeTime - $a.KernelModeTime - $a.UserModeTime) / 1e7
    Write-Output ("pid " + $a.ProcessId + " cpu+ " + [math]::Round($dt,1) + "s/30s")
  }
}
