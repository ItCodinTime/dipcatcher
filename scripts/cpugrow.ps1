$s1 = @{}
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { $s1[$_.Id] = $_.CPU }
Start-Sleep -Seconds 45
$total = 0.0; $n = 0
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
  if ($s1.ContainsKey($_.Id)) {
    $d = $_.CPU - $s1[$_.Id]
    if ($d -gt 0.5) { $n++; $total += $d }
  }
}
Write-Output ("python procs burning CPU: " + $n + ", total cpu-sec in 45s: " + [math]::Round($total,1))
