$s1 = @{}
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { $s1[$_.Id] = $_.CPU }
Start-Sleep -Seconds 20
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
  if ($s1.ContainsKey($_.Id) -and ($_.CPU - $s1[$_.Id]) -gt 0.5) {
    $c = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $_.Id)
    Write-Output ("BUSY pid " + $_.Id + " +" + [math]::Round($_.CPU - $s1[$_.Id],1) + "s : " + $c.CommandLine.Substring(0,[Math]::Min(200,$c.CommandLine.Length)))
  }
}
