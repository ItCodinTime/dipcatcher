"alive $(Get-Date -Format o)" | Out-File D:\dipcatcher\.dsh-24x7\eval-full\mini2_alive.txt
Start-Sleep -Seconds 25
"still-alive $(Get-Date -Format o)" | Out-File D:\dipcatcher\.dsh-24x7\eval-full\mini2_alive.txt -Append
