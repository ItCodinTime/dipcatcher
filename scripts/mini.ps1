"D:\dipcatcher\.dsh-24x7\eval-full\mini_alive.txt" | Out-Null
"alive $(Get-Date -Format o)" | Out-File D:\dipcatcher\.dsh-24x7\eval-full\mini_alive.txt
Start-Sleep -Seconds 30
"still-alive $(Get-Date -Format o)" | Out-File D:\dipcatcher\.dsh-24x7\eval-full\mini_alive.txt -Append
