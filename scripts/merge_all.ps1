Set-Location D:\dipcatcher
$py = "D:\dipcatcher\.venv\Scripts\python.exe"
$dir = ".dsh-24x7\eval-full"

$d1 = (Get-ChildItem "$dir\d1_*.fixed.npz").FullName
& $py scripts\sota_eval_kronos.py --merge-parts @d1 --merge-out "$dir\d1_merged.json" --seed 7 --n-boot 2000 2>&1 | Select-Object -Last 45
Write-Output "=========== H4 ==========="
$h4 = (Get-ChildItem "$dir\h4_*.cc.npz").FullName
& $py scripts\sota_eval_kronos.py --merge-parts @h4 --merge-out "$dir\h4_merged.json" --seed 7 --n-boot 2000 2>&1 | Select-Object -Last 45
Write-Output "MERGE DONE"
