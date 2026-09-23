# Merge per-asset shards into pooled receipts + print pooled CRPS/MCS tables.
$ErrorActionPreference = "Continue"
Set-Location D:\dipcatcher
$py = "D:\dipcatcher\.venv\Scripts\python.exe"
$dir = ".dsh-24x7\eval-full"

function Merge-Group($pattern, $outName) {
  $parts = Get-ChildItem "$dir\$pattern" -Filter *.losses.npz |
    Where-Object { $_.Name -notmatch 'merged|smoke|probe' }
  if ($parts.Count -eq 0) { Write-Output "no parts for $pattern"; return }
  Write-Output "== merging $($parts.Count) parts -> $outName"
  & $py scripts\sota_eval_kronos.py --merge-parts $parts.FullName --merge-out "$dir\$outName" --seed 7 --n-boot 2000 2>&1 |
    Select-Object -Last 40
}

Merge-Group "d1_*.losses.npz" "d1_merged.json"
Merge-Group "h4_*.losses.npz" "h4_merged.json"
Merge-Group "seed11_*.losses.npz" "seed11_merged.json"
Merge-Group "seed23_*.losses.npz" "seed23_merged.json"
Write-Output "DONE"
