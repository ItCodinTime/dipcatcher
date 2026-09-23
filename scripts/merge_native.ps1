# Merge native-protocol shards per frequency with protocol guards.
# Daily: nd_* (main, 15 models) + ns_* (supplement, 4 new challengers)
# 4h:    nh_* (main) + ns4_* (supplement)
Set-Location D:\dipcatcher
$py = "D:\evalenv\Scripts\python.exe"
$nat = "D:\dipcatcher\.dsh-24x7\native"
$ev = "D:\dipcatcher\.dsh-24x7"

$d1 = Get-ChildItem "$nat\nd_*.paths.npz","$nat\ns_*.paths.npz" | ForEach-Object { $_.FullName }
$h4 = Get-ChildItem "$nat\nh_*.paths.npz","$nat\ns4_*.paths.npz" | ForEach-Object { $_.FullName }
Write-Host "d1 parts: $($d1.Count) | h4 parts: $($h4.Count)"

& $py scripts\sota_eval_native.py --merge-parts $d1 --merge-out "$ev\evidence-sota-native-d1.json" --expect-freq 1d --expect-lookback 40 --expect-horizon 12
& $py scripts\sota_eval_native.py --merge-parts $h4 --merge-out "$ev\evidence-sota-native-h4.json" --expect-freq 4h --expect-lookback 90 --expect-horizon 18
