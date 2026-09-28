# fleet_manifest_sota.ps1 — emit the canonical SOTA fleet as a fleet_spawn
# manifest. This is the same job list spawn_sota_all.ps1 built inline
# (11 daily + 5 intraday books, plus the s11/s23 seed-replication cells),
# rendered to JSON so the parametrized launcher + watchdog can drive it.
#
# Usage:
#   powershell -NoProfile -File scripts\fleet_manifest_sota.ps1 -Out jobs.json
[CmdletBinding()]
param([string]$Out = ".dsh-24x7\fleet_sota_manifest.json")

$ErrorActionPreference = "Stop"

$targets = "--kronos-repo third_party\kronos " +
  "--kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-2k " +
  "--kronos kronos_base=data\models\Kronos-base,data\models\Kronos-Tokenizer-base " +
  "--chronos2 chronos2=data\models\chronos-2 " +
  "--bolt bolt_small=data\models\chronos-bolt-small " +
  "--timesfm timesfm=data\models\timesfm-2.5-200m-pytorch"
$kronosOnly = "--kronos-repo third_party\kronos " +
  "--kronos kronos_small=data\models\Kronos-small,data\models\Kronos-Tokenizer-2k"
$common = "--origins 400 --samples 16 --seed 7 --n-boot 2000 --torch-threads 1 --checkpoint-every 50"
$env = [ordered]@{ OMP_NUM_THREADS = "1"; MKL_NUM_THREADS = "1"; TOKENIZERS_PARALLELISM = "false" }
$outdir = ".dsh-24x7\eval-full"

$jobs = @()

$daily = @(
  "btcusdt_1d_deep", "ethusdt_1d_deep", "solusdt_1d_deep", "bnbusdt_1d_deep",
  "xrpusdt_1d_deep", "adausdt_1d", "avaxusdt_1d", "dogeusdt_1d",
  "linkusdt_1d", "ltcusdt_1d", "trxusdt_1d"
)
foreach ($n in $daily) {
    $jobs += [ordered]@{
        name    = "d1_$n"
        command = ".venv\Scripts\python.exe scripts\sota_eval_kronos.py --bars data\raw\sources\$n.parquet $targets $common --out $outdir\d1_$n.json"
        env     = $env
        out     = "$outdir\d1_$n.json"
    }
}

$intraday = @(
  "btcusdt_4h_deep", "ethusdt_4h_deep", "solusdt_4h_deep", "bnbusdt_4h", "xrpusdt_4h"
)
foreach ($n in $intraday) {
    $jobs += [ordered]@{
        name    = "h4_$n"
        command = ".venv\Scripts\python.exe scripts\sota_eval_kronos.py --bars data\raw\sources\$n.parquet $targets $common --out $outdir\h4_$n.json"
        env     = $env
        out     = "$outdir\h4_$n.json"
    }
}

foreach ($seed in @(11, 23)) {
    foreach ($n in @("btcusdt_1d_deep", "ethusdt_1d_deep", "solusdt_1d_deep")) {
        $seedargs = "--origins 200 --samples 16 --seed $seed --n-boot 2000 --torch-threads 1 --checkpoint-every 50"
        $jobs += [ordered]@{
            name    = "seed${seed}_$n"
            command = ".venv\Scripts\python.exe scripts\sota_eval_kronos.py --bars data\raw\sources\$n.parquet $kronosOnly $seedargs --out $outdir\seed${seed}_$n.json"
            env     = $env
            out     = "$outdir\seed${seed}_$n.json"
        }
    }
}

[ordered]@{
    workdir = "D:\dipcatcher"
    logroot = ".dsh-24x7\fleet-logs"
    jobs    = $jobs
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $Out

Write-Output "manifest=$Out jobs=$($jobs.Count)"
