# fleet_spawn.ps1 — parametrized WMI launcher for the remote fleet.
#
# Replaces the per-fleet spawn_*.ps1 one-offs: every job is described in a JSON
# manifest (scripts/fleet_manifest.json or a generated file), then each entry
# is spawned as a WMI-owned process (survives ssh session teardown) wrapped in
# cmd /c for stdout/stderr redirection — the same pattern every spawn_* script
# used, expressed once.
#
# Manifest schema:
#   {
#     "workdir": "D:\\dipcatcher",                    # spawn working directory
#     "jobs": [
#       {
#         "name": "d1_btcusdt",                       # unique job id
#         "command": ".venv\\Scripts\\python.exe scripts\\sota_eval_kronos.py --bars ... --out ...",
#         "env": {"OMP_NUM_THREADS": "1", ...},       # optional per-job env
#         "out": ".dsh-24x7\\eval-full\\d1_btcusdt.json"  # optional expected output (watchdog signal)
#       }
#     ]
#   }
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fleet_spawn.ps1 -Manifest jobs.json [-DryRun]
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [string]$LogRoot = ".dsh-24x7\fleet-logs",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Read-Manifest {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "manifest not found: $Path"
    }
    $m = Get-Content -Raw $Path | ConvertFrom-Json
    if ($null -eq $m.jobs -or $m.jobs.Count -eq 0) {
        throw "manifest has no jobs"
    }
    if (-not $m.workdir) {
        throw "manifest requires 'workdir'"
    }
    $seen = @{}
    foreach ($j in $m.jobs) {
        if (-not $j.name -or -not $j.command) {
            throw "every job needs 'name' and 'command'"
        }
        if ($seen.ContainsKey($j.name)) {
            throw "duplicate job name: $($j.name)"
        }
        $seen[$j.name] = $true
        if ($j.command -match "[&|;]") {
            # cmd /c wrappers make shell metacharacters dangerous; the manifest
            # is trusted input but the guard keeps accidents out of the fleet.
            throw "job '$($j.name)' command contains shell metacharacters"
        }
    }
    return $m
}

function New-JobCommandLine {
    param($Job, [string]$Workdir, [string]$LogRoot)
    $name = $Job.name
    $log = Join-Path $LogRoot "$name.stdout.log"
    $elog = Join-Path $LogRoot "$name.stderr.log"
    $env = ""
    if ($Job.env) {
        foreach ($k in $Job.env.PSObject.Properties.Name) {
            $env += "set $k=$($Job.env.$k)&& "
        }
    }
    $inner = "cd /d $Workdir && ${env}$($Job.command) 1> `"$log`" 2> `"$elog`""
    return "cmd /c `"$inner`""
}

$manifest = Read-Manifest -Path $Manifest
$workdir = $manifest.workdir
if (-not $PSBoundParameters.ContainsKey('LogRoot') -and $manifest.logroot) {
    $LogRoot = $manifest.logroot
}
$logrootAbs = Join-Path $workdir $LogRoot

if ($DryRun) {
    Write-Output "DRY-RUN: $($manifest.jobs.Count) jobs from $Manifest"
}

New-Item -ItemType Directory -Force $logrootAbs | Out-Null

$spawned = @()
foreach ($job in $manifest.jobs) {
    $cmdline = New-JobCommandLine -Job $job -Workdir $workdir -LogRoot $logrootAbs
    if ($DryRun) {
        Write-Output "  $($job.name): $cmdline"
        continue
    }
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $workdir
    }
    $entry = [ordered]@{
        name = $job.name
        pid  = $r.ProcessId
        rc   = $r.ReturnValue
        out  = $job.out
    }
    $spawned += $entry
    Write-Output ("{0} rc={1} pid={2}" -f $job.name, $r.ReturnValue, $r.ProcessId)
    if ($r.ReturnValue -ne 0) {
        throw "spawn failed for $($job.name): Win32_Process rc=$($r.ReturnValue)"
    }
}

if (-not $DryRun) {
    $receiptPath = Join-Path $logrootAbs "spawn_receipt.json"
    [ordered]@{
        spawned_at = (Get-Date).ToString("o")
        manifest   = (Resolve-Path $Manifest).Path
        workdir    = $workdir
        jobs       = $spawned
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $receiptPath
    Write-Output "SPAWNED $($spawned.Count) jobs; receipt=$receiptPath"
}
