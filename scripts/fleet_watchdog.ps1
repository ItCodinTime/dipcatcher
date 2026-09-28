# fleet_watchdog.ps1 — heartbeat + auto-respawn for manifest-spawned fleet jobs.
#
# Pair with fleet_spawn.ps1: the watchdog reads the same manifest, and for each
# job checks (a) whether a spawned process is still alive by PID command line,
# and (b) whether the job's expected output file appeared. Dead jobs are
# respawned through the same Win32_Process path (bounded by -MaxRespawns).
# Every poll writes a heartbeat file with per-job liveness so operators (and
# merge tooling) can see fleet state without a remote desktop session.
#
# Usage:
#   # one pass (for Scheduled Tasks):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fleet_watchdog.ps1 -Manifest jobs.json -Once
#   # continuous supervision:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fleet_watchdog.ps1 -Manifest jobs.json -IntervalSeconds 60
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [string]$LogRoot = ".dsh-24x7\fleet-logs",
    [string]$Heartbeat = ".dsh-24x7\fleet_heartbeat.json",
    [int]$IntervalSeconds = 60,
    [int]$StaleMinutes = 15,
    [int]$MaxRespawns = 3,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Read-Manifest {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "manifest not found: $Path"
    }
    $m = Get-Content -Raw $Path | ConvertFrom-Json
    if ($null -eq $m.jobs -or $m.jobs.Count -eq 0) { throw "manifest has no jobs" }
    if (-not $m.workdir) { throw "manifest requires 'workdir'" }
    return $m
}

function Get-JobProcess {
    # Match a live process whose command line contains the job's marker —
    # the job name appears in its log paths, making it a stable handle.
    param([string]$JobName, [string]$LogRootAbs)
    $needle = ($JobName + ".stdout.log")
    return Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($needle) } |
        Select-Object -First 1
}

function Invoke-Respawn {
    param($Job, [string]$Workdir, [string]$LogRootAbs)
    $log = Join-Path $LogRootAbs "$($Job.name).stdout.log"
    $elog = Join-Path $LogRootAbs "$($Job.name).stderr.log"
    $env = ""
    if ($Job.env) {
        foreach ($k in $Job.env.PSObject.Properties.Name) {
            $env += "set $k=$($Job.env.$k)&& "
        }
    }
    $inner = "cd /d $Workdir && ${env}$($Job.command) 1> `"$log`" 2> `"$elog`""
    $cmdline = "cmd /c `"$inner`""
    return Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $Workdir
    }
}

$manifest = Read-Manifest -Path $Manifest
$workdir = $manifest.workdir
if (-not $PSBoundParameters.ContainsKey('LogRoot') -and $manifest.logroot) {
    $LogRoot = $manifest.logroot
}
$logrootAbs = Join-Path $workdir $LogRoot
$heartbeatAbs = Join-Path $workdir $Heartbeat
New-Item -ItemType Directory -Force $logrootAbs | Out-Null
New-Item -ItemType Directory -Force (Split-Path $heartbeatAbs) | Out-Null

$respawns = @{}
foreach ($j in $manifest.jobs) { $respawns[$j.name] = 0 }

while ($true) {
    $now = Get-Date
    $states = @()
    foreach ($job in $manifest.jobs) {
        $proc = Get-JobProcess -JobName $job.name -LogRootAbs $logrootAbs
        $alive = $null -ne $proc
        $outExists = $job.out -and (Test-Path (Join-Path $workdir $job.out))
        $outStale = $false
        $lastWrite = $null
        if ($outExists) {
            $lastWrite = (Get-Item (Join-Path $workdir $job.out)).LastWriteTime
            $outStale = (($now - $lastWrite).TotalMinutes -gt $StaleMinutes)
        }
        $action = "none"
        # Done = output file exists and no live process; respawn only when the
        # process is dead AND (no output OR output still expected/stale).
        if (-not $alive -and -not $outExists -and $respawns[$job.name] -lt $MaxRespawns) {
            $r = Invoke-Respawn -Job $job -Workdir $workdir -LogRootAbs $logrootAbs
            $respawns[$job.name] += 1
            $action = "respawned pid=$($r.ProcessId) rc=$($r.ReturnValue) n=$($respawns[$job.name])"
            if ($r.ReturnValue -ne 0) { $action = "respawn_failed rc=$($r.ReturnValue)" }
        } elseif (-not $alive -and -not $outExists) {
            $action = "dead_exhausted_respawns"
        }
        $states += [ordered]@{
            name            = $job.name
            alive           = $alive
            pid             = if ($proc) { $proc.ProcessId } else { $null }
            out_exists      = [bool]$outExists
            out_stale       = $outStale
            out_last_write  = if ($lastWrite) { $lastWrite.ToString("o") } else { $null }
            respawn_count   = $respawns[$job.name]
            action          = $action
        }
    }
    [ordered]@{
        checked_at = $now.ToString("o")
        manifest   = (Resolve-Path $Manifest).Path
        jobs       = $states
    } | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $heartbeatAbs

    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
