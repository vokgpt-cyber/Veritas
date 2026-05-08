param(
    [int[]]$Ports = @(8765, 5173),
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "SilentlyContinue"
$blocked = @()
$seen = @{}
$projectRootLower = $ProjectRoot.ToLowerInvariant()

foreach ($port in $Ports) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen
    foreach ($listener in $listeners) {
        $owner = [int]$listener.OwningProcess
        if ($seen.ContainsKey($owner)) {
            continue
        }
        $seen[$owner] = $true

        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$owner"
        if (-not $proc) {
            continue
        }

        $cmd = [string]$proc.CommandLine
        $cmdLower = $cmd.ToLowerInvariant()
        $exe = [string]$proc.ExecutablePath

        $isLocalVeritas =
            $cmdLower.Contains($projectRootLower) -or
            $cmdLower.Contains("backend.app.main:app") -or
            ($cmdLower.Contains("vite") -and $cmdLower.Contains("veritas"))

        if ($isLocalVeritas) {
            Write-Host "[CLEANUP] Stopping old VERITAS process on port $port (PID $owner)"
            Stop-Process -Id $owner -Force
        } else {
            $blocked += [pscustomobject]@{
                Port = $port
                PID = $owner
                Executable = $exe
                CommandLine = $cmd
            }
        }
    }
}

Start-Sleep -Seconds 2

foreach ($port in $Ports) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen
    foreach ($listener in $listeners) {
        $owner = [int]$listener.OwningProcess
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$owner"
        $cmd = if ($proc) { [string]$proc.CommandLine } else { "" }
        $cmdLower = $cmd.ToLowerInvariant()
        $isLocalVeritas =
            $cmdLower.Contains($projectRootLower) -or
            $cmdLower.Contains("backend.app.main:app") -or
            ($cmdLower.Contains("vite") -and $cmdLower.Contains("veritas"))
        if (-not $isLocalVeritas) {
            $blocked += [pscustomobject]@{
                Port = $port
                PID = $owner
                Executable = if ($proc) { [string]$proc.ExecutablePath } else { "" }
                CommandLine = $cmd
            }
        }
    }
}

if ($blocked.Count -gt 0) {
    Write-Host "[ERROR] The following ports are used by non-VERITAS processes:"
    $blocked | Sort-Object Port, PID -Unique | Format-Table -AutoSize
    exit 4
}

exit 0
