param(
    [ValidateSet("Menu", "Start", "Stop", "Restart", "Status", "Open")]
    [string]$Action = "Menu"
)

$ErrorActionPreference = "Stop"
$Workspace = if ($PSScriptRoot) {
    $PSScriptRoot
}
elseif ($env:NOVEL_AGENTS_WORKSPACE) {
    $env:NOVEL_AGENTS_WORKSPACE.TrimEnd("\", "/")
}
else {
    (Get-Location).Path
}
$HostAddress = "127.0.0.1"
$Port = 43127
$Url = "http://${HostAddress}:${Port}/"
$HealthUrl = "${Url}api/health"
$RuntimeDir = Join-Path $Workspace ".novel_agents"
$PidFile = Join-Path $RuntimeDir "web-43127.pid"

function Get-PythonWindowless {
    $fixed = "D:\python\pythonw.exe"
    if (Test-Path -LiteralPath $fixed) {
        return $fixed
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        $candidate = Join-Path (Split-Path $python.Source) "pythonw.exe"
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    throw "pythonw.exe was not found. Install Python or update the configured path."
}

function Test-NovelAgentsServer {
    try {
        $result = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
        return $result.ok -eq $true
    }
    catch {
        return $false
    }
}

function Get-ListenerPid {
    $pattern = "^\s*TCP\s+${HostAddress}:$Port\s+\S+\s+LISTENING\s+(\d+)\s*$"
    foreach ($line in (& netstat -ano -p tcp 2>$null)) {
        if ($line -match $pattern) {
            return [int]$Matches[1]
        }
    }
    return $null
}

function Save-Pid([int]$ProcessId) {
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
    Set-Content -LiteralPath $PidFile -Value $ProcessId -Encoding ASCII
}

function Get-ManagedPid {
    $listener = Get-ListenerPid
    $healthy = Test-NovelAgentsServer
    if (Test-Path -LiteralPath $PidFile) {
        $saved = 0
        if ([int]::TryParse((Get-Content -Raw -LiteralPath $PidFile).Trim(), [ref]$saved)) {
            $process = Get-Process -Id $saved -ErrorAction SilentlyContinue
            if (
                $healthy -and
                $listener -eq $saved -and
                $process -and
                $process.ProcessName -in @("python", "pythonw")
            ) {
                return $saved
            }
        }
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }

    if ($listener -and $healthy) {
        $process = Get-Process -Id $listener -ErrorAction SilentlyContinue
        if ($process -and $process.ProcessName -in @("python", "pythonw")) {
            Save-Pid $listener
            return $listener
        }
    }
    return $null
}

function Start-NovelAgents {
    if (Test-NovelAgentsServer) {
        $runningPid = Get-ManagedPid
        Write-Host "Novel Agents is already running. PID: $runningPid" -ForegroundColor Green
        Write-Host $Url
        return
    }

    $listener = Get-ListenerPid
    if ($listener) {
        throw "Port $Port is already used by PID $listener."
    }

    $pythonw = Get-PythonWindowless
    $arguments = @(
        "-m", "novel_agents.web",
        "--host", $HostAddress,
        "--port", "$Port",
        "--workspace", $Workspace
    )
    $process = Start-Process `
        -FilePath $pythonw `
        -ArgumentList $arguments `
        -WorkingDirectory $Workspace `
        -WindowStyle Hidden `
        -PassThru

    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 200
        if (Test-NovelAgentsServer) {
            Save-Pid $process.Id
            Write-Host "Novel Agents started. PID: $($process.Id)" -ForegroundColor Green
            Write-Host $Url
            return
        }
        if ($process.HasExited) {
            break
        }
    }

    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    throw "Novel Agents failed to start. Run the tests from the project directory."
}

function Stop-NovelAgents {
    $managedPid = Get-ManagedPid
    if (-not $managedPid) {
        if (Test-NovelAgentsServer) {
            throw "A server was detected, but it could not be verified as this project."
        }
        Write-Host "Novel Agents is not running." -ForegroundColor Yellow
        return
    }

    Stop-Process -Id $managedPid -Force
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 100
        if (-not (Get-Process -Id $managedPid -ErrorAction SilentlyContinue)) {
            break
        }
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "Novel Agents stopped." -ForegroundColor Green
}

function Show-NovelAgentsStatus {
    if (Test-NovelAgentsServer) {
        $runningPid = Get-ManagedPid
        Write-Host "Status: running" -ForegroundColor Green
        Write-Host "PID: $runningPid"
        Write-Host "URL: $Url"
    }
    else {
        Get-ManagedPid | Out-Null
        Write-Host "Status: stopped" -ForegroundColor Yellow
    }
}

function Open-NovelAgents {
    if (-not (Test-NovelAgentsServer)) {
        Start-NovelAgents
    }
    Start-Process $Url
}

function Invoke-Action([string]$SelectedAction) {
    switch ($SelectedAction) {
        "Start" { Start-NovelAgents }
        "Stop" { Stop-NovelAgents }
        "Restart" {
            Stop-NovelAgents
            Start-NovelAgents
        }
        "Status" { Show-NovelAgentsStatus }
        "Open" { Open-NovelAgents }
        default { throw "Unsupported action: $SelectedAction" }
    }
}

if ($Action -ne "Menu") {
    Invoke-Action $Action
    exit 0
}

while ($true) {
    Clear-Host
    Write-Host "Novel Agents Control"
    Write-Host ""
    Show-NovelAgentsStatus
    Write-Host ""
    Write-Host "1. Start"
    Write-Host "2. Stop"
    Write-Host "3. Restart"
    Write-Host "4. Status"
    Write-Host "5. Open page"
    Write-Host "0. Exit"
    Write-Host ""
    $choice = Read-Host "Select"
    try {
        switch ($choice) {
            "1" { Start-NovelAgents }
            "2" { Stop-NovelAgents }
            "3" { Invoke-Action "Restart" }
            "4" { Show-NovelAgentsStatus }
            "5" { Open-NovelAgents }
            "0" { exit 0 }
            default { Write-Host "Invalid option." -ForegroundColor Red }
        }
    }
    catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
    Write-Host ""
    Read-Host "Press Enter to continue" | Out-Null
}
