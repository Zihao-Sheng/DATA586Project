$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$guiScript = Join-Path $projectRoot "scripts\maintenance\ensure_packages_gui.pyw"
$rebuildScript = Join-Path $projectRoot "scripts\maintenance\rebuild_gui_shortcuts.ps1"
$launchScript = Join-Path $projectRoot "scripts\maintenance\launch_check_requirements.ps1"
$powershellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

function Resolve-PythonLauncher {
    # Try pythonw directly (never a Store stub)
    $pythonwCmd = Get-Command pythonw -ErrorAction SilentlyContinue
    if ($pythonwCmd -and $pythonwCmd.Source -notlike "*WindowsApps*") {
        return @{ FilePath = $pythonwCmd.Source; PrefixArgs = @() }
    }

    # Try python but skip Windows Store stubs (WindowsApps)
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source -notlike "*WindowsApps*") {
        $pythonwCandidate = Join-Path (Split-Path -Parent $pythonCmd.Source) "pythonw.exe"
        if (Test-Path $pythonwCandidate) {
            return @{ FilePath = $pythonwCandidate; PrefixArgs = @() }
        }
        return @{ FilePath = $pythonCmd.Source; PrefixArgs = @() }
    }

    # Try 'py' launcher — resolve real exe path
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $realExe = & py -c "import sys; print(sys.executable)" 2>$null
        if ($realExe -and (Test-Path $realExe.Trim())) {
            $realExe = $realExe.Trim()
            $pythonwCandidate = Join-Path (Split-Path -Parent $realExe) "pythonw.exe"
            if (Test-Path $pythonwCandidate) {
                return @{ FilePath = $pythonwCandidate; PrefixArgs = @() }
            }
            return @{ FilePath = $realExe; PrefixArgs = @() }
        }
        return @{ FilePath = $pyCmd.Source; PrefixArgs = @("-3") }
    }

    $pywCmd = Get-Command pyw -ErrorAction SilentlyContinue
    if ($pywCmd) {
        return @{ FilePath = $pywCmd.Source; PrefixArgs = @("-3") }
    }

    return $null
}

function Normalize-PathString([string]$pathValue) {
    if (-not $pathValue) {
        return ""
    }
    return $pathValue.Trim().Replace("/", "\").ToLowerInvariant()
}

function Read-Shortcut([string]$shortcutPath, $wsh) {
    if (-not (Test-Path $shortcutPath)) {
        return $null
    }
    return $wsh.CreateShortcut($shortcutPath)
}

function Needs-ShortcutRepair($resolvedLauncher) {
    $wsh = New-Object -ComObject WScript.Shell
    $checkShortcutPath = Join-Path $projectRoot "Check Requirements.lnk"
    $trainingShortcutPath = Join-Path $projectRoot "Launch Training GUI.lnk"

    $checkShortcut = Read-Shortcut $checkShortcutPath $wsh
    $trainingShortcut = Read-Shortcut $trainingShortcutPath $wsh
    if (-not $checkShortcut -or -not $trainingShortcut) {
        return $true
    }

    $checkTarget = Normalize-PathString $checkShortcut.TargetPath
    $checkArgs = Normalize-PathString $checkShortcut.Arguments
    $expectedCheckTarget = Normalize-PathString $powershellExe
    $expectedLaunchRef = Normalize-PathString $launchScript

    if ($checkTarget -ne $expectedCheckTarget -or $checkArgs -notlike "*$expectedLaunchRef*") {
        return $true
    }

    $trainingTarget = $trainingShortcut.TargetPath
    if (-not $trainingTarget -or -not (Test-Path $trainingTarget)) {
        return $true
    }

    # Only enforce exact target match when we resolved a concrete python executable path.
    if ($resolvedLauncher -and $resolvedLauncher.PrefixArgs.Count -eq 0) {
        $expectedTrainingTarget = Normalize-PathString $resolvedLauncher.FilePath
        if ((Normalize-PathString $trainingTarget) -ne $expectedTrainingTarget) {
            return $true
        }
    }

    return $false
}

$launcher = Resolve-PythonLauncher
if (Needs-ShortcutRepair $launcher) {
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $rebuildScript | Out-Null
    } catch {
    }
}

if (-not $launcher) {
    $launcher = Resolve-PythonLauncher
}

if (-not $launcher) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        "No Python interpreter was found. Please install Python/Conda and then run scripts\maintenance\rebuild_gui_shortcuts.ps1.",
        "DATA586 Requirements Checker"
    ) | Out-Null
    exit 1
}

$args = @()
$args += $launcher.PrefixArgs
$args += @($guiScript)

Start-Process -FilePath $launcher.FilePath -ArgumentList $args -WorkingDirectory $projectRoot | Out-Null
