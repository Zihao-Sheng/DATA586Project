$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$launcherPath = Join-Path $projectRoot "scripts\maintenance\launch_check_requirements.ps1"
$powershellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$pythonwExe = $null

# Resolve real Python executable, skipping the Windows Store stub (WindowsApps).
function Resolve-RealPythonExe {
    # 1. Try 'python' but skip WindowsApps stubs
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source -notlike "*WindowsApps*") {
        return $pythonCmd.Source
    }
    # 2. Try 'py' launcher (Python Launcher for Windows)
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $realExe = & py -c "import sys; print(sys.executable)" 2>$null
        if ($realExe -and (Test-Path $realExe)) {
            return $realExe.Trim()
        }
    }
    # 3. Try 'python3'
    $py3Cmd = Get-Command python3 -ErrorAction SilentlyContinue
    if ($py3Cmd -and $py3Cmd.Source -notlike "*WindowsApps*") {
        return $py3Cmd.Source
    }
    return $null
}

try {
    $pythonExe = Resolve-RealPythonExe
    if ($pythonExe) {
        $pythonwCandidate = Join-Path (Split-Path -Parent $pythonExe) "pythonw.exe"
        if (Test-Path $pythonwCandidate) {
            $pythonwExe = $pythonwCandidate
        } else {
            $pythonwExe = $pythonExe
        }
    } else {
        Write-Warning "Python was not found in PATH. Launch Training GUI shortcut will not be updated this run."
    }
} catch {
    Write-Warning "Python was not found in PATH. Launch Training GUI shortcut will not be updated this run."
}

$wsh = New-Object -ComObject WScript.Shell

# Check Requirements shortcut: always use the resilient PowerShell launcher.
$checkShortcutPath = Join-Path $projectRoot "Check Requirements.lnk"
$checkShortcut = $wsh.CreateShortcut($checkShortcutPath)
$checkShortcut.TargetPath = $powershellExe
$checkShortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcherPath`""
$checkShortcut.WorkingDirectory = $projectRoot
$checkIconPath = Join-Path $projectRoot "scripts\assets\requirements_launcher_icon.ico"
if (Test-Path $checkIconPath) {
    $checkShortcut.IconLocation = "$checkIconPath,0"
}
$checkShortcut.Save()
Write-Host "Updated Check Requirements.lnk -> $powershellExe"

# Training shortcut: keep direct Python launch behavior.
if ($pythonwExe) {
    $trainingScriptPath = Join-Path $projectRoot "scripts\app\training_gui.py"
    $trainingShortcutPath = Join-Path $projectRoot "Launch Training GUI.lnk"
    $trainingShortcut = $wsh.CreateShortcut($trainingShortcutPath)
    $trainingShortcut.TargetPath = $pythonwExe
    $trainingShortcut.Arguments = "`"$trainingScriptPath`""
    $trainingShortcut.WorkingDirectory = $projectRoot
    $trainingIconPath = Join-Path $projectRoot "scripts\assets\training_launcher_icon.ico"
    if (Test-Path $trainingIconPath) {
        $trainingShortcut.IconLocation = "$trainingIconPath,0"
    }
    $trainingShortcut.Save()
    Write-Host "Updated Launch Training GUI.lnk -> $pythonwExe"
}
