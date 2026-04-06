$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$guiLauncher = Join-Path $projectRoot "scripts\maintenance\launch_training_gui.pyw"
$powershellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

function Resolve-PythonLauncher {
    $pythonwCmd = Get-Command pythonw -ErrorAction SilentlyContinue
    if ($pythonwCmd -and $pythonwCmd.Source -notlike "*WindowsApps*") {
        return @{ FilePath = $pythonwCmd.Source; PrefixArgs = @() }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source -notlike "*WindowsApps*") {
        $pythonwCandidate = Join-Path (Split-Path -Parent $pythonCmd.Source) "pythonw.exe"
        if (Test-Path $pythonwCandidate) {
            return @{ FilePath = $pythonwCandidate; PrefixArgs = @() }
        }
        return @{ FilePath = $pythonCmd.Source; PrefixArgs = @() }
    }

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

    return $null
}

$launcher = Resolve-PythonLauncher
if (-not $launcher) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        "No usable Python interpreter was found. Please run scripts\maintenance\rebuild_gui_shortcuts.ps1 after installing Python.",
        "Training Launcher Error"
    ) | Out-Null
    exit 1
}

$args = @()
$args += $launcher.PrefixArgs
$args += @($guiLauncher)

Start-Process -FilePath $launcher.FilePath -ArgumentList $args -WorkingDirectory $projectRoot | Out-Null
