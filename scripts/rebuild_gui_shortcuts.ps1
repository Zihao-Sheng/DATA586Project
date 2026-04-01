$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = (Get-Command python -ErrorAction Stop).Source
$pythonwExe = Join-Path (Split-Path -Parent $pythonExe) "pythonw.exe"

if (-not (Test-Path $pythonwExe)) {
    $pythonwExe = $pythonExe
}

$shortcutSpecs = @(
    @{
        ShortcutName = "Check Requirements.lnk"
        ScriptPath = Join-Path $projectRoot "scripts\ensure_packages_gui.pyw"
        IconPath = Join-Path $projectRoot "scripts\assets\requirements_launcher_icon.ico"
    },
    @{
        ShortcutName = "Launch Training GUI.lnk"
        ScriptPath = Join-Path $projectRoot "scripts\training_gui.py"
        IconPath = Join-Path $projectRoot "scripts\assets\training_launcher_icon.ico"
    }
)

$wsh = New-Object -ComObject WScript.Shell

foreach ($spec in $shortcutSpecs) {
    $shortcutPath = Join-Path $projectRoot $spec.ShortcutName
    $shortcut = $wsh.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $pythonwExe
    $shortcut.Arguments = "`"$($spec.ScriptPath)`""
    $shortcut.WorkingDirectory = $projectRoot
    if (Test-Path $spec.IconPath) {
        $shortcut.IconLocation = "$($spec.IconPath),0"
    }
    $shortcut.Save()
    Write-Host "Updated $($spec.ShortcutName) -> $pythonwExe"
}
