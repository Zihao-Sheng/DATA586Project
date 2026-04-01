$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$guiScript = Join-Path $PSScriptRoot "training_gui.py"
$iconPath = Join-Path $PSScriptRoot "assets\\training_launcher_icon.ico"

pyinstaller `
  --noconfirm `
  --onefile `
  --windowed `
  --name DATA586TrainingLauncher `
  --icon $iconPath `
  --paths $projectRoot `
  $guiScript
