$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$guiScript = Join-Path $projectRoot "scripts\\app\\training_gui.py"
$iconPath = Join-Path $projectRoot "scripts\\assets\\training_launcher_icon.ico"

pyinstaller `
  --noconfirm `
  --onefile `
  --windowed `
  --name DATA586TrainingLauncher `
  --icon $iconPath `
  --paths $projectRoot `
  $guiScript
