$ErrorActionPreference = "Stop"

$target = Join-Path (Split-Path -Parent $PSScriptRoot) "build\build_training_gui.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $target
