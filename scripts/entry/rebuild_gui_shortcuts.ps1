$ErrorActionPreference = "Stop"

$target = Join-Path (Split-Path -Parent $PSScriptRoot) "maintenance\rebuild_gui_shortcuts.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $target
