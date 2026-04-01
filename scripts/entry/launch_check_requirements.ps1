$ErrorActionPreference = "Stop"

$target = Join-Path (Split-Path -Parent $PSScriptRoot) "maintenance\launch_check_requirements.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $target
