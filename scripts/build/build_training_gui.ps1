$ErrorActionPreference = "Stop"

$portableScript = Join-Path $PSScriptRoot "build_portable_windows.ps1"
if (!(Test-Path $portableScript)) {
    throw "Missing portable build script: $portableScript"
}

& $portableScript
