$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$studioPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $studioPython)) {
    Write-Host 'Create the project environment and install .[demo,phase2] first. See docs/studio_demo.md.'
    exit 1
}
Write-Host 'Beauty Template Studio: http://127.0.0.1:8765'
& $studioPython -m beauty_video_templates.demo.app --repo $PSScriptRoot
