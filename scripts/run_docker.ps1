# PowerShell Docker stack launcher for Tesseract BioMechanics
$RootDir = Split-Path $PSScriptRoot -Parent
Set-Location $RootDir
Write-Host "🐳 Building and starting Dockerized Tesseracts & Dashboard..." -ForegroundColor Cyan
docker compose up --build
