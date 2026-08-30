# PowerShell Docker Runner for Tesseract BioMechanics
# Architecture: Runs Dual Tesseract Engines in Docker (Port 8000 & 8001) + Native Dashboard (Port 8501)
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"
$env:FEM_TESSERACT_URL = "http://127.0.0.1:8000"
$env:GEOMETRY_TESSERACT_URL = "http://127.0.0.1:8001"

$RootDir = Split-Path $PSScriptRoot -Parent
Set-Location $RootDir

Write-Host "🐳 1. Starting Dual Tesseract Microservices in Docker (Port 8000 & 8001)..." -ForegroundColor Cyan
docker compose up -d --build fem_tesseract geometry_tesseract

Start-Sleep -Seconds 2

$VenvPython = Join-Path $RootDir ".venv\Scripts\streamlit.exe"
if (Test-Path $VenvPython) {
    Write-Host "🚀 2. Launching Tesseract Dashboard on http://localhost:8501..." -ForegroundColor Green
    & $VenvPython run app.py --server.port 8501
} else {
    Write-Host "🚀 2. Launching Tesseract Dashboard on http://localhost:8501..." -ForegroundColor Green
    streamlit run app.py --server.port 8501
}

Write-Host "🛑 Stopping Docker Microservices..." -ForegroundColor Yellow
docker compose stop fem_tesseract geometry_tesseract
