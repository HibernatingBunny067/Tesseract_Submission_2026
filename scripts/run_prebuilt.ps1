# PowerShell Pre-Built Launcher for Judges
# Pulls & Runs Pre-Built Docker Images from GHCR (Zero Build Time) + Native Dashboard (Port 8501)
$env:FEM_TESSERACT_URL = "http://127.0.0.1:8000"
$env:GEOMETRY_TESSERACT_URL = "http://127.0.0.1:8001"

$RootDir = Split-Path $PSScriptRoot -Parent
Set-Location $RootDir

Write-Host "📦 1. Pulling & Starting Pre-Built Tesseract Engines from GHCR (Zero Build Time)..." -ForegroundColor Cyan
docker compose -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.ghcr.yml up -d

Start-Sleep -Seconds 2

$VenvPython = Join-Path $RootDir ".venv\Scripts\streamlit.exe"
if (Test-Path $VenvPython) {
    Write-Host "🚀 2. Launching Tesseract Dashboard on http://localhost:8501..." -ForegroundColor Green
    & $VenvPython run app.py --server.port 8501
} else {
    Write-Host "🚀 2. Launching Tesseract Dashboard on http://localhost:8501..." -ForegroundColor Green
    streamlit run app.py --server.port 8501
}

Write-Host "🛑 Stopping Pre-Built Docker Microservices..." -ForegroundColor Yellow
docker compose -f docker-compose.ghcr.yml stop
