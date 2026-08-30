# PowerShell launch script for Tesseract BioMechanics
Write-Host "🚀 Launching Tesseract BioMechanics on http://localhost:8501..." -ForegroundColor Cyan

$RootDir = Split-Path $PSScriptRoot -Parent
$venvStreamlit = Join-Path $RootDir ".venv\Scripts\streamlit.exe"
$appPath = Join-Path $RootDir "app.py"

Set-Location $RootDir

if (Test-Path $venvStreamlit) {
    & $venvStreamlit run $appPath --server.port 8501 --server.headless false
} else {
    streamlit run $appPath --server.port 8501 --server.headless false
}
