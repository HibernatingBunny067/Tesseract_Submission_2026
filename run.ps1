# PowerShell launch script for Tesseract BioMechanics
Write-Host "🚀 Launching Tesseract BioMechanics on http://localhost:8501..." -ForegroundColor Cyan

$venvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
if (Test-Path $venvStreamlit) {
    & $venvStreamlit run (Join-Path $PSScriptRoot "app.py") --server.port 8501 --server.headless false
} else {
    streamlit run (Join-Path $PSScriptRoot "app.py") --server.port 8501 --server.headless false
}
