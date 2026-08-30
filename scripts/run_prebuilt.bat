@echo off
setlocal
set FEM_TESSERACT_URL=http://127.0.0.1:8000
set GEOMETRY_TESSERACT_URL=http://127.0.0.1:8001

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

echo 📦 1. Pulling & Starting Pre-Built Tesseract Engines from GHCR (Zero Build Time)...
docker compose -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.ghcr.yml up -d

timeout /t 2 /nobreak >nul

echo 🚀 2. Launching Tesseract Dashboard on http://localhost:8501...
if exist ".venv\Scripts\streamlit.exe" (
    .venv\Scripts\streamlit.exe run app.py --server.port 8501
) else (
    streamlit run app.py --server.port 8501
)

echo 🛑 Stopping Pre-Built Docker Microservices...
docker compose -f docker-compose.ghcr.yml stop
