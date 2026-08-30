@echo off
setlocal
set DOCKER_BUILDKIT=1
set COMPOSE_DOCKER_CLI_BUILD=1
set FEM_TESSERACT_URL=http://127.0.0.1:8000
set GEOMETRY_TESSERACT_URL=http://127.0.0.1:8001

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

echo 🐳 1. Starting Dual Tesseract Microservices in Docker (Port 8000 & 8001)...
docker compose up -d --build fem_tesseract geometry_tesseract

timeout /t 2 /nobreak >nul

echo 🚀 2. Launching Tesseract Dashboard on http://localhost:8501...
if exist ".venv\Scripts\streamlit.exe" (
    .venv\Scripts\streamlit.exe run app.py --server.port 8501
) else (
    streamlit run app.py --server.port 8501
)

echo 🛑 Stopping Docker Microservices...
docker compose stop fem_tesseract geometry_tesseract
