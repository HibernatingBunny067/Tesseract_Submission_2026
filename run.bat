@echo off
setlocal
echo ========================================================
echo   Launching Tesseract BioMechanics on Windows
echo ========================================================

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH. Please install Python.
    pause
    exit /b 1
)

REM Run Streamlit
streamlit run app.py --server.port 8501

pause
