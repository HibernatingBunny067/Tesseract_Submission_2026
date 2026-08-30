@echo off
setlocal
echo 🚀 Launching Tesseract BioMechanics on http://localhost:8501...

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless false
) else (
    streamlit run app.py --server.port 8501 --server.headless false
)
