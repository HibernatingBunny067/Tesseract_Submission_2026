#!/bin/bash

# Robust launch script for Tesseract BioMechanics with instant Ctrl+C cleanup
set -m

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_STREAMLIT="$ROOT_DIR/.venv/bin/streamlit"
APP_PATH="$ROOT_DIR/app.py"

cleanup() {
    # Disable trap to avoid recursive loops
    trap - INT TERM EXIT HUP
    echo ""
    echo "🛑 Shutting down Streamlit & Tesseract servers..."
    
    # 1. Kill the main Streamlit process immediately
    if [ -n "$APP_PID" ]; then
        kill -9 "$APP_PID" 2>/dev/null || true
    fi
    
    # 2. Force kill any remaining processes holding ports 8501, 8000, or 8001
    lsof -ti:8501,8000,8001 2>/dev/null | xargs kill -9 2>/dev/null || true
    pkill -9 -f "tesseract_server.py" 2>/dev/null || true
    pkill -9 -f "streamlit run" 2>/dev/null || true
    
    echo "✅ Shutdown complete. Ports 8501, 8000 & 8001 are clean."
    exit 0
}

# Trap interrupt and termination signals
trap cleanup INT TERM EXIT HUP

# Pre-execution cleanup: make sure ports are free
lsof -ti:8501,8000,8001 2>/dev/null | xargs kill -9 2>/dev/null || true
pkill -9 -f "tesseract_server.py" 2>/dev/null || true
pkill -9 -f "streamlit run" 2>/dev/null || true
sleep 0.3

RUN_CMD="streamlit"
if [ -f "$VENV_STREAMLIT" ]; then
    RUN_CMD="$VENV_STREAMLIT"
fi

echo "🚀 Launching Tesseract BioMechanics on http://localhost:8501..."

# Run in background so bash immediately receives and handles Ctrl+C without lag
cd "$ROOT_DIR"
"$RUN_CMD" run "$APP_PATH" --server.port 8501 --server.headless false &
APP_PID=$!

# Wait for process (will be unblocked instantly on Ctrl+C)
wait "$APP_PID"
