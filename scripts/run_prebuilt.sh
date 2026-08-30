#!/bin/bash

# Quick Launch Script for Judges & Users
# Runs Pre-Built Docker Images from GitHub Container Registry (GHCR) with Zero Build Time
set -m

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_STREAMLIT="$ROOT_DIR/.venv/bin/streamlit"
APP_PATH="$ROOT_DIR/app.py"

cleanup() {
    trap - INT TERM EXIT HUP
    echo ""
    echo "🛑 Shutting down Streamlit & Pre-Built Tesseract microservices..."
    
    if [ -n "$APP_PID" ]; then
        kill -9 "$APP_PID" 2>/dev/null || true
    fi
    
    cd "$ROOT_DIR"
    docker compose -f docker-compose.ghcr.yml stop 2>/dev/null || true
    
    lsof -ti:8501 2>/dev/null | xargs kill -9 2>/dev/null || true
    pkill -9 -f "streamlit run" 2>/dev/null || true
    
    echo "✅ Shutdown complete. Ports 8501, 8000 & 8001 are clean."
    exit 0
}

trap cleanup INT TERM EXIT HUP

cd "$ROOT_DIR"

echo "📦 1. Pulling & Starting Pre-Built Tesseract Engines from GHCR (Zero Build Time)..."
docker compose -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.ghcr.yml up -d

echo "⏳ Waiting for Tesseract Engines health checks (Port 8000 & 8001)..."
sleep 2

RUN_CMD="streamlit"
if [ -f "$VENV_STREAMLIT" ]; then
    RUN_CMD="$VENV_STREAMLIT"
fi

export FEM_TESSERACT_URL="http://127.0.0.1:8000"
export GEOMETRY_TESSERACT_URL="http://127.0.0.1:8001"

echo "🚀 2. Launching Tesseract Dashboard on http://localhost:8501..."
"$RUN_CMD" run "$APP_PATH" --server.port 8501 --server.headless false &
APP_PID=$!

wait "$APP_PID"
