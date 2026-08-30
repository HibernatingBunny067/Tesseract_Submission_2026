#!/bin/bash

# Launcher for running the full BioMechanics Multi-Tesseract stack in Docker
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🐳 Building and starting Dockerized Tesseracts & Dashboard..."
docker compose up --build
