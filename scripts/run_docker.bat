@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."
echo 🐳 Building and starting Dockerized Tesseracts & Dashboard...
docker compose up --build
