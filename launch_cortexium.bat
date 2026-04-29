@echo off
setlocal

echo =======================================================
echo            CORTEXIUM SOCIAL INTELLIGENCE
echo =======================================================
echo.

REM 1. Cleanup existing processes
echo [CLEANUP] Stopping existing Cortexium services...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :45678 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1

REM 2. Start FastAPI Backend
echo [START] Launching API Backend (Port 45678)...
start "Cortexium API" cmd /c "venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 45678"

REM 3. Start Frontend Dashboard
echo [START] Launching Dashboard (Port 3000)...
start "Cortexium Dashboard" cmd /c "cd dashboard && npm run dev"

REM 4. Start AI Core & HUD
echo [START] Launching AI Vision Pipeline & HUD...
echo.
venv\Scripts\python.exe main.py

pause
