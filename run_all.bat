@echo off
REM Cortexium Orchestrator

echo =======================================================
echo            CORTEXIUM SOCIAL INTELLIGENCE
echo =======================================================
echo.

REM 1. Check for .env
if exist .env goto env_ok
echo [ERROR] .env file missing!
echo Please copy .env.example to .env and configure it.
pause
exit /b
:env_ok

REM 2. Check for Virtual Environment
if exist venv\Scripts\activate.bat goto venv_ok
echo [ERROR] Python virtual environment (venv) not found!
echo Please create it with: python -m venv venv
pause
exit /b
:venv_ok

REM 3. Check for external services
echo [CHECK] Checking external services...

REM Ollama check
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 goto no_ollama
echo [OK] Ollama is running.
goto ollama_ok
:no_ollama
echo [WARNING] Ollama does not seem to be running on port 11434.
echo           LLM features may fail.
:ollama_ok

REM Neo4j check
curl -s http://localhost:7474 >nul 2>&1
if errorlevel 1 goto no_neo4j
echo [OK] Neo4j is running.
goto neo4j_ok
:no_neo4j
echo [WARNING] Neo4j does not seem to be running on port 7474.
:neo4j_ok

REM 4. Dashboard setup check
if exist dashboard\node_modules goto npm_ok
echo [INFO] Dashboard node_modules missing. running npm install...
cd dashboard && call npm install && cd ..
:npm_ok

echo.
echo [PRE-START] Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8770 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1

echo.
echo [START] Launching components...
echo.

REM Launch FastAPI
start "Cortexium API" cmd /c "venv\Scripts\activate && uvicorn api.main:app --host 0.0.0.0 --port 8770"

REM Launch Dashboard
start "Cortexium Dashboard" cmd /c "cd dashboard && npm run dev"

REM Launch AI Core (Main Loop)
echo Starting AI Vision Pipeline...
call venv\Scripts\activate
python main.py
