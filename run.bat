
@echo off
setlocal
title Smart Camera - Starting
color 0B

echo.
echo ============================================================
echo   SMART CAMERA - Real-Time Pose Suggestion and Alignment
echo ============================================================
echo.

REM ============================================================
REM 1. SET PROJECT DIRECTORY
REM ============================================================

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM ============================================================
REM 2. CHECK PYTHON
REM ============================================================

echo [CHECK] Checking Python...

python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM ============================================================
REM 3. CHECK NODE.JS
REM ============================================================

echo.
echo [CHECK] Checking Node.js...

node --version
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo Install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM ============================================================
REM 4. CHECK REQUIRED PROJECT FILES
REM ============================================================

if not exist "%PROJECT_DIR%requirements.txt" (
    echo [ERROR] requirements.txt not found.
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%backend\main.py" (
    echo [ERROR] backend\main.py not found.
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%frontend\package.json" (
    echo [ERROR] frontend\package.json not found.
    pause
    exit /b 1
)

REM ============================================================
REM 5. CREATE VIRTUAL ENVIRONMENT
REM ============================================================

echo.
echo [SETUP] Checking Python virtual environment...

if not exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment...

    python -m venv "%PROJECT_DIR%.venv"

    if errorlevel 1 (
        echo [ERROR] Could not create virtual environment.
        pause
        exit /b 1
    )
)

set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"

echo [OK] Virtual environment ready.

REM ============================================================
REM 6. INSTALL PYTHON DEPENDENCIES
REM ============================================================

echo.
echo [SETUP] Installing Python dependencies...

"%PYTHON_EXE%" -m pip install --upgrade pip

if errorlevel 1 (
    echo [ERROR] Pip upgrade failed.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m pip install -r "%PROJECT_DIR%requirements.txt"

if errorlevel 1 (
    echo [ERROR] Python dependency installation failed.
    pause
    exit /b 1
)

echo [OK] Python dependencies installed.

REM ============================================================
REM 7. CREATE ENVIRONMENT FILE
REM ============================================================

if not exist "%PROJECT_DIR%.env" (
    if exist "%PROJECT_DIR%.env.example" (
        copy "%PROJECT_DIR%.env.example" "%PROJECT_DIR%.env" >nul
        echo [OK] Environment file created.
    ) else (
        echo [WARNING] .env.example not found.
    )
)

REM ============================================================
REM 8. CREATE CAPTURES DIRECTORY
REM ============================================================

if not exist "%PROJECT_DIR%captures" (
    mkdir "%PROJECT_DIR%captures"
)

REM ============================================================
REM 9. INITIALIZE DATABASE
REM ============================================================

echo.
echo [SETUP] Initializing database...

if exist "%PROJECT_DIR%init_db.py" (
    "%PYTHON_EXE%" "%PROJECT_DIR%init_db.py"

    if errorlevel 1 (
        echo [ERROR] Database initialization failed.
        pause
        exit /b 1
    )
) else (
    echo [WARNING] init_db.py not found.
    echo Ensure backend startup initializes the database.
)

echo [OK] Database initialization step completed.

REM ============================================================
REM 10. INSTALL FRONTEND DEPENDENCIES
REM ============================================================

echo.
echo [SETUP] Checking frontend dependencies...

if not exist "%PROJECT_DIR%frontend\node_modules" (
    echo [SETUP] Installing frontend packages...

    pushd "%PROJECT_DIR%frontend"

    call npm install

    if errorlevel 1 (
        popd
        echo [ERROR] Frontend installation failed.
        pause
        exit /b 1
    )

    popd
)

echo [OK] Frontend dependencies ready.

REM ============================================================
REM 11. CHECK AND CLEAR PROJECT PORTS
REM ============================================================

echo.
echo [SETUP] Checking ports 8000 and 5173...

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /R /C:":8000 .*LISTENING"') do (
    if not "%%a"=="0" taskkill /PID %%a /F >nul 2>&1
)

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /R /C:":5173 .*LISTENING"') do (
    if not "%%a"=="0" taskkill /PID %%a /F >nul 2>&1
)

REM ============================================================
REM 12. START BACKEND
REM ============================================================

echo.
echo [START] Launching FastAPI backend...

start "SmartCamera-Backend" cmd /k "cd /d ""%PROJECT_DIR%"" && ""%PYTHON_EXE%"" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

REM ============================================================
REM 13. WAIT FOR BACKEND
REM ============================================================

echo [WAIT] Waiting for backend...

timeout /t 3 /nobreak >nul

REM ============================================================
REM 14. START FRONTEND
REM ============================================================

echo.
echo [START] Launching React frontend...

start "SmartCamera-Frontend" cmd /k "cd /d ""%PROJECT_DIR%frontend"" && npm run dev -- --host 127.0.0.1 --port 5173"

REM ============================================================
REM 15. DISPLAY APPLICATION
REM ============================================================

echo.
echo ============================================================
echo.
echo   SMART CAMERA STARTUP REQUESTS SENT
echo.
echo   Frontend:
echo   http://localhost:5173
echo.
echo   Backend API:
echo   http://127.0.0.1:8000/docs
echo.
echo   Backend and frontend run in separate CMD windows.
echo.
echo   Keep both server windows open.
echo.
echo   To stop the application, use stop.bat.
echo.
echo ============================================================
echo.

REM ============================================================
REM 16. OPEN BROWSER
REM ============================================================

start "" "http://localhost:5173"

echo Browser launch requested.
echo Check both server windows for startup errors.
echo.
pause

endlocal
