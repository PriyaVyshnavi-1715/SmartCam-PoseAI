@echo off
title Smart Camera - Stopping
color 0C
echo.
echo ============================================================
echo   SMART CAMERA - Stopping Application
echo ============================================================
echo.

REM ── Kill backend on port 8000 ──
set "KILLED=0"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
    set "KILLED=1"
)
if "!KILLED!"=="1" ( echo [OK] Backend stopped. ) else ( echo [--] Backend was not running. )

REM ── Kill frontend on port 5173 ──
set "KILLED=0"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5173 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
    set "KILLED=1"
)
if "!KILLED!"=="1" ( echo [OK] Frontend stopped. ) else ( echo [--] Frontend was not running. )

REM ── Also kill by window title ──
taskkill /FI "WINDOWTITLE eq SmartCamera-Backend*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq SmartCamera-Frontend*" /F >nul 2>&1

echo.
echo [DONE] All Smart Camera servers stopped.
echo.
pause
