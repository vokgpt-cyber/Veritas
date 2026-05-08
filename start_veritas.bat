@echo off
title EPAM VERITAS - Starting...
color 0C
echo.
echo  ========================================
echo    EPAM VERITAS - Meeting Intelligence
echo  ========================================
echo.

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run the setup guide first.
    echo.
    pause
    exit /b 1
)

:: Activate venv
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11.
    pause
    exit /b 1
)

:: Set environment variables
if exist ".env" (
    echo [2/3] Loading environment variables...
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        set "%%a=%%b"
    )
) else (
    echo [WARNING] No .env file found. Using defaults.
    set EPAM_AUTH_SECRET_KEY=veritas-mvp-secret-change-me
)

:: Start backend
echo [3/3] Starting VERITAS backend on http://127.0.0.1:8000 ...
echo.
echo  ----------------------------------------
echo   Backend is starting. First run will
echo   download ML models (~6 GB). Please wait.
echo  ----------------------------------------
echo.
echo   To stop: press Ctrl+C
echo.

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

pause
