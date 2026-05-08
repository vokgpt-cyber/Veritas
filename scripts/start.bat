@echo off
REM EPAM VERITAS - Windows Start Script
REM Verbal Intelligence, Transcription and Summarization v2.0

echo.
echo ====================================
echo EPAM VERITAS - Starting Application
echo ====================================
echo.

REM Change to the project root directory
cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if dependencies are installed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Run the application
echo.
echo Starting EPAM VERITAS API Server...
echo Server will be available at: http://127.0.0.1:8000
echo API Documentation: http://127.0.0.1:8000/api/docs
echo.

uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo Error: Failed to start the application
    pause
    exit /b 1
)

pause
