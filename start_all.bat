@echo off
title EPAM VERITAS - Full Stack
color 0C
echo.
echo  ========================================
echo    EPAM VERITAS - Meeting Intelligence
echo  ========================================
echo.

:: On-premise: block all HuggingFace network access (models loaded from cache)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

:: Reduce CUDA memory fragmentation (recommended by PyTorch for sequential model loading)
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

:: Activate venv if it exists (optional — works without venv too)
if exist "venv\Scripts\activate.bat" (
    echo [OK] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No venv found, using system Python.
)

:: Load .env
if exist ".env" (
    echo [OK] Loading .env...
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
) else (
    echo [WARNING] No .env file. Using defaults.
    set EPAM_AUTH_SECRET_KEY=veritas-mvp-secret-change-me
)

:: Check if node_modules exist for frontend
if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
)

:: Start backend in a new window
echo [1/2] Starting backend on http://localhost:8000 ...
if exist "venv\Scripts\activate.bat" (
    start "VERITAS Backend" cmd /k "set HF_HUB_OFFLINE=1 && set TRANSFORMERS_OFFLINE=1 && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && call venv\Scripts\activate.bat && python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 "
) else (
    start "VERITAS Backend" cmd /k "set HF_HUB_OFFLINE=1 && set TRANSFORMERS_OFFLINE=1 && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 "
)

:: Wait for backend to be ready
echo      Waiting for backend...
timeout /t 3 /nobreak >nul

:: Start frontend in a new window
echo [2/2] Starting frontend on http://localhost:3000 ...
start "VERITAS Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo  ========================================
echo   VERITAS is starting!
echo.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo   API docs: http://localhost:8000/api/docs
echo.
echo   Login: admin / admin
echo.
echo   Close this window when done.
echo   Backend and Frontend windows will
echo   stay open independently.
echo  ========================================
echo.
pause
