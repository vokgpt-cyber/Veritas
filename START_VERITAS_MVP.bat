@echo off
title EPAM VERITAS
color 0C

echo.
echo  ======================================================
echo.
echo   VERITAS - Meeting Transcription ^& Summarization
echo   RTX 3090 (24GB)  ASR: GigaAM v3 / Whisper-RU (auto-route)
echo                    Diar: pyannote 4.0 community-1
echo                    LLM:  Gemma 4 26B (Ollama)
echo.
echo  ======================================================
echo.

echo [trace] cd to script dir
cd /d "%~dp0"

echo [trace] setting HF offline env vars
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo [trace] default auth secret
if not defined EPAM_AUTH_SECRET_KEY set EPAM_AUTH_SECRET_KEY=veritas-mvp-secret-change-me

echo [trace] loading .env if present
if exist ".env" (
    call :load_env
)

echo [trace] checking python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.11+.
    pause
    exit /b 1
)

echo [trace] checking venv
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv not found. Run REBUILD_VENV.bat first to set up the
    echo         Python environment ^(~20 min, one-time^).
    pause
    exit /b 1
)

echo [trace] checking npm
REM IMPORTANT: `npm` on Windows is npm.cmd (a batch script), not npm.exe.
REM When one batch invokes another WITHOUT `call`, control never returns
REM to the caller and the parent batch silently terminates. `where npm`
REM is an .exe and doesn't have this problem.
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found on PATH. Install Node.js 20+ from
    echo         https://nodejs.org. Frontend requires npm to run.
    pause
    exit /b 1
)

echo [trace] checking Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 goto ollama_missing

:: Ollama is running. Check gemma4:26b is pulled.
curl -s http://localhost:11434/api/tags | findstr /C:"gemma4:26b" >nul 2>&1
if errorlevel 1 goto model_missing

echo [OK] Ollama running, gemma4:26b available.
goto ollama_check_done

:ollama_missing
echo [WARNING] Ollama not running at localhost:11434.
echo           Start Ollama from Start Menu or run: ollama serve
echo           Then pull the model: ollama pull gemma4:26b
echo           Summarization WILL FAIL without Ollama.
echo.
echo Press any key to continue anyway, or close this window to abort.
pause >nul
goto ollama_check_done

:model_missing
echo [WARNING] Ollama is running but 'gemma4:26b' is NOT pulled.
echo           The default summarization model will fail to load.
echo           Pull it now: ollama pull gemma4:26b
echo           ^(~16GB download, one-time^)
echo.
echo Press any key to continue anyway, or close this window to abort.
pause >nul

:ollama_check_done

:: Install frontend deps if needed
if not exist "frontend\node_modules" (
    echo [SETUP] Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
    echo.
)

:: Backend port. Default 8765. Set EPAM_BACKEND_PORT in your .env
:: file (or in Windows env vars) to use a different port.
:: 8765 was picked to avoid collisions with FastAPI/Django (8000),
:: alternative HTTP servers (8080), and common dev tools (5000).
if not defined EPAM_BACKEND_PORT set EPAM_BACKEND_PORT=8765

:: Pre-flight: is the backend port already in use? Don't try to
:: start uvicorn on a port that's already bound — the user might
:: have another project running on 8000, 8080, or 8765.
echo.
echo [trace] checking port %EPAM_BACKEND_PORT% is free
netstat -ano | findstr "LISTENING" | findstr ":%EPAM_BACKEND_PORT% " >nul 2>&1
if not errorlevel 1 (
    echo.
    echo [ERROR] Port %EPAM_BACKEND_PORT% is already in use by another process.
    echo.
    echo Either:
    echo   1. Stop the other process using port %EPAM_BACKEND_PORT%, OR
    echo   2. Edit .env and set a different port, e.g.:
    echo        EPAM_BACKEND_PORT=8766
    echo      Then run START_VERITAS_MVP.bat again.
    echo.
    echo To find what is using the port, run:
    echo   netstat -ano ^| findstr :%EPAM_BACKEND_PORT%
    echo.
    pause
    exit /b 5
)
echo [OK] Port %EPAM_BACKEND_PORT% is free.

:: Frontend port. Default 3000. If 3000 is taken (user has another
:: project running on it — VERITAS users typically juggle 3 projects
:: in parallel), auto-pick the next free port in 3000..3010 and pass
:: it through to vite via EPAM_FRONTEND_PORT. The browser-open step
:: at the end uses the resolved port.
::
:: Without this scan vite would silently fall back to 3001 / 3002
:: while the launcher kept opening 3000 — broken UX reported
:: 2026-05-04.
if not defined EPAM_FRONTEND_PORT set EPAM_FRONTEND_PORT=3000
echo.
echo [trace] checking frontend port %EPAM_FRONTEND_PORT% is free
set _FRONT_FREE=
for /l %%P in (%EPAM_FRONTEND_PORT%, 1, 3010) do (
    if not defined _FRONT_FREE (
        netstat -ano | findstr "LISTENING" | findstr ":%%P " >nul 2>&1
        if errorlevel 1 (
            set _FRONT_FREE=%%P
        )
    )
)
if not defined _FRONT_FREE (
    echo [ERROR] No free port found between 3000 and 3010 for the frontend.
    echo         Close some dev servers and try again.
    pause
    exit /b 6
)
if not "%_FRONT_FREE%"=="%EPAM_FRONTEND_PORT%" (
    echo [INFO] Port %EPAM_FRONTEND_PORT% is busy; using %_FRONT_FREE% instead.
)
set EPAM_FRONTEND_PORT=%_FRONT_FREE%
echo [OK] Frontend port: %EPAM_FRONTEND_PORT%

:: Start backend
echo [1/2] Starting backend on port %EPAM_BACKEND_PORT%...
start "VERITAS Backend" cmd /k "cd /d "%~dp0" && call venv\Scripts\activate && set HF_HUB_OFFLINE=1 && set TRANSFORMERS_OFFLINE=1 && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && set EPAM_AUTH_SECRET_KEY=%EPAM_AUTH_SECRET_KEY% && set EPAM_BACKEND_PORT=%EPAM_BACKEND_PORT% && python -m uvicorn backend.app.main:app --host 127.0.0.1 --port %EPAM_BACKEND_PORT%"

:: Wait for backend health endpoint
echo      Waiting for backend to start...
:wait_backend
timeout /t 2 /nobreak >nul
curl -s http://localhost:%EPAM_BACKEND_PORT%/api/health >nul 2>&1
if errorlevel 1 goto wait_backend
echo      Backend is ready.

:: Start frontend. Pass both ports through:
::   EPAM_BACKEND_PORT — vite.config.ts proxies /api and /ws here.
::   EPAM_FRONTEND_PORT — vite.config.ts binds here (with strictPort,
::                        so vite errors out instead of silently
::                        switching to a port we won't open).
echo [2/2] Starting frontend on port %EPAM_FRONTEND_PORT%...
start "VERITAS Frontend" cmd /k "cd /d "%~dp0\frontend" && set EPAM_BACKEND_PORT=%EPAM_BACKEND_PORT% && set EPAM_FRONTEND_PORT=%EPAM_FRONTEND_PORT% && npm run dev"

:: Wait for Vite
timeout /t 3 /nobreak >nul

echo.
echo  ======================================================
echo.
echo   VERITAS is running!
echo.
echo   Open in browser:  http://localhost:%EPAM_FRONTEND_PORT%
echo   Login:            admin / admin
echo.
echo   Backend API:      http://localhost:%EPAM_BACKEND_PORT%/api/docs
echo.
echo  ======================================================
echo.

:: Open browser
start http://localhost:%EPAM_FRONTEND_PORT%

echo Press any key to stop both servers...
pause >nul

:: Kill both windows
taskkill /FI "WINDOWTITLE eq VERITAS Backend*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq VERITAS Frontend*" /F >nul 2>&1
echo Servers stopped.
goto :eof


REM =====================================================================
REM Subroutine: load .env into current environment.
REM Using a subroutine isolates for-loop parse failures. If a .env line
REM has % or special chars that crash the for loop, only the subroutine
REM dies, not the whole batch.
REM =====================================================================
:load_env
for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
    if not "%%a"=="" if not "%%b"=="" set "%%a=%%b"
)
exit /b 0
