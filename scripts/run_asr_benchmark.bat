@echo off
REM ASR multi-engine benchmark runner (ASCII-only; Cyrillic resolved in helper).
setlocal enableextensions
cd /d "%~dp0.."
set LOGFILE=scripts\run_asr_benchmark_log.txt

echo ============================================================ > "%LOGFILE%"
echo ASR benchmark run - %DATE% %TIME% >> "%LOGFILE%"
echo Working dir: %CD% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

echo [launcher] Working dir: %CD%
echo [launcher] Log file:    %CD%\%LOGFILE%
echo.

if not exist "venv\Scripts\python.exe" (
    echo ERROR: venv not found at venv\Scripts\python.exe
    echo ERROR: venv not found >> "%LOGFILE%"
    goto :end
)
echo [launcher] venv python:  OK

if not exist "scripts\run_asr_benchmark_helper.py" (
    echo ERROR: helper not found at scripts\run_asr_benchmark_helper.py
    echo ERROR: helper not found >> "%LOGFILE%"
    goto :end
)
echo [launcher] helper:       OK
echo.

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [launcher] Starting multi-engine ASR benchmark. Expect 10-30 minutes for a 1hr audio.
echo [launcher] Progress is written to %LOGFILE%.
echo.

venv\Scripts\python.exe -u scripts\run_asr_benchmark_helper.py >> "%LOGFILE%" 2>&1

set RC=%ERRORLEVEL%
echo.
echo ============================================================
echo [launcher] Harness finished with exit code: %RC%
echo [launcher] Harness exit code: %RC% >> "%LOGFILE%"
echo ============================================================
echo.
echo Last 80 lines of log:
echo ----------------------------------------------------------------
powershell -NoProfile -Command "Get-Content -Path '%LOGFILE%' -Tail 80"
echo ----------------------------------------------------------------
echo.
echo Full log: %CD%\%LOGFILE%

:end
echo.
echo Press any key to close this window...
pause >nul
endlocal
