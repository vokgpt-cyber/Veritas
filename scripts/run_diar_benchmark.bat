@echo off
REM Diarization benchmark runner — uses the project venv.
REM Usage:
REM     scripts\run_diar_benchmark.bat [engine ...]
REM If no engines given, runs: pyannote_community1 pyannote_31 speechbrain
REM To run everything: scripts\run_diar_benchmark.bat --all

setlocal
cd /d "%~dp0.."
call venv\Scripts\activate.bat

if "%~1"=="" (
    python scripts\run_diar_benchmark.py --audio "data\test\Court hearing 129_01.mp4"
) else (
    python scripts\run_diar_benchmark.py --audio "data\test\Court hearing 129_01.mp4" --engines %*
)

endlocal
