@echo off
title VERITAS - Verify WhisperX setup
color 0E

echo.
echo  ======================================================
echo.
echo   Checking that WhisperX is installed AND its models
echo   are cached locally so it can run offline.
echo.
echo   This is read-only — no installation, no downloads,
echo   no risk. Takes about 30 seconds.
echo.
echo  ======================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run REBUILD_VENV.bat first.
    pause
    exit /b 1
)

REM Force offline mode for the verification — that way we know the
REM check passes only if both models are truly cached and reachable
REM without network. If a model was only partially downloaded last
REM time, this will surface as a clean failure instead of silently
REM completing it via the network.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo [1/3] Importing whisperx...
venv\Scripts\python.exe -c "import whisperx; print('  OK, version:', getattr(whisperx, '__version__', 'unknown'))"
if errorlevel 1 (
    echo.
    echo [FAIL] whisperx is NOT installed.
    echo Run INSTALL_WHISPERX.bat first.
    pause
    exit /b 2
)

echo.
echo [2/3] Loading Whisper large-v3 from cache (offline mode)...
venv\Scripts\python.exe -c "import whisperx; m = whisperx.load_model('large-v3', device='cuda', compute_type='float16', language='ru'); print('  Whisper OK')"
if errorlevel 1 (
    echo.
    echo [FAIL] Whisper large-v3 is NOT cached locally.
    echo The runtime would try to download it and fail behind
    echo HF_HUB_OFFLINE=1.
    echo.
    echo Fix: re-run INSTALL_WHISPERX.bat with a stable network.
    pause
    exit /b 3
)

echo.
echo [3/3] Loading Russian wav2vec2 alignment model from cache...
venv\Scripts\python.exe -c "import whisperx; a, meta = whisperx.load_align_model(language_code='ru', device='cuda', model_name='jonatasgrosman/wav2vec2-large-xlsr-53-russian'); print('  wav2vec2 OK')"
if errorlevel 1 (
    echo.
    echo [FAIL] Russian wav2vec2 model is NOT cached locally.
    echo This is the model that does word-level alignment in
    echo WhisperX. Without it the engine fails at runtime.
    echo.
    echo Fix: re-run INSTALL_WHISPERX.bat with a stable network.
    echo If huggingface.co is unreachable, try a mirror:
    echo   set HF_ENDPOINT=https://hf-mirror.com
    echo   INSTALL_WHISPERX.bat
    pause
    exit /b 4
)

echo.
echo  ======================================================
echo.
echo   ALL GOOD - WhisperX can run fully offline.
echo.
echo   You can now:
echo     1. Restart the backend (close VERITAS Backend window
echo        and re-run START_VERITAS_MVP.bat).
echo     2. In the UI, expand "Advanced options" on upload.
echo     3. Pick "WhisperX" as the ASR engine and submit.
echo.
echo  ======================================================
echo.
pause
