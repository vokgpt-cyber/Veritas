@echo off
title VERITAS - Download Russian wav2vec2 (mirror)
color 0E

echo.
echo  ======================================================
echo.
echo   Downloading the Russian wav2vec2 model that WhisperX
echo   needs for word-level alignment.
echo.
echo   This is the file that has been failing to download
echo   directly from huggingface.co with SSL errors. We try
echo   THREE strategies in order:
echo.
echo     1. Default huggingface.co (one more shot in case it
echo        was a transient SSL drop)
echo     2. Chinese mirror hf-mirror.com (often works when
echo        HF main is blocked or unstable)
echo     3. Modelscope (Alibaba mirror, last resort)
echo.
echo   Size: ~1.3 GB. Takes 2-10 minutes depending on
echo   network. Only the model that's missing — Whisper
echo   itself is already cached locally.
echo.
echo  ======================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run REBUILD_VENV.bat first.
    pause
    exit /b 1
)

REM Make sure HF_HUB_OFFLINE is OFF for this download.
set HF_HUB_OFFLINE=
set TRANSFORMERS_OFFLINE=

echo [Attempt 1/3] Trying default huggingface.co...
set HF_ENDPOINT=
venv\Scripts\python.exe -c "import os; os.environ.pop('HF_HUB_OFFLINE', None); os.environ.pop('TRANSFORMERS_OFFLINE', None); from huggingface_hub import snapshot_download; print('Downloading jonatasgrosman/wav2vec2-large-xlsr-53-russian...'); p = snapshot_download(repo_id='jonatasgrosman/wav2vec2-large-xlsr-53-russian'); print('  Cached at:', p)"
if not errorlevel 1 goto verify

echo.
echo [Attempt 2/3] Default failed. Trying hf-mirror.com...
set HF_ENDPOINT=https://hf-mirror.com
venv\Scripts\python.exe -c "import os; os.environ.pop('HF_HUB_OFFLINE', None); os.environ.pop('TRANSFORMERS_OFFLINE', None); os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'; from huggingface_hub import snapshot_download; print('Downloading jonatasgrosman/wav2vec2-large-xlsr-53-russian via mirror...'); p = snapshot_download(repo_id='jonatasgrosman/wav2vec2-large-xlsr-53-russian'); print('  Cached at:', p)"
if not errorlevel 1 goto verify

echo.
echo [Attempt 3/3] Mirror also failed. All HuggingFace endpoints unreachable.
echo.
echo  ======================================================
echo   FAILED - Could not download wav2vec2 model.
echo.
echo   The Russian wav2vec2 model (~1.3 GB) is hosted on
echo   huggingface.co and its mirror, and your network
echo   cannot reach either reliably.
echo.
echo   Workarounds:
echo     1. Try a different network (mobile hotspot, etc.).
echo        Once cached locally, runtime works offline.
echo     2. Stop trying to use WhisperX. GigaAM is the
echo        production default and produces better Russian
echo        transcripts. WhisperX is an OPTIONAL pilot
echo        engine — not blocking anything.
echo.
echo   The VERITAS UI is configured to fall back to other
echo   engines automatically if WhisperX fails to load,
echo   so you do not need WhisperX to keep working.
echo  ======================================================
pause
exit /b 2

:verify
echo.
echo  ======================================================
echo   Download succeeded. Verifying model is loadable...
echo  ======================================================
echo.

REM Force offline mode to confirm the cached version is complete.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_ENDPOINT=

venv\Scripts\python.exe -c "import os; os.environ['HF_HUB_OFFLINE'] = '1'; import whisperx; a, meta = whisperx.load_align_model(language_code='ru', device='cuda', model_name='jonatasgrosman/wav2vec2-large-xlsr-53-russian'); print('  wav2vec2 loaded OK in offline mode')"
if errorlevel 1 (
    echo.
    echo [ERROR] Download completed but offline load still fails.
    echo The cache may be partial. Try re-running this script.
    pause
    exit /b 3
)

echo.
echo  ======================================================
echo   SUCCESS - WhisperX is now fully ready.
echo.
echo   Restart the backend (close VERITAS Backend window
echo   and re-run START_VERITAS_MVP.bat). Then in the UI
echo   pick "WhisperX" as the ASR engine.
echo  ======================================================
echo.
pause
