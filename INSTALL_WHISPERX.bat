@echo off
title VERITAS - Install WhisperX
color 0E

echo.
echo  ======================================================
echo.
echo   Installing WhisperX (optional ASR engine)
echo.
echo   WhisperX = Whisper transcription + wav2vec2 word-level
echo   alignment. Slightly worse Russian word accuracy than
echo   GigaAM, but materially better speaker attribution at
echo   interruption boundaries because of phoneme-level
echo   timestamps.
echo.
echo   Install size: ~3 GB pip download + ~1.5 GB wav2vec2
echo   model on first use. One-time.
echo.
echo  ======================================================
echo.

cd /d "%~dp0"

REM ---------------------------------------------------------
REM Pre-flight: venv must exist
REM ---------------------------------------------------------
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found at venv\Scripts\python.exe
    echo Run REBUILD_VENV.bat first to create the Python env.
    pause
    exit /b 1
)

echo [1/3] Installing whisperx package...
echo This may take 5-10 minutes (downloads pyannote, faster-whisper, etc.)
echo.

venv\Scripts\python.exe -m pip install whisperx
if errorlevel 1 (
    echo.
    echo [ERROR] pip install whisperx failed. See errors above.
    echo.
    echo Common causes:
    echo   * No network or PyPI blocked
    echo   * pip resolver conflict with existing torch/torchaudio
    echo   * Out of disk space
    echo.
    echo If it's a torch conflict, try:
    echo   venv\Scripts\python.exe -m pip install whisperx --no-deps
    echo (skips re-installing torch / torchaudio that we already have)
    echo.
    pause
    exit /b 2
)

echo.
echo [2/3] Verifying import...
venv\Scripts\python.exe -c "import whisperx; print('  whisperx OK, version:', getattr(whisperx, '__version__', 'unknown'))"
if errorlevel 1 (
    echo.
    echo [ERROR] whisperx installed but import fails.
    echo This usually means a torch/torchaudio version mismatch.
    pause
    exit /b 3
)

echo.
echo [3/4] Verifying CUDA still works after install...
venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA broken'; print('  cuda:', torch.cuda.is_available()); print('  torch:', torch.__version__)"
if errorlevel 1 (
    echo.
    echo [ERROR] CUDA detection broke after install.
    echo whisperx may have replaced your CUDA torch with a CPU build.
    echo Re-run REBUILD_VENV.bat to restore.
    pause
    exit /b 4
)

echo.
echo [4/4] Pre-downloading WhisperX models (~4.5 GB).
echo This MUST happen now while you have a good network — the
echo runtime call (from the UI) will be offline-only otherwise.
echo Expect 5-15 minutes depending on your connection.
echo.

REM IMPORTANT: temporarily unset HF_HUB_OFFLINE for the download.
REM The launcher (START_VERITAS_MVP.bat) sets HF_HUB_OFFLINE=1 for
REM security, which blocks downloads. We need network ACCESS during
REM install to populate the local model cache, after which the
REM runtime can be offline.
set HF_HUB_OFFLINE=
set TRANSFORMERS_OFFLINE=

venv\Scripts\python.exe -c "import os; os.environ.pop('HF_HUB_OFFLINE', None); os.environ.pop('TRANSFORMERS_OFFLINE', None); import whisperx; print('Loading Whisper large-v3 (faster-whisper backbone)...'); m = whisperx.load_model('large-v3', device='cuda', compute_type='float16', language='ru'); print('  Whisper OK'); print('Loading Russian wav2vec2 alignment model...'); a, meta = whisperx.load_align_model(language_code='ru', device='cuda', model_name='jonatasgrosman/wav2vec2-large-xlsr-53-russian'); print('  wav2vec2 OK'); print('Models cached for offline use.')"
if errorlevel 1 (
    echo.
    echo [ERROR] Pre-download failed. Common causes:
    echo.
    echo   * Flaky network or VPN dropping connections
    echo   * Corporate firewall doing SSL inspection
    echo   * HuggingFace temporarily unreachable
    echo   * Disk full
    echo.
    echo Workarounds:
    echo   1. Try again — first SSL drops are sometimes transient.
    echo      Re-run INSTALL_WHISPERX.bat.
    echo.
    echo   2. If your network can't reach huggingface.co, use a
    echo      mirror. Set this BEFORE re-running:
    echo        set HF_ENDPOINT=https://hf-mirror.com
    echo        INSTALL_WHISPERX.bat
    echo.
    echo   3. Use a phone hotspot / different network for the
    echo      install. After models are cached locally, the
    echo      runtime works fully offline.
    echo.
    echo   4. Skip WhisperX. GigaAM is the production default
    echo      and produces excellent Russian transcripts. WhisperX
    echo      is a pilot-only optional engine.
    pause
    exit /b 5
)

echo.
echo  ======================================================
echo   SUCCESS - WhisperX installed AND models cached.
echo.
echo   To use it:
echo     1. Restart the backend (close VERITAS Backend window
echo        and re-run START_VERITAS_MVP.bat).
echo     2. Open the VERITAS UI (http://localhost:3000).
echo     3. Upload a file.
echo     4. In "Advanced options", pick "WhisperX" as the ASR
echo        engine.
echo.
echo   The WhisperX option in the UI should now be enabled
echo   (it was greyed out before because models weren't cached).
echo  ======================================================
echo.
pause
