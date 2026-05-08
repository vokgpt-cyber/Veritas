@echo off
title VERITAS - ASR engine comparison
color 0E

echo.
echo  ======================================================
echo.
echo   ASR engine side-by-side comparison
echo.
echo   Runs the same court-hearing audio through TWO ASR
echo   engines (GigaAM and WhisperX), scores both against
echo   the gold reference (court_hearing_129, 281 turns of
echo   manually-corrected ground truth), and produces a
echo   3-column HTML report with WER/CER metrics and
echo   side-by-side text.
echo.
echo   Total runtime: 10-25 minutes on RTX 3090.
echo     - GigaAM run:    ~3-5 min
echo     - WhisperX run:  ~5-15 min (slower due to wav2vec2 alignment)
echo     - Comparison:    ~10s
echo.
echo   Summarization is disabled in both engines: not part of
echo   the comparison, and would add 5-10 min per side.
echo.
echo  ======================================================
echo.

cd /d "%~dp0"

REM ---------------------------------------------------------
REM Pre-flight: venv must exist
REM ---------------------------------------------------------
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run REBUILD_VENV.bat first.
    pause
    exit /b 1
)

REM ---------------------------------------------------------
REM Pre-flight: gold + audio must exist
REM ---------------------------------------------------------
if not exist "benchmarks\gold\court_hearing_129.jsonl" (
    echo [ERROR] Gold reference not found:
    echo   benchmarks\gold\court_hearing_129.jsonl
    pause
    exit /b 2
)
if not exist "benchmarks\audio_cache\Court hearing 129_01_16k_mono.wav" (
    echo [ERROR] Audio not found:
    echo   benchmarks\audio_cache\Court hearing 129_01_16k_mono.wav
    echo.
    echo Make sure the test audio is in place. The court_hearing_129
    echo benchmark requires the 16kHz mono WAV preprocessed copy.
    pause
    exit /b 3
)

REM ---------------------------------------------------------
REM Pre-flight: WER scoring deps (jiwer, rapidfuzz).
REM Both are pure-Python and tiny, but missing them blows up
REM the run AFTER waiting 10+ min for ASR. Install on demand.
REM
REM Flat structure with goto labels - cmd's paren-counter
REM mis-parses pip arguments containing version specifiers
REM (jiwer>=4.0 etc.) when they appear inside an if-block,
REM because the unescaped > looks like a redirect.
REM ---------------------------------------------------------
echo [trace] Checking WER scoring deps...
venv\Scripts\python.exe -c "import jiwer, rapidfuzz" >nul 2>&1
if not errorlevel 1 goto deps_ok

echo Installing jiwer and rapidfuzz, one-time setup...
venv\Scripts\python.exe -m pip install jiwer rapidfuzz
if errorlevel 1 goto deps_failed

REM Confirm the installed versions actually import cleanly.
venv\Scripts\python.exe -c "import jiwer, rapidfuzz" >nul 2>&1
if errorlevel 1 goto deps_failed
goto deps_ok

:deps_failed
echo.
echo [ERROR] Could not install jiwer or rapidfuzz.
echo WER scoring will not work without them.
echo.
echo Try manually:
echo   venv\Scripts\python.exe -m pip install jiwer rapidfuzz
echo.
pause
exit /b 4

:deps_ok
echo [OK] WER scoring deps available.

REM ---------------------------------------------------------
REM Pre-flight: WhisperX models must be cached for offline use
REM ---------------------------------------------------------
echo [trace] Checking WhisperX is ready...
set HF_HUB_OFFLINE=1
venv\Scripts\python.exe -c "import whisperx; whisperx.load_align_model(language_code='ru', device='cuda', model_name='jonatasgrosman/wav2vec2-large-xlsr-53-russian')" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] WhisperX models are not cached or not loadable offline.
    echo The comparison will fall back to GigaAM if WhisperX cannot load.
    echo.
    echo If you want a full GigaAM-vs-WhisperX comparison:
    echo   1. Run INSTALL_WHISPERX.bat to install whisperx
    echo   2. Run DOWNLOAD_WAV2VEC2_RU.bat to download the alignment model
    echo   3. Run VERIFY_WHISPERX.bat to confirm both are loadable offline
    echo.
    echo Press any key to continue anyway, or close this window to abort.
    pause >nul
)
set HF_HUB_OFFLINE=

REM ---------------------------------------------------------
REM The comparison itself. Each pipeline run is full E2E
REM (preprocess + ASR + diarization + alignment + post-process)
REM but with summarization disabled so we don't waste 5-10 min
REM per side on the LLM stage that we're not even comparing.
REM ---------------------------------------------------------
echo.
echo  ======================================================
echo   Starting comparison run.
echo   Don't close this window - it will auto-open the
echo   report in your browser when done.
echo  ======================================================
echo.

REM --reuse-existing: if a previous run for an engine already produced
REM aligned.json (e.g. GigaAM completed before a missing scoring dep
REM killed the overall script), don't redo it - just rescore.
venv\Scripts\python.exe benchmarks\scripts\compare_asr_engines.py --engines gigaam whisperx --reuse-existing
if errorlevel 1 (
    echo.
    echo [ERROR] Comparison run failed. See messages above.
    pause
    exit /b %errorlevel%
)

REM ---------------------------------------------------------
REM Open the most recent HTML report in the default browser.
REM ---------------------------------------------------------
echo.
echo  ======================================================
echo   Opening report in browser...
echo  ======================================================
for /f "delims=" %%f in ('dir /b /o-d benchmarks\reports\asr_compare_*.html 2^>nul') do (
    start "" "benchmarks\reports\%%f"
    goto :done
)
echo [WARNING] No report file found.

:done
echo.
echo  ======================================================
echo   Comparison complete.
echo.
echo   Report saved at:  benchmarks\reports\asr_compare_*.html
echo   Raw stats JSON:   benchmarks\reports\asr_compare_*.json
echo  ======================================================
echo.
pause
