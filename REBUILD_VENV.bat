@echo off
REM =====================================================================
REM  VERITAS - Clean venv rebuild
REM
REM  Throws away the existing venv and rebuilds from requirements.txt
REM  with all four traps learned 2026-04-22 locked in as pins:
REM    - CUDA torch via --extra-index-url (not CPU default)
REM    - huggingface-hub<1.0 (transformers compat)
REM    - lightning only (not pytorch-lightning) — kernel collision
REM    - no torchcodec (torch 2.8 DLL incompatibility on Windows)
REM
REM  Local model caches in models/ are untouched. Audio in data/ untouched.
REM  Only venv/ is deleted and rebuilt.
REM
REM  Total time: ~15-25 min depending on internet speed (~3GB download).
REM =====================================================================

cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo =====================================================================
echo   STEP 1/6  Deleting existing venv
echo =====================================================================
if exist venv (
    echo   Removing venv\ ^(this may take a minute^)...
    rmdir /s /q venv
    if exist venv (
        echo ERROR: Could not delete venv\. A Python process may still be holding it.
        echo Close all cmd windows and editors with this project open, then retry.
        pause
        exit /b 10
    )
    echo   venv\ deleted.
) else (
    echo   No existing venv\ found. Proceeding to create.
)

echo.
echo =====================================================================
echo   STEP 2/6  Creating fresh venv
echo =====================================================================
python -m venv venv
if errorlevel 1 goto ERR_VENV_CREATE
echo   venv\ created.

call "venv\Scripts\activate.bat"
if errorlevel 1 goto ERR_VENV_ACTIVATE

echo.
echo   Python location and version:
python --version
where python

echo.
echo =====================================================================
echo   STEP 3/6  Upgrading pip
echo =====================================================================
python -m pip install --upgrade pip
if errorlevel 1 goto ERR_PIP_UPGRADE

echo.
echo =====================================================================
echo   STEP 4/6  Installing from requirements.txt
echo   (CUDA torch ~3GB, will take 5-15 min)
echo =====================================================================
pip install -r requirements.txt
if errorlevel 1 goto ERR_REQUIREMENTS

echo.
echo   [defensive] checking torchcodec was not pulled in transitively:
pip show torchcodec >nul 2>&1
if not errorlevel 1 (
    echo   torchcodec found in venv ^(pulled by pyannote.audio^) — uninstalling to prevent DLL crash
    pip uninstall torchcodec -y
) else (
    echo   torchcodec not present. Good.
)

echo.
echo =====================================================================
echo   STEP 5/6  Installing GigaAM v3 from GitHub
echo   (PyPI 'gigaam' only has v1/v2. v3_e2e_rnnt requires GitHub install.)
echo =====================================================================
pip install --force-reinstall --no-deps https://github.com/salute-developers/GigaAM/archive/refs/heads/main.zip
if errorlevel 1 goto ERR_GIGAAM

echo.
echo =====================================================================
echo   STEP 6/6  Verifying CUDA + key imports
echo =====================================================================

echo.
echo   [check 1/4] torch + CUDA:
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print('  cuda:', torch.cuda.is_available()); print('  version:', torch.__version__); print('  device:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto ERR_CUDA_CHECK

echo.
echo   [check 2/4] pyannote.audio:
python -c "import pyannote.audio; print('  pyannote.audio:', pyannote.audio.__version__)"
if errorlevel 1 goto ERR_PYANNOTE_CHECK

echo.
echo   [check 3/4] speechbrain:
python -c "import speechbrain; print('  speechbrain:', speechbrain.__version__)"
if errorlevel 1 goto ERR_SPEECHBRAIN_CHECK

echo.
echo   [check 4/4] gigaam:
python -c "import gigaam; print('  gigaam: importable')"
if errorlevel 1 goto ERR_GIGAAM_CHECK

echo.
echo =====================================================================
echo   SUCCESS - venv rebuilt and verified
echo =====================================================================
echo   All four traps locked in:
echo     - torch 2.8.0+cu126 installed with CUDA
echo     - huggingface-hub pinned ^<1.0
echo     - lightning only, no pytorch-lightning
echo     - torchcodec not present
echo.
echo   Next step: run your pipeline script of choice, e.g.
echo     START_VERITAS_MVP.bat
echo     benchmarks\scripts\run_admin_e2e.py
echo.
pause
goto :EOF


:ERR_VENV_CREATE
echo.
echo ERROR: python -m venv failed. Is Python installed on PATH?
pause
exit /b 11

:ERR_VENV_ACTIVATE
echo.
echo ERROR: venv\Scripts\activate.bat failed. Venv may be corrupt.
pause
exit /b 12

:ERR_PIP_UPGRADE
echo.
echo ERROR: pip upgrade failed. Check internet connection.
pause
exit /b 13

:ERR_REQUIREMENTS
echo.
echo ERROR: pip install -r requirements.txt FAILED.
echo Most common causes:
echo   - No internet / firewall blocking PyPI or pytorch.org
echo   - Disk full (torch CUDA wheels are ~3GB)
echo   - --extra-index-url being ignored by pip config
echo.
echo If torch installed as +cpu instead of +cu126, edit pip.ini or try:
echo   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
echo.
pause
exit /b 14

:ERR_GIGAAM
echo.
echo ERROR: GigaAM GitHub install FAILED.
echo Check internet connection and GitHub availability.
pause
exit /b 15

:ERR_CUDA_CHECK
echo.
echo ERROR: CUDA check FAILED. Torch installed without CUDA support.
echo This means pip pulled the CPU-only torch wheel despite --extra-index-url.
echo Try explicit: pip install torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
pause
exit /b 16

:ERR_PYANNOTE_CHECK
echo.
echo ERROR: pyannote.audio import FAILED. Check log above for details.
pause
exit /b 17

:ERR_SPEECHBRAIN_CHECK
echo.
echo ERROR: speechbrain import FAILED. Check log above.
pause
exit /b 18

:ERR_GIGAAM_CHECK
echo.
echo ERROR: gigaam import FAILED. GitHub install may not have replaced the PyPI version.
pause
exit /b 19
