@echo off
REM =====================================================================
REM  VERITAS - Admin 13-04 three-column comparison
REM
REM  Runs the full pipeline on the Admin audio file with the current
REM  post-Phase-A+ orchestrator, then builds a side-by-side HTML
REM  against a) the IT team reference transcript and b) our old
REM  pre-Phase-A+ archive. Opens the HTML in the default browser.
REM
REM  Prerequisites:
REM    - Ollama running with T-Pro 2.0 model pulled
REM    - HF_TOKEN in .env for pyannote-community-1 download
REM    - Python venv at venv\ with requirements.txt installed
REM    - RTX 3090 or comparable GPU visible via CUDA
REM =====================================================================

cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM -------- Activate the project venv -----------------------------------
if exist "venv\Scripts\activate.bat" goto VENV_FOUND
if exist ".venv\Scripts\activate.bat" goto DOT_VENV_FOUND

echo WARNING: No venv found at venv\ or .venv\.
echo Continuing with system Python - this will likely fail with
echo ModuleNotFoundError. Create a venv and install deps first:
echo     python -m venv venv
echo     call venv\Scripts\activate.bat
echo     pip install -r requirements.txt
echo.
goto VENV_DONE

:VENV_FOUND
echo [breadcrumb 1] Activating venv: venv\Scripts\activate.bat
call "venv\Scripts\activate.bat"
echo [breadcrumb 2] venv activation returned
goto VENV_DONE

:DOT_VENV_FOUND
echo [breadcrumb 1] Activating venv: .venv\Scripts\activate.bat
call ".venv\Scripts\activate.bat"
echo [breadcrumb 2] venv activation returned
goto VENV_DONE

:VENV_DONE

echo [breadcrumb 3] Python version check:
python --version
echo [breadcrumb 4] Python location:
where python

REM -------- Preflight: Python on PATH -----------------------------------
where python >nul 2>&1
if errorlevel 1 goto ERR_NO_PYTHON

REM -------- Preflight: pyyaml importable --------------------------------
echo [breadcrumb 5] Checking that yaml is importable...
python -c "import yaml; print('yaml ok')"
if errorlevel 1 goto ERR_NO_YAML
echo [breadcrumb 6] yaml import succeeded, about to run E2E pipeline.

echo.
echo =====================================================================
echo   STEP 1/3  Running full pipeline E2E on Admin audio
echo   This takes 10-25 minutes on RTX 3090. Live progress below.
echo =====================================================================
python benchmarks\scripts\run_admin_e2e.py
if errorlevel 1 goto ERR_PIPELINE

echo.
echo =====================================================================
echo   STEP 2/3  Building three-column HTML comparison
echo =====================================================================
python benchmarks\scripts\build_admin_compare_html.py
if errorlevel 1 goto ERR_HTML

echo.
echo =====================================================================
echo   STEP 3/3  Opening comparison in your browser
echo =====================================================================
set REPORT=benchmarks\reports\admin_13_04_compare.html
if not exist "%REPORT%" goto ERR_NO_REPORT

start "" "%REPORT%"
echo Opened: %REPORT%

echo.
echo Done. If you want to re-build the HTML without re-running the pipeline:
echo     python benchmarks\scripts\build_admin_compare_html.py
echo.
pause
goto :EOF


:ERR_NO_PYTHON
echo.
echo ERROR: Python not found on PATH.
echo Activate your venv first, or install Python and add it to PATH.
echo.
pause
exit /b 10

:ERR_NO_YAML
echo.
echo ERROR: import yaml failed - the active Python does not have the
echo VERITAS dependencies installed. Install them with:
echo     pip install -r requirements.txt
echo Make sure your venv is active first.
echo.
pause
exit /b 11

:ERR_PIPELINE
echo.
echo ERROR: E2E pipeline FAILED. Not building comparison HTML.
echo Most common causes:
echo   - Ollama not running        - run: ollama serve
echo   - Model not pulled          - run: ollama pull t-tech/T-pro-it-2.0:q4_K_M
echo   - Audio file missing        - put it at data\test\Admin...mp3
echo   - Python deps missing       - pip install -r requirements.txt
echo.
pause
exit /b 1

:ERR_HTML
echo.
echo ERROR: HTML builder FAILED. Check the log above.
echo.
pause
exit /b 2

:ERR_NO_REPORT
echo.
echo ERROR: Expected report not found at %REPORT%
echo.
pause
exit /b 3
