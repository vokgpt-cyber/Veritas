@echo off
setlocal enabledelayedexpansion
title EPAM VERITAS Docker

cd /d "%~dp0"

echo.
echo  ======================================================
echo   VERITAS Docker launcher
echo  ======================================================
echo.

where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not on PATH.
    echo         Install Docker Desktop and enable WSL2 + NVIDIA GPU support.
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Desktop is not running.
    echo         Start Docker Desktop, wait until it is ready, then run this again.
    pause
    exit /b 2
)

if not exist ".env.docker" (
    echo [SETUP] Creating .env.docker with local random secrets...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$rng=[Security.Cryptography.RandomNumberGenerator]::Create();" ^
      "function New-Secret { $b=New-Object byte[] 32; $rng.GetBytes($b); [Convert]::ToHexString($b).ToLowerInvariant() }" ^
      "$auth=New-Secret; $enc=New-Secret;" ^
      "Set-Content -Encoding ASCII '.env.docker' @('EPAM_BACKEND_PORT=8765','EPAM_FRONTEND_PORT=5173','EPAM_AUTH_SECRET_KEY='+$auth,'EPAM_ENCRYPTION_SECRET_KEY='+$enc,'EPAM_SUMMARIZATION_OLLAMA_BASE_URL=http://host.docker.internal:11434','EPAM_SUMMARIZATION_OLLAMA_MODEL=gemma4:26b','HF_TOKEN=','PYANNOTE_METRICS_ENABLED=0','HF_HUB_OFFLINE=0','TRANSFORMERS_OFFLINE=0','HF_DATASETS_OFFLINE=0')"
)

for /f "tokens=1,* delims==" %%A in (.env.docker) do (
    if "%%A"=="EPAM_BACKEND_PORT" set EPAM_BACKEND_PORT=%%B
    if "%%A"=="EPAM_FRONTEND_PORT" set EPAM_FRONTEND_PORT=%%B
)
if not defined EPAM_BACKEND_PORT set EPAM_BACKEND_PORT=8765
if not defined EPAM_FRONTEND_PORT set EPAM_FRONTEND_PORT=5173

echo [CHECK] Frontend will open on http://localhost:%EPAM_FRONTEND_PORT%
echo [CHECK] Backend API will be on http://localhost:%EPAM_BACKEND_PORT%/api/health
echo.

echo [CLEANUP] Removing old Docker containers, if any...
docker compose --env-file .env.docker down --remove-orphans >nul 2>&1

echo [CLEANUP] Checking local VERITAS processes on ports %EPAM_BACKEND_PORT% and %EPAM_FRONTEND_PORT%...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_local_veritas_ports.ps1" -Ports %EPAM_BACKEND_PORT%,%EPAM_FRONTEND_PORT%
if errorlevel 1 (
    echo.
    echo [ERROR] One of the required ports is used by another application.
    echo         Close that application, or edit .env.docker and change:
    echo           EPAM_BACKEND_PORT=%EPAM_BACKEND_PORT%
    echo           EPAM_FRONTEND_PORT=%EPAM_FRONTEND_PORT%
    pause
    exit /b 4
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:11434/api/tags' -TimeoutSec 2; if ($r.Content -notmatch 'gemma4:26b') { exit 3 } else { exit 0 } } catch { exit 2 }"
if errorlevel 3 (
    echo [WARNING] Ollama is running, but gemma4:26b was not found.
    echo           Run: ollama pull gemma4:26b
    echo.
) else if errorlevel 2 (
    echo [WARNING] Ollama is not reachable at localhost:11434.
    echo           Start Ollama before processing meetings.
    echo.
) else (
    echo [OK] Ollama is reachable and gemma4:26b is available.
)

echo [START] Building and starting Docker containers...
docker compose --env-file .env.docker up --build -d
if errorlevel 1 (
    echo.
    echo [ERROR] Docker Compose failed.
    echo         Most common cause: old local VERITAS backend/frontend still uses
    echo         ports %EPAM_BACKEND_PORT% or %EPAM_FRONTEND_PORT%.
    echo         Close old VERITAS command windows or run STOP_VERITAS_DOCKER.bat
    echo         for the Docker stack, then try again.
    pause
    exit /b 3
)

echo.
echo [WAIT] Waiting for backend health...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$port='%EPAM_BACKEND_PORT%'; for ($i=0; $i -lt 90; $i++) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri ('http://localhost:'+$port+'/api/health') -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Seconds 2 }; exit 1"
if errorlevel 1 (
    echo [WARNING] Containers started, but backend health is not ready yet.
    echo           Check logs with: docker compose --env-file .env.docker logs -f backend
) else (
    echo [OK] Backend is healthy.
)

echo.
echo  ======================================================
echo   VERITAS Docker is running
echo.
echo   Open:       http://localhost:%EPAM_FRONTEND_PORT%
echo   Login:      admin / admin
echo   Stop:       STOP_VERITAS_DOCKER.bat
echo   Logs:       docker compose --env-file .env.docker logs -f
echo  ======================================================
echo.

start http://localhost:%EPAM_FRONTEND_PORT%
pause
