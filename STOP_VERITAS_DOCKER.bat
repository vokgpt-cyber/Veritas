@echo off
setlocal
cd /d "%~dp0"

if exist ".env.docker" (
    docker compose --env-file .env.docker down --remove-orphans
) else (
    docker compose down --remove-orphans
)

echo.
echo VERITAS Docker stack stopped.
pause
