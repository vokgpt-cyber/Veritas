@echo off
title EPAM VERITAS - Frontend
color 0C
echo.
echo  ========================================
echo    EPAM VERITAS - Frontend UI
echo  ========================================
echo.

:: Check if node_modules exists
if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend dependencies first...
    cd frontend
    npm install
    cd ..
)

echo Starting frontend on http://localhost:5173 ...
echo.
echo  Open your browser and go to:
echo  http://localhost:5173
echo.
echo  To stop: press Ctrl+C
echo.

cd frontend
npm run dev
