@echo off
echo 🤖 BroadAxis RFP/RFQ Management Platform
echo =====================================
echo Starting all services...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.12+
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js 18+
    pause
    exit /b 1
)

REM Install frontend dependencies if needed
if not exist "frontend\node_modules" (
    echo 📦 Installing frontend dependencies...
    cd frontend
    npm install
    cd ..
)

REM Start the platform
echo 🚀 Starting BroadAxis Platform...
python start_all.py

pause