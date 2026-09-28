@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"
title Solfeggio OCR Studio - 1-Click Installer

echo ============================================================
echo   Solfeggio OCR Studio - 1-Click Automated Setup
echo ============================================================
echo.

:: 1. Check Python installation
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python 3.10-3.12 (check "Add Python to PATH" during installation)
    echo Download: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/4] Checking Python environment...
python --version

:: 2. Create virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/4] Virtual environment (.venv) already exists.
)

:: 3. Upgrade pip and install PyTorch
echo [3/4] Installing PyTorch with CUDA acceleration...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1

where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] NVIDIA GPU detected. Installing PyTorch with CUDA 12.8...
    ".venv\Scripts\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cu128
) else (
    echo [WARN] NVIDIA GPU not detected. Installing PyTorch CPU build...
    ".venv\Scripts\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

:: 4. Install pipeline dependencies
echo [4/4] Installing required dependencies...
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation encountered errors.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   INSTALLATION SUCCESSFUL!
echo ============================================================
echo.
echo Everything is ready. You can now launch the application anytime
echo by simply double-clicking:
echo.
echo     start.bat
echo.
pause
