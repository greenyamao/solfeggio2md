@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Solfeggio OCR Studio - Windows 11 Workbench

echo ============================================================
echo   Solfeggio OCR Studio - Windows 11 Workbench
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Launching Solfeggio OCR Studio via .venv...
    ".venv\Scripts\python.exe" run_native_app.py
    goto :after_run
)

echo [WARN] Virtual environment (.venv) not found.
echo.
echo Would you like to run the automated installer now?
set /p RUN_INSTALL="Run install.bat? (Y/n): "
if /i "%RUN_INSTALL%"=="n" (
    echo Launching using system Python...
    python run_native_app.py
    goto :after_run
)

call install.bat
if exist ".venv\Scripts\python.exe" (
    echo.
    echo [INFO] Launching application...
    ".venv\Scripts\python.exe" run_native_app.py
)

:after_run
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)
