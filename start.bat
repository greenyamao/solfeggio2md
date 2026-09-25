@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title PDF to Markdown Music - Windows 11 Native Workbench

echo ============================================================
echo   PDF to Markdown Music - Windows 11 Native Workbench
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Launching via .venv virtual environment...
    ".venv\Scripts\python.exe" run_native_app.py
    goto :after_run
)

echo [INFO] Virtual environment not found, launching via system python...
python run_native_app.py


:after_run
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)
