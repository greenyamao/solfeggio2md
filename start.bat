@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Solfeggio OCR Studio

echo ============================================================
echo   Solfeggio OCR Studio - Standalone Desktop Application
echo ============================================================
echo.

set "VENV_PYTHON=.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    echo [INFO] Запуск через виртуальное окружение .venv...
    "%VENV_PYTHON%" run_workbench.py
) else (
    echo [INFO] Виртуальное окружение не найдено, запуск через системный python...
    python run_workbench.py
)

if errorlevel 1 (
    echo.
    echo [ERROR] Приложение завершилось с ошибкой.
    pause
)