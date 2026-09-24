@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title PDF to Markdown Music - Legacy Web Workbench

echo ============================================================
echo   PDF to Markdown Music - Legacy Web Workbench (WebView2)
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Запуск через виртуальное окружение .venv...
    ".venv\Scripts\python.exe" run_workbench.py
    goto :after_run
)

echo [INFO] Виртуальное окружение не найдено, запуск через системный python...
python run_workbench.py

:after_run
if errorlevel 1 (
    echo.
    echo [ERROR] Приложение завершилось с кодом ошибки %errorlevel%.
    pause
)
