@echo off
REM JARVIS launcher - uses the dedicated Jarvis venv (Python 3.13, CPU torch)
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo JARVIS venv missing. Run: powershell -ExecutionPolicy Bypass -File setup_jarvis.ps1
    pause
    exit /b 1
)
set PYTHONPATH=%~dp0
"venv\Scripts\python.exe" main.py %*
