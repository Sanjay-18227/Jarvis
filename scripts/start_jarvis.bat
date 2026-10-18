@echo off
:: Starts JARVIS in the Windows background without a console window
cd /d "%~dp0\.."
echo Starting JARVIS in background...
start "" "C:\Users\acer\Desktop\YAKI\venv\Scripts\pythonw.exe" main.py
echo JARVIS background process launched.
