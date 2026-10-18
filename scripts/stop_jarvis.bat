@echo off
:: Stops running background JARVIS processes
echo Stopping JARVIS background process...
taskkill /F /IM pythonw.exe 2>nul
echo JARVIS stopped successfully.
