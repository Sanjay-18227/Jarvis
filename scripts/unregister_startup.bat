@echo off
:: Removes JARVIS from Windows Startup
echo Removing JARVIS from Windows Startup...
set "SHORTCUT_PATH=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\JARVIS.lnk"

if exist "%SHORTCUT_PATH%" (
    del /f /q "%SHORTCUT_PATH%"
    echo [SUCCESS] JARVIS startup shortcut removed.
) else (
    echo [INFO] JARVIS was not registered in Startup.
)
pause
