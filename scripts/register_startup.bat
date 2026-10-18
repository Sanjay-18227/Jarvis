@echo off
:: Registers JARVIS to run automatically upon Windows user logon
echo Registering JARVIS to Windows Startup...

set "TARGET_BAT=%~dp0start_jarvis.bat"
set "SHORTCUT_PATH=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\JARVIS.lnk"

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT_PATH%'); $s.TargetPath = '%TARGET_BAT%'; $s.WorkingDirectory = '%~dp0..'; $s.WindowStyle = 7; $s.Save()"

if exist "%SHORTCUT_PATH%" (
    echo [SUCCESS] JARVIS registered to Windows Startup!
    echo It will now run silently in the background on system boot.
) else (
    echo [ERROR] Failed to register startup shortcut.
)
pause
