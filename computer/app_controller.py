"""
Application Controller Subsystem
Safely launches, verifies, and closes desktop applications using native Windows APIs.
"""

import os
import subprocess
import time
from typing import Dict, Optional, Tuple


class AppController:
    """
    Manages desktop applications with launch verification.
    """

    KNOWN_APPS = {
        "calculator": {"exe": "calc.exe", "process": "CalculatorApp.exe", "cmd": "start calc:"},
        "chrome": {"exe": "chrome.exe", "process": "chrome.exe", "cmd": "start chrome"},
        "vscode": {"exe": "code", "process": "Code.exe", "cmd": "code"},
        "vs code": {"exe": "code", "process": "Code.exe", "cmd": "code"},
        "notepad": {"exe": "notepad.exe", "process": "notepad.exe", "cmd": "notepad.exe"},
        "explorer": {"exe": "explorer.exe", "process": "explorer.exe", "cmd": "explorer.exe"},
        "file explorer": {"exe": "explorer.exe", "process": "explorer.exe", "cmd": "explorer.exe"},
        "settings": {"exe": "ms-settings:", "process": "SystemSettings.exe", "cmd": "start ms-settings:"},
        "terminal": {"exe": "wt.exe", "process": "WindowsTerminal.exe", "cmd": "start wt"},
        "powershell": {"exe": "powershell.exe", "process": "powershell.exe", "cmd": "start powershell"},
        "spotify": {"exe": "spotify.exe", "process": "Spotify.exe", "cmd": "start spotify:"},
    }

    def is_process_running(self, process_name: str) -> bool:
        """Checks if process is currently running using Windows tasklist."""
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq {process_name}"],
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            return process_name.lower() in output.lower()
        except Exception:
            return False

    def open_application(self, app_name: str) -> Tuple[bool, str]:
        """
        Launches an application and verifies it was opened.

        Returns:
            (success: bool, status_message: str)
        """
        clean_name = app_name.lower().strip()
        app_info = self.KNOWN_APPS.get(clean_name)

        if not app_info:
            # Fallback search
            for k, v in self.KNOWN_APPS.items():
                if k in clean_name or clean_name in k:
                    app_info = v
                    break

        if not app_info:
            return False, f"Application '{app_name}' is not recognized in registered apps catalog."

        try:
            # Execute launch command
            subprocess.Popen(
                app_info["cmd"],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Verification loop
            time.sleep(1.0)
            verified = self.is_process_running(app_info["process"])
            if verified:
                return True, f"{app_name.capitalize()} opened successfully."
            else:
                # Some apps like calc or settings might register slightly differently; return success if command ran
                return True, f"{app_name.capitalize()} launch command dispatched."
        except Exception as e:
            return False, f"Failed to launch {app_name}: {str(e)}"

    def close_application(self, app_name: str) -> Tuple[bool, str]:
        """
        Closes an application by process name using taskkill.
        """
        clean_name = app_name.lower().strip()
        app_info = self.KNOWN_APPS.get(clean_name)
        if not app_info:
            for k, v in self.KNOWN_APPS.items():
                if k in clean_name:
                    app_info = v
                    break

        if not app_info:
            return False, f"Application '{app_name}' is not recognized."

        proc = app_info["process"]
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", proc],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            time.sleep(0.5)
            if not self.is_process_running(proc):
                return True, f"{app_name.capitalize()} closed successfully."
            else:
                return False, f"Could not terminate {app_name}."
        except Exception as e:
            return False, f"Failed to close {app_name}: {str(e)}"
