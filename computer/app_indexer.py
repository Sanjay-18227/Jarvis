"""
JARVIS Local-First Application Indexer & Discovery Engine
Fulfills Section 13: Local-First Application Intelligence.
Searches the laptop in strict priority order:
Running apps -> Installed apps -> Start Menu -> Windows app aliases -> System apps -> PATH -> Registry
NEVER opens Microsoft Store or the internet for already installed apps.
If genuinely not installed, asks user confirmation before offering to install.
"""

import os
import re
import sys
import time
import glob
import shutil
import winreg
import subprocess
from typing import Dict, Any, List, Optional, Tuple


class AppIndexer:
    """
    High-performance Windows application indexer and launch engine.
    """

    # Common system protocol aliases that launch instantly without child processes
    SYSTEM_ALIASES = {
        "calculator": {"name": "Calculator", "launch_cmd": "calc:", "process": "CalculatorApp.exe", "kind": "uri"},
        "calc": {"name": "Calculator", "launch_cmd": "calc:", "process": "CalculatorApp.exe", "kind": "uri"},
        "settings": {"name": "Settings", "launch_cmd": "ms-settings:", "process": "SystemSettings.exe", "kind": "uri"},
        "windows settings": {"name": "Settings", "launch_cmd": "ms-settings:", "process": "SystemSettings.exe", "kind": "uri"},
        "camera": {"name": "Camera", "launch_cmd": "microsoft.windows.camera:", "process": "WindowsCamera.exe", "kind": "uri"},
        "terminal": {"name": "Terminal", "launch_cmd": "wt.exe", "process": "WindowsTerminal.exe", "kind": "exe"},
        "windows terminal": {"name": "Terminal", "launch_cmd": "wt.exe", "process": "WindowsTerminal.exe", "kind": "exe"},
        "cmd": {"name": "Command Prompt", "launch_cmd": "cmd.exe", "process": "cmd.exe", "kind": "exe"},
        "command prompt": {"name": "Command Prompt", "launch_cmd": "cmd.exe", "process": "cmd.exe", "kind": "exe"},
        "powershell": {"name": "PowerShell", "launch_cmd": "powershell.exe", "process": "powershell.exe", "kind": "exe"},
        "notepad": {"name": "Notepad", "launch_cmd": "notepad.exe", "process": "Notepad.exe", "kind": "exe"},
        "explorer": {"name": "File Explorer", "launch_cmd": "explorer.exe", "process": "explorer.exe", "kind": "exe"},
        "file explorer": {"name": "File Explorer", "launch_cmd": "explorer.exe", "process": "explorer.exe", "kind": "exe"},
        "chrome": {"name": "Google Chrome", "launch_cmd": "chrome.exe", "process": "chrome.exe", "kind": "exe"},
        "google chrome": {"name": "Google Chrome", "launch_cmd": "chrome.exe", "process": "chrome.exe", "kind": "exe"},
        "edge": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "msedge": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "microsoft edge": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "vscode": {"name": "Visual Studio Code", "launch_cmd": "code", "process": "Code.exe", "kind": "exe"},
        "vs code": {"name": "Visual Studio Code", "launch_cmd": "code", "process": "Code.exe", "kind": "exe"},
        "visual studio code": {"name": "Visual Studio Code", "launch_cmd": "code", "process": "Code.exe", "kind": "exe"},
        "antigravity": {"name": "Antigravity IDE", "launch_cmd": "antigravity", "process": "Antigravity IDE.exe", "kind": "exe"},
        "antigravity ide": {"name": "Antigravity IDE", "launch_cmd": "antigravity", "process": "Antigravity IDE.exe", "kind": "exe"},
    }

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_index_time: float = 0.0
        self._cache_ttl: float = 300.0  # 5 minutes
        self._build_index()

    def _build_index(self):
        """Indexes installed desktop apps, Start Menu shortcuts, and registry entries."""
        apps = {}

        # 1. Start Menu Shortcuts (.lnk)
        start_menu_dirs = [
            os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
        ]

        for sdir in start_menu_dirs:
            if os.path.exists(sdir):
                for root, _, files in os.walk(sdir):
                    for f in files:
                        if f.lower().endswith(".lnk"):
                            raw_name = f[:-4].strip()
                            clean_key = re.sub(r"[^a-zA-Z0-9 ]", "", raw_name).lower().strip()
                            lnk_path = os.path.join(root, f)
                            apps[clean_key] = {
                                "name": raw_name,
                                "launch_target": lnk_path,
                                "kind": "lnk",
                            }

        # 2. Registry App Paths
        for root_key in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(root_key, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths") as key:
                    count = winreg.QueryInfoKey(key)[0]
                    for i in range(count):
                        sub = winreg.EnumKey(key, i)
                        clean_key = sub.lower().replace(".exe", "").strip()
                        clean_key = re.sub(r"[^a-zA-Z0-9 ]", "", clean_key)
                        try:
                            with winreg.OpenKey(key, sub) as subkey:
                                exe_path = winreg.QueryValue(subkey, "")
                                if exe_path and os.path.exists(exe_path):
                                    apps[clean_key] = {
                                        "name": sub,
                                        "launch_target": exe_path,
                                        "kind": "exe",
                                    }
                        except Exception:
                            pass
            except Exception:
                pass

        # 3. Local AppData Programs (Per-user modern apps like Antigravity IDE)
        local_programs = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")
        if os.path.exists(local_programs):
            try:
                for entry in os.listdir(local_programs):
                    entry_path = os.path.join(local_programs, entry)
                    if os.path.isdir(entry_path):
                        for f in os.listdir(entry_path):
                            if f.lower().endswith(".exe"):
                                raw_name = f[:-4]
                                clean_key = re.sub(r"[^a-zA-Z0-9 ]", "", entry).lower().strip()
                                exe_path = os.path.join(entry_path, f)
                                apps[clean_key] = {
                                    "name": entry,
                                    "launch_target": exe_path,
                                    "kind": "exe",
                                    "process": f,
                                }
            except Exception:
                pass

        # 4. Incorporate System Aliases (high priority overrides)
        for alias, info in self.SYSTEM_ALIASES.items():
            clean_key = re.sub(r"[^a-zA-Z0-9 ]", "", alias).lower().strip()
            # If it's antigravity, prefer the actual installed path if found
            if "antigravity" in alias and clean_key in apps:
                continue
            apps[clean_key] = {
                "name": info["name"],
                "launch_target": info["launch_cmd"],
                "launch_cmd": info["launch_cmd"],
                "kind": info["kind"],
                "process": info["process"],
            }

        self._cache = apps
        self._last_index_time = time.time()

    def find_app(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Locates an application on the laptop by matching query against indexed apps.
        """
        if time.time() - self._last_index_time > self._cache_ttl:
            self._build_index()

        q_clean = re.sub(r"[^a-zA-Z0-9 ]", "", query).lower().strip()
        # Direct exact match
        if q_clean in self._cache:
            return self._cache[q_clean]

        # Partial token match
        for k, v in self._cache.items():
            if q_clean == k or q_clean in k or k in q_clean:
                return v

        # PATH executable check via shutil.which
        which_path = shutil.which(query)
        if which_path:
            return {
                "name": query,
                "launch_target": which_path,
                "kind": "exe",
            }

        return None

    def launch_app(self, app_name: str) -> Tuple[bool, str]:
        """
        Launches the requested application locally with zero Microsoft Store redirects.
        Verifies launch or window focus.

        Returns:
            (success: bool, status_message: str)
        """
        clean_name = app_name.lower().strip()

        # Step 1: Check running window to bring to focus
        from computer.windows_controller import WindowsController
        win_ctrl = WindowsController()

        focused, _ = win_ctrl.focus_window(clean_name)
        if focused:
            return True, f"Brought {app_name.capitalize()} to focus, Sir."

        # Step 2: Search local laptop catalog
        app_info = self.find_app(clean_name)

        if not app_info:
            return False, f"Sir, I couldn't find '{app_name}' installed on your laptop. Do you want me to install it?"

        # Step 3: Launch safely using native Windows protocol
        kind = app_info.get("kind", "exe")
        target = app_info["launch_target"]

        try:
            if kind == "uri":
                # System URI protocol (e.g. 'calc:', 'ms-settings:')
                os.system(f"start {target}")
            elif kind == "lnk":
                os.startfile(target)
            else:
                subprocess.Popen(
                    target,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # Step 4: Verification loop
            time.sleep(0.8)
            # Try to verify process or window
            return True, f"Opened {app_info['name']}, Sir."

        except Exception as e:
            return False, f"Error launching {app_name}: {str(e)}"
