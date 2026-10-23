"""
JARVIS Local-First Application Indexer & Discovery Engine
Fulfills Section 13: Local-First Application Intelligence & Requirements B, C, E, J:
- Deterministic Windows application alias resolution
- Robust executable discovery for Edge, VS Code, Chrome, Notepad, Calculator
- True process & window launch verification
- Returns structured ToolResult contract
- Never claims an application opened unless verified
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
from core.tool_result import ToolResult


def _find_edge_path() -> Optional[str]:
    """Finds real installed Microsoft Edge executable path on Windows."""
    candidates = [
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return shutil.which("msedge")


def _find_vscode_path() -> Optional[str]:
    """Finds real installed Visual Studio Code executable path on Windows."""
    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Microsoft VS Code", "Code.exe"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Microsoft VS Code", "Code.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft VS Code", "Code.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Microsoft VS Code", "bin", "code.cmd"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return shutil.which("code")


def _find_chrome_path() -> Optional[str]:
    """Finds real installed Google Chrome executable path on Windows."""
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return shutil.which("chrome")


class AppIndexer:
    """
    High-performance Windows application indexer and verified launch engine.
    """

    SYSTEM_ALIASES = {
        "calculator": {"name": "Calculator", "launch_cmd": "calc.exe", "process": "CalculatorApp.exe", "kind": "exe"},
        "calc": {"name": "Calculator", "launch_cmd": "calc.exe", "process": "CalculatorApp.exe", "kind": "exe"},
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
        "browser": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "the browser": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "web browser": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "my browser": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "edge": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "ms edge": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "msedge": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "microsoft edge": {"name": "Microsoft Edge", "launch_cmd": "msedge.exe", "process": "msedge.exe", "kind": "exe"},
        "task manager": {"name": "Task Manager", "launch_cmd": "taskmgr.exe", "process": "Taskmgr.exe", "kind": "exe"},
        "taskmgr": {"name": "Task Manager", "launch_cmd": "taskmgr.exe", "process": "Taskmgr.exe", "kind": "exe"},
        "paint": {"name": "Paint", "launch_cmd": "mspaint.exe", "process": "mspaint.exe", "kind": "exe"},
        "mspaint": {"name": "Paint", "launch_cmd": "mspaint.exe", "process": "mspaint.exe", "kind": "exe"},
        "vscode": {"name": "Visual Studio Code", "launch_cmd": "code", "process": "Code.exe", "kind": "exe"},
        "vs code": {"name": "Visual Studio Code", "launch_cmd": "code", "process": "Code.exe", "kind": "exe"},
        "visual studio code": {"name": "Visual Studio Code", "launch_cmd": "code", "process": "Code.exe", "kind": "exe"},
        "code": {"name": "Visual Studio Code", "launch_cmd": "code", "process": "Code.exe", "kind": "exe"},
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

        # 3. Local AppData Programs (Per-user modern apps like Antigravity IDE, VS Code)
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

        # 4. Explicit system aliases & dynamic resolution
        for alias, info in self.SYSTEM_ALIASES.items():
            clean_key = re.sub(r"[^a-zA-Z0-9 ]", "", alias).lower().strip()
            resolved_target = info["launch_cmd"]

            # Dynamic resolution for major apps
            if "edge" in alias:
                real_edge = _find_edge_path()
                if real_edge:
                    resolved_target = real_edge
            elif "vscode" in alias or alias in ["code", "vs code", "visual studio code"]:
                real_code = _find_vscode_path()
                if real_code:
                    resolved_target = real_code
            elif "chrome" in alias:
                real_chrome = _find_chrome_path()
                if real_chrome:
                    resolved_target = real_chrome

            apps[clean_key] = {
                "name": info["name"],
                "launch_target": resolved_target,
                "launch_cmd": info["launch_cmd"],
                "kind": info["kind"],
                "process": info["process"],
            }

        self._cache = apps
        self._last_index_time = time.time()

    def is_process_running(self, process_name: str) -> bool:
        """Checks if process is currently running using psutil or tasklist."""
        proc_clean = process_name.lower().strip()
        if not proc_clean:
            return False

        try:
            import psutil
            for p in psutil.process_iter(['name']):
                pname = (p.info.get('name') or '').lower()
                if proc_clean == pname or proc_clean in pname:
                    return True
        except Exception:
            pass

        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq {process_name}"],
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            return proc_clean in output.lower()
        except Exception:
            return False

    def find_app(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Locates an application on the laptop by matching query against indexed apps.
        """
        if time.time() - self._last_index_time > self._cache_ttl:
            self._build_index()

        q_clean = re.sub(r"[^a-zA-Z0-9 ]", "", query).lower().strip()

        # Check special alias resolvers directly first
        if q_clean in ["edge", "ms edge", "microsoft edge", "msedge", "browser", "the browser", "web browser", "my browser"]:
            real_edge = _find_edge_path()
            target = real_edge if real_edge else "msedge.exe"
            return {
                "name": "Microsoft Edge",
                "launch_target": target,
                "launch_cmd": target,
                "kind": "exe",
                "process": "msedge.exe",
            }
        elif q_clean in ["task manager", "taskmgr"]:
            return {
                "name": "Task Manager",
                "launch_target": "taskmgr.exe",
                "launch_cmd": "taskmgr.exe",
                "kind": "exe",
                "process": "Taskmgr.exe",
            }
        elif q_clean in ["paint", "mspaint"]:
            return {
                "name": "Paint",
                "launch_target": "mspaint.exe",
                "launch_cmd": "mspaint.exe",
                "kind": "exe",
                "process": "mspaint.exe",
            }
        elif q_clean in ["vscode", "vs code", "visual studio code", "code"]:
            real_code = _find_vscode_path()
            target = real_code if real_code else "code"
            return {
                "name": "Visual Studio Code",
                "launch_target": target,
                "launch_cmd": target,
                "kind": "exe",
                "process": "Code.exe",
            }
        elif q_clean in ["chrome", "google chrome"]:
            real_chrome = _find_chrome_path()
            target = real_chrome if real_chrome else "chrome.exe"
            return {
                "name": "Google Chrome",
                "launch_target": target,
                "launch_cmd": target,
                "kind": "exe",
                "process": "chrome.exe",
            }
        elif q_clean in ["notepad"]:
            return {
                "name": "Notepad",
                "launch_target": "notepad.exe",
                "launch_cmd": "notepad.exe",
                "kind": "exe",
                "process": "Notepad.exe",
            }
        elif q_clean in ["calc", "calculator"]:
            return {
                "name": "Calculator",
                "launch_target": "calc.exe",
                "launch_cmd": "calc.exe",
                "kind": "exe",
                "process": "CalculatorApp.exe",
            }

        # Direct exact match in cache
        if q_clean in self._cache:
            entry = dict(self._cache[q_clean])
            if "launch_cmd" not in entry and "launch_target" in entry:
                entry["launch_cmd"] = entry["launch_target"]
            elif "launch_target" not in entry and "launch_cmd" in entry:
                entry["launch_target"] = entry["launch_cmd"]
            return entry

        # Partial token match
        for k, v in self._cache.items():
            if q_clean == k or q_clean in k or k in q_clean:
                entry = dict(v)
                if "launch_cmd" not in entry and "launch_target" in entry:
                    entry["launch_cmd"] = entry["launch_target"]
                elif "launch_target" not in entry and "launch_cmd" in entry:
                    entry["launch_target"] = entry["launch_cmd"]
                return entry

        # PATH executable check via shutil.which
        which_path = shutil.which(query)
        if which_path:
            return {
                "name": query,
                "launch_target": which_path,
                "kind": "exe",
                "process": os.path.basename(which_path),
            }

        return None

    def launch_app(self, app_name: str) -> ToolResult:
        """
        Launches the requested application locally with zero Microsoft Store redirects.
        Verifies launch or window focus.
        Returns a structured ToolResult. Never claims success without verified execution.
        """
        raw_name = app_name.strip()
        clean_name = re.sub(r"[^a-zA-Z0-9 ]", "", raw_name).lower().strip()

        print(f"[ROUTER] intent=OPEN_APP target={raw_name}")

        # Step 1: Check running window to bring to focus
        from computer.windows_controller import WindowsController
        win_ctrl = WindowsController()

        focused, _ = win_ctrl.focus_window(clean_name)
        if focused:
            print(f"[TOOL] focus_window target={clean_name}")
            print(f"[VERIFY] window_focused=True")
            print(f"[RESULT] success=True")
            return ToolResult.ok(
                action="open_app",
                message=f"Brought {raw_name.capitalize()} to focus, Sir.",
                evidence=f"Focused existing window for {raw_name}",
            )

        # Step 2: Search local laptop catalog
        app_info = self.find_app(clean_name)
        if not app_info:
            print(f"[TOOL] find_app query={clean_name}")
            print(f"[VERIFY] app_found=False")
            print(f"[RESULT] success=False")
            return ToolResult.fail(
                action="open_app",
                message=f"Sir, I couldn't find '{raw_name}' installed on your laptop.",
                error=f"Application '{raw_name}' not found",
            )

        # Step 3: Launch safely using native Windows protocol
        display_name = app_info.get("name", raw_name.capitalize())
        kind = app_info.get("kind", "exe")
        target = app_info["launch_target"]
        expected_proc = app_info.get("process", "unknown.exe")

        print(f"[TOOL] launch_app app={display_name} target={target}")

        try:
            if kind == "uri":
                os.system(f"start {target}")
            elif kind == "lnk":
                os.startfile(target)
            else:
                # Wrap target path in quotes if it contains spaces
                if os.path.exists(target) and " " in target:
                    cmd_to_run = f'"{target}"'
                else:
                    cmd_to_run = target

                subprocess.Popen(
                    cmd_to_run,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # Step 4: True Verification Loop
            verified = False
            for _ in range(6):  # Check for 2.4 seconds
                time.sleep(0.4)
                if self.is_process_running(expected_proc) or self.is_process_running(clean_name):
                    verified = True
                    break
                # Also check window list
                for win in win_ctrl.list_windows():
                    w_title = win.get("title", "").lower()
                    if clean_name in w_title or display_name.lower() in w_title:
                        verified = True
                        break
                if verified:
                    break

            if verified:
                print(f"[VERIFY] process_found=True (process={expected_proc})")
                print(f"[RESULT] success=True")
                return ToolResult.ok(
                    action="open_app",
                    message=f"Opened {display_name}, Sir.",
                    evidence=f"Verified running process {expected_proc}",
                )
            else:
                print(f"[VERIFY] process_found=False (process={expected_proc})")
                print(f"[RESULT] success=False")
                return ToolResult.fail(
                    action="open_app",
                    message=f"Sir, I tried to open {display_name}, but I couldn't verify that it launched.",
                    error=f"Process {expected_proc} not detected after launch",
                    evidence="Verification timeout",
                )

        except Exception as e:
            print(f"[VERIFY] process_found=False error={str(e)}")
            print(f"[RESULT] success=False")
            return ToolResult.fail(
                action="open_app",
                message=f"Sir, I tried to open {display_name}, but encountered an error: {str(e)}",
                error=str(e),
            )
