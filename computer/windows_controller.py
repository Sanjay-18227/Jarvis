"""
Windows Native System Controller
Controls system volume, media keys, battery query, screen lock, and Google searches.
"""

import os
import subprocess
import urllib.parse
from typing import Dict, Any, Tuple


class WindowsController:
    """
    Native Windows hardware, media, and system operations.
    """

    def search_google(self, query: str) -> Tuple[bool, str]:
        """Opens default browser and searches Google."""
        try:
            encoded = urllib.parse.quote_plus(query)
            url = f"https://www.google.com/search?q={encoded}"
            os.system(f"start {url}")
            return True, f"Searching Google for: '{query}'"
        except Exception as e:
            return False, f"Failed to open browser search: {e}"

    def control_media(self, action: str) -> Tuple[bool, str]:
        """
        Sends Windows virtual media key commands.
        action: play_pause | next | prev | mute | volume_up | volume_down
        """
        import win32api
        import win32con

        KEY_MAP = {
            "play_pause": win32con.VK_MEDIA_PLAY_PAUSE,
            "next": win32con.VK_MEDIA_NEXT_TRACK,
            "prev": win32con.VK_MEDIA_PREV_TRACK,
            "mute": win32con.VK_VOLUME_MUTE,
            "volume_up": win32con.VK_VOLUME_UP,
            "volume_down": win32con.VK_VOLUME_DOWN,
        }

        vk = KEY_MAP.get(action.lower().strip())
        if not vk:
            return False, f"Unknown media action: {action}"

        try:
            # Simulate key down and up
            win32api.keybd_event(vk, 0, 0, 0)
            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
            return True, f"Media action '{action}' executed."
        except Exception as e:
            return False, f"Error controlling media: {e}"

    def get_battery_status(self) -> Dict[str, Any]:
        """Queries Windows battery status."""
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery:
                return {
                    "percent": battery.percent,
                    "power_plugged": battery.power_plugged,
                    "status": "Charging" if battery.power_plugged else "Discharging",
                }
        except Exception:
            pass

        # Fallback to WMIC
        try:
            output = subprocess.check_output(
                ["WMIC", "Path", "Win32_Battery", "Get", "EstimatedChargeRemaining"],
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
            if len(lines) > 1 and lines[1].isdigit():
                return {"percent": int(lines[1]), "status": "Operational"}
        except Exception:
            pass

        return {"percent": 100, "status": "AC Power"}

    def lock_workstation(self) -> Tuple[bool, str]:
        """Locks the Windows workstation screen."""
        try:
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
            return True, "Workstation locked, Sir."
        except Exception as e:
            return False, f"Failed to lock workstation: {e}"

    def list_windows(self) -> list:
        """Enumerates visible application windows with titles."""
        import win32gui

        windows = []

        def enum_handler(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).strip()
                if title and title not in ["Program Manager", "Default IME", "MSCTFIME UI"]:
                    windows.append({"hwnd": hwnd, "title": title})

        try:
            win32gui.EnumWindows(enum_handler, None)
        except Exception:
            pass
        return windows

    def focus_window(self, title_keyword: str) -> Tuple[bool, str]:
        """
        Finds a visible window matching keyword (e.g. 'antigravity', 'vscode', 'chrome')
        and brings it to the foreground.
        """
        import win32gui
        import win32con

        kw = title_keyword.lower().strip()
        matched_hwnd = None
        matched_title = ""

        for win in self.list_windows():
            if kw in win["title"].lower():
                matched_hwnd = win["hwnd"]
                matched_title = win["title"]
                break

        if not matched_hwnd:
            return False, f"No active window matching '{title_keyword}' found."

        try:
            # If minimized, restore it
            if win32gui.IsIconic(matched_hwnd):
                win32gui.ShowWindow(matched_hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(matched_hwnd, win32con.SW_SHOW)

            win32gui.SetForegroundWindow(matched_hwnd)
            return True, f"Brought window '{matched_title}' to focus."
        except Exception as e:
            return False, f"Could not focus window '{matched_title}': {e}"

    def close_window(self, title_keyword: str) -> Tuple[bool, str]:
        """Closes a window matching title keyword cleanly."""
        import win32gui
        import win32con

        kw = title_keyword.lower().strip()
        for win in self.list_windows():
            if kw in win["title"].lower():
                try:
                    win32gui.PostMessage(win["hwnd"], win32con.WM_CLOSE, 0, 0)
                    return True, f"Sent close signal to '{win['title']}'."
                except Exception as e:
                    return False, f"Failed closing window: {e}"

        return False, f"Window matching '{title_keyword}' not found."
