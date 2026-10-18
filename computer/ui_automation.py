"""
UI Automation Subsystem
Safe, controlled mouse, keyboard, and scrolling operations using Win32 API.
"""

import time
from typing import Tuple, List, Optional
import win32api
import win32con


class UIAutomation:
    """
    Native Win32 input simulator for desktop automation.
    """

    VK_MAP = {
        "ctrl": win32con.VK_CONTROL,
        "control": win32con.VK_CONTROL,
        "alt": win32con.VK_MENU,
        "shift": win32con.VK_SHIFT,
        "win": win32con.VK_LWIN,
        "enter": win32con.VK_RETURN,
        "esc": win32con.VK_ESCAPE,
        "tab": win32con.VK_TAB,
        "space": win32con.VK_SPACE,
        "backspace": win32con.VK_BACK,
        "save": 0x53,  # 'S'
    }

    def type_text(self, text: str) -> Tuple[bool, str]:
        """Simulates keyboard character input."""
        try:
            for char in text:
                vk = win32api.VkKeyScan(char)
                if vk != -1:
                    code = vk & 0xFF
                    win32api.keybd_event(code, 0, 0, 0)
                    win32api.keybd_event(code, 0, win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(0.01)
            return True, f"Typed {len(text)} characters."
        except Exception as e:
            return False, f"Failed typing text: {e}"

    def press_hotkey(self, keys: List[str]) -> Tuple[bool, str]:
        """Simulates combined keypresses (e.g. ['ctrl', 's'])."""
        vks = []
        for k in keys:
            k_low = k.lower().strip()
            if k_low in self.VK_MAP:
                vks.append(self.VK_MAP[k_low])
            elif len(k_low) == 1:
                vks.append(ord(k_low.upper()))

        if not vks:
            return False, "No valid keys found."

        try:
            for vk in vks:
                win32api.keybd_event(vk, 0, 0, 0)
            for vk in reversed(vks):
                win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
            return True, f"Pressed hotkey: {'+'.join(keys)}"
        except Exception as e:
            return False, f"Error pressing hotkey: {e}"

    def scroll(self, clicks: int = -5) -> Tuple[bool, str]:
        """
        Simulates mouse wheel scrolling.
        Negative = scroll down, Positive = scroll up.
        """
        try:
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, clicks * 120, 0)
            direction = "down" if clicks < 0 else "up"
            return True, f"Scrolled {direction}."
        except Exception as e:
            return False, f"Error scrolling: {e}"

    def click(self, x: Optional[int] = None, y: Optional[int] = None) -> Tuple[bool, str]:
        """Clicks at specific screen coordinates or current cursor location."""
        try:
            if x is not None and y is not None:
                win32api.SetCursorPos((x, y))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return True, f"Mouse clicked."
        except Exception as e:
            return False, f"Mouse click failed: {e}"
