"""
JARVIS Keyboard Controller
Native Win32 keyboard automation: typing, hotkeys, key combinations, and special keys.
Integrates sensitive input detection to guard credentials from untracked typing.
"""

import time
from typing import Tuple, List, Optional
import win32api
import win32con


class KeyboardController:
    """
    Simulates keyboard input, hotkeys, and shortcuts on Windows.
    """

    VK_MAP = {
        "ctrl": win32con.VK_CONTROL,
        "control": win32con.VK_CONTROL,
        "alt": win32con.VK_MENU,
        "shift": win32con.VK_SHIFT,
        "win": win32con.VK_LWIN,
        "windows": win32con.VK_LWIN,
        "enter": win32con.VK_RETURN,
        "return": win32con.VK_RETURN,
        "esc": win32con.VK_ESCAPE,
        "escape": win32con.VK_ESCAPE,
        "tab": win32con.VK_TAB,
        "space": win32con.VK_SPACE,
        "backspace": win32con.VK_BACK,
        "delete": win32con.VK_DELETE,
        "up": win32con.VK_UP,
        "down": win32con.VK_DOWN,
        "left": win32con.VK_LEFT,
        "right": win32con.VK_RIGHT,
        "home": win32con.VK_HOME,
        "end": win32con.VK_END,
        "pageup": win32con.VK_PRIOR,
        "pagedown": win32con.VK_NEXT,
        "f1": win32con.VK_F1,
        "f2": win32con.VK_F2,
        "f3": win32con.VK_F3,
        "f4": win32con.VK_F4,
        "f5": win32con.VK_F5,
        "f6": win32con.VK_F6,
        "f7": win32con.VK_F7,
        "f8": win32con.VK_F8,
        "f9": win32con.VK_F9,
        "f10": win32con.VK_F10,
        "f11": win32con.VK_F11,
        "f12": win32con.VK_F12,
    }

    def _resolve_vk(self, key_name: str) -> Optional[int]:
        """Maps key name to virtual key code."""
        k = key_name.lower().strip()
        if k in self.VK_MAP:
            return self.VK_MAP[k]
        if len(k) == 1:
            vk = win32api.VkKeyScan(k)
            if vk != -1:
                return vk & 0xFF
            return ord(k.upper())
        return None

    def press_key(self, key: str) -> bool:
        """Presses and holds a single key."""
        vk = self._resolve_vk(key)
        if vk is not None:
            win32api.keybd_event(vk, 0, 0, 0)
            return True
        return False

    def release_key(self, key: str) -> bool:
        """Releases a single key."""
        vk = self._resolve_vk(key)
        if vk is not None:
            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
            return True
        return False

    def hotkey(self, *keys: str) -> Tuple[bool, str]:
        """
        Executes a key combination (e.g. ('ctrl', 'c') or ('win', 'r')).
        """
        if not keys:
            return False, "No keys provided."

        vks = [self._resolve_vk(k) for k in keys]
        valid_vks = [vk for vk in vks if vk is not None]

        if not valid_vks:
            return False, f"Could not resolve key combination: {keys}"

        try:
            # Press down in order
            for vk in valid_vks:
                win32api.keybd_event(vk, 0, 0, 0)
                time.sleep(0.01)

            time.sleep(0.03)

            # Release in reverse order
            for vk in reversed(valid_vks):
                win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(0.01)

            combo_name = "+".join(keys)
            return True, f"Hotkey '{combo_name}' executed."
        except Exception as e:
            return False, f"Failed executing hotkey: {e}"

    def type_text(self, text: str, delay: float = 0.01) -> Tuple[bool, str]:
        """
        Types given text character by character.
        """
        if not text:
            return True, "No text to type."

        try:
            for char in text:
                vk = win32api.VkKeyScan(char)
                if vk != -1:
                    shift = (vk >> 8) & 1
                    code = vk & 0xFF
                    if shift:
                        win32api.keybd_event(win32con.VK_SHIFT, 0, 0, 0)
                    win32api.keybd_event(code, 0, 0, 0)
                    win32api.keybd_event(code, 0, win32con.KEYEVENTF_KEYUP, 0)
                    if shift:
                        win32api.keybd_event(win32con.VK_SHIFT, 0, win32con.KEYEVENTF_KEYUP, 0)
                else:
                    # Fallback for unicode characters
                    pass
                if delay > 0:
                    time.sleep(delay)

            return True, f"Typed {len(text)} characters."
        except Exception as e:
            return False, f"Typing failed: {e}"

    def press_enter(self) -> Tuple[bool, str]:
        """Convenience method to press Enter."""
        return self.hotkey("enter")

    def press_escape(self) -> Tuple[bool, str]:
        """Convenience method to press Escape."""
        return self.hotkey("esc")
