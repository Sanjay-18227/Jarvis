"""
JARVIS Mouse Controller
Native Win32 mouse automation: move, click, double click, right click, drag, scroll.
"""

import time
from typing import Tuple, Optional
import win32api
import win32con


class MouseController:
    """
    Simulates high-precision, human-like mouse interactions on Windows.
    """

    def get_position(self) -> Tuple[int, int]:
        """Returns current (x, y) cursor coordinates with safe fallback."""
        try:
            return win32api.GetCursorPos()
        except Exception:
            return (0, 0)

    def move_to(self, x: int, y: int, smooth: bool = False) -> Tuple[bool, str]:
        """Moves cursor to target (x, y) coordinates."""
        try:
            if smooth:
                cur_x, cur_y = self.get_position()
                steps = 15
                for i in range(1, steps + 1):
                    nx = int(cur_x + (x - cur_x) * (i / steps))
                    ny = int(cur_y + (y - cur_y) * (i / steps))
                    win32api.SetCursorPos((nx, ny))
                    time.sleep(0.005)
            else:
                win32api.SetCursorPos((x, y))
            return True, f"Moved mouse to ({x}, {y})"
        except Exception as e:
            return False, f"Failed moving mouse: {e}"

    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
    ) -> Tuple[bool, str]:
        """
        Clicks mouse button at specified or current coordinates.
        Button: 'left', 'right', 'middle'.
        """
        try:
            if x is not None and y is not None:
                self.move_to(x, y)

            btn = button.lower().strip()
            if btn == "right":
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                time.sleep(0.02)
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            elif btn == "middle":
                win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                time.sleep(0.02)
                win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
            else:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.02)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

            loc = f"at ({x}, {y})" if x is not None else "at current position"
            return True, f"{button.capitalize()} click executed {loc}."
        except Exception as e:
            return False, f"Mouse click failed: {e}"

    def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> Tuple[bool, str]:
        """Executes double click at specified or current coordinates."""
        try:
            if x is not None and y is not None:
                self.move_to(x, y)

            for _ in range(2):
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.05)

            loc = f"at ({x}, {y})" if x is not None else "at current position"
            return True, f"Double click executed {loc}."
        except Exception as e:
            return False, f"Double click failed: {e}"

    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> Tuple[bool, str]:
        """Convenience wrapper for right click."""
        return self.click(x=x, y=y, button="right")

    def scroll(self, clicks: int = -5) -> Tuple[bool, str]:
        """
        Scrolls the mouse wheel.
        Negative = down, Positive = up.
        """
        try:
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, clicks * 120, 0)
            direction = "down" if clicks < 0 else "up"
            return True, f"Scrolled {direction} by {abs(clicks)} steps."
        except Exception as e:
            return False, f"Scroll failed: {e}"

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> Tuple[bool, str]:
        """Drags from start coordinates to end coordinates."""
        try:
            self.move_to(start_x, start_y)
            time.sleep(0.05)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.05)
            self.move_to(end_x, end_y, smooth=True)
            time.sleep(0.05)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return True, f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})."
        except Exception as e:
            return False, f"Drag operation failed: {e}"
