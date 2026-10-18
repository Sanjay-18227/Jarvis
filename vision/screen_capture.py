"""
Screen Capture & Active Window Detection Subsystem
Captures window contents and checks active window privacy context.
"""

import os
import io
import time
from typing import Optional, Dict, Any, Tuple
from PIL import Image


class ScreenCapture:
    """
    Manages display capture and active foreground window detection.
    """

    SENSITIVE_WINDOW_TITLES = [
        "keepass", "1password", "bitwarden", "lastpass", "credential",
        "incognito", "inprivate", "authenticator", "keychain"
    ]

    def get_active_window_info(self) -> Dict[str, Any]:
        """Queries the current active foreground window title and rect."""
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) if hwnd else ""
            rect = win32gui.GetWindowRect(hwnd) if hwnd else (0, 0, 0, 0)
            return {
                "hwnd": hwnd,
                "title": title,
                "rect": rect,
                "is_sensitive": any(s in title.lower() for s in self.SENSITIVE_WINDOW_TITLES),
            }
        except Exception:
            return {"hwnd": 0, "title": "Unknown", "rect": (0, 0, 0, 0), "is_sensitive": False}

    def capture_screen(self, save_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Captures the display screen.
        """
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                img.save(save_path)
            return img
        except Exception as e:
            # Fallback mock image for testing/headless environments
            img = Image.new("RGB", (1280, 720), color=(30, 30, 30))
            if save_path:
                img.save(save_path)
            return img

    def capture_bytes(self) -> bytes:
        """Captures display and returns JPEG bytes."""
        img = self.capture_screen()
        if not img:
            return b""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
