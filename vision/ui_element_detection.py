"""
JARVIS UI Element Detection Subsystem
Detects visible UI elements, windows, buttons, controls, and active coordinates using Win32 API
and accessibility APIs for semantic computer operation.
"""

from typing import List, Dict, Any, Optional, Tuple
import win32gui
import win32con


class UIElementDetector:
    """
    Locates on-screen UI elements, controls, and coordinates semantically.
    """

    def enumerate_window_controls(self, hwnd: int) -> List[Dict[str, Any]]:
        """Enumerates child controls (buttons, edit fields, labels) inside a given window."""
        controls = []

        def child_enum_proc(child_hwnd, _):
            try:
                if win32gui.IsWindowVisible(child_hwnd):
                    text = win32gui.GetWindowText(child_hwnd).strip()
                    class_name = win32gui.GetClassName(child_hwnd).strip()
                    rect = win32gui.GetWindowRect(child_hwnd)
                    left, top, right, bottom = rect
                    width = right - left
                    height = bottom - top

                    if width > 0 and height > 0:
                        controls.append({
                            "hwnd": child_hwnd,
                            "class": class_name,
                            "text": text,
                            "rect": rect,
                            "center": (left + width // 2, top + height // 2),
                        })
            except Exception:
                pass
            return True

        try:
            win32gui.EnumChildWindows(hwnd, child_enum_proc, None)
        except Exception:
            pass

        return controls

    def find_element_by_text(self, text_query: str, hwnd: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """
        Searches for a control with matching text and returns its (x, y) center coordinates.
        """
        target_hwnd = hwnd or win32gui.GetForegroundWindow()
        if not target_hwnd:
            return None

        query = text_query.lower().strip()
        controls = self.enumerate_window_controls(target_hwnd)

        for ctrl in controls:
            if query in ctrl["text"].lower():
                return ctrl["center"]

        return None

    def get_focused_control_info(self) -> Dict[str, Any]:
        """Returns details on the current foreground window and active control."""
        try:
            fg_hwnd = win32gui.GetForegroundWindow()
            if not fg_hwnd:
                return {"title": "Desktop", "class": "Progman", "rect": (0, 0, 1920, 1080)}

            title = win32gui.GetWindowText(fg_hwnd)
            class_name = win32gui.GetClassName(fg_hwnd)
            rect = win32gui.GetWindowRect(fg_hwnd)

            return {
                "hwnd": fg_hwnd,
                "title": title,
                "class": class_name,
                "rect": rect,
            }
        except Exception:
            return {"title": "Desktop", "class": "Progman", "rect": (0, 0, 1920, 1080)}
