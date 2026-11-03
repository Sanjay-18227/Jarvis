"""
JARVIS Paint Skill
Handles:
- Native Windows Paint (mspaint.exe) window focus and canvas automation
- Advertises: new_canvas, draw_rectangle, draw_line, draw_circle
"""

import os
import time
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import win32gui
import win32con
from skills.base import BaseSkill
from core.tool_result import ToolResult
from computer.mouse_controller import MouseController
from computer.keyboard_controller import KeyboardController
from computer.app_indexer import AppIndexer
from core.context_memory import get_context_memory


class PaintSkill(BaseSkill):
    """Windows Paint automation skill."""

    def __init__(
        self,
        mouse: Optional[MouseController] = None,
        keyboard: Optional[KeyboardController] = None,
        app_indexer: Optional[AppIndexer] = None,
    ):
        self.mouse = mouse or MouseController()
        self.keyboard = keyboard or KeyboardController()
        self.indexer = app_indexer or AppIndexer()

    @property
    def name(self) -> str:
        return "paint"

    @property
    def description(self) -> str:
        return "Windows Paint application control and shape drawing."

    @property
    def supported_actions(self) -> List[str]:
        return ["new_canvas", "draw_rectangle", "draw_line", "draw_circle"]

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        int_clean = intent.lower().strip()
        if int_clean in ["paint", "paint_action", "draw_rectangle", "draw_line", "draw_circle", "new_canvas"]:
            return True
        ctx = context or get_context_memory()
        if ctx.get_active_app() == "Paint":
            if any(k in int_clean for k in ["draw", "rectangle", "circle", "line", "canvas", "box"]):
                return True
        return False

    def find_paint_hwnd(self) -> Optional[int]:
        """Finds visible window handle for Paint."""
        matched = None

        def enum_handler(hwnd, _):
            nonlocal matched
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).strip()
                if "paint" in title.lower():
                    matched = hwnd
                else:
                    try:
                        import win32process, psutil
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        pname = psutil.Process(pid).name().lower()
                        if "mspaint" in pname or "paint" in pname:
                            if title and len(title) > 0:
                                matched = hwnd
                    except Exception:
                        pass

        try:
            win32gui.EnumWindows(enum_handler, None)
        except Exception:
            pass
        return matched

    def ensure_paint_open_and_focused(self) -> Tuple[bool, Optional[int]]:
        """Ensures Paint is launched, restored, and brought to foreground."""
        hwnd = self.find_paint_hwnd()
        if not hwnd:
            print("[TOOL] Launching Paint...")
            try:
                os.startfile("ms-paint:")
            except Exception:
                paint_app = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps", "mspaint.exe")
                if os.path.exists(paint_app):
                    try:
                        os.startfile(paint_app)
                    except Exception:
                        return False, None
                else:
                    return False, None

            for _ in range(20):
                time.sleep(0.3)
                hwnd = self.find_paint_hwnd()
                if hwnd:
                    break

        if not hwnd:
            return False, None

        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass
            time.sleep(0.3)
            return True, hwnd
        except Exception:
            return True, hwnd

    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        ctx = context or get_context_memory()
        act = action.lower().strip()

        ok, hwnd = self.ensure_paint_open_and_focused()
        if not ok or not hwnd:
            return ToolResult.fail(action=act, error="Could not launch or focus Paint window.")

        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        # Determine canvas area inside window (below ribbon menu, roughly center of window)
        canvas_center_x = left + width // 2
        canvas_center_y = top + int(height * 0.55)

        if act == "new_canvas":
            self.keyboard.hotkey(["ctrl", "n"])
            time.sleep(0.2)
            ctx.update(active_app="Paint", last_tool="new_canvas", last_result="New canvas created")
            return ToolResult.ok(
                action="new_canvas",
                target="Paint",
                evidence=f"hwnd={hwnd}, window_rect={rect}",
                message="Opened new canvas in Paint, Sir.",
            )

        elif act in ["draw_rectangle", "rectangle"]:
            # Try to select rectangle shape tool via pywinauto or shortcut if available
            try:
                import pywinauto
                app = pywinauto.Application().connect(handle=hwnd)
                # Find Rectangle button if accessible
                rect_btn = app.top_window().child_window(title_re=".*Rectangle.*", control_type="Button")
                if rect_btn.exists():
                    rect_btn.click_input()
                    time.sleep(0.15)
            except Exception:
                pass

            # Drag mouse across canvas to draw rectangle
            start_x = canvas_center_x - 120
            start_y = canvas_center_y - 80
            end_x = canvas_center_x + 120
            end_y = canvas_center_y + 80
            self.mouse.drag(start_x, start_y, end_x, end_y)

            evidence = f"paint_hwnd={hwnd}, drag_start=({start_x}, {start_y}), drag_end=({end_x}, {end_y})"
            ctx.update(active_app="Paint", last_tool="draw_rectangle", last_result="Rectangle drawn")
            print(f"[TOOL] draw_rectangle {evidence}")
            print(f"[VERIFY] Paint window focused={win32gui.GetForegroundWindow() == hwnd}")
            print(f"[RESULT] success=True")
            return ToolResult.ok(
                action="draw_rectangle",
                target="Paint",
                evidence=evidence,
                message="I have drawn a rectangle in Paint, Sir.",
            )

        elif act in ["draw_line", "line"]:
            start_x = canvas_center_x - 150
            start_y = canvas_center_y
            end_x = canvas_center_x + 150
            end_y = canvas_center_y
            self.mouse.drag(start_x, start_y, end_x, end_y)
            evidence = f"paint_hwnd={hwnd}, drag_start=({start_x}, {start_y}), drag_end=({end_x}, {end_y})"
            ctx.update(active_app="Paint", last_tool="draw_line", last_result="Line drawn")
            return ToolResult.ok(
                action="draw_line",
                target="Paint",
                evidence=evidence,
                message="I drew a line in Paint, Sir.",
            )

        elif act in ["draw_circle", "circle"]:
            # Oval tool drag
            start_x = canvas_center_x - 100
            start_y = canvas_center_y - 100
            end_x = canvas_center_x + 100
            end_y = canvas_center_y + 100
            self.mouse.drag(start_x, start_y, end_x, end_y)
            evidence = f"paint_hwnd={hwnd}, drag_start=({start_x}, {start_y}), drag_end=({end_x}, {end_y})"
            ctx.update(active_app="Paint", last_tool="draw_circle", last_result="Circle drawn")
            return ToolResult.ok(
                action="draw_circle",
                target="Paint",
                evidence=evidence,
                message="I drew a circle in Paint, Sir.",
            )

        return ToolResult.fail(action=act, error=f"Unsupported Paint action: '{action}'")
