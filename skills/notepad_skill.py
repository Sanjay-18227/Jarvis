"""
JARVIS Notepad Skill
Handles:
- Launching and focusing Windows Notepad
- Typing text into Notepad (e.g. "type hello")
- Saving documents
- Advertises: open, type_text, save_file
"""

import time
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import win32gui
import win32con
from skills.base import BaseSkill
from core.tool_result import ToolResult
from computer.keyboard_controller import KeyboardController
from computer.app_indexer import AppIndexer
from core.context_memory import get_context_memory


class NotepadSkill(BaseSkill):
    """Windows Notepad automation skill."""

    def __init__(
        self,
        keyboard: Optional[KeyboardController] = None,
        app_indexer: Optional[AppIndexer] = None,
    ):
        self.keyboard = keyboard or KeyboardController()
        self.indexer = app_indexer or AppIndexer()

    @property
    def name(self) -> str:
        return "notepad"

    @property
    def description(self) -> str:
        return "Windows Notepad application control, text typing, and file saving."

    @property
    def supported_actions(self) -> List[str]:
        return ["open", "type_text", "save_file"]

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        int_clean = intent.lower().strip()
        if int_clean in ["notepad", "open_notepad", "notepad_action", "notepad_type_save"]:
            return True
        ctx = context or get_context_memory()
        if ctx.get_active_app() == "Notepad" and int_clean in ["type", "type_text", "write", "save", "save_file"]:
            return True
        return False

    def find_notepad_hwnd(self) -> Optional[int]:
        """Finds visible window handle for Notepad."""
        matched = None

        def enum_handler(hwnd, _):
            nonlocal matched
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).strip()
                if "notepad" in title.lower():
                    matched = hwnd

        try:
            win32gui.EnumWindows(enum_handler, None)
        except Exception:
            pass
        return matched

    def ensure_notepad_open_and_focused(self) -> Tuple[bool, Optional[int]]:
        """Ensures Notepad is launched and brought to foreground."""
        hwnd = self.find_notepad_hwnd()
        if not hwnd:
            try:
                subprocess.Popen("notepad.exe", shell=True)
            except Exception:
                return False, None

            for _ in range(10):
                time.sleep(0.25)
                hwnd = self.find_notepad_hwnd()
                if hwnd:
                    break

        if not hwnd:
            return False, None

        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.25)
            return True, hwnd
        except Exception:
            return False, hwnd

    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        ctx = context or get_context_memory()
        act = action.lower().strip()

        if act in ["open", "open_app", "launch"]:
            res = self.indexer.launch_app("notepad")
            if res.success:
                ctx.update(active_app="Notepad", last_tool="open_notepad", last_result="Notepad opened")
            return res

        ok, hwnd = self.ensure_notepad_open_and_focused()
        if not ok or not hwnd:
            return ToolResult.fail(action=act, error="Could not focus Notepad window.")

        if act in ["type_text", "type", "write"]:
            text_to_type = params.get("text") or params.get("query") or params.get("content") or ""
            if not text_to_type:
                return ToolResult.fail(action="type_text", error="No text specified to type.")

            # Try UIAutomation / pywinauto direct typing if possible
            typed_ok = False
            try:
                import pywinauto
                app = pywinauto.Application().connect(handle=hwnd)
                # In modern Windows 11 Notepad, edit area is Edit / RichEditD2DPT
                win = app.top_window()
                edit = win.child_window(control_type="Edit") or win.child_window(control_type="Document")
                if edit.exists():
                    edit.type_keys(text_to_type + "{ENTER}", with_spaces=True)
                    typed_ok = True
            except Exception:
                pass

            if not typed_ok:
                # Fallback to KeyboardController
                self.keyboard.type_string(text_to_type)
                self.keyboard.press_key("enter")

            evidence = f"notepad_hwnd={hwnd}, text_typed='{text_to_type}'"
            ctx.update(active_app="Notepad", last_tool="type_text", last_result=f"Typed '{text_to_type}'")
            return ToolResult.ok(
                action="type_text",
                target="Notepad",
                evidence=evidence,
                message=f"I typed '{text_to_type}' in Notepad, Sir.",
            )

        elif act in ["save_file", "save"]:
            filename = params.get("filename") or params.get("file") or "Untitled.txt"
            self.keyboard.hotkey(["ctrl", "s"])
            time.sleep(0.5)
            self.keyboard.type_string(filename)
            self.keyboard.press_key("enter")
            ctx.update(active_app="Notepad", active_file=filename, last_tool="save_file", last_result=f"Saved {filename}")
            return ToolResult.ok(
                action="save_file",
                target="Notepad",
                evidence=f"saved_filename='{filename}'",
                message=f"Saved file as {filename}, Sir.",
            )

        return ToolResult.fail(action=act, error=f"Unsupported Notepad action: '{action}'")
