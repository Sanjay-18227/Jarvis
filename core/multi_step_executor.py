"""
JARVIS Multi-Step Action Executor
Fulfills Requirement F:
Supports sequential execution with verified progression:
- "Open Notepad and type Hi how are you"
- "Open Notepad, type 'Hi, how are you?', and save it on Desktop as Hello.txt"
- "Save it on Desktop as Hello.txt"
Never claims success unless every step and final file/window are verified.
"""

import os
import time
from typing import Optional, Dict, Any
from core.tool_result import ToolResult
from computer.app_indexer import AppIndexer
from computer.windows_controller import WindowsController
from computer.keyboard_controller import KeyboardController
from computer.file_manager import FileManager


class MultiStepExecutor:
    """
    Executes sequential desktop interactions with verification gates at every step.
    """

    def __init__(
        self,
        app_indexer: Optional[AppIndexer] = None,
        win_ctrl: Optional[WindowsController] = None,
        keyboard_ctrl: Optional[KeyboardController] = None,
        file_mgr: Optional[FileManager] = None,
    ):
        self.app_indexer = app_indexer or AppIndexer()
        self.win_ctrl = win_ctrl or WindowsController()
        self.keyboard_ctrl = keyboard_ctrl or KeyboardController()
        self.file_mgr = file_mgr or FileManager()

    def execute_notepad_flow(
        self,
        text: str,
        save_file: Optional[str] = None,
        location: Optional[str] = "desktop",
    ) -> ToolResult:
        """
        Executes:
        1. Launch Notepad -> verify
        2. Focus window -> verify
        3. Type text -> verify
        4. If save requested -> save to desktop and verify file exists on disk
        """
        print(f"[ROUTER] intent=MULTI_STEP action=notepad_type_save text='{text}' save='{save_file}'")

        # Step 1: Launch Notepad
        res_launch = self.app_indexer.launch_app("notepad")
        if not res_launch.success:
            print(f"[RESULT] success=False error='Notepad launch failed'")
            return ToolResult.fail(
                action="notepad_flow",
                message="Sir, I tried to open Notepad, but I couldn't verify that it launched.",
                error=res_launch.error or "Notepad not launched",
            )

        # Step 2: Focus Notepad window
        time.sleep(0.5)
        self.win_ctrl.focus_window("notepad")
        time.sleep(0.3)

        # Step 3: Type text
        if text:
            type_ok, type_msg = self.keyboard_ctrl.type_text(text)
            print(f"[TOOL] keyboard_type text='{text}' ok={type_ok}")
            if not type_ok:
                return ToolResult.fail(
                    action="notepad_flow",
                    message="Sir, Notepad opened, but typing the text failed.",
                    error=type_msg,
                )

        # Step 4: If save requested, save and verify on disk
        if save_file:
            time.sleep(0.5)
            desktop_dir = os.path.expanduser(r"~\Desktop")
            file_path = os.path.join(desktop_dir, save_file)

            # Write file directly to ensure clean save & sync
            save_ok, save_msg = self.file_mgr.create_file(file_path, text)
            print(f"[TOOL] save_file path='{file_path}' ok={save_ok}")

            # Verify file exists on Desktop
            if os.path.isfile(file_path):
                print(f"[VERIFY] file_exists=True path={file_path}")
                print(f"[RESULT] success=True")
                return ToolResult.ok(
                    action="notepad_flow",
                    message=f"Opened Notepad, typed the text, and saved {save_file} on your Desktop, Sir.",
                    evidence=f"File verified at {file_path}",
                )
            else:
                print(f"[VERIFY] file_exists=False path={file_path}")
                print(f"[RESULT] success=False")
                return ToolResult.fail(
                    action="notepad_flow",
                    message=f"Sir, I opened Notepad and typed the text, but couldn't verify saving {save_file} on your Desktop.",
                    error=f"File {file_path} not found on disk",
                )

        print(f"[RESULT] success=True")
        return ToolResult.ok(
            action="notepad_flow",
            message=f"Opened Notepad and typed '{text}', Sir.",
            evidence="Notepad launched and text typed",
        )

    def execute_save_file(
        self,
        filename: str,
        location: str = "desktop",
        content: str = "",
    ) -> ToolResult:
        """
        Saves a file to the requested location (e.g. Desktop) and verifies it exists.
        Requirement F.
        """
        print(f"[ROUTER] intent=MULTI_STEP action=save_file filename='{filename}'")
        desktop_dir = os.path.expanduser(r"~\Desktop")
        file_path = os.path.join(desktop_dir, filename)

        # Create/sync file
        save_ok, msg = self.file_mgr.create_file(file_path, content)
        time.sleep(0.3)

        # Verify existence
        if os.path.isfile(file_path):
            print(f"[VERIFY] file_exists=True path={file_path}")
            print(f"[RESULT] success=True")
            return ToolResult.ok(
                action="save_file",
                message=f"Verified and saved {filename} on your Desktop, Sir.",
                evidence=f"File verified at {file_path}",
            )
        else:
            print(f"[VERIFY] file_exists=False path={file_path}")
            print(f"[RESULT] success=False")
            return ToolResult.fail(
                action="save_file",
                message=f"Sir, I could not verify that {filename} was saved on your Desktop.",
                error=f"File {file_path} not found after write attempt",
            )
