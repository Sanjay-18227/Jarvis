"""
JARVIS VS Code Skill
Handles:
- VS Code workspace and project launching (e.g. "open Kumar in VS Code")
- Local deterministic folder resolution
- File creation and execution inside active workspace
- Advertises: open_folder, open_file, create_file, run_file
"""

import os
import sys
import time
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from skills.base import BaseSkill
from core.tool_result import ToolResult
from computer.app_indexer import AppIndexer, _find_vscode_path
from computer.file_manager import FileManager
from core.context_memory import get_context_memory


class VSCodeSkill(BaseSkill):
    """Visual Studio Code automation and workspace manager."""

    def __init__(
        self,
        app_indexer: Optional[AppIndexer] = None,
        file_mgr: Optional[FileManager] = None,
    ):
        self.indexer = app_indexer or AppIndexer()
        self.file_mgr = file_mgr or FileManager()

    @property
    def name(self) -> str:
        return "vscode"

    @property
    def description(self) -> str:
        return "Visual Studio Code workspace management, project loading, and file execution."

    @property
    def supported_actions(self) -> List[str]:
        return ["open_folder", "open_file", "create_file", "run_file"]

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        int_clean = intent.lower().strip()
        params = params or {}
        if int_clean == "open_folder_in_app" and params.get("app") in ["vscode", "code"]:
            return True
        if int_clean in ["vscode", "open_vscode"]:
            return True
        ctx = context or get_context_memory()
        if ctx.get_active_app() == "VS Code" and int_clean in ["open_folder", "open_file", "create_file", "run_file"]:
            return True
        return False

    @staticmethod
    def resolve_folder_path(folder_name: str) -> Optional[str]:
        """
        Deterministically searches likely locations for target folder:
        Desktop, Documents, user profile home, current workspace.
        """
        if not folder_name:
            return None

        # Clean folder name
        clean_name = folder_name.strip().strip("'\"")
        if clean_name.lower().endswith(" folder"):
            clean_name = clean_name[:-7].strip()

        # Check if already absolute path
        if os.path.isabs(clean_name) and os.path.isdir(clean_name):
            return os.path.normpath(clean_name)

        search_roots = [
            os.path.expanduser(r"~\Desktop"),
            os.path.expanduser(r"~\Documents"),
            os.path.expanduser("~"),
            os.getcwd(),
            os.path.dirname(os.getcwd()),
        ]

        # 1. Exact match in search roots
        for root in search_roots:
            candidate = os.path.join(root, clean_name)
            if os.path.isdir(candidate):
                return os.path.normpath(candidate)

        # 2. Case-insensitive match in search roots (immediate children)
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            try:
                for entry in os.listdir(root):
                    if entry.lower() == clean_name.lower():
                        full = os.path.join(root, entry)
                        if os.path.isdir(full):
                            return os.path.normpath(full)
            except Exception:
                pass

        # 3. If not found, default to Desktop/<folder_name>
        desktop_target = os.path.join(os.path.expanduser(r"~\Desktop"), clean_name)
        return os.path.normpath(desktop_target)

    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        ctx = context or get_context_memory()
        act = action.lower().strip()
        vscode_bin = _find_vscode_path() or "code"

        if act in ["open_folder", "open_folder_in_app", "load_project"]:
            raw_folder = params.get("folder") or params.get("folder_name") or params.get("target") or ctx.get_active_folder()
            if not raw_folder:
                # Launch empty VS Code
                res = self.indexer.launch_app("vscode")
                ctx.update(active_app="VS Code", last_tool="open_vscode", last_result="VS Code launched")
                return res

            target_path = self.resolve_folder_path(raw_folder)
            folder_display = os.path.basename(target_path) if target_path else raw_folder

            # If folder does not exist, create it cleanly
            if target_path and not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)

            print(f"[TOOL] Launching VS Code with folder: '{target_path}'")
            try:
                subprocess.Popen(
                    [vscode_bin, target_path],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                return ToolResult.fail(action="open_folder", target=target_path, error=str(e))

            # Verify VS Code is running
            verified = False
            for _ in range(10):
                time.sleep(0.3)
                if self.indexer.is_process_running("Code.exe"):
                    verified = True
                    break

            evidence = f"folder='{target_path}', process_running={verified}"
            ctx.update(
                active_app="VS Code",
                active_folder=folder_display,
                last_tool="open_folder",
                last_result=f"Opened {folder_display} in VS Code",
            )
            print(f"[VERIFY] vscode_process_verified={verified}")
            print(f"[RESULT] success={verified}")

            return ToolResult.ok(
                action="open_folder",
                target=target_path,
                evidence=evidence,
                message=f"I opened the {folder_display} folder in VS Code, Sir.",
            )

        elif act == "create_file":
            filename = params.get("filename") or params.get("file")
            folder = params.get("folder") or ctx.get_active_folder()
            content = params.get("content", "")

            if not filename:
                return ToolResult.fail(action="create_file", error="No filename specified.")

            target_dir = self.resolve_folder_path(folder) if folder else os.path.expanduser(r"~\Desktop")
            file_path = os.path.join(target_dir, filename)

            os.makedirs(target_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Open the new file in VS Code
            subprocess.Popen([vscode_bin, file_path], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            evidence = f"file='{file_path}', exists={os.path.isfile(file_path)}"
            ctx.update(
                active_app="VS Code",
                active_folder=os.path.basename(target_dir),
                active_file=filename,
                last_tool="create_file",
                last_result=f"Created {filename}",
            )
            return ToolResult.ok(
                action="create_file",
                target=file_path,
                evidence=evidence,
                message=f"Created {filename} inside {os.path.basename(target_dir)} and opened it in VS Code, Sir.",
            )

        elif act == "run_file":
            filename = params.get("filename") or params.get("file") or ctx.get_active_file()
            folder = params.get("folder") or ctx.get_active_folder()

            if not filename:
                return ToolResult.fail(action="run_file", error="No file specified to run.")

            target_dir = self.resolve_folder_path(folder) if folder else os.path.expanduser(r"~\Desktop")
            file_path = os.path.join(target_dir, filename)

            if not os.path.isfile(file_path):
                # Search directly inside Desktop
                alt_path = os.path.join(os.path.expanduser(r"~\Desktop"), filename)
                if os.path.isfile(alt_path):
                    file_path = alt_path

            if not os.path.isfile(file_path):
                return ToolResult.fail(action="run_file", target=filename, error=f"File '{filename}' not found.")

            try:
                proc = subprocess.run(
                    [sys.executable, file_path],
                    capture_output=True,
                    text=True,
                    timeout=15.0,
                    cwd=os.path.dirname(file_path),
                )
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()
                evidence = f"file='{file_path}', returncode={proc.returncode}, stdout='{stdout}', stderr='{stderr}'"
                ctx.update(active_file=filename, last_tool="run_file", last_result=stdout or "Script executed")
                msg = f"Executed {filename}, Sir." + (f" Output: {stdout}" if stdout else "")
                return ToolResult.ok(action="run_file", target=file_path, evidence=evidence, message=msg)
            except Exception as e:
                return ToolResult.fail(action="run_file", target=file_path, error=str(e))

        return ToolResult.fail(action=act, error=f"Unsupported VS Code action: '{action}'")
