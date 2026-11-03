"""
JARVIS File Skill
Handles:
- Folder creation, search, and navigation (e.g. "Create a folder called Demo on Desktop", "Open Demo")
- File creation (e.g. "Create agent.py inside Kumar")
- Python execution (e.g. "Run agent.py")
- Advertises: create_folder, create_file, rename_file, delete_file, list_dir, search_folder, run_python
"""

import os
import sys
import subprocess
from typing import Dict, Any, List, Optional
from skills.base import BaseSkill
from core.tool_result import ToolResult
from computer.file_manager import FileManager
from core.context_memory import get_context_memory


class FileSkill(BaseSkill):
    """Local filesystem operations and script execution skill."""

    def __init__(self, file_mgr: Optional[FileManager] = None):
        self.file_mgr = file_mgr or FileManager()

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return "Local filesystem folder and file creation, management, and script execution."

    @property
    def supported_actions(self) -> List[str]:
        return ["create_folder", "create_file", "rename_file", "delete_file", "list_dir", "search_folder", "run_python"]

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        int_clean = intent.lower().strip()
        return int_clean in [
            "create_folder", "create_file", "rename_file", "delete_file",
            "list_dir", "search_folder", "run_python", "file_action", "open_folder"
        ]

    @staticmethod
    def resolve_target_dir(folder_name: Optional[str] = None, location: Optional[str] = None) -> str:
        """Resolves target directory using Desktop default and fuzzy match."""
        desktop = os.path.expanduser(r"~\Desktop")
        if not folder_name:
            if location and "doc" in location.lower():
                return os.path.expanduser(r"~\Documents")
            return desktop

        clean = folder_name.strip().strip("'\"")
        if clean.lower().endswith(" folder"):
            clean = clean[:-7].strip()

        # Check search locations
        search_roots = [
            desktop,
            os.path.expanduser(r"~\Documents"),
            os.path.expanduser("~"),
            os.getcwd(),
        ]

        for root in search_roots:
            cand = os.path.join(root, clean)
            if os.path.isdir(cand):
                return os.path.normpath(cand)

        for root in search_roots:
            if not os.path.isdir(root):
                continue
            try:
                for entry in os.listdir(root):
                    if entry.lower() == clean.lower():
                        full = os.path.join(root, entry)
                        if os.path.isdir(full):
                            return os.path.normpath(full)
            except Exception:
                pass

        # Default: on Desktop
        loc_root = os.path.expanduser(r"~\Documents") if (location and "doc" in location.lower()) else desktop
        return os.path.normpath(os.path.join(loc_root, clean))

    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        ctx = context or get_context_memory()
        act = action.lower().strip()

        if act == "create_folder":
            raw_name = params.get("folder") or params.get("folder_name") or params.get("name") or "New Folder"
            location = params.get("location")
            target_path = self.resolve_target_dir(raw_name, location)

            ok, msg = self.file_mgr.create_folder(target_path)
            if ok:
                folder_base = os.path.basename(target_path)
                evidence = f"folder='{target_path}', exists={os.path.isdir(target_path)}"
                ctx.update(active_folder=folder_base, last_tool="create_folder", last_result=f"Created {folder_base}")
                return ToolResult.ok(
                    action="create_folder",
                    target=target_path,
                    evidence=evidence,
                    message=f"I have created the '{folder_base}' folder on your Desktop, Sir.",
                )
            return ToolResult.fail(action="create_folder", target=target_path, error=msg)

        elif act in ["open_folder", "explore_folder"]:
            raw_name = params.get("folder") or params.get("folder_name") or params.get("name") or ctx.get_active_folder()
            if not raw_name:
                return ToolResult.fail(action="open_folder", error="No folder specified.")

            target_path = self.resolve_target_dir(raw_name)
            if not os.path.exists(target_path):
                # If Demo or requested folder, create it
                os.makedirs(target_path, exist_ok=True)

            # Open in Windows File Explorer
            subprocess.Popen(["explorer.exe", target_path])
            folder_base = os.path.basename(target_path)
            evidence = f"folder='{target_path}', exists={os.path.isdir(target_path)}"
            ctx.update(active_folder=folder_base, last_tool="open_folder", last_result=f"Opened {folder_base}")
            return ToolResult.ok(
                action="open_folder",
                target=target_path,
                evidence=evidence,
                message=f"Opened {folder_base} in File Explorer, Sir.",
            )

        elif act == "create_file":
            filename = params.get("filename") or params.get("file") or params.get("name")
            folder = params.get("folder") or ctx.get_active_folder()
            content = params.get("content", "")

            if not filename:
                return ToolResult.fail(action="create_file", error="No filename specified.")

            parent_dir = self.resolve_target_dir(folder) if folder else os.path.expanduser(r"~\Desktop")
            os.makedirs(parent_dir, exist_ok=True)
            file_path = os.path.join(parent_dir, filename)

            ok, msg = self.file_mgr.create_file(file_path, content=content)
            if ok:
                evidence = f"file='{file_path}', size={os.path.getsize(file_path)}"
                ctx.update(
                    active_folder=os.path.basename(parent_dir),
                    active_file=filename,
                    last_tool="create_file",
                    last_result=f"Created {filename}",
                )
                folder_str = f" inside {os.path.basename(parent_dir)}" if folder else ""
                return ToolResult.ok(
                    action="create_file",
                    target=file_path,
                    evidence=evidence,
                    message=f"Created {filename}{folder_str}, Sir.",
                )
            return ToolResult.fail(action="create_file", target=file_path, error=msg)

        elif act == "run_python":
            filename = params.get("filename") or params.get("file") or ctx.get_active_file()
            folder = params.get("folder") or ctx.get_active_folder()

            if not filename:
                return ToolResult.fail(action="run_python", error="No python file specified.")

            parent_dir = self.resolve_target_dir(folder) if folder else os.path.expanduser(r"~\Desktop")
            file_path = os.path.join(parent_dir, filename)

            if not os.path.isfile(file_path):
                # Search directly inside Desktop
                alt_path = os.path.join(os.path.expanduser(r"~\Desktop"), filename)
                if os.path.isfile(alt_path):
                    file_path = alt_path

            if not os.path.isfile(file_path):
                return ToolResult.fail(action="run_python", target=filename, error=f"File '{filename}' not found.")

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
                ctx.update(active_file=filename, last_tool="run_python", last_result=stdout or "Script executed")
                msg = f"Executed {filename}, Sir." + (f" Output: {stdout}" if stdout else "")
                return ToolResult.ok(action="run_python", target=file_path, evidence=evidence, message=msg)
            except Exception as e:
                return ToolResult.fail(action="run_python", target=file_path, error=str(e))

        elif act in ["rename_file", "rename"]:
            old_name = params.get("old_name") or params.get("old_path") or params.get("src")
            new_name = params.get("new_name") or params.get("new_path") or params.get("dst")
            folder = params.get("folder") or ctx.get_active_folder()

            if not old_name or not new_name:
                return ToolResult.fail(action="rename_file", error="Both old and new file names required.")

            parent_dir = self.resolve_target_dir(folder) if folder else os.path.expanduser(r"~\Desktop")
            src_path = os.path.join(parent_dir, old_name) if not os.path.isabs(old_name) else old_name
            dst_path = os.path.join(parent_dir, new_name) if not os.path.isabs(new_name) else new_name

            if not os.path.exists(src_path):
                alt_src = os.path.join(os.path.expanduser(r"~\Desktop"), old_name)
                if os.path.exists(alt_src):
                    src_path = alt_src
                    dst_path = os.path.join(os.path.expanduser(r"~\Desktop"), new_name)

            ok, msg = self.file_mgr.rename_file(src_path, dst_path)
            print(f"[TOOL] rename_file src='{src_path}' dst='{dst_path}'")
            if ok:
                evidence = f"renamed '{src_path}' to '{dst_path}'"
                print("[VERIFY] renamed=True")
                print("[RESULT] success=True")
                ctx.update(active_file=new_name, last_tool="rename_file", last_result=f"Renamed {old_name} to {new_name}")
                return ToolResult.ok(
                    action="rename_file",
                    target=dst_path,
                    evidence=evidence,
                    message=f"Renamed {old_name} to {new_name}, Sir.",
                )
            print(f"[VERIFY] renamed=False error='{msg}'")
            print("[RESULT] success=False")
            return ToolResult.fail(action="rename_file", target=src_path, error=msg)

        return ToolResult.fail(action=act, error=f"Unsupported file action: '{action}'")
