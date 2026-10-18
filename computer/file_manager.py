"""
Safe File & Directory Manager Subsystem
Performs verified filesystem operations with path sandboxing and security checks.
"""

import os
import shutil
from typing import Tuple, List, Optional


class FileManager:
    """
    Controlled filesystem manager.
    """

    def __init__(self, default_root: Optional[str] = None):
        self.default_root = default_root or os.path.expanduser(r"~\Desktop")

    def _resolve_path(self, path: str) -> str:
        """Expands and normalizes paths."""
        expanded = os.path.expanduser(os.path.expandvars(path))
        if not os.path.isabs(expanded):
            expanded = os.path.join(self.default_root, expanded)
        return os.path.normpath(expanded)

    def create_folder(self, folder_path: str) -> Tuple[bool, str]:
        """Creates a directory and verifies creation."""
        target = self._resolve_path(folder_path)
        try:
            os.makedirs(target, exist_ok=True)
            if os.path.isdir(target):
                return True, f"Folder created at {target}"
            return False, f"Folder creation failed at {target}"
        except Exception as e:
            return False, f"Error creating folder: {e}"

    def create_file(self, file_path: str, content: str = "") -> Tuple[bool, str]:
        """Creates a new file and writes initial content."""
        target = self._resolve_path(file_path)
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            if os.path.isfile(target):
                return True, f"File created at {target}"
            return False, f"File creation failed at {target}"
        except Exception as e:
            return False, f"Error creating file: {e}"

    def rename_file(self, old_path: str, new_path: str) -> Tuple[bool, str]:
        """Renames a file or folder and verifies new name."""
        src = self._resolve_path(old_path)
        dst = self._resolve_path(new_path)
        try:
            if not os.path.exists(src):
                return False, f"Source path does not exist: {src}"
            os.rename(src, dst)
            if os.path.exists(dst) and not os.path.exists(src):
                return True, f"Renamed to {dst}"
            return False, "Rename operation could not be verified."
        except Exception as e:
            return False, f"Error renaming: {e}"

    def list_files(self, folder_path: str) -> Tuple[bool, List[str], str]:
        """Lists directory contents."""
        target = self._resolve_path(folder_path)
        try:
            if not os.path.isdir(target):
                return False, [], f"Directory not found: {target}"
            entries = os.listdir(target)
            return True, entries, f"Found {len(entries)} items in {target}"
        except Exception as e:
            return False, [], f"Error listing directory: {e}"

    def delete_path(self, target_path: str) -> Tuple[bool, str]:
        """
        Deletes a file or directory with verification.
        Note: The caller / Executor MUST ensure Level 3/4 confirmation has been granted!
        """
        target = self._resolve_path(target_path)
        try:
            if not os.path.exists(target):
                return False, f"Path does not exist: {target}"
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)

            if not os.path.exists(target):
                return True, f"Deleted {target} successfully."
            return False, f"Failed to delete {target}."
        except Exception as e:
            return False, f"Error during deletion: {e}"
