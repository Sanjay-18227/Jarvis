"""
VS Code Controller Subsystem
Opens workspaces, folders, and files directly in Visual Studio Code.
"""

import os
import subprocess
from typing import Tuple, Optional


class VSCodeController:
    """
    Manages VS Code workspace launches.
    """

    def open_workspace(self, path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Opens a folder or workspace in VS Code.
        """
        target = os.path.abspath(path) if path else os.getcwd()
        try:
            subprocess.Popen(
                ["code", target],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, f"VS Code opened for workspace: {os.path.basename(target)}"
        except Exception as e:
            return False, f"Error launching VS Code: {e}"

    def open_file(self, file_path: str, line: Optional[int] = None) -> Tuple[bool, str]:
        """
        Opens a specific file at an optional line number in VS Code.
        """
        target = os.path.abspath(file_path)
        cmd = ["code", "-g", f"{target}:{line}"] if line else ["code", target]
        try:
            subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, f"Opened file in VS Code: {os.path.basename(target)}"
        except Exception as e:
            return False, f"Failed opening file in VS Code: {e}"
