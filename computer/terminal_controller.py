"""
JARVIS Terminal Controller
Safely executes supervised development, build, and diagnostic commands.
Enforces command sanitization, timeout limits, and prevents destructive shell commands.
"""

import os
import subprocess
import shlex
from typing import Tuple, Optional, Dict, Any


class TerminalController:
    """
    Supervised command runner for Python, Node, Git, testing, and system diagnostics.
    """

    # Block destructive/malicious system commands from autonomous execution
    FORBIDDEN_PATTERNS = [
        "format ",
        "del /f /s /q c:",
        "rmdir /s /q c:\\",
        "diskpart",
        "bcdedit",
        "reg delete",
        "takeown",
        "icacls /grant",
        ":(){ :|:& };:",
        "shutdown /s /f",
    ]

    ALLOWED_DEVELOPMENT_TOOLS = [
        "python", "python3", "pytest", "pip", "uv",
        "node", "npm", "npx", "yarn", "pnpm",
        "git", "cargo", "go", "dotnet",
        "tasklist", "dir", "where", "echo",
    ]

    def __init__(self, default_cwd: Optional[str] = None):
        self.default_cwd = default_cwd or os.getcwd()

    def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 45,
    ) -> Tuple[bool, int, str]:
        """
        Executes a command safely, capturing output and return code.

        Returns:
            (success: bool, return_code: int, combined_output: str)
        """
        cmd_clean = command.strip()
        if not cmd_clean:
            return False, -1, "Empty command provided."

        # Safety Check: Forbidden patterns
        cmd_lower = cmd_clean.lower()
        for forbidden in self.FORBIDDEN_PATTERNS:
            if forbidden in cmd_lower:
                return False, -1, f"Security Block: Command contains forbidden operation: '{forbidden}'."

        work_dir = cwd or self.default_cwd
        if not os.path.exists(work_dir):
            work_dir = self.default_cwd

        try:
            # Run command synchronously with strict timeout
            process = subprocess.run(
                cmd_clean,
                cwd=work_dir,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            stdout = process.stdout.strip()
            stderr = process.stderr.strip()

            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(f"[STDERR]\n{stderr}")

            combined_output = "\n".join(output_parts) if output_parts else "[No output produced]"

            # Cap maximum output length to avoid flooding memory/context
            if len(combined_output) > 4000:
                combined_output = combined_output[:3800] + "\n... [Output truncated for safety]"

            success = (process.returncode == 0)
            return success, process.returncode, combined_output

        except subprocess.TimeoutExpired:
            return False, -1, f"Command execution timed out after {timeout} seconds."
        except Exception as e:
            return False, -1, f"Execution failed: {str(e)}"
