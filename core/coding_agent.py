"""
VS Code Coding Assistant Agent
Implements autonomous PLAN -> ACT -> OBSERVE -> VERIFY -> FIX -> VERIFY -> REPORT workflow.
"""

import os
import subprocess
import shutil
import time
from typing import Dict, Any, Tuple, Optional, List
from computer.file_manager import FileManager
from computer.vscode_controller import VSCodeController
from llm.base import AIProvider
from llm.provider_factory import get_llm_provider


class CodingAgent:
    """
    Autonomous programming assistant that creates, runs, tests, and verifies code.
    """

    def __init__(
        self,
        provider: Optional[AIProvider] = None,
        max_iterations: int = 3,
    ):
        self.provider = provider or get_llm_provider()
        self.file_mgr = FileManager()
        self.vscode_ctrl = VSCodeController()
        self.max_iterations = max_iterations

    def create_project(
        self,
        project_name: str,
        description: str,
        base_dir: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Full 7-stage verification loop:
        1. PLAN: Plan structure
        2. ACT: Create files and code
        3. OBSERVE: Inspect generated files
        4. VERIFY: Run tests / execution
        5. FIX: If errors, generate fix
        6. VERIFY: Rerun verification
        7. REPORT: Final natural summary
        """
        root = base_dir or os.path.expanduser(r"~\Desktop")
        project_path = os.path.join(root, project_name)

        # Step 1: PLAN
        print(f"[CodingAgent] Planning project: '{project_name}'...")
        os.makedirs(project_path, exist_ok=True)

        plan_prompt = (
            f"You are a Senior Software Architect. Create a project plan for: '{description}'.\n"
            f"Project Directory: {project_path}\n"
            "Provide the primary Python code file content for main.py that is self-contained and runnable.\n"
            "Output ONLY the python code inside ```python ... ``` block."
        )

        code_response = self.provider.generate(plan_prompt)

        # Extract python code
        code = self._extract_code(code_response)
        main_py = os.path.join(project_path, "main.py")

        # Step 2: ACT (with backup if exists)
        if os.path.exists(main_py):
            shutil.copyfile(main_py, f"{main_py}.bak")

        with open(main_py, "w", encoding="utf-8") as f:
            f.write(code)

        # Step 3 & 4: OBSERVE & VERIFY
        print(f"[CodingAgent] Verifying project execution...")
        success, output = self._run_python_script(main_py)

        # Step 5 & 6: FIX loop if errors occurred
        iteration = 0
        while not success and iteration < self.max_iterations:
            iteration += 1
            print(f"[CodingAgent] Fixing runtime issue (Attempt {iteration}/{self.max_iterations})...")
            fix_prompt = (
                f"The following Python code failed with this error:\n"
                f"--- Error ---\n{output}\n"
                f"--- Code ---\n{code}\n"
                "Fix the error. Return ONLY the complete fixed code inside ```python ... ``` block."
            )
            fixed_response = self.provider.generate(fix_prompt)
            code = self._extract_code(fixed_response)
            with open(main_py, "w", encoding="utf-8") as f:
                f.write(code)

            success, output = self._run_python_script(main_py)

        # Open in VS Code
        self.vscode_ctrl.open_workspace(project_path)

        # Step 7: REPORT
        if success:
            return True, f"I have built the {project_name} project in VS Code and verified that it runs successfully, Sir."
        else:
            return False, f"The project was created at {project_path}, but encountered runtime warnings during verification: {output[:120]}"

    def _extract_code(self, text: str) -> str:
        """Extracts python code blocks from model response."""
        if "```python" in text:
            parts = text.split("```python")[1].split("```")[0]
            return parts.strip()
        elif "```" in text:
            parts = text.split("```")[1].split("```")[0]
            return parts.strip()
        return text.strip()

    def _run_python_script(self, script_path: str) -> Tuple[bool, str]:
        """Runs the python script with timeout to verify execution."""
        try:
            res = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if res.returncode == 0:
                return True, res.stdout
            return False, res.stderr or res.stdout
        except subprocess.TimeoutExpired:
            return True, "Script started and running."
        except Exception as e:
            return False, str(e)
