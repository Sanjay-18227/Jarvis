"""
JARVIS Project Detector Subsystem
Fulfills Section 21 & 22: Project Detection & Autonomous Goal Execution.
Inspects directories for build files, configuration, and dependencies to accurately
identify frameworks (FastAPI, Flask, Django, Streamlit, React, Next.js, Vite, Node)
and determines the correct run and test commands.
"""

import os
import json
from typing import Dict, Any, Optional, List


class ProjectDetector:
    """
    Analyzes software repositories to detect framework, entry points, and runner commands.
    """

    def detect_project(self, project_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Inspects directory contents and returns structured project profile.
        """
        work_dir = project_dir or os.getcwd()
        if not os.path.isdir(work_dir):
            return {"type": "unknown", "framework": "unknown", "run_cmd": "echo No project found"}

        files = os.listdir(work_dir)
        lower_files = [f.lower() for f in files]

        profile = {
            "root": work_dir,
            "type": "unknown",
            "framework": "unknown",
            "entry_point": None,
            "run_cmd": None,
            "test_cmd": None,
            "has_venv": False,
        }

        # Check virtual environments
        for venv_dir in ["venv", ".venv", "env"]:
            if venv_dir in files and os.path.isdir(os.path.join(work_dir, venv_dir)):
                profile["has_venv"] = True
                break

        # 1. Python Project Detection
        if any(f in lower_files for f in ["requirements.txt", "pyproject.toml", "setup.py", "pipfile"]) or any(f.endswith(".py") for f in files):
            profile["type"] = "python"
            profile["test_cmd"] = "pytest"

            # Check specific frameworks
            reqs_content = ""
            if "requirements.txt" in lower_files:
                try:
                    with open(os.path.join(work_dir, "requirements.txt"), "r", encoding="utf-8", errors="replace") as rf:
                        reqs_content = rf.read().lower()
                except Exception:
                    pass

            if "streamlit" in reqs_content or any("streamlit" in f for f in lower_files):
                profile["framework"] = "streamlit"
                entry = "app.py" if "app.py" in lower_files else "main.py"
                profile["entry_point"] = entry
                profile["run_cmd"] = f"streamlit run {entry}"

            elif "fastapi" in reqs_content:
                profile["framework"] = "fastapi"
                profile["entry_point"] = "main.py"
                profile["run_cmd"] = "uvicorn main:app --reload"

            elif "flask" in reqs_content:
                profile["framework"] = "flask"
                profile["entry_point"] = "app.py" if "app.py" in lower_files else "main.py"
                profile["run_cmd"] = f"python {profile['entry_point']}"

            elif "django" in reqs_content or "manage.py" in lower_files:
                profile["framework"] = "django"
                profile["entry_point"] = "manage.py"
                profile["run_cmd"] = "python manage.py runserver"

            else:
                profile["framework"] = "standard_python"
                entry = "main.py" if "main.py" in lower_files else ("app.py" if "app.py" in lower_files else None)
                if not entry:
                    py_files = [f for f in files if f.endswith(".py")]
                    entry = py_files[0] if py_files else "main.py"
                profile["entry_point"] = entry
                profile["run_cmd"] = f"python {entry}"

            return profile

        # 2. Node / Web Project Detection
        if "package.json" in lower_files:
            profile["type"] = "node"
            profile["test_cmd"] = "npm test"

            try:
                with open(os.path.join(work_dir, "package.json"), "r", encoding="utf-8") as pf:
                    pkg_data = json.load(pf)
                    scripts = pkg_data.get("scripts", {})
                    deps = {**pkg_data.get("dependencies", {}), **pkg_data.get("devDependencies", {})}

                    if "next" in deps or any("next.config" in f for f in lower_files):
                        profile["framework"] = "nextjs"
                        profile["run_cmd"] = "npm run dev"
                    elif "vite" in deps or any("vite.config" in f for f in lower_files):
                        profile["framework"] = "vite"
                        profile["run_cmd"] = "npm run dev"
                    elif "react" in deps:
                        profile["framework"] = "react"
                        profile["run_cmd"] = "npm start"
                    elif "dev" in scripts:
                        profile["run_cmd"] = "npm run dev"
                    elif "start" in scripts:
                        profile["run_cmd"] = "npm start"
                    else:
                        profile["run_cmd"] = "node index.js"
            except Exception:
                profile["run_cmd"] = "npm run dev"

            return profile

        return profile
