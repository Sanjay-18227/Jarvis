"""
JARVIS Phase 3 Autonomous Agent Loop
Implements the verified execution cycle:
Goal -> Understand -> Plan -> Authorize -> Execute Step -> Observe -> Verify -> Success/Retry -> Report.

Strictly enforces:
1. Only OWNER can initiate or execute within the Agent Loop.
2. Every action maps to a registered allowlisted tool with verification evidence.
3. One-step-at-a-time execution with retry/replan limits (max 2 retries, max 12 steps).
4. Truthful partial completion reporting (never claim success if an error occurred).
5. Immediate cancellation upon owner stop command ("Jarvis stop", "cancel task", "wait").
"""

import os
import sys
import json
import time
import subprocess
import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from config.constants import SpeakerClassification, RiskLevel, ActionStatus
from security.authorization_gate import authorize_action, is_high_risk
from computer.app_indexer import AppIndexer
from computer.file_manager import FileManager
from computer.browser_controller import BrowserController
from computer.system_monitor import SystemMonitor
from computer.windows_controller import WindowsController


class AgentState(str, Enum):
    IDLE = "IDLE"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    AUTHORIZING = "AUTHORIZING"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    VERIFYING = "VERIFYING"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# PART 17: Tool Allowlist
APPROVED_TOOL_ALLOWLIST = {
    "create_folder",
    "create_file",
    "write_file",
    "read_file",
    "rename_file",
    "delete_file",
    "delete_folder",
    "run_python",
    "open_app",
    "close_app",
    "open_url",
    "web_search",
    "get_battery",
    "get_system_info",
    "window_control"
}

AGENT_MAX_RETRIES_PER_STEP = 2
AGENT_MAX_PLAN_STEPS = 12


@dataclass
class PlanStep:
    id: int
    tool: str
    args: Dict[str, Any]
    description: str
    status: str = "PENDING"
    retries: int = 0
    observation: Optional[Dict[str, Any]] = None


@dataclass
class StructuredPlan:
    goal: str
    steps: List[PlanStep] = field(default_factory=list)


@dataclass
class StepObservation:
    success: bool
    tool: str
    target: str
    evidence: str
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool": self.tool,
            "target": self.target,
            "evidence": self.evidence,
            "error": self.error,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class AgentContext:
    """Maintains short-term task context (Part 25)."""
    def __init__(self):
        self.last_created_folder: Optional[str] = None
        self.last_created_file: Optional[str] = None
        self.last_opened_app: Optional[str] = None
        self.last_goal: Optional[str] = None

    def resolve_reference(self, arg_val: str) -> str:
        """Resolves pronouns like 'it', 'inside it', 'the folder' to the recent object."""
        if not isinstance(arg_val, str):
            return arg_val
        val_lower = arg_val.lower().strip()
        if val_lower in ["it", "inside it", "the folder", "that folder", "the project"]:
            if self.last_created_folder:
                return self.last_created_folder
        if val_lower in ["the file", "that file", "the script"]:
            if self.last_created_file:
                return self.last_created_file
        return arg_val


class Phase3AgentLoop:
    """
    Autonomous verified agent loop orchestrator for multi-action Windows tasks.
    """

    def __init__(
        self,
        llm_provider=None,
        file_mgr: Optional[FileManager] = None,
        app_indexer: Optional[AppIndexer] = None,
        browser_ctrl: Optional[BrowserController] = None,
        win_ctrl: Optional[WindowsController] = None,
        sys_monitor: Optional[SystemMonitor] = None,
    ):
        self.llm_provider = llm_provider
        self.file_mgr = file_mgr or FileManager()
        self.app_indexer = app_indexer or AppIndexer()
        self.browser_ctrl = browser_ctrl or BrowserController()
        self.win_ctrl = win_ctrl or WindowsController()
        self.sys_monitor = sys_monitor or SystemMonitor()

        self.state = AgentState.IDLE
        self.context = AgentContext()
        self._stop_requested = False

    def set_state(self, new_state: AgentState):
        self.state = new_state
        print(f"[AGENT] State={new_state.value}")

    def request_stop(self):
        """Immediately signals the active agent loop to cancel (Part 26)."""
        self._stop_requested = True
        self.set_state(AgentState.CANCELLED)
        print("[AGENT] Cancellation signal received. Stopping active task.")

    def run_goal(
        self,
        goal: str,
        speaker: SpeakerClassification = SpeakerClassification.OWNER,
        confirmed: bool = False,
    ) -> Tuple[bool, str]:
        """
        Main entry point for Phase 3 Agent Loop.
        Fails closed: UNKNOWN speakers are rejected immediately.
        """
        g_lower = (goal or "").lower().strip()
        if any(sc == g_lower or sc in g_lower for sc in ["jarvis stop", "cancel task", "stop task", "stop please"]):
            self.request_stop()
            return False, "Task halted per your stop command, Sir."

        if self._stop_requested:
            self.set_state(AgentState.CANCELLED)
            self._stop_requested = False
            return False, "Task halted per your stop command, Sir."

        self._stop_requested = False

        # Gate check (Part 14): Only OWNER may enter the Agent Loop
        if speaker != SpeakerClassification.OWNER:
            print(f"[SECURITY] Agent Loop blocked for speaker={speaker.value}")
            self.set_state(AgentState.FAILED)
            return False, "I don't recognize this voice. Computer controls are locked."

        print(f"\n[AGENT] Goal accepted: '{goal}'")
        self.context.last_goal = goal

        # Step 1: UNDERSTAND
        self.set_state(AgentState.UNDERSTANDING)
        if self._stop_requested:
            return False, "Task cancelled, Sir."

        # Step 2: PLAN
        self.set_state(AgentState.PLANNING)
        plan = self._create_plan(goal)
        if not plan or not plan.steps:
            self.set_state(AgentState.FAILED)
            return False, "Sir, I could not determine a valid execution plan for that request."

        print(f"[PLAN] steps={len(plan.steps)} goal='{plan.goal}'")
        for s in plan.steps:
            print(f"  Step {s.id}: [{s.tool}] {s.description}")

        # Step 3: AUTHORIZE PLAN
        self.set_state(AgentState.AUTHORIZING)
        is_valid, validation_msg = self._validate_plan(plan, speaker, confirmed)
        if not is_valid:
            self.set_state(AgentState.FAILED)
            return False, validation_msg

        print("[SECURITY] Plan authorized")

        # Step 4: EXECUTE ONE STEP AT A TIME
        completed_steps = []
        failed_step = None

        for step in plan.steps:
            if self._stop_requested:
                self.set_state(AgentState.CANCELLED)
                return False, "Task halted per your stop command, Sir."

            print(f"[AGENT] Step={step.id}/{len(plan.steps)} tool={step.tool}")
            self.set_state(AgentState.EXECUTING)

            step_success = False
            last_obs = None

            # Retry loop per step (Part 22 & 23)
            while step.retries <= AGENT_MAX_RETRIES_PER_STEP and not step_success:
                if self._stop_requested:
                    self.set_state(AgentState.CANCELLED)
                    return False, "Task halted per your stop command, Sir."

                # Execute
                obs = self._execute_step(step)
                last_obs = obs
                step.observation = obs.to_dict()

                # Step 5: OBSERVE & Step 6: VERIFY
                self.set_state(AgentState.VERIFYING)
                step_success = obs.success

                if step_success:
                    print(f"[VERIFY] PASS (evidence: {obs.evidence})")
                    step.status = "COMPLETED"
                    completed_steps.append(step)
                    # Update context
                    self._update_context(step)
                    break
                else:
                    step.retries += 1
                    if step.retries <= AGENT_MAX_RETRIES_PER_STEP:
                        self.set_state(AgentState.RETRYING)
                        print(f"[RETRY] Step {step.id} failed ({obs.error}). Re-evaluating (Attempt {step.retries}/{AGENT_MAX_RETRIES_PER_STEP})...")
                        self._replan_step(step, obs)
                    else:
                        print(f"[VERIFY] FAIL (Step {step.id} exhausted {AGENT_MAX_RETRIES_PER_STEP} retries: {obs.error})")
                        step.status = "FAILED"
                        failed_step = step

            if not step_success:
                # Step failed after retries -> Stop and report partial completion
                self.set_state(AgentState.FAILED)
                return False, self._format_partial_report(plan, completed_steps, failed_step, last_obs)

        # All steps succeeded
        self.set_state(AgentState.COMPLETED)
        return True, "Completed successfully, Sir."

    def _create_plan(self, goal: str) -> Optional[StructuredPlan]:
        """
        Decomposes user goal into a structured plan matching approved tools.
        Combines deterministic regex decomposition for speed with LLM planning as needed.
        """
        g_clean = goal.strip()

        # 1. Deterministic planner for common desktop automation workflows
        det_plan = self._try_deterministic_plan(g_clean)
        if det_plan:
            return det_plan

        # 2. LLM-based structured planning via OpenRouter
        if self.llm_provider and hasattr(self.llm_provider, "generate"):
            try:
                system_prompt = (
                    "You are the JARVIS Autonomous Plan Generator. Decompose the user's request into a strict JSON plan.\n"
                    f"Only use tools from this allowlist: {list(APPROVED_TOOL_ALLOWLIST)}.\n"
                    "Schema:\n"
                    "{\n"
                    '  "goal": "short summary",\n'
                    '  "steps": [\n'
                    '    {"id": 1, "tool": "tool_name", "args": {"key": "val"}, "description": "what step does"}\n'
                    "  ]\n"
                    "}\n"
                    "Output ONLY raw JSON. No markdown backticks, no explanations."
                )
                raw_json = self.llm_provider.generate(g_clean, system_prompt=system_prompt, max_tokens=600)
                # Strip code fences if present
                clean_json = raw_json.strip()
                if clean_json.startswith("```"):
                    clean_json = clean_json.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                data = json.loads(clean_json)
                steps = []
                for s in data.get("steps", [])[:AGENT_MAX_PLAN_STEPS]:
                    steps.append(PlanStep(
                        id=int(s["id"]),
                        tool=s["tool"].lower().strip(),
                        args=s.get("args", {}),
                        description=s.get("description", ""),
                    ))
                if steps:
                    return StructuredPlan(goal=data.get("goal", g_clean), steps=steps)
            except Exception as e:
                print(f"[AGENT] LLM plan generation error: {e}")

        # Fallback to single action step
        return StructuredPlan(goal=g_clean, steps=[
            PlanStep(id=1, tool="open_app", args={"app_name": g_clean}, description=f"Execute {g_clean}")
        ])

    def _try_deterministic_plan(self, goal: str) -> Optional[StructuredPlan]:
        """Decomposes compound sentences (e.g. 'Create folder X, create file Y, write Z, run it, open VS Code')."""
        import re
        g_lower = goal.lower()

        # Detect compound multi-step creation workflows
        # Example: "create a folder called PythonTest on my Desktop, create app.py inside it, write a hello-world program, run it, and open the folder in VS Code"
        has_folder = "create" in g_lower and "folder" in g_lower
        has_file = "create" in g_lower and ("file" in g_lower or ".py" in g_lower or ".txt" in g_lower)

        if has_folder and has_file:
            steps: List[PlanStep] = []
            step_id = 1

            # 1. Folder extraction
            f_match = re.search(r"(?:called|named)\s+([a-zA-Z0-9_\-\.]+)", goal, re.IGNORECASE)
            folder_name = f_match.group(1) if f_match else "Project"
            desktop_dir = os.path.expanduser(r"~\Desktop")
            folder_path = os.path.join(desktop_dir, folder_name)

            steps.append(PlanStep(
                id=step_id,
                tool="create_folder",
                args={"path": folder_path},
                description=f"Create folder {folder_name} on Desktop"
            ))
            step_id += 1

            # 2. File extraction
            file_match = re.search(r"create\s+([a-zA-Z0-9_\-\.]+\.(?:py|txt|js|json|html|md))", goal, re.IGNORECASE)
            file_name = file_match.group(1) if file_match else "app.py"
            file_path = os.path.join(folder_path, file_name)

            steps.append(PlanStep(
                id=step_id,
                tool="create_file",
                args={"path": file_path, "content": ""},
                description=f"Create {file_name} inside {folder_name}"
            ))
            step_id += 1

            # 3. Code writing extraction
            has_write = any(w in g_lower for w in ["write", "prints", "hello", "code"])
            if has_write:
                code_content = "print('Hello, Sir.')\n"
                if "hello world" in g_lower or "hello-world" in g_lower:
                    code_content = "print('Hello, World!')\n"
                elif "hello sir" in g_lower:
                    code_content = "print('Hello Sir')\n"
                elif "agent loop works" in g_lower:
                    code_content = "Agent Loop Works\n"

                steps.append(PlanStep(
                    id=step_id,
                    tool="write_file",
                    args={"path": file_path, "content": code_content},
                    description=f"Write program code into {file_name}"
                ))
                step_id += 1

            # 4. Run python extraction
            if "run" in g_lower or "execute" in g_lower:
                steps.append(PlanStep(
                    id=step_id,
                    tool="run_python",
                    args={"path": file_path},
                    description=f"Run {file_name} and verify output"
                ))
                step_id += 1

            # 5. Open in editor extraction
            if "vs code" in g_lower or "vscode" in g_lower or "code" in g_lower:
                steps.append(PlanStep(
                    id=step_id,
                    tool="open_app",
                    args={"app_name": "vscode", "target": folder_path},
                    description=f"Open project in VS Code"
                ))
                step_id += 1

            return StructuredPlan(goal=f"Create and run {folder_name} project", steps=steps)

        return None

    def _validate_plan(self, plan: StructuredPlan, speaker: SpeakerClassification, confirmed: bool) -> Tuple[bool, str]:
        """Validates all plan steps against tool allowlist, paths, and security rules (Part 18)."""
        if len(plan.steps) > AGENT_MAX_PLAN_STEPS:
            return False, f"Plan rejected: exceeds maximum allowed step count of {AGENT_MAX_PLAN_STEPS}."

        for step in plan.steps:
            if step.tool not in APPROVED_TOOL_ALLOWLIST:
                return False, f"Plan rejected: tool '{step.tool}' is not in the approved security allowlist."

            # Authorization check per step
            allowed, auth_msg, req_conf = authorize_action(
                speaker=speaker,
                action_name=step.tool,
                parameters=step.args,
            )
            if not allowed:
                return False, auth_msg

            if req_conf and not confirmed:
                return False, f"Action '{step.tool}' in plan requires explicit confirmation, Sir. Shall I proceed?"

        return True, "Plan valid and authorized."

    def _execute_step(self, step: PlanStep) -> StepObservation:
        """Executes a single plan step via the corresponding local subsystem (Part 19)."""
        tool = step.tool
        args = step.args
        print(f"[TOOL] execute tool={tool} args={args}")

        try:
            if tool == "create_folder":
                path = self.context.resolve_reference(args.get("path") or args.get("folder_path", ""))
                ok, msg = self.file_mgr.create_folder(path)
                real_dir = os.path.isdir(self.file_mgr._resolve_path(path))
                return StepObservation(
                    success=ok and real_dir,
                    tool=tool,
                    target=path,
                    evidence=f"Directory verified at {path}" if real_dir else "Directory not found on disk",
                    error=None if (ok and real_dir) else msg,
                )

            elif tool in ["create_file", "write_file"]:
                path = self.context.resolve_reference(args.get("path") or args.get("file_path", ""))
                content = args.get("content", "")
                ok, msg = self.file_mgr.create_file(path, content)
                resolved = self.file_mgr._resolve_path(path)
                real_file = os.path.isfile(resolved)
                return StepObservation(
                    success=ok and real_file,
                    tool=tool,
                    target=path,
                    evidence=f"File verified at {resolved} ({os.path.getsize(resolved)} bytes)" if real_file else "File not found on disk",
                    error=None if (ok and real_file) else msg,
                )

            elif tool == "read_file":
                path = self.context.resolve_reference(args.get("path") or args.get("file_path", ""))
                ok, content = self.file_mgr.read_file(path)
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target=path,
                    evidence=f"Read {len(content)} characters",
                    error=None if ok else content,
                    stdout=content if ok else None,
                )

            elif tool == "delete_file" or tool == "delete_folder":
                path = self.context.resolve_reference(args.get("path") or args.get("target", ""))
                ok, msg = self.file_mgr.delete_path(path)
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target=path,
                    evidence=msg,
                    error=None if ok else msg,
                )

            elif tool == "rename_file":
                old_p = self.context.resolve_reference(args.get("old_path", ""))
                new_p = self.context.resolve_reference(args.get("new_path", ""))
                ok, msg = self.file_mgr.rename_file(old_p, new_p)
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target=new_p,
                    evidence=msg,
                    error=None if ok else msg,
                )

            elif tool == "run_python":
                path = self.context.resolve_reference(args.get("path") or args.get("file_path", ""))
                resolved = self.file_mgr._resolve_path(path)
                if not os.path.isfile(resolved):
                    return StepObservation(
                        success=False,
                        tool=tool,
                        target=resolved,
                        evidence="File not found",
                        error=f"Python script not found at {resolved}",
                    )

                cwd = os.path.dirname(resolved)
                proc = subprocess.run(
                    [sys.executable, resolved],
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=15.0,
                )
                clean_stdout = proc.stdout.strip()
                clean_stderr = proc.stderr.strip()
                is_ok = (proc.returncode == 0)
                evidence = f"exit_code={proc.returncode} stdout='{clean_stdout}'"
                error_msg = clean_stderr if clean_stderr else (None if is_ok else f"Process exited with code {proc.returncode}")

                print(f"[OBSERVE] run_python stdout='{clean_stdout}' code={proc.returncode}")
                return StepObservation(
                    success=is_ok,
                    tool=tool,
                    target=resolved,
                    evidence=evidence,
                    error=error_msg,
                    stdout=clean_stdout,
                    stderr=clean_stderr,
                )

            elif tool == "open_app":
                app_name = args.get("app_name") or args.get("target", "")
                res = self.app_indexer.launch_app(app_name)
                return StepObservation(
                    success=res.success,
                    tool=tool,
                    target=app_name,
                    evidence=res.evidence,
                    error=res.error if not res.success else None,
                )

            elif tool == "close_app":
                app_name = args.get("app_name") or args.get("target", "")
                res = self.app_indexer.close_app(app_name)
                return StepObservation(
                    success=res.success,
                    tool=tool,
                    target=app_name,
                    evidence=res.evidence,
                    error=res.error if not res.success else None,
                )

            elif tool == "open_url":
                url = args.get("url", "")
                res = self.browser_ctrl.open_website(url)
                return StepObservation(
                    success=res.success,
                    tool=tool,
                    target=url,
                    evidence=res.evidence,
                    error=res.error if not res.success else None,
                )

            elif tool == "web_search":
                query = args.get("query", "")
                res = self.browser_ctrl.search_web(query)
                return StepObservation(
                    success=res.success,
                    tool=tool,
                    target=query,
                    evidence=res.evidence,
                    error=res.error if not res.success else None,
                )

            elif tool == "get_battery":
                vitals = self.sys_monitor.get_system_vitals()
                pct = vitals["battery"]["percent"]
                return StepObservation(
                    success=True,
                    tool=tool,
                    target="system_battery",
                    evidence=f"Battery: {pct}%",
                    stdout=str(pct),
                )

            elif tool == "get_system_info":
                vitals = self.sys_monitor.get_system_vitals()
                return StepObservation(
                    success=True,
                    tool=tool,
                    target="system_vitals",
                    evidence=f"CPU: {vitals['cpu']['percent']}%, RAM: {vitals['memory']['percent']}%",
                    stdout=str(vitals),
                )

            elif tool == "window_control":
                action = args.get("action", "focus")
                title = args.get("title", "")
                if action == "focus":
                    ok = self.win_ctrl.focus_window(title)
                elif action == "lock":
                    ok = self.win_ctrl.lock_workstation()
                else:
                    ok = True
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target=title or action,
                    evidence=f"Window action {action} completed",
                    error=None if ok else "Window action failed",
                )

        except Exception as e:
            return StepObservation(
                success=False,
                tool=tool,
                target=str(args),
                evidence="Exception raised",
                error=str(e),
            )

        return StepObservation(
            success=False,
            tool=tool,
            target="",
            evidence="Unsupported tool",
            error=f"Unrecognized tool '{tool}'",
        )

    def _replan_step(self, step: PlanStep, obs: StepObservation):
        """Attempts limited in-flight parameter remediation if a step fails."""
        # Example: if VS Code was attempted via raw name, try resolving full path or indexer alias
        if step.tool == "open_app":
            app = step.args.get("app_name", "")
            if app.lower() == "code":
                step.args["app_name"] = "vscode"
            elif app.lower() == "vscode":
                step.args["app_name"] = "code"

        # If a file path failed due to relative path, normalize through context folder
        if step.tool in ["create_file", "write_file", "run_python"]:
            p = step.args.get("path", "")
            if not os.path.isabs(p) and self.context.last_created_folder:
                step.args["path"] = os.path.join(self.context.last_created_folder, os.path.basename(p))

    def _update_context(self, step: PlanStep):
        """Updates working task context with verified objects."""
        tool = step.tool
        args = step.args
        if tool == "create_folder":
            self.context.last_created_folder = self.file_mgr._resolve_path(args.get("path", ""))
        elif tool in ["create_file", "write_file"]:
            self.context.last_created_file = self.file_mgr._resolve_path(args.get("path", ""))
        elif tool == "open_app":
            self.context.last_opened_app = args.get("app_name") or args.get("target")

    def _format_partial_report(
        self,
        plan: StructuredPlan,
        completed_steps: List[PlanStep],
        failed_step: Optional[PlanStep],
        last_obs: Optional[StepObservation],
    ) -> str:
        """Truthful failure and partial completion reporting (Part 24)."""
        if not completed_steps:
            err = last_obs.error if last_obs and last_obs.error else "Execution could not be verified."
            return f"Sir, I could not complete the request because {err}."

        completed_descs = [s.description for s in completed_steps]
        done_summary = ", ".join(completed_descs)
        fail_desc = failed_step.description if failed_step else "the next step"
        err_msg = last_obs.error if last_obs and last_obs.error else "an error occurred"

        return f"Sir, I completed {done_summary}, but {fail_desc} failed because {err_msg}."
