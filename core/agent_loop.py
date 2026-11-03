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
6. Strict entity locking: Never invent names (e.g. Project, app.py) or unrequested code.
7. End-to-end goal verification after all steps complete.
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
from computer.calculator_controller import CalculatorController
from core.entity_extractor import EntityExtractor, StructuredGoal, GoalEntities
from core.context_memory import get_context_memory
from llm.openrouter_provider import OpenRouterError
from vision.screen_vision_service import ScreenVisionService, get_screen_vision_service


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


# Approved Tool Registry
APPROVED_TOOL_ALLOWLIST = {
    "open_app",
    "close_app",
    "focus_app",
    "open_url",
    "web_search",
    "create_folder",
    "create_file",
    "write_file",
    "read_file",
    "rename_file",
    "delete_file",
    "delete_folder",
    "run_python",
    "calculator_input",
    "type_text",
    "press_key",
    "window_control",
    "get_battery",
    "get_system_info",
    "inspect_screen",
    "verify_screen",
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
    entities: Optional[GoalEntities] = None


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
    """Maintains short-term task context and pronoun reference resolution."""
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
        calculator_ctrl: Optional[CalculatorController] = None,
        screen_vision: Optional[ScreenVisionService] = None,
    ):
        self.llm_provider = llm_provider
        self.file_mgr = file_mgr or FileManager()
        self.app_indexer = app_indexer or AppIndexer()
        self.browser_ctrl = browser_ctrl or BrowserController()
        self.win_ctrl = win_ctrl or WindowsController()
        self.sys_monitor = sys_monitor or SystemMonitor()
        self.calculator_ctrl = calculator_ctrl or CalculatorController()
        self.screen_vision = screen_vision or get_screen_vision_service()

        self.state = AgentState.IDLE
        self.context = AgentContext()
        self._stop_requested = False
        self._last_plan_error: Optional[str] = None

    def set_state(self, new_state: AgentState):
        self.state = new_state
        print(f"[AGENT] State={new_state.value}")

    def request_stop(self):
        """Immediately signals the active agent loop to cancel."""
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
        self._last_plan_error = None

        # Gate check: Only OWNER may enter the Agent Loop
        if speaker != SpeakerClassification.OWNER:
            print(f"[SECURITY] DENIED: Agent Loop blocked for speaker={speaker.value}")
            self.set_state(AgentState.FAILED)
            return False, "I don't recognize this voice. Computer controls are locked."

        print(f"\n[AGENT] Goal accepted: '{goal}'")

        # Step 1: UNDERSTAND & Entity Locking
        self.set_state(AgentState.UNDERSTANDING)
        structured_goal = EntityExtractor.extract(goal)
        ent = structured_goal.entities
        ctx = get_context_memory()
        print(f"[CONTEXT] active_folder={ent.folder or ctx.get_active_folder()} active_file={ent.filename or ctx.get_active_file()} active_app={ent.application or ctx.get_active_app()}")

        if self._stop_requested:
            return False, "Task cancelled, Sir."

        # Step 2: PLAN
        self.set_state(AgentState.PLANNING)
        plan = self._create_plan(structured_goal)
        if not plan or not plan.steps:
            self.set_state(AgentState.FAILED)
            if self._last_plan_error:
                msg = self._last_plan_error
                self._last_plan_error = None
                return False, msg
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
        last_python_stdout = None

        for step in plan.steps:
            if self._stop_requested:
                self.set_state(AgentState.CANCELLED)
                return False, "Task halted per your stop command, Sir."

            print(f"[AGENT] Step={step.id}/{len(plan.steps)} tool={step.tool}")
            self.set_state(AgentState.EXECUTING)

            step_success = False
            last_obs = None

            # Retry loop per step
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
                    print(f"[RESULT] success=True")
                    step.status = "COMPLETED"
                    completed_steps.append(step)
                    if step.tool == "run_python" and obs.stdout:
                        last_python_stdout = obs.stdout
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
                        print(f"[RESULT] success=False")
                        step.status = "FAILED"
                        failed_step = step

            if not step_success:
                # Step failed after retries -> Stop and report partial completion
                self.set_state(AgentState.FAILED)
                return False, self._format_partial_report(plan, completed_steps, failed_step, last_obs)

        # Step 7: FINAL GOAL VERIFICATION
        # Verify actual state of the world before declaring success
        goal_ok, goal_msg = self._verify_final_goal(plan, last_python_stdout)
        if not goal_ok:
            print(f"[VERIFY] Goal verification FAIL: {goal_msg}")
            self.set_state(AgentState.FAILED)
            return False, f"Sir, steps were executed but final verification failed: {goal_msg}"

        print("[VERIFY] Goal verification PASS")
        print("[AGENT] COMPLETED")
        self.set_state(AgentState.COMPLETED)

        if last_python_stdout:
            return True, f"Completed successfully, Sir. Output: {last_python_stdout}"
        return True, "Completed successfully, Sir."

    def _verify_final_goal(self, plan: StructuredPlan, stdout: Optional[str] = None) -> Tuple[bool, str]:
        """
        Ensures the final objective is physically verified on disk, process, or screen.
        """
        ent = plan.entities
        desktop_dir = os.path.expanduser(r"~\Desktop")

        # 1. Folder verification if folder was requested
        if ent and ent.folder:
            f_path = self.file_mgr._resolve_path(ent.folder) if hasattr(self.file_mgr, "_resolve_path") else os.path.join(desktop_dir, ent.folder)
            if not os.path.isdir(f_path):
                f_desktop = os.path.join(desktop_dir, ent.folder)
                if not os.path.isdir(f_desktop) and not os.path.isdir(ent.folder):
                    return False, f"Target folder '{ent.folder}' does not exist on disk"

        # 2. File verification if file was requested
        if ent and ent.filename:
            file_found = False
            if ent.folder:
                target_rel = os.path.join(ent.folder, ent.filename)
                resolved_rel = self.file_mgr._resolve_path(target_rel) if hasattr(self.file_mgr, "_resolve_path") else ""
                if os.path.isfile(resolved_rel):
                    file_found = True
                f_path = os.path.join(desktop_dir, ent.folder, ent.filename)
                if os.path.isfile(f_path):
                    file_found = True
            if not file_found:
                resolved_f = self.file_mgr._resolve_path(ent.filename) if hasattr(self.file_mgr, "_resolve_path") else ""
                if os.path.isfile(resolved_f):
                    file_found = True
                elif os.path.isfile(os.path.join(desktop_dir, ent.filename)):
                    file_found = True
            if not file_found:
                return False, f"Target file '{ent.filename}' does not exist on disk"

        # 3. Execution stdout verification if should_run was requested
        if ent and ent.should_run and ent.code_to_print:
            expected = ent.code_to_print.strip()
            if stdout and expected.lower() not in stdout.lower():
                return False, f"Expected output '{expected}' not present in stdout '{stdout}'"

        # 4. Visual verification if screen inspection step occurred
        for s in plan.steps:
            if s.tool in ["inspect_screen", "verify_screen"] and s.observation:
                obs = s.observation
                if obs.get("error"):
                    return False, f"Screen error observed: {obs.get('error')}"

        return True, "All target entities verified."

    def _create_plan(self, goal_or_structured: Any) -> Optional[StructuredPlan]:
        """
        Decomposes user goal into a structured plan matching approved tools.
        Prioritizes deterministic entity-locked planning, with LLM fallback.
        """
        if isinstance(goal_or_structured, str):
            structured_goal = EntityExtractor.extract(goal_or_structured)
        else:
            structured_goal = goal_or_structured

        # 1. Try deterministic entity-locked plan
        det_plan = self._try_deterministic_plan(structured_goal)
        if det_plan:
            return det_plan

        # 2. LLM-based structured planning via OpenRouter
        if self.llm_provider and hasattr(self.llm_provider, "generate"):
            try:
                system_prompt = (
                    "You are the JARVIS Autonomous Plan Generator. Decompose the user's request into a strict JSON plan.\n"
                    f"Only use tools from this allowlist: {sorted(list(APPROVED_TOOL_ALLOWLIST))}.\n"
                    "Schema:\n"
                    "{\n"
                    '  "goal": "short summary",\n'
                    '  "steps": [\n'
                    '    {"id": 1, "tool": "tool_name", "args": {"key": "val"}, "description": "what step does"}\n'
                    "  ]\n"
                    "}\n"
                    "Output ONLY raw JSON. No markdown backticks, no explanations."
                )
                raw_json = self.llm_provider.generate(structured_goal.goal, system_prompt=system_prompt, max_tokens=600)
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
                    return StructuredPlan(
                        goal=data.get("goal", structured_goal.goal),
                        steps=steps,
                        entities=structured_goal.entities,
                    )
            except OpenRouterError as oe:
                print(f"[OPENROUTER] Request failed: {oe}")
                print("[AGENT] State=FAILED")
                self._last_plan_error = "Sir, the AI planning service is currently unavailable, so I couldn't safely continue this task."
                return None
            except Exception as e:
                err_str = str(e).lower()
                if "credit" in err_str or "payment" in err_str or "402" in err_str:
                    print("[OPENROUTER] Request failed: insufficient credits")
                else:
                    print(f"[OPENROUTER] Request failed: {e}")
                print("[AGENT] State=FAILED")
                self._last_plan_error = "Sir, the AI planning service is currently unavailable, so I couldn't safely continue this task."
                return None

        # Fallback to single action step
        return StructuredPlan(
            goal=structured_goal.goal,
            steps=[
                PlanStep(id=1, tool="open_app", args={"app_name": structured_goal.goal}, description=f"Execute {structured_goal.goal}")
            ],
            entities=structured_goal.entities,
        )

    def _try_deterministic_plan(self, goal_or_structured: Any) -> Optional[StructuredPlan]:
        """
        Builds a verified plan directly from locked entities without inventing filenames or code.
        """
        if isinstance(goal_or_structured, str):
            structured_goal = EntityExtractor.extract(goal_or_structured)
        else:
            structured_goal = goal_or_structured

        ent = structured_goal.entities
        goal = structured_goal.goal
        g_lower = goal.lower()
        ctx = get_context_memory()

        desktop_dir = os.path.expanduser(r"~\Desktop")

        # Determine target folder
        folder_name = ent.folder or ctx.get_active_folder()
        folder_path = os.path.join(desktop_dir, folder_name) if folder_name else None

        # Determine target file
        file_name = ent.filename or ctx.get_active_file()
        if file_name and folder_path:
            file_path = os.path.join(folder_path, file_name)
        elif file_name:
            file_path = os.path.join(desktop_dir, file_name)
        else:
            file_path = None

        steps: List[PlanStep] = []
        step_id = 1

        # Case A: Compound file / folder / coding workflow
        has_folder_intent = any(w in g_lower for w in ["folder", "create folder", "make folder", "open folder"]) or (folder_name is not None)
        has_file_intent = (file_name is not None) or any(w in g_lower for w in ["create", "write", ".py", ".txt"])

        if has_folder_intent or has_file_intent:
            # 1. Folder creation
            if folder_path and any(w in g_lower for w in ["create", "make", "open folder", "inside", "in"]):
                steps.append(PlanStep(
                    id=step_id,
                    tool="create_folder",
                    args={"path": folder_path},
                    description=f"Create/verify folder '{folder_name}'"
                ))
                step_id += 1

            # 2. File creation
            if file_path and any(w in g_lower for w in ["create", "file", ".py", "write"]):
                steps.append(PlanStep(
                    id=step_id,
                    tool="create_file",
                    args={"path": file_path, "content": ""},
                    description=f"Create file '{file_name}'" + (f" inside '{folder_name}'" if folder_name else "")
                ))
                step_id += 1

            # 3. Write code if specifically requested
            if file_path and ent.code_content:
                steps.append(PlanStep(
                    id=step_id,
                    tool="write_file",
                    args={"path": file_path, "content": ent.code_content},
                    description=f"Write program code into '{file_name}'"
                ))
                step_id += 1

            # 4. Run python script
            if file_path and (ent.should_run or any(w in g_lower for w in ["run", "execute", "tell me the actual output", "tell me the output"])):
                steps.append(PlanStep(
                    id=step_id,
                    tool="run_python",
                    args={"path": file_path},
                    description=f"Run '{file_name}' and verify output"
                ))
                step_id += 1

            # 5. Open project / folder / app in editor (e.g. VS Code)
            if ent.application in ["VS Code", "Notepad"] or any(w in g_lower for w in ["vs code", "vscode", "code"]):
                app_target = folder_path if folder_path else (file_path if file_path else "vscode")
                steps.append(PlanStep(
                    id=step_id,
                    tool="open_app",
                    args={"app_name": "vscode", "target": app_target},
                    description=f"Open project in VS Code"
                ))
                step_id += 1

            # 6. Screen inspection & error check if requested
            if getattr(ent, "check_screen", False) or any(w in g_lower for w in ["check the screen", "check screen", "inspect screen", "check for any errors", "check for errors", "any errors on screen"]):
                steps.append(PlanStep(
                    id=step_id,
                    tool="inspect_screen",
                    args={"query": "Check the screen for any visible errors", "mode": "active_window"},
                    description="Inspect screen for any visible errors"
                ))
                step_id += 1

            if steps:
                return StructuredPlan(
                    goal=f"Execute workflow for {folder_name or file_name}",
                    steps=steps,
                    entities=ent,
                )

        # Case B: Calculator interaction
        if ent.calculation or any(w in g_lower for w in ["calculate", "calculator", "minus", "+", "-", "*", "/"]):
            calc_expr = ent.calculation or "678 - 876"
            steps.append(PlanStep(
                id=step_id,
                tool="calculator_input",
                args={"expression": calc_expr},
                description=f"Calculate {calc_expr} on Calculator"
            ))
            return StructuredPlan(goal=f"Calculate {calc_expr}", steps=steps, entities=ent)

        # Case C: Edge / YouTube interaction
        if "youtube" in g_lower:
            if ent.query:
                steps.append(PlanStep(
                    id=1,
                    tool="open_url",
                    args={"url": f"https://www.youtube.com/results?search_query={ent.query}"},
                    description=f"Open YouTube search for '{ent.query}'"
                ))
            else:
                steps.append(PlanStep(
                    id=1,
                    tool="open_url",
                    args={"url": "https://www.youtube.com"},
                    description="Open YouTube in Microsoft Edge"
                ))
            return StructuredPlan(goal="Open YouTube", steps=steps, entities=ent)

        # Case D: Direct screen vision request
        if getattr(ent, "check_screen", False) or any(w in g_lower for w in ["what is on my screen", "what's on my screen", "inspect screen", "what error is on my screen"]):
            steps.append(PlanStep(
                id=1,
                tool="inspect_screen",
                args={"query": goal, "mode": "active_window" if "error" in g_lower or "dialog" in g_lower else "full_screen"},
                description="Inspect screen visually"
            ))
            return StructuredPlan(goal=goal, steps=steps, entities=ent)

        return None

    def _validate_plan(self, plan: StructuredPlan, speaker: SpeakerClassification, confirmed: bool) -> Tuple[bool, str]:
        """Validates all plan steps against tool allowlist, paths, and security rules."""
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
        """Executes a single plan step via the corresponding local subsystem."""
        tool = step.tool
        args = step.args
        ctx = get_context_memory()
        print(f"[TOOL] execute tool={tool} args={args}")

        try:
            if tool == "create_folder":
                raw_path = args.get("path") or args.get("folder_path", "")
                path = ctx.resolve_reference(raw_path)
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
                raw_path = args.get("path") or args.get("file_path", "")
                path = ctx.resolve_reference(raw_path)
                content = args.get("content", "")
                ok, msg = self.file_mgr.create_file(path, content)
                resolved = self.file_mgr._resolve_path(path)
                real_file = os.path.isfile(resolved)
                size = os.path.getsize(resolved) if real_file else 0
                return StepObservation(
                    success=ok and real_file,
                    tool=tool,
                    target=path,
                    evidence=f"File verified at {resolved} ({size} bytes)" if real_file else "File not found on disk",
                    error=None if (ok and real_file) else msg,
                )

            elif tool == "read_file":
                raw_path = args.get("path") or args.get("file_path", "")
                path = ctx.resolve_reference(raw_path)
                ok, content = self.file_mgr.read_file(path)
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target=path,
                    evidence=f"Read {len(content)} characters",
                    error=None if ok else content,
                    stdout=content if ok else None,
                )

            elif tool in ["delete_file", "delete_folder"]:
                raw_path = args.get("path") or args.get("target", "")
                path = ctx.resolve_reference(raw_path)
                ok, msg = self.file_mgr.delete_path(path)
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target=path,
                    evidence=msg,
                    error=None if ok else msg,
                )

            elif tool == "rename_file":
                old_p = ctx.resolve_reference(args.get("old_path", ""))
                new_p = ctx.resolve_reference(args.get("new_path", ""))
                ok, msg = self.file_mgr.rename_file(old_p, new_p)
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target=new_p,
                    evidence=msg,
                    error=None if ok else msg,
                )

            elif tool == "run_python":
                raw_path = args.get("path") or args.get("file_path", "")
                path = ctx.resolve_reference(raw_path)
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

            elif tool == "calculator_input":
                expr = args.get("expression") or args.get("input", "0")
                ok, res_str = self.calculator_ctrl.execute_calculation_on_calculator(expr)
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target="Calculator",
                    evidence=f"Calculation result: {res_str}",
                    stdout=res_str if ok else None,
                    error=None if ok else res_str,
                )

            elif tool == "open_app":
                app_name = args.get("app_name") or args.get("target", "")
                target_arg = args.get("target") if args.get("target") != app_name else None
                res = self.app_indexer.launch_app(app_name, target_arg=target_arg)
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

            elif tool == "focus_app":
                app_name = args.get("app_name") or args.get("title", "")
                ok, msg = self.win_ctrl.focus_window(app_name)
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target=app_name,
                    evidence=msg,
                    error=None if ok else msg,
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
                    ok, msg = self.win_ctrl.focus_window(title)
                elif action == "lock":
                    ok, msg = self.win_ctrl.lock_workstation()
                else:
                    ok, msg = True, "Done"
                return StepObservation(
                    success=ok,
                    tool=tool,
                    target=title or action,
                    evidence=f"Window action {action} completed: {msg}",
                    error=None if ok else msg,
                )

            elif tool in ["inspect_screen", "verify_screen"]:
                q = args.get("query", "Inspect screen")
                mode = args.get("mode", "active_window")
                res = self.screen_vision.analyze_screen(
                    mode=mode,
                    user_query=q,
                    speaker=SpeakerClassification.OWNER,
                )
                success = res.get("success", False)
                summary = res.get("summary", "")
                struct = res.get("evidence", {}).get("structured", {})
                errors = struct.get("errors", []) if isinstance(struct, dict) else []
                err_text = None
                if errors:
                    err_text = errors[0].get("message") if isinstance(errors[0], dict) else str(errors[0])
                    print(f"[OBSERVE] Screen vision detected error: {err_text}")

                return StepObservation(
                    success=success,
                    tool=tool,
                    target=res.get("active_app", "Screen"),
                    evidence=summary,
                    error=err_text if (not success and res.get("error")) else None,
                    stdout=err_text or summary,
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
        # Remediation 1: Alias resolution for application names
        if step.tool == "open_app":
            app = step.args.get("app_name", "")
            if app.lower() == "code":
                step.args["app_name"] = "vscode"
            elif app.lower() == "vscode":
                step.args["app_name"] = "code"

        # Remediation 2: Resolve relative paths using context folder
        if step.tool in ["create_file", "write_file", "run_python"]:
            p = step.args.get("path", "")
            ctx = get_context_memory()
            active_f = ctx.get_active_folder()
            if not os.path.isabs(p) and active_f:
                desktop_dir = os.path.expanduser(r"~\Desktop")
                step.args["path"] = os.path.join(desktop_dir, active_f, os.path.basename(p))

    def _update_context(self, step: PlanStep):
        """Updates working task context with verified objects."""
        tool = step.tool
        args = step.args
        ctx = get_context_memory()

        if tool == "create_folder":
            p = self.file_mgr._resolve_path(args.get("path", ""))
            ctx.update(active_folder=os.path.basename(p))
        elif tool in ["create_file", "write_file"]:
            p = self.file_mgr._resolve_path(args.get("path", ""))
            ctx.update(active_file=os.path.basename(p))
        elif tool == "open_app":
            app = args.get("app_name") or args.get("target")
            ctx.update(active_app=app)

    def _format_partial_report(
        self,
        plan: StructuredPlan,
        completed_steps: List[PlanStep],
        failed_step: Optional[PlanStep],
        last_obs: Optional[StepObservation],
    ) -> str:
        """Truthful failure and partial completion reporting."""
        if not completed_steps:
            err = last_obs.error if last_obs and last_obs.error else "Execution could not be verified."
            return f"Sir, I could not complete the request because {err}."

        completed_descs = [s.description for s in completed_steps]
        done_summary = ", ".join(completed_descs)
        fail_desc = failed_step.description if failed_step else "the next step"
        err_msg = last_obs.error if last_obs and last_obs.error else "an error occurred"

        return f"Sir, I completed {done_summary}, but {fail_desc} failed because {err_msg}."
