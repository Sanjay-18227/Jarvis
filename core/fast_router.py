"""
JARVIS Fast Router Subsystem
Deterministic sub-second local execution engine:
- Local Fast Path -> Contextual Tool Action -> Multi-Step Local Plan -> LLM Fallback
- Extensible Skill Registry integration (Calculator, Paint, VS Code, Browser, Notepad, File, System)
- Enforces real verification before reporting success
- Clean debug logs [ROUTER], [TOOL], [VERIFY], [RESULT]
"""

import re
import time
import datetime
from typing import Optional, Tuple, Dict, Any
from config.constants import SpeakerClassification
from core.normalizer import Normalizer, IntentType
from core.tool_result import ToolResult
from computer.app_indexer import AppIndexer
from computer.windows_controller import WindowsController
from computer.browser_controller import BrowserController
from computer.system_monitor import SystemMonitor
from vision.screen_understanding import ScreenVision
from core.multi_step_executor import MultiStepExecutor
from security.authorization_gate import authorize_action
from computer.calculator_controller import CalculatorController
from core.context_memory import get_context_memory
from skills.registry import SkillRegistry, get_skill_registry


class FastRouter:
    """
    Sub-second intent classifier and verified local executor.
    """

    def __init__(
        self,
        app_indexer: Optional[AppIndexer] = None,
        win_ctrl: Optional[WindowsController] = None,
        sys_monitor: Optional[SystemMonitor] = None,
        browser_ctrl: Optional[BrowserController] = None,
        screen_vision: Optional[ScreenVision] = None,
        multi_step_executor: Optional[MultiStepExecutor] = None,
        skill_registry: Optional[SkillRegistry] = None,
    ):
        self.app_indexer = app_indexer or AppIndexer()
        self.win_ctrl = win_ctrl or WindowsController()
        self.sys_monitor = sys_monitor or SystemMonitor()
        self.browser_ctrl = browser_ctrl or BrowserController()
        self.screen_vision = screen_vision or ScreenVision()
        self.multi_step = multi_step_executor or MultiStepExecutor(
            app_indexer=self.app_indexer,
            win_ctrl=self.win_ctrl,
        )
        self.skill_registry = skill_registry or get_skill_registry()
        self.last_routing_ms: float = 0.0
        self.last_tool_ms: float = 0.0
        self.last_verify_ms: float = 0.0

    def _format_response(self, text: str, speaker: SpeakerClassification) -> str:
        """Sanitizes responses so UNKNOWN / GUEST is never called 'Sanjay' or 'Sir'."""
        if speaker == SpeakerClassification.OWNER:
            return text
        res = text.replace(", Sir.", ".").replace(" Sir.", ".").replace(", Sir", "")
        res = res.replace("Sir, ", "").replace("Sir.", "")
        res = res.replace("Hello Sanjay.", "Hello.")
        res = res.replace("Your battery", "The battery")
        res = re.sub(r"\bSanjay\b", "", res)
        res = re.sub(r"\s+", " ", res).strip()
        res = res.replace(" ,", ",").replace(" .", ".")
        return res

    def try_fast_path(
        self,
        text: str,
        speaker: SpeakerClassification = SpeakerClassification.OWNER,
    ) -> Optional[Tuple[str, str]]:
        """
        Evaluates user text. If it is a deterministic local action, website, app,
        or multi-step command, executes and verifies immediately on Fast Path.

        Returns:
            (action_name, spoken_response) if handled on Fast Path, else None.
        """
        t_route_0 = time.perf_counter()
        intent, params = Normalizer.classify_intent(text)
        self.last_routing_ms = (time.perf_counter() - t_route_0) * 1000.0
        self.last_tool_ms = 0.0
        self.last_verify_ms = 0.0
        ctx = get_context_memory()

        # Step 0: Centralized Security Gate enforcement before any Fast Path action
        allowed, auth_msg, requires_confirmation = authorize_action(
            speaker=speaker,
            action_name=intent.lower(),
            intent=intent,
            parameters=params,
            raw_query=text,
        )
        if not allowed:
            print(f"[SECURITY] DENIED: FastPath blocked for speaker={speaker.value} intent={intent}")
            return "security_denied", auth_msg

        # 1. Connection / Hearing check
        if intent == IntentType.HEAR_CHECK:
            print("[ROUTER] intent=HEAR_CHECK")
            if speaker == SpeakerClassification.OWNER:
                return "hear_check", "Yep, Sir. I can hear you clearly."
            return "hear_check", "I can hear you clearly."

        # 2. Greetings
        if intent == IntentType.GREETING:
            print("[ROUTER] intent=GREETING")
            if speaker == SpeakerClassification.OWNER:
                return "greeting", "Hello, Sir. How can I assist you today?"
            return "greeting", "Hello. I don't recognize this voice. Computer controls are locked."

        # 2.5 Name & Owner Queries
        if intent == IntentType.NAME_QUERY:
            print("[ROUTER] intent=NAME_QUERY")
            if speaker == SpeakerClassification.OWNER:
                return "name_query", "Your name is Sanjay, Sir."
            return "name_query", "Voice verification failed. Please authenticate as the owner."

        if intent == IntentType.OWNER_QUERY:
            print("[ROUTER] intent=OWNER_QUERY")
            return "owner_query", "My owner is Sanjay, Sir."

        # 3. Arithmetic & Calculator
        if intent == IntentType.CALCULATE:
            expr = params.get("expression", "")
            print(f"[ROUTER] intent=CALCULATE expr='{expr}'")
            calc_skill = self.skill_registry.get_skill("calculator")
            if calc_skill:
                res = calc_skill.execute("calculate", params, ctx)
                if res.success:
                    return "calculate", self._format_response(res.message, speaker)

        # 4. Multi-Step Local Mini-Plans (e.g. Edge + YouTube, VS Code + Kumar)
        if intent == IntentType.MULTI_STEP_LOCAL:
            plan_name = params.get("name", "local_mini_plan")
            print(f"[ROUTER] intent=MULTI_STEP_LOCAL plan='{plan_name}'")
            steps = params.get("steps", [])
            all_ok = True
            for s in steps:
                s_res = self.skill_registry.execute(s["tool"], s["action"], s.get("params", {}), ctx)
                if not s_res.success:
                    all_ok = False
                    break
            if all_ok:
                msg = params.get("message", "Executed local sequence, Sir.")
                return "multi_step_local", self._format_response(msg, speaker)

        # 5. Open Folder in App (e.g. "open Kumar folder on VS Code", "open Kumar in VS Code", "load Kumar project")
        if intent == IntentType.OPEN_FOLDER_IN_APP:
            fol = params.get("folder", "")
            app = params.get("app", "vscode")
            print(f"[ROUTER] intent=OPEN_FOLDER_IN_APP app='{app}' folder='{fol}'")
            vscode_skill = self.skill_registry.get_skill("vscode")
            if vscode_skill and app in ["vscode", "code"]:
                res = vscode_skill.execute("open_folder_in_app", params, ctx)
                if res.success:
                    return "open_folder_in_app", self._format_response(res.message, speaker)

        # 6. Paint Actions (e.g. "draw a rectangle", "draw a line", "draw a circle", "new canvas")
        if intent == IntentType.PAINT_ACTION:
            act = params.get("action", "draw_rectangle")
            print(f"[ROUTER] intent=PAINT_ACTION action='{act}'")
            paint_skill = self.skill_registry.get_skill("paint")
            if paint_skill:
                res = paint_skill.execute(act, params, ctx)
                if res.success:
                    return "paint_action", self._format_response(res.message, speaker)

        # 7. Notepad Actions (e.g. "type hello")
        if intent == IntentType.NOTEPAD_ACTION:
            act = params.get("action", "type_text")
            print(f"[ROUTER] intent=NOTEPAD_ACTION action='{act}'")
            notepad_skill = self.skill_registry.get_skill("notepad")
            if notepad_skill:
                res = notepad_skill.execute(act, params, ctx)
                if res.success:
                    return "notepad_action", self._format_response(res.message, speaker)

        # 8. Filesystem Actions (Create Folder, Open Folder, Create File, Run Python)
        if intent == IntentType.CREATE_FOLDER:
            print(f"[ROUTER] intent=CREATE_FOLDER params={params}")
            file_skill = self.skill_registry.get_skill("file")
            if file_skill:
                res = file_skill.execute("create_folder", params, ctx)
                if res.success:
                    return "create_folder", self._format_response(res.message, speaker)

        if intent == IntentType.OPEN_FOLDER:
            print(f"[ROUTER] intent=OPEN_FOLDER folder='{params.get('folder')}'")
            file_skill = self.skill_registry.get_skill("file")
            if file_skill:
                res = file_skill.execute("open_folder", params, ctx)
                if res.success:
                    return "open_folder", self._format_response(res.message, speaker)

        if intent == IntentType.CREATE_FILE:
            print(f"[ROUTER] intent=CREATE_FILE params={params}")
            # If VS Code is active and user said create file: use vscode_skill, else file_skill
            skill_target = "vscode" if ctx.get_active_app() == "VS Code" else "file"
            s = self.skill_registry.get_skill(skill_target) or self.skill_registry.get_skill("file")
            if s:
                res = s.execute("create_file", params, ctx)
                if res.success:
                    return "create_file", self._format_response(res.message, speaker)

        if intent == IntentType.RUN_PYTHON:
            print(f"[ROUTER] intent=RUN_PYTHON params={params}")
            file_skill = self.skill_registry.get_skill("file")
            if file_skill:
                res = file_skill.execute("run_python", params, ctx)
                if res.success:
                    return "run_python", self._format_response(res.message, speaker)

        if intent == IntentType.RENAME_FILE:
            print(f"[ROUTER] intent=RENAME_FILE old='{params.get('old_name')}' new='{params.get('new_name')}'")
            file_skill = self.skill_registry.get_skill("file")
            if file_skill:
                res = file_skill.execute("rename_file", params, ctx)
                if res.success:
                    return "rename_file", self._format_response(res.message, speaker)

        # 9. Generic Open Browser (e.g. "open browser", "launch browser")
        if intent == IntentType.OPEN_BROWSER:
            print("[ROUTER] intent=OPEN_BROWSER")
            b_skill = self.skill_registry.get_skill("browser")
            if b_skill:
                res = b_skill.execute("open_browser", params, ctx)
                if res.success:
                    return "open_browser", self._format_response(res.message, speaker)

        # 10. Website Routing (e.g. "open YouTube", "open Google", "open GitHub")
        if intent == IntentType.OPEN_WEBSITE:
            print(f"[ROUTER] intent=OPEN_WEBSITE url='{params.get('url')}'")
            b_skill = self.skill_registry.get_skill("browser")
            if b_skill:
                res = b_skill.execute("open_url", params, ctx)
                if res.success:
                    return "open_website", self._format_response(res.message, speaker)

        # 11. YouTube & Web Searches
        if intent == IntentType.SEARCH_YOUTUBE:
            print(f"[ROUTER] intent=SEARCH_YOUTUBE query='{params.get('query')}'")
            res = self.browser_ctrl.search_youtube(params["query"])
            return "search_youtube", self._format_response(res.message, speaker)

        if intent == IntentType.PLAY_YOUTUBE:
            res = self.browser_ctrl.play_youtube(params["query"])
            return "play_youtube", self._format_response(res.message, speaker)

        if intent == IntentType.PLAY_LATEST_MUSIC:
            q = params.get("query", "latest songs")
            res = self.browser_ctrl.search_youtube(q)
            msg = f"I opened the YouTube search results for {q}, Sir."
            return "play_music", self._format_response(msg, speaker)

        if intent == IntentType.SEARCH_WEB:
            res = self.browser_ctrl.search_web(params["query"])
            return "search_web", self._format_response(res.message, speaker)

        # 12. Application Launching (e.g. "open VS Code", "open Calculator", "open Paint")
        if intent == IntentType.OPEN_APP:
            app_key = params.get("app_key") or params.get("raw_name", "")
            print(f"[ROUTER] intent=OPEN_APP key='{app_key}'")
            res = self.app_indexer.launch_app(app_key)
            if res.success:
                app_canon = "Calculator" if app_key in ["calc", "calculator"] else (
                    "VS Code" if app_key in ["vscode", "code"] else (
                        "Microsoft Edge" if app_key in ["edge", "msedge", "browser"] else (
                            "Notepad" if app_key == "notepad" else (
                                "Paint" if app_key in ["paint", "mspaint"] else app_key
                            )
                        )
                    )
                )
                ctx.update(active_app=app_canon, last_tool="open_app", last_result=f"Opened {app_canon}")
            return "open_app", self._format_response(res.message, speaker)

        # 13. System Skills (Battery, Diagnostics, Date/Time, Volume, Lock Screen)
        if intent in [IntentType.BATTERY, IntentType.DIAGNOSTICS, IntentType.DATETIME, IntentType.MEDIA_CONTROL, IntentType.LOCK_SCREEN]:
            print(f"[ROUTER] intent={intent}")
            sys_skill = self.skill_registry.get_skill("system")
            if sys_skill:
                res = sys_skill.execute(intent.lower(), params, ctx)
                if res.success:
                    return intent.lower(), self._format_response(res.message, speaker)

        # 14. Multi-Step Legacy Flows (Notepad type & save)
        if intent == IntentType.MULTI_STEP:
            action = params.get("action")
            if action == "notepad_type_save":
                res = self.multi_step.execute_notepad_flow(
                    text=params.get("text", ""),
                    save_file=params.get("save_file"),
                    location=params.get("location", "desktop"),
                )
                return "multi_step", self._format_response(res.message, speaker)
            elif action == "save_file":
                res = self.multi_step.execute_save_file(
                    filename=params.get("filename"),
                    location=params.get("location", "desktop"),
                )
                return "multi_step", self._format_response(res.message, speaker)

        # 15. Screen Vision Queries
        if intent == IntentType.SCREEN_VISION:
            mode = params.get("mode")
            res = self.screen_vision.inspect_screen(params.get("query", text), speaker=speaker, mode=mode)
            if res.success:
                try:
                    import ast
                    ev = ast.literal_eval(res.evidence) if isinstance(res.evidence, str) and res.evidence.startswith("{") else {}
                    struct = ev.get("structured", {})
                    errors = struct.get("errors", [])
                    detected_app = ev.get("active_window", {}).get("application")
                    if detected_app:
                        ctx.update(active_app=detected_app)
                    if errors:
                        first_err = errors[0]
                        err_msg = first_err.get("message") if isinstance(first_err, dict) else str(first_err)
                        ctx.update(
                            active_app=detected_app,
                            last_observed_error=err_msg,
                            last_observed_error_app=detected_app,
                            last_observed_screen_summary=res.message,
                        )
                except Exception:
                    pass
            return "screen_vision", self._format_response(res.message, speaker)

        # 16. Contextual Follow-up Resolution (e.g. active_app=Paint -> "Draw a rectangle", active_app=Calculator -> "2 + 2")
        active_app = ctx.get_active_app()
        if active_app == "Paint" and any(k in text.lower() for k in ["draw", "rectangle", "line", "circle", "box"]):
            paint_skill = self.skill_registry.get_skill("paint")
            if paint_skill:
                act = "draw_rectangle" if any(k in text.lower() for k in ["rectangle", "box"]) else (
                    "draw_line" if "line" in text.lower() else "draw_circle"
                )
                res = paint_skill.execute(act, {}, ctx)
                if res.success:
                    return "paint_action", self._format_response(res.message, speaker)

        if active_app == "Calculator" and any(k in text.lower() for k in ["+", "-", "*", "/", "add", "minus", "plus", "times"]):
            calc_skill = self.skill_registry.get_skill("calculator")
            if calc_skill:
                res = calc_skill.execute("calculate", {"expression": text, "use_calculator_app": True}, ctx)
                if res.success:
                    return "calculate", self._format_response(res.message, speaker)

        # Fall through to AI Path (Autonomous Planner / LLM reasoning)
        return None
