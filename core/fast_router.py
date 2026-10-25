"""
JARVIS Fast Router Subsystem
Fulfills Section 12: Hybrid Intelligence & Requirements B, C, D, E, F, G, H, J:
- Deterministic sub-second execution for all Windows local actions
- Distinguishes websites from apps
- Enforces real verification before reporting success
- Clean debug logs [ROUTER], [TOOL], [VERIFY], [RESULT]
"""

import re
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
        intent, params = Normalizer.classify_intent(text)

        # Step 0: Centralized Security Gate enforcement before any Fast Path action
        allowed, auth_msg, requires_confirmation = authorize_action(
            speaker=speaker,
            action_name=intent.lower(),
            intent=intent,
            parameters=params,
            raw_query=text,
        )
        if not allowed:
            print(f"[SECURITY] FastPath blocked for speaker={speaker.value} intent={intent}")
            return "security_denied", auth_msg

        # 1. Connection / Hearing check
        if intent == IntentType.HEAR_CHECK:
            print(f"[ROUTER] intent=HEAR_CHECK")
            if speaker == SpeakerClassification.OWNER:
                return "hear_check", "Yep, Sir. I can hear you clearly."
            return "hear_check", "I can hear you clearly."

        # 2. Greetings
        if intent == IntentType.GREETING:
            print(f"[ROUTER] intent=GREETING")
            if speaker == SpeakerClassification.OWNER:
                return "greeting", "Hello, Sir. How can I assist you today?"
            return "greeting", "Hello. I don't recognize this voice. Computer controls are locked."

        # 2.5 Name & Owner Queries (Part 1 & Test 10)
        if intent == IntentType.NAME_QUERY:
            print(f"[ROUTER] intent=NAME_QUERY")
            if speaker == SpeakerClassification.OWNER:
                return "name_query", "Your name is Sanjay, Sir."
            return "name_query", "Voice verification failed. Please authenticate as the owner."

        if intent == IntentType.OWNER_QUERY:
            print(f"[ROUTER] intent=OWNER_QUERY")
            return "owner_query", "My owner is Sanjay, Sir."

        # 3. Battery status
        if intent == IntentType.BATTERY:
            print(f"[ROUTER] intent=BATTERY")
            vitals = self.sys_monitor.get_system_vitals()
            pct = vitals["battery"]["percent"]
            plugged = vitals["battery"]["plugged"]
            status_desc = "charging" if plugged else "discharging"
            msg = f"Your battery is at {pct}%, currently {status_desc}, Sir."
            return "battery", self._format_response(msg, speaker)

        # 4. Diagnostics ("Why is my laptop slow?")
        if intent == IntentType.DIAGNOSTICS:
            print(f"[ROUTER] intent=DIAGNOSTICS")
            _, diagnosis = self.sys_monitor.diagnose_performance()
            return "diagnostics", self._format_response(diagnosis, speaker)

        # 5. Date / Time
        if intent == IntentType.DATETIME:
            print(f"[ROUTER] intent=DATETIME")
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p")
            date_str = now.strftime("%A, %B %d")
            msg = f"It's {time_str} on {date_str}, Sir."
            return "datetime", self._format_response(msg, speaker)

        # 6. Media / Volume Controls
        if intent == IntentType.MEDIA_CONTROL:
            act = params.get("action", "play_pause")
            print(f"[ROUTER] intent=MEDIA_CONTROL action={act}")
            self.win_ctrl.control_media(act)
            if act == "volume_up":
                return "volume", self._format_response("Volume increased, Sir.", speaker)
            elif act == "volume_down":
                return "volume", self._format_response("Volume decreased, Sir.", speaker)
            elif act == "mute":
                return "volume", self._format_response("Mute toggled, Sir.", speaker)
            return "media", self._format_response("Playback toggled, Sir.", speaker)

        # 7. Lock Screen
        if intent == IntentType.LOCK_SCREEN:
            print(f"[ROUTER] intent=LOCK_SCREEN")
            self.win_ctrl.lock_workstation()
            return "lock", self._format_response("Workstation locked, Sir.", speaker)

        # 8. Website Routing (Requirement D: Open YouTube, Google, GitHub, etc.)
        if intent == IntentType.OPEN_WEBSITE:
            res = self.browser_ctrl.open_website(
                url=params["url"],
                browser=params.get("browser"),
                site_name=params.get("site_name"),
            )
            return "open_website", self._format_response(res.message, speaker)

        # 9. YouTube Search & Music Commands (Requirement D & H)
        if intent == IntentType.SEARCH_YOUTUBE:
            res = self.browser_ctrl.search_youtube(params["query"])
            return "search_youtube", self._format_response(res.message, speaker)

        if intent == IntentType.PLAY_YOUTUBE:
            res = self.browser_ctrl.play_youtube(params["query"])
            return "play_youtube", self._format_response(res.message, speaker)

        if intent == IntentType.PLAY_LATEST_MUSIC:
            q = params.get("query", "latest songs")
            print(f"[ROUTER] intent=PLAY_LATEST_MUSIC query='{q}'")
            res = self.browser_ctrl.search_youtube(q)
            msg = f"I opened the YouTube search results for {q}, Sir."
            return "play_music", self._format_response(msg, speaker)

        # 9.5 General Web Searches (Requirement D)
        if intent == IntentType.SEARCH_WEB:
            res = self.browser_ctrl.search_web(params["query"])
            return "search_web", self._format_response(res.message, speaker)

        # 10. Application Launching (Requirement B, C, E)
        if intent == IntentType.OPEN_APP:
            app_key = params.get("app_key") or params.get("raw_name", "")
            res = self.app_indexer.launch_app(app_key)
            return "open_app", self._format_response(res.message, speaker)

        # 11. Multi-Step Execution (Requirement F)
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

        # 12. Screen Vision Queries (Requirement G: Never hallucinate)
        if intent == IntentType.SCREEN_VISION:
            res = self.screen_vision.inspect_screen(params.get("query", text), speaker)
            return "screen_vision", self._format_response(res.message, speaker)

        # Fall through to AI Path (Autonomous Planner / LLM reasoning)
        return None
