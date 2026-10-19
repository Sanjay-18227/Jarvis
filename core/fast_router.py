"""
JARVIS Fast Router Subsystem
Fulfills Section 12: Hybrid Intelligence.
Splits incoming intents into:
- FAST PATH: Sub-second local actions (open app, battery, volume, time, "Can you hear me?", system diagnostics)
- AI PATH: Autonomous Gemini multi-step reasoning, coding, screen vision, research, and planning.
Avoids rigid keyword lists by matching semantic action patterns.
"""

import re
import datetime
from typing import Optional, Tuple, Dict, Any
from config.constants import SpeakerClassification
from computer.app_indexer import AppIndexer
from computer.windows_controller import WindowsController
from computer.system_monitor import SystemMonitor


class FastRouter:
    """
    Sub-second intent classifier and fast local executor.
    """

    def __init__(
        self,
        app_indexer: Optional[AppIndexer] = None,
        win_ctrl: Optional[WindowsController] = None,
        sys_monitor: Optional[SystemMonitor] = None,
    ):
        self.app_indexer = app_indexer or AppIndexer()
        self.win_ctrl = win_ctrl or WindowsController()
        self.sys_monitor = sys_monitor or SystemMonitor()

    def try_fast_path(
        self,
        text: str,
        speaker: SpeakerClassification = SpeakerClassification.OWNER,
    ) -> Optional[Tuple[str, str]]:
        """
        Evaluates user text. If it is a common local OS action or conversational greeting,
        executes immediately on Fast Path.

        Returns:
            (action_name, spoken_response) if handled on Fast Path, else None.
        """
        # Normalize and strip trailing punctuation
        t = text.lower().strip()
        t = re.sub(r"[?!.,]+$", "", t).strip()

        # 1. Connection / Hearing checks ("Can you hear me?", "Are you there?")
        if any(phrase in t or t == phrase for phrase in [
            "can you hear me", "are you there", "do you hear me", "you hear me", "can u hear me", "are you listening", "hear me"
        ]):
            return "hear_check", "Yep, Sir. I can hear you clearly."

        # Strip polite conversational prefixes & wake words
        for prefix in ["hey jarvis ", "jarvis ", "ok jarvis ", "okay jarvis ", "please ", "can you ", "could you "]:
            if t.startswith(prefix):
                t = t[len(prefix):].strip()

        # Strip trailing wake word e.g. "hello jarvis" -> "hello"
        if t.endswith(" jarvis"):
            t = t[:-7].strip()

        # 2. Greetings ("Hello", "Good morning", "Hi Jarvis")
        if t in ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "vanakkam", "வணக்கம்"] or any(t == g or t.startswith(g + " ") for g in ["hello", "hi", "hey", "good morning", "good evening", "vanakkam"]):
            return "greeting", "Hello Sanjay. How can I assist you today, Sir?"

        # 3. Battery status
        if any(k in t for k in ["battery status", "battery percentage", "what is my battery", "how much battery", "battery percent", "charge remaining"]):
            vitals = self.sys_monitor.get_system_vitals()
            pct = vitals["battery"]["percent"]
            plugged = vitals["battery"]["plugged"]
            status_desc = "charging" if plugged else "discharging"
            return "battery", f"Your battery is at {pct}%, currently {status_desc}, Sir."

        # 4. Laptop Performance / Slow Diagnosis ("Why is my laptop slow?")
        if any(k in t for k in ["why is my laptop slow", "laptop slow", "why is the pc lagging", "check performance", "system performance", "system vitals"]):
            _, diagnosis = self.sys_monitor.diagnose_performance()
            return "diagnostics", diagnosis

        # 5. Current Time & Date
        if any(t == phrase or t.startswith(phrase) for phrase in ["what time is it", "what is the time", "tell me the time", "current time", "what day is it", "whats the date", "what is today's date"]):
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p")
            date_str = now.strftime("%A, %B %d")
            return "datetime", f"It's {time_str} on {date_str}, Sir."

        # 6. Media / Volume Controls
        if t in ["volume up", "satham yethu", "sound up"]:
            self.win_ctrl.control_media("volume_up")
            return "volume", "Volume increased, Sir."
        elif t in ["volume down", "satham kura", "sound down"]:
            self.win_ctrl.control_media("volume_down")
            return "volume", "Volume decreased, Sir."
        elif t in ["mute", "unmute"]:
            self.win_ctrl.control_media("mute")
            return "volume", "Mute toggled, Sir."
        elif t in ["play music", "pause music", "pause", "resume"]:
            self.win_ctrl.control_media("play_pause")
            return "media", "Playback toggled, Sir."

        # 7. Lock Screen
        if t in ["lock screen", "lock workstation", "lock the laptop", "lock system", "screen lock"]:
            self.win_ctrl.lock_workstation()
            return "lock", "Workstation locked, Sir."

        # 8. Fast App Launch ("Open <app>")
        # Avoid routing multi-step tasks like "Open Antigravity and build this project" to fast path!
        if (t.startswith("open ") or t.startswith("launch ")) and not any(k in t for k in ["and", "then", "project", "why", "error", "build"]):
            target = t.split(" ", 1)[1].replace("the ", "").replace("application", "").replace("app", "").strip()
            if target and len(target) < 30:
                success, msg = self.app_indexer.launch_app(target)
                return "open_app", msg

        # 9. Tanglish / Tamil simple app launch (e.g. "Chrome open pannu", "calc open pannu")
        tamil_app_match = re.search(r"([a-zA-Z0-9 ]+?)\s*(open\s*பண்ணு|open\s*pan+u|open\s*karo)", t)
        if tamil_app_match and not any(k in t for k in ["and", "then", "project", "check", "error"]):
            target = tamil_app_match.group(1).strip()
            if target:
                success, msg = self.app_indexer.launch_app(target)
                return "open_app", msg

        # No fast path: Delegate to AI Path (Gemini with tools)
        return None
