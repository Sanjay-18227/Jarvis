"""
JARVIS Input Normalizer & Deterministic Intent Classifier
Fulfills Requirements C, D, E, G, H, I, N:
- Normalizes natural language variations and strips conversational filler/wake words
- Distinguishes Websites (OPEN_WEBSITE) from Applications (OPEN_APP)
- Distinguishes Information Questions ("How do I...") from Actions ("Open...")
- Resolves deterministic application aliases (VS Code, Edge, Chrome, Notepad, Calculator, Paint, Task Manager)
- Detects YouTube searches, playback, and general web searches (using Microsoft Edge)
- Detects multi-step sequential actions (e.g. open -> type -> save)
- Detects screen inspection queries with truthfulness enforcement
"""

import re
import urllib.parse
from typing import Dict, Any, Optional, Tuple, List


class IntentType:
    OPEN_WEBSITE = "OPEN_WEBSITE"
    SEARCH_YOUTUBE = "SEARCH_YOUTUBE"
    PLAY_YOUTUBE = "PLAY_YOUTUBE"
    PLAY_LATEST_MUSIC = "PLAY_LATEST_MUSIC"
    SEARCH_WEB = "SEARCH_WEB"
    OPEN_APP = "OPEN_APP"
    MULTI_STEP = "MULTI_STEP"
    SCREEN_VISION = "SCREEN_VISION"
    HEAR_CHECK = "HEAR_CHECK"
    GREETING = "GREETING"
    BATTERY = "BATTERY"
    DIAGNOSTICS = "DIAGNOSTICS"
    DATETIME = "DATETIME"
    MEDIA_CONTROL = "MEDIA_CONTROL"
    LOCK_SCREEN = "LOCK_SCREEN"
    GENERAL_AI = "GENERAL_AI"


class Normalizer:
    """
    Normalizes spoken/typed text and extracts structured intents.
    """

    KNOWN_WEBSITES = {
        "youtube": "https://www.youtube.com",
        "yt": "https://www.youtube.com",
        "youtube.com": "https://www.youtube.com",
        "google": "https://www.google.com",
        "google.com": "https://www.google.com",
        "github": "https://github.com",
        "github.com": "https://github.com",
        "gmail": "https://mail.google.com",
        "reddit": "https://www.reddit.com",
        "reddit.com": "https://www.reddit.com",
        "twitter": "https://x.com",
        "x": "https://x.com",
        "chatgpt": "https://chatgpt.com",
        "claude": "https://claude.ai",
        "wikipedia": "https://www.wikipedia.org",
        "linkedin": "https://www.linkedin.com",
        "stackoverflow": "https://stackoverflow.com",
    }

    APP_ALIASES = {
        "vscode": "vscode",
        "vs code": "vscode",
        "visual studio code": "vscode",
        "code": "vscode",
        "edge": "edge",
        "ms edge": "edge",
        "microsoft edge": "edge",
        "msedge": "edge",
        "browser": "edge",
        "the browser": "edge",
        "web browser": "edge",
        "my browser": "edge",
        "chrome": "chrome",
        "google chrome": "chrome",
        "notepad": "notepad",
        "calculator": "calc",
        "calc": "calc",
        "terminal": "terminal",
        "windows terminal": "terminal",
        "wt": "terminal",
        "cmd": "cmd",
        "command prompt": "cmd",
        "powershell": "powershell",
        "explorer": "explorer",
        "file explorer": "explorer",
        "settings": "settings",
        "windows settings": "settings",
        "task manager": "taskmgr",
        "taskmgr": "taskmgr",
        "paint": "mspaint",
        "mspaint": "mspaint",
        "antigravity": "antigravity",
        "antigravity ide": "antigravity",
    }

    FILLER_PREFIXES = [
        "hey jarvis ", "jarvis ", "ok jarvis ", "okay jarvis ",
        "please ", "kindly ", "can you ", "could you ", "would you ", "will you ",
        "i want you to ", "help me ", "just ", "for me ",
    ]

    FILLER_SUFFIXES = [
        " bro", " bhai", " man", " mate", " sir", " please", " jarvis",
        " for me", " right now", " now", " quickly", " pannu", " pannunga", " karo",
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Strips punctuation, lowercases, and removes wake words and polite fillers."""
        t = text.lower().strip()
        t = re.sub(r"[?!.,;:]+$", "", t).strip()

        # Strip prefixes repeatedly until clean
        changed = True
        while changed:
            changed = False
            for prefix in cls.FILLER_PREFIXES:
                if t.startswith(prefix):
                    t = t[len(prefix):].strip()
                    changed = True

        # Strip suffixes repeatedly
        changed = True
        while changed:
            changed = False
            for suffix in cls.FILLER_SUFFIXES:
                if t.endswith(suffix):
                    t = t[:-len(suffix)].strip()
                    changed = True

        return t.strip()

    @classmethod
    def classify_intent(cls, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Classifies incoming input into a normalized intent type with parameters.
        """
        raw_clean = raw_text.strip()
        t = cls.clean_text(raw_text)

        # 1. Connection / Hearing checks
        if any(p in t or p in raw_clean.lower() for p in [
            "can you hear me", "hear me", "are you there", "do you hear me",
            "you hear me", "are you listening", "hear me clearly"
        ]):
            return IntentType.HEAR_CHECK, {}

        # 2. Greetings
        if t in ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "vanakkam", "வணக்கம்"]:
            return IntentType.GREETING, {}

        # 3. Screen Vision Queries (Requirement G & Test 10)
        raw_lower = raw_text.lower().strip()
        if any(p in t or p in raw_lower for p in [
            "see my screen", "see the screen", "see screen", "look at my screen",
            "look at screen", "see my desktop", "see the desktop", "check screen",
            "inspect screen", "what is on my screen", "what is on screen", "what's on my screen",
            "what do you see on my screen", "can you see my screen", "do you see my screen"
        ]):
            return IntentType.SCREEN_VISION, {"query": raw_clean}

        # 4. Information Question vs Action Distinction (Requirement N)
        # e.g., "How do I open Task Manager?" -> GENERAL_AI explanation
        # e.g., "How do I search YouTube?" -> GENERAL_AI explanation
        if any(t.startswith(q_pre) for q_pre in [
            "how do i ", "how can i ", "how to ", "what is the way to ",
            "explain how to ", "tell me how to ", "can you explain how to "
        ]):
            return IntentType.GENERAL_AI, {"query": raw_text}

        # 5. Multi-Step Execution (Requirement F)
        multi_step_match = cls._match_multi_step(t, raw_text)
        if multi_step_match:
            return IntentType.MULTI_STEP, multi_step_match

        # 6. YouTube Search & Playback (Requirement D & H)
        # "search <query> on youtube" or "search youtube for <query>"
        yt_search_match = re.search(r"search\s+(?:on\s+youtube\s+for\s+|youtube\s+for\s+|for\s+)?(.+?)(?:\s+on\s+youtube|\s+in\s+youtube)?$", t)
        if "youtube" in t and ("search" in t or t.startswith("find ")) and yt_search_match:
            query = yt_search_match.group(1).replace("on youtube", "").replace("in youtube", "").strip()
            if query and query != "youtube":
                return IntentType.SEARCH_YOUTUBE, {"query": query}

        # "play <song> on youtube" or "play <song>"
        if t.startswith("play "):
            song_target = t[5:].replace("on youtube", "").replace("in youtube", "").strip()
            if any(term in song_target for term in ["latest", "trending", "current", "new", "top"]) and any(term in song_target for term in ["song", "music", "track"]):
                return IntentType.PLAY_LATEST_MUSIC, {"query": song_target}
            elif song_target:
                return IntentType.PLAY_YOUTUBE, {"query": song_target}

        # 7. General Web Searches (Requirement D)
        # e.g., "search Python FastAPI", "search AI news", "search python on google"
        # e.g., "open my browser and search machine learning tutorials"
        browser_search_match = re.search(r"(?:open\s+(?:my\s+)?browser\s+and\s+search\s+|search\s+(?:for\s+)?)(.+?)(?:\s+on\s+browser|\s+in\s+browser|\s+on\s+google|\s+in\s+google|\s+on\s+bing)?$", t)
        if t.startswith("search ") or "browser and search" in t:
            if browser_search_match:
                search_q = browser_search_match.group(1).replace("on google", "").replace("on browser", "").replace("in browser", "").strip()
                if search_q and "youtube" not in search_q:
                    return IntentType.SEARCH_WEB, {"query": search_q}

        # 8. Browser Explicit Specification & Website Routing (Requirement D & I)
        target_browser = None
        if " in microsoft edge" in t or " in edge" in t or " on edge" in t or " on microsoft edge" in t:
            target_browser = "edge"
            t = re.sub(r"\s+(?:in|on)\s+(?:microsoft\s+edge|edge)", "", t).strip()
        elif " in chrome" in t or " in google chrome" in t or " on chrome" in t or " on google chrome" in t:
            target_browser = "chrome"
            t = re.sub(r"\s+(?:in|on)\s+(?:google\s+chrome|chrome)", "", t).strip()
        elif " in browser" in t or " on browser" in t or " using browser" in t:
            t = re.sub(r"\s+(?:in|on|using)\s+browser", "", t).strip()

        # Check launch website words
        for act in ["open ", "launch ", "go to ", "navigate to "]:
            if t.startswith(act):
                target = t[len(act):].strip()
                # Strip filler words from target (e.g. "the youtube website" -> "youtube", "the browser" -> "browser")
                target = re.sub(r"^(?:the\s+|my\s+)", "", target).strip()
                target = re.sub(r"\s+(?:website|site|web\s+app)$", "", target).strip()
                target = re.sub(r"\s+(?:on\s+browser|in\s+browser)$", "", target).strip()

                # Check known websites
                if target in cls.KNOWN_WEBSITES:
                    return IntentType.OPEN_WEBSITE, {
                        "site_name": target,
                        "url": cls.KNOWN_WEBSITES[target],
                        "browser": target_browser,
                    }
                # Check domain pattern (e.g. "reddit.com", "example.org")
                if re.match(r"^[a-zA-Z0-9-]+\.(com|org|net|io|in|co|ai|dev|app|edu|gov)(/.*)?$", target):
                    url = f"https://{target}"
                    return IntentType.OPEN_WEBSITE, {
                        "site_name": target,
                        "url": url,
                        "browser": target_browser,
                    }

        # Direct domain mention without "open" (e.g. "youtube.com")
        if t in cls.KNOWN_WEBSITES or re.match(r"^[a-zA-Z0-9-]+\.(com|org|net|io|in|co|ai|dev|app|edu|gov)$", t):
            target = t
            url = cls.KNOWN_WEBSITES.get(target, f"https://{target}")
            return IntentType.OPEN_WEBSITE, {
                "site_name": target,
                "url": url,
                "browser": target_browser,
            }

        # 9. Application Opening (Requirement C, E, I)
        for act in ["open ", "launch ", "start "]:
            if t.startswith(act):
                target = t[len(act):].replace("the ", "").replace("application", "").replace("app", "").strip()
                target = re.sub(r"^(?:my\s+)", "", target).strip()
                if target in cls.APP_ALIASES:
                    return IntentType.OPEN_APP, {"app_key": cls.APP_ALIASES[target], "raw_name": target}
                for alias_k, canonical in cls.APP_ALIASES.items():
                    if target == alias_k:
                        return IntentType.OPEN_APP, {"app_key": canonical, "raw_name": target}

        # Generic single-word app aliases like "notepad", "calculator", "calc", "edge", "vscode"
        if t in cls.APP_ALIASES:
            return IntentType.OPEN_APP, {"app_key": cls.APP_ALIASES[t], "raw_name": t}

        # 10. Battery
        if any(k in t for k in ["battery status", "battery percentage", "what is my battery", "how much battery", "check battery"]):
            return IntentType.BATTERY, {}

        # 11. Diagnostics
        if any(k in t for k in ["why is my laptop slow", "laptop slow", "check performance", "system vitals"]):
            return IntentType.DIAGNOSTICS, {}

        # 12. Date / Time
        if any(t == p or t.startswith(p) for p in ["what time is it", "what is the time", "tell me the time", "what day is it", "whats the date"]):
            return IntentType.DATETIME, {}

        # 13. Media / Volume Controls
        if t in ["volume up", "sound up", "satham yethu", "increase volume"]:
            return IntentType.MEDIA_CONTROL, {"action": "volume_up"}
        if t in ["volume down", "sound down", "satham kura", "decrease volume"]:
            return IntentType.MEDIA_CONTROL, {"action": "volume_down"}
        if t in ["mute", "unmute", "mute volume", "unmute volume"]:
            return IntentType.MEDIA_CONTROL, {"action": "mute"}
        if t in ["play music", "pause music", "pause", "resume"]:
            return IntentType.MEDIA_CONTROL, {"action": "play_pause"}
        if t in ["lock screen", "lock workstation", "lock laptop"]:
            return IntentType.LOCK_SCREEN, {}

        # 14. General AI / Complex multi-step reasoning
        return IntentType.GENERAL_AI, {"query": raw_text}

    @classmethod
    def _match_multi_step(cls, clean_t: str, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Detects sequential multi-step requests like:
        - "Open Notepad and type Hi how are you"
        - "Open Notepad, type 'Hi, how are you?', and save it on Desktop as Hello.txt"
        - "Save it on Desktop as Hello.txt"
        """
        # Standalone save command: "save it on desktop as hello.txt" or "save that text as hello.txt on desktop"
        save_match = re.search(r"save\s+(?:it|that|this|the\s+file|the\s+text)?\s*(?:on|to|in)?\s*(?:the\s+)?(desktop)?\s*(?:as|named|with\s+name)?\s*([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)", clean_t)
        if save_match:
            filename = save_match.group(2)
            location = "desktop" if "desktop" in clean_t else "desktop"
            return {
                "action": "save_file",
                "filename": filename,
                "location": location,
                "raw_text": raw_text,
            }

        # Multi-step: "open notepad ... and type ..."
        if ("open notepad" in clean_t or "launch notepad" in clean_t) and ("type " in clean_t or "write " in clean_t):
            # Extract text to type
            text_to_type = ""
            type_match = re.search(r"(?:type|write)\s+['\"]?([^'\".,]+)['\"]?", clean_t)
            if type_match:
                text_to_type = type_match.group(1).strip()
                if " and save" in text_to_type:
                    text_to_type = text_to_type.split(" and save")[0].strip()

            # Check if there is also a save component
            save_file = None
            if "save" in clean_t:
                fn_match = re.search(r"as\s+([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)", clean_t)
                if fn_match:
                    save_file = fn_match.group(1)

            return {
                "action": "notepad_type_save",
                "app": "notepad",
                "text": text_to_type,
                "save_file": save_file,
                "location": "desktop" if "desktop" in clean_t else None,
                "raw_text": raw_text,
            }

        return None
