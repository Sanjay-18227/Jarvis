"""
JARVIS Input Normalizer & Deterministic Intent Classifier
Normalizes spoken/typed text, removes fillers, applies high-confidence command vocabulary correction,
and classifies into local deterministic intents before LLM escalation.
"""

import re
import urllib.parse
from typing import Dict, Any, Optional, Tuple, List


class IntentType:
    OPEN_WEBSITE = "OPEN_WEBSITE"
    OPEN_BROWSER = "OPEN_BROWSER"
    SEARCH_YOUTUBE = "SEARCH_YOUTUBE"
    PLAY_YOUTUBE = "PLAY_YOUTUBE"
    PLAY_LATEST_MUSIC = "PLAY_LATEST_MUSIC"
    SEARCH_WEB = "SEARCH_WEB"
    OPEN_APP = "OPEN_APP"
    OPEN_FOLDER_IN_APP = "OPEN_FOLDER_IN_APP"
    OPEN_FOLDER = "OPEN_FOLDER"
    CREATE_FOLDER = "CREATE_FOLDER"
    CREATE_FILE = "CREATE_FILE"
    RUN_PYTHON = "RUN_PYTHON"
    MULTI_STEP = "MULTI_STEP"
    MULTI_STEP_LOCAL = "MULTI_STEP_LOCAL"
    PAINT_ACTION = "PAINT_ACTION"
    NOTEPAD_ACTION = "NOTEPAD_ACTION"
    SCREEN_VISION = "SCREEN_VISION"
    HEAR_CHECK = "HEAR_CHECK"
    GREETING = "GREETING"
    BATTERY = "BATTERY"
    DIAGNOSTICS = "DIAGNOSTICS"
    DATETIME = "DATETIME"
    MEDIA_CONTROL = "MEDIA_CONTROL"
    LOCK_SCREEN = "LOCK_SCREEN"
    GENERAL_AI = "GENERAL_AI"
    NAME_QUERY = "NAME_QUERY"
    OWNER_QUERY = "OWNER_QUERY"
    CALCULATE = "CALCULATE"
    RENAME_FILE = "RENAME_FILE"


class Normalizer:
    """
    Normalizes natural speech and extracts structured deterministic intents.
    """

    KNOWN_WEBSITES = {
        "youtube": "https://www.youtube.com",
        "yt": "https://www.youtube.com",
        "youtube.com": "https://www.youtube.com",
        "google": "https://www.google.com",
        "google.com": "https://www.google.com",
        "github": "https://github.com",
        "github.com": "https://github.com",
        "chatgpt": "https://chatgpt.com",
        "chatgpt.com": "https://chatgpt.com",
        "claude": "https://claude.ai",
        "claude.ai": "https://claude.ai",
        "gmail": "https://mail.google.com",
        "reddit": "https://www.reddit.com",
        "reddit.com": "https://www.reddit.com",
        "twitter": "https://x.com",
        "x": "https://x.com",
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
        "paint": "mspaint",
        "mspaint": "mspaint",
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
        "antigravity": "antigravity",
        "antigravity ide": "antigravity",
    }

    FILLER_PREFIXES = [
        "hey jarvis ", "jarvis ", "ok jarvis ", "okay jarvis ",
        "bro ", "bhai ", "ok ", "okay ", "hey ", "hi ",
        "please ", "kindly ", "can you ", "could you ", "would you ", "will you ",
        "i want you to ", "help me ", "just ", "for me ",
        "an added ", "an add ",
    ]

    FILLER_SUFFIXES = [
        " bro", " bhai", " man", " mate", " sir", " please", " jarvis",
        " for me", " right now", " now", " quickly", " pannu", " pannunga", " karo",
    ]

    # High-confidence vocabulary correction patterns for STT errors
    STT_CORRECTIONS = [
        (r"\bhijab\s+is\b", "hey jarvis"),
        (r"\bhijab\b", "hey jarvis"),
        (r"\bhey\s+java\b", "hey jarvis"),
        (r"\bjar\s+vis\b", "jarvis"),
        (r"\bopen\s+vs\s+code\b", "open vs code"),
    ]

    @classmethod
    def correct_speech_vocabulary(cls, text: str) -> str:
        """Applies high-confidence corrections for known command vocabulary without aggressive rewriting."""
        t = text
        for pat, repl in cls.STT_CORRECTIONS:
            t = re.sub(pat, repl, t, flags=re.IGNORECASE)
        return t

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Strips punctuation, lowercases, applies token correction, and removes wake words & fillers."""
        t = cls.correct_speech_vocabulary(text)
        t = t.lower().strip()
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
        raw_lower = raw_clean.lower()

        # 1. Connection / Hearing checks
        if any(p in t or p in raw_lower for p in [
            "can you hear me", "hear me", "are you there", "do you hear me",
            "you hear me", "are you listening", "hear me clearly"
        ]):
            return IntentType.HEAR_CHECK, {}

        # 2. Greetings
        if t in ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "vanakkam", "வணக்கம்"]:
            return IntentType.GREETING, {}

        # 2.5 Name & Owner Queries
        if any(nq in t or nq in raw_lower for nq in [
            "what is my name", "whats my name", "what's my name",
            "tell me my name", "do you know my name", "who am i",
            "my name"
        ]):
            return IntentType.NAME_QUERY, {}

        if any(oq in t or oq in raw_lower for oq in [
            "who is your owner", "who's your owner", "who is the owner",
            "who owns you", "who created you", "who do you belong to"
        ]):
            return IntentType.OWNER_QUERY, {}

        # 3. Arithmetic & Calculator Queries
        # e.g. "2 + 2", "2+2", "calculate 678 - 876", "678 - 876", "25*4", "100/5", "what is 2 plus 2", "calculate two plus two", "an added 2 + 2"
        use_calc_app = "calculator" in t or "calc" in t
        has_calc_word = any(w in t or w in raw_lower for w in [
            "calculate", "math", "minus", "plus", "times", "divided by", "what is", "add", "an added", "sum"
        ])
        pure_math_pattern = re.match(r"^[0-9\s\+\-\*\/\%\(\)\.]+$", t)

        # Convert word numbers to digits
        word_nums = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        }
        math_str = t
        for w, d in word_nums.items():
            math_str = re.sub(rf"\b{w}\b", d, math_str, flags=re.IGNORECASE)

        math_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?\s*(?:[\+\-\*\/]|minus|plus|times|divided\s+by|\bx\b)\s*[0-9]+(?:\.[0-9]+)?(?:\s*(?:[\+\-\*\/]|minus|plus|times|divided\s+by|\bx\b)\s*[0-9]+(?:\.[0-9]+)?)*)",
            math_str,
            re.IGNORECASE,
        )

        if pure_math_pattern or (has_calc_word and math_match) or math_match:
            # Avoid matching dates/times as math
            if not any(d in t for d in ["what time", "what date", "battery", "pm", "am"]):
                raw_expr = math_match.group(1).strip() if math_match else t
                norm_expr = (
                    raw_expr.replace("minus", "-")
                    .replace("plus", "+")
                    .replace("times", "*")
                    .replace("divided by", "/")
                    .replace("x", "*")
                )
                return IntentType.CALCULATE, {
                    "expression": norm_expr,
                    "use_calculator_app": use_calc_app,
                    "raw_text": raw_text,
                }

        # 4. Local Mini-Plans (Compound safe local actions)
        # e.g. "open edge and open youtube", "open edge and youtube"
        if ("open edge" in t or "launch edge" in t) and ("youtube" in t):
            return IntentType.MULTI_STEP_LOCAL, {
                "name": "edge_and_youtube",
                "steps": [
                    {"tool": "browser", "action": "open_browser", "params": {"browser": "edge"}},
                    {"tool": "browser", "action": "open_url", "params": {"url": "https://www.youtube.com", "site_name": "YouTube"}},
                ],
                "message": "Opened Microsoft Edge and navigated to YouTube, Sir.",
            }

        # e.g. "open vs code and open kumar folder", "open vs code and open kumar"
        if ("open vs code" in t or "open vscode" in t or "open code" in t) and ("kumar" in t):
            return IntentType.MULTI_STEP_LOCAL, {
                "name": "vscode_and_kumar",
                "steps": [
                    {"tool": "vscode", "action": "open_folder", "params": {"folder": "Kumar"}},
                ],
                "message": "Opened Kumar folder in VS Code, Sir.",
            }

        # 5. Open Folder in App (e.g. "open Kumar folder on VS Code", "open Kumar in VS Code", "load Kumar project")
        folder_app_match = re.search(
            r"(?:open|load)\s+(?:the\s+)?([a-zA-Z0-9_-]+)(?:\s+folder|\s+project)?\s+(?:on|in|using|with)\s+(?:vs\s*code|visual\s+studio\s+code|code|vscode)",
            t,
            re.IGNORECASE,
        )
        if not folder_app_match:
            folder_app_match = re.search(
                r"open\s+(?:vs\s*code|visual\s+studio\s+code|code|vscode)\s+(?:with|in|for|and\s+load)\s+(?:the\s+)?([a-zA-Z0-9_-]+)(?:\s+folder|\s+project)?",
                t,
                re.IGNORECASE,
            )
        if not folder_app_match:
            folder_app_match = re.search(
                r"(?:load|open)\s+(?:the\s+)?([a-zA-Z0-9_-]+)\s+project",
                t,
                re.IGNORECASE,
            )
        if folder_app_match:
            folder_name = folder_app_match.group(1).strip()
            # Extract case from raw_clean if possible
            raw_match = re.search(re.escape(folder_name), raw_clean, re.IGNORECASE)
            if raw_match:
                folder_name = raw_match.group(0)
            return IntentType.OPEN_FOLDER_IN_APP, {"app": "vscode", "folder": folder_name}

        # 6. Paint actions (e.g. "draw a rectangle", "draw a line", "draw a circle", "new canvas")
        if any(p in t for p in [
            "draw a rectangle", "draw rectangle", "draw a box", "draw box",
            "draw a line", "draw line", "draw a circle", "draw circle", "new canvas"
        ]):
            act = "draw_rectangle" if any(r in t for r in ["rectangle", "box"]) else (
                "draw_line" if "line" in t else (
                    "draw_circle" if "circle" in t else "new_canvas"
                )
            )
            return IntentType.PAINT_ACTION, {"action": act}

        # 7. Notepad actions (e.g. "type hello", "type 'hello'")
        type_match = re.search(r"^type\s+['\"]?(.+?)['\"]?$", t)
        if type_match:
            text_val = type_match.group(1).strip()
            return IntentType.NOTEPAD_ACTION, {"action": "type_text", "text": text_val}

        # 8. Create Folder (e.g. "create a folder called Demo on Desktop", "create folder Demo on Desktop", "create folder Demo")
        create_folder_match = re.search(
            r"create\s+(?:a\s+)?folder\s+(?:called|named)?\s*([a-zA-Z0-9_-]+)(?:\s+on|\s+in)?\s*(the\s+)?(desktop)?",
            t,
            re.IGNORECASE,
        )
        if create_folder_match:
            folder_name = create_folder_match.group(1).strip()
            loc = "desktop" if create_folder_match.group(3) else "desktop"
            return IntentType.CREATE_FOLDER, {"folder": folder_name, "location": loc}

        # 9. Create File (e.g. "create agent.py inside Kumar", "create agent.py in Kumar", "create agent.py")
        create_file_match = re.search(
            r"create\s+([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)(?:\s+(?:inside|in)\s+([a-zA-Z0-9_-]+))?",
            t,
            re.IGNORECASE,
        )
        if create_file_match:
            fname = create_file_match.group(1).strip()
            folder_target = create_file_match.group(2).strip() if create_file_match.group(2) else None
            return IntentType.CREATE_FILE, {"filename": fname, "folder": folder_target}

        # 10. Run Python script (e.g. "run agent.py", "execute agent.py")
        run_file_match = re.search(r"^(?:run|execute)\s+([a-zA-Z0-9_.-]+\.py)$", t, re.IGNORECASE)
        if run_file_match:
            return IntentType.RUN_PYTHON, {"filename": run_file_match.group(1).strip()}

        # 10.5 Rename File (e.g. "rename agent.py to agent_v2.py", "rename file old.txt to new.txt")
        rename_match = re.search(r"rename\s+(?:file\s+)?([a-zA-Z0-9_.-]+)\s+to\s+([a-zA-Z0-9_.-]+)", t, re.IGNORECASE)
        if rename_match:
            return IntentType.RENAME_FILE, {
                "old_name": rename_match.group(1).strip(),
                "new_name": rename_match.group(2).strip(),
            }

        # 11. Screen Vision Queries
        screen_vision_patterns = [
            "see my screen", "see the screen", "see screen", "look at my screen",
            "look at screen", "see my desktop", "see the desktop", "check screen",
            "inspect screen", "what is on my screen", "what is on screen", "what's on my screen",
            "what do you see on my screen", "can you see my screen", "do you see my screen",
            "tell me what's on the screen", "show me what's on the screen", "what's on the screen",
            "what error is on my screen", "what's this error", "what is this error", "what is the error",
            "why did this application fail", "why did this fail", "what does this dialog say",
            "what is this dialog", "which application am i looking at", "which application is open",
            "what window is active", "what's visible in the terminal", "what is visible in the terminal",
            "check the screen and tell me what happened", "check the screen", "is my python program running",
        ]
        if any(p in t or p in raw_lower for p in screen_vision_patterns) or (
            ("can you see " in t or "check whether " in t or "check if " in t) and any(app in t for app in ["vs code", "vscode", "edge", "notepad", "calculator", "opened", "running"])
        ):
            mode = "ACTIVE_WINDOW"
            if any(p in t for p in ["what is on my screen", "what's on my screen", "tell me what's on the screen", "show me what's on the screen", "see my screen"]):
                mode = "FULL_SCREEN"
            return IntentType.SCREEN_VISION, {"query": raw_clean, "mode": mode}

        # 12. Information Question vs Action Distinction
        if any(t.startswith(q_pre) for q_pre in [
            "how do i ", "how can i ", "how to ", "what is the way to ",
            "explain how to ", "tell me how to ", "can you explain how to "
        ]):
            return IntentType.GENERAL_AI, {"query": raw_text}

        # 13. Multi-Step Execution (e.g. open notepad -> type -> save)
        multi_step_match = cls._match_multi_step(t, raw_text)
        if multi_step_match:
            return IntentType.MULTI_STEP, multi_step_match

        # 14. YouTube Search & Playback
        yt_search_match = re.search(r"search\s+(?:on\s+youtube\s+for\s+|youtube\s+for\s+|for\s+)?(.+?)(?:\s+on\s+youtube|\s+in\s+youtube)?$", t)
        if "youtube" in t and ("search" in t or t.startswith("find ")) and yt_search_match:
            query = yt_search_match.group(1).replace("on youtube", "").replace("in youtube", "").strip()
            if query and query != "youtube":
                return IntentType.SEARCH_YOUTUBE, {"query": query}

        if t.startswith("play "):
            song_target = t[5:].replace("on youtube", "").replace("in youtube", "").strip()
            if any(term in song_target for term in ["latest", "trending", "current", "new", "top"]) and any(term in song_target for term in ["song", "music", "track"]):
                return IntentType.PLAY_LATEST_MUSIC, {"query": song_target}
            elif song_target:
                return IntentType.PLAY_YOUTUBE, {"query": song_target}

        # 15. General Web Searches
        browser_search_match = re.search(r"(?:open\s+(?:my\s+)?browser\s+and\s+search\s+|search\s+(?:for\s+)?)(.+?)(?:\s+on\s+browser|\s+in\s+browser|\s+on\s+google|\s+in\s+google|\s+on\s+bing)?$", t)
        if t.startswith("search ") or "browser and search" in t:
            if browser_search_match:
                search_q = browser_search_match.group(1).replace("on google", "").replace("on browser", "").replace("in browser", "").strip()
                if search_q and "youtube" not in search_q:
                    return IntentType.SEARCH_WEB, {"query": search_q}

        # 16. Browser Explicit Launch & Website Routing
        target_browser = "edge"
        if " in microsoft edge" in t or " in edge" in t or " on edge" in t or " on microsoft edge" in t:
            target_browser = "edge"
            t = re.sub(r"\s+(?:in|on)\s+(?:microsoft\s+edge|edge)", "", t).strip()
        elif " in chrome" in t or " in google chrome" in t or " on chrome" in t or " on google chrome" in t:
            target_browser = "chrome"
            t = re.sub(r"\s+(?:in|on)\s+(?:google\s+chrome|chrome)", "", t).strip()
        elif " in browser" in t or " on browser" in t or " using browser" in t:
            t = re.sub(r"\s+(?:in|on|using)\s+browser", "", t).strip()

        # Check generic open browser -> OPEN_APP(edge)
        if t in ["open browser", "launch browser", "launch the browser", "start browser", "open my browser", "open edge", "launch edge"]:
            return IntentType.OPEN_APP, {"app_key": "edge", "raw_name": "browser"}

        # Check launch website words
        for act in ["open ", "launch ", "go to ", "navigate to "]:
            if t.startswith(act):
                target = t[len(act):].strip()
                target = re.sub(r"^(?:the\s+|my\s+)", "", target).strip()
                target = re.sub(r"\s+(?:website|site|web\s+app)$", "", target).strip()
                target = re.sub(r"\s+(?:on\s+browser|in\s+browser)$", "", target).strip()

                if target in cls.KNOWN_WEBSITES:
                    return IntentType.OPEN_WEBSITE, {
                        "site_name": target,
                        "url": cls.KNOWN_WEBSITES[target],
                        "browser": target_browser,
                    }
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

        # 17. Application Opening
        for act in ["open ", "launch ", "start "]:
            if t.startswith(act):
                target = t[len(act):].replace("the ", "").replace("application", "").replace("app", "").strip()
                target = re.sub(r"^(?:my\s+)", "", target).strip()
                if target in cls.APP_ALIASES:
                    return IntentType.OPEN_APP, {"app_key": cls.APP_ALIASES[target], "raw_name": target}
                for alias_k, canonical in cls.APP_ALIASES.items():
                    if target == alias_k:
                        return IntentType.OPEN_APP, {"app_key": canonical, "raw_name": target}

        if t in cls.APP_ALIASES:
            return IntentType.OPEN_APP, {"app_key": cls.APP_ALIASES[t], "raw_name": t}

        if t.endswith(" open"):
            target = t[:-5].replace("the ", "").replace("application", "").replace("app", "").strip()
            if target in cls.APP_ALIASES:
                return IntentType.OPEN_APP, {"app_key": cls.APP_ALIASES[target], "raw_name": target}
            if target in cls.KNOWN_WEBSITES:
                return IntentType.OPEN_WEBSITE, {"site_name": target, "url": cls.KNOWN_WEBSITES[target]}

        # 18. Open Folder (e.g. "open Demo", "open Demo folder")
        if t.startswith("open "):
            target_fol = t[5:].strip()
            if target_fol.endswith(" folder"):
                target_fol = target_fol[:-7].strip()
            # If target is a single word or path and not an app/website
            if target_fol and target_fol not in cls.APP_ALIASES and target_fol not in cls.KNOWN_WEBSITES:
                return IntentType.OPEN_FOLDER, {"folder": target_fol}

        # 19. Battery
        if any(k in t for k in ["battery status", "battery percentage", "what is my battery", "how much battery", "check battery"]):
            return IntentType.BATTERY, {}

        # 20. Diagnostics & System Info
        if any(k in t for k in ["why is my laptop slow", "laptop slow", "check performance", "system vitals", "system info", "system information", "system specs"]):
            return IntentType.DIAGNOSTICS, {}

        # 21. Date / Time
        if any(t == p or t.startswith(p) for p in ["what time is it", "what is the time", "tell me the time", "what day is it", "whats the date"]):
            return IntentType.DATETIME, {}

        # 22. Media / Volume Controls
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

        # 23. General AI / Complex reasoning fallback
        return IntentType.GENERAL_AI, {"query": raw_text}

    @classmethod
    def _match_multi_step(cls, clean_t: str, raw_text: str) -> Optional[Dict[str, Any]]:
        """Matches multi-step sequential commands."""
        if ("open notepad" in clean_t or "launch notepad" in clean_t) and ("type " in clean_t or "write " in clean_t):
            text_to_type = ""
            type_match = re.search(r"(?:type|write)\s+['\"]?([^'\".,]+)['\"]?", clean_t)
            if type_match:
                text_to_type = type_match.group(1).strip()
                if " and save" in text_to_type:
                    text_to_type = text_to_type.split(" and save")[0].strip()

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

        return None
