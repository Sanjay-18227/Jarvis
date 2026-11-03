"""
JARVIS Entity Extractor & Goal Preservation Engine
Extracts and strictly locks user-specified named entities BEFORE planning:
- folder (e.g. Kumar)
- filename (e.g. agent.py)
- application (e.g. VS Code, Calculator, Microsoft Edge)
- code_content & code_to_print (e.g. print('Agent Ready'))
- should_run (whether execution of file is requested)
- calculation (e.g. '678 - 876')
- query / website (e.g. YouTube, 'Wanna Be Yours')

RULE: Never silently invent or replace explicit user entities (e.g. Project, app.py, test.py).
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from core.context_memory import get_context_memory


@dataclass
class GoalEntities:
    folder: Optional[str] = None
    filename: Optional[str] = None
    application: Optional[str] = None
    code_content: Optional[str] = None
    code_to_print: Optional[str] = None
    should_run: bool = False
    calculation: Optional[str] = None
    website: Optional[str] = None
    query: Optional[str] = None
    check_screen: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in {
                "folder": self.folder,
                "filename": self.filename,
                "application": self.application,
                "code_content": self.code_content,
                "code_to_print": self.code_to_print,
                "should_run": self.should_run,
                "calculation": self.calculation,
                "website": self.website,
                "query": self.query,
                "check_screen": self.check_screen,
            }.items() if v is not None
        }


@dataclass
class StructuredGoal:
    goal: str
    entities: GoalEntities = field(default_factory=GoalEntities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "entities": self.entities.to_dict()
        }


class EntityExtractor:
    """
    Extracts and locks domain entities from natural language requests.
    """

    KNOWN_APPS = {
        "vs code": "VS Code",
        "vscode": "VS Code",
        "code": "VS Code",
        "visual studio code": "VS Code",
        "calculator": "Calculator",
        "calc": "Calculator",
        "notepad": "Notepad",
        "edge": "Microsoft Edge",
        "msedge": "Microsoft Edge",
        "microsoft edge": "Microsoft Edge",
        "browser": "Microsoft Edge",
        "chrome": "Google Chrome",
        "google chrome": "Google Chrome",
    }

    @classmethod
    def extract(cls, raw_text: str) -> StructuredGoal:
        ctx = get_context_memory()
        entities = GoalEntities()
        t = raw_text.strip()
        t_low = t.lower()

        # 1. Filename extraction (strict regex for file with extension)
        # e.g. "create agent.py", "open app.py", "test.txt"
        file_match = re.search(r"\b([a-zA-Z0-9_\-\.]+\.(?:py|txt|js|json|html|md|cpp|c|sh|bat))\b", t, re.IGNORECASE)
        if file_match:
            entities.filename = file_match.group(1).strip()
        elif any(p in t_low for p in ["the file", "that file", "the script", "it"]):
            entities.filename = ctx.get_active_file()

        # 2. Folder extraction
        # Examples:
        # "open Kumar folder", "open the Kumar folder", "in Kumar folder", "inside Kumar folder"
        # "open Kumar in VS Code", "create agent.py inside Kumar", "create a Python file in Kumar"
        # "folder called Kumar", "folder named Kumar", "create folder Kumar"
        folder_patterns = [
            r"(?:open\s+(?:the\s+)?|in\s+(?:the\s+)?|inside\s+(?:the\s+)?|into\s+(?:the\s+)?)?([a-zA-Z0-9_\-\.]+)\s+folder",
            r"(?:folder\s+(?:called|named)\s+|create\s+(?:a\s+)?folder\s+(?:called|named)?\s*)([a-zA-Z0-9_\-\.]+)",
            r"(?:open\s+(?:the\s+)?)([a-zA-Z0-9_\-\.]+)\s+in\s+vs\s*code",
            r"(?:inside|in)\s+([a-zA-Z0-9_\-\.]+)(?:\s+and|\s+in\s+vs|\s*,|$)",
            r"(?:create\s+[a-zA-Z0-9_\-\.]+\s+(?:inside|in)\s+)([a-zA-Z0-9_\-\.]+)",
        ]

        for pat in folder_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                # Ignore common keywords that are not folder names
                if cand.lower() not in ["the", "a", "an", "this", "that", "my", "it", "vscode", "vs", "code", "desktop", "calculator", "notepad", "edge", "chrome", "youtube"]:
                    entities.folder = cand
                    break

        if not entities.folder:
            if any(p in t_low for p in ["the folder", "that folder", "the project", "inside it", "in it"]):
                entities.folder = ctx.get_active_folder()

        # 3. Application extraction
        for alias, app_name in cls.KNOWN_APPS.items():
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, t_low):
                entities.application = app_name
                break

        if not entities.application:
            # Check context if user is issuing an action on currently open application
            if any(w in t_low for w in ["calculate", "type", "save"]) and ctx.get_active_app():
                entities.application = ctx.get_active_app()

        # 4. Code & Print Content extraction
        # Example: "write a simple Python program that prints Agent Ready"
        # Example: "print Agent Ready"
        print_match = re.search(r"prints?\s+['\"]?([^'\".,]+?)['\"]?(?:\s*,\s*run|\s+and\s+run|\s+and\s+tell|$)", t, re.IGNORECASE)
        if print_match:
            print_text = print_match.group(1).strip()
            # Clean trailing words like "run it"
            print_text = re.sub(r"\s+and\s+run.*$", "", print_text, flags=re.IGNORECASE).strip()
            entities.code_to_print = print_text
            entities.code_content = f"print('{print_text}')\n"
        elif "hello world" in t_low or "hello-world" in t_low:
            entities.code_to_print = "Hello, World!"
            entities.code_content = "print('Hello, World!')\n"
        elif "hello sir" in t_low:
            entities.code_to_print = "Hello Sir"
            entities.code_content = "print('Hello Sir')\n"
        elif any(w in t_low for w in ["write", "create"]) and entities.filename:
            # If user explicitly said "create agent.py" without code instructions:
            # Mandatory rule: DO NOT invent a program. Leave code_content as empty string or None.
            entities.code_content = ""

        # 5. Should run check
        if any(w in t_low for w in ["run it", "run the python file", "run the file", "run agent.py", "execute it", "tell me the actual output", "tell me the output"]):
            entities.should_run = True

        # 6. Calculation extraction
        # e.g. "calculate 678 - 876", "calculate 678 minus 876", "678 - 876"
        calc_match = re.search(r"(?:calculate\s+|what\s+is\s+)?([0-9]+\s*(?:[\+\-\*\/]|minus|plus|times|divided\s+by)\s*[0-9]+(?:\s*(?:[\+\-\*\/]|minus|plus|times|divided\s+by)\s*[0-9]+)*)", t, re.IGNORECASE)
        if calc_match:
            raw_calc = calc_match.group(1).strip()
            # Normalize word operators
            norm_calc = (
                raw_calc.replace("minus", "-")
                .replace("plus", "+")
                .replace("times", "*")
                .replace("divided by", "/")
                .replace("x", "*")
            )
            entities.calculation = norm_calc

        # 7. Website & YouTube Queries
        if "youtube" in t_low:
            entities.website = "youtube"
            play_match = re.search(r"play\s+['\"]?(.+?)['\"]?(?:\s+on\s+youtube|\s+in\s+youtube)?$", t, re.IGNORECASE)
            if play_match:
                q = play_match.group(1).replace("on youtube", "").replace("in youtube", "").strip()
                if q:
                    entities.query = q

        # 8. Screen Inspection & Error Verification (Phase 4)
        if any(w in t_low for w in [
            "check the screen", "check screen", "inspect screen", "look at the screen",
            "look at screen", "check for any errors", "check for errors", "any errors on screen",
            "see if it opened", "check whether it opened", "verify it opened"
        ]):
            entities.check_screen = True

        # Update context memory with newly locked entities
        ctx.update(
            active_app=entities.application,
            active_folder=entities.folder,
            active_file=entities.filename,
            last_goal=t,
            last_calculation=entities.calculation,
        )

        return StructuredGoal(goal=t, entities=entities)
