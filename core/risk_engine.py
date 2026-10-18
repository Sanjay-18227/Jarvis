"""
Action Risk Classification Engine
Classifies requested computer actions into Risk Levels 0 to 4.
"""

from typing import Dict, Any, Tuple
from config.constants import RiskLevel


class RiskEngine:
    """
    Evaluates intent and tools to determine risk severity.
    """

    CRITICAL_KEYWORDS = [
        "format", "drop database", "delete system", "reg delete", "shutdown /r",
        "rmdir /s", "del /f", "transfer money", "buy", "purchase", "password",
        "api key", "wipe", "truncate table", "kill -9"
    ]

    SENSITIVE_KEYWORDS = [
        "delete", "delete_path", "delete file", "remove file", "remove",
        "send email", "upload", "git push --force",
        "install software", "pip install", "modify registry", "change setting"
    ]

    MODERATE_KEYWORDS = [
        "create file", "write code", "edit file", "create folder", "save file",
        "npm install", "git commit", "run script"
    ]

    SAFE_KEYWORDS = [
        "open application", "open chrome", "open vs code", "open calculator",
        "open notepad", "take screenshot", "volume", "battery", "search google",
        "scroll down", "type text"
    ]

    def classify_action(self, action_name: str, parameters: Dict[str, Any] = None) -> Tuple[RiskLevel, str]:
        """
        Calculates risk level for an action or tool call.

        Returns:
            (RiskLevel, rationale)
        """
        act_lower = action_name.lower()
        param_str = str(parameters or {}).lower()
        combined = f"{act_lower} {param_str}"

        # Level 4 - Critical
        for kw in self.CRITICAL_KEYWORDS:
            if kw in combined:
                return RiskLevel.LEVEL_4_CRITICAL, f"Critical operation detected: contains '{kw}'"

        # Level 3 - Sensitive
        for kw in self.SENSITIVE_KEYWORDS:
            if kw in combined:
                return RiskLevel.LEVEL_3_SENSITIVE, f"Sensitive action detected: contains '{kw}'"

        # Level 2 - Moderate
        for kw in self.MODERATE_KEYWORDS:
            if kw in combined:
                return RiskLevel.LEVEL_2_MODERATE, f"Moderate system modification: contains '{kw}'"

        # Level 1 - Safe
        for kw in self.SAFE_KEYWORDS:
            if kw in combined:
                return RiskLevel.LEVEL_1_SAFE, "Standard safe computer control"

        # Level 0 - Informational
        return RiskLevel.LEVEL_0_INFORMATIONAL, "Conversational / Informational request"
