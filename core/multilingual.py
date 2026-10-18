"""
Multilingual & Code-Mixed NLP Subsystem
Handles Tamil, Tanglish, Hindi, and English intent mapping and dynamic voice selection.
"""

import re
from typing import Tuple, Dict, Any, Optional


class MultilingualProcessor:
    """
    Parses code-mixed expressions (Tamil, Tanglish, Hindi) into normalized actions.
    """

    # Keyword mappings for Tanglish / Tamil commands
    ACTION_PATTERNS = [
        # Open Application
        (r"(chrome|browser)\s*(open\s*பண்ணு|open\s*pan+u|open\s*karo|thira|திற)", "open_application", "chrome"),
        (r"(vs\s*code|vscode|code)\s*(open\s*பண்ணு|open\s*pan+u|open\s*karo|thira|திற)", "open_application", "vscode"),
        (r"(calculator|calc)\s*(open\s*பண்ணு|open\s*pan+u|open\s*karo)", "open_application", "calculator"),
        (r"(notepad)\s*(open\s*பண்ணு|open\s*pan+u|open\s*karo)", "open_application", "notepad"),
        (r"(settings)\s*(open\s*பண்ணு|open\s*pan+u|open\s*karo)", "open_application", "settings"),

        # Close Application
        (r"(chrome|browser)\s*(close\s*பண்ணு|close\s*pan+u|moodu|மூடு)", "close_application", "chrome"),
        (r"(calculator|calc)\s*(close\s*பண்ணு|close\s*pan+u|moodu|மூடு)", "close_application", "calculator"),
        (r"(vs\s*code|vscode)\s*(close\s*பண்ணு|close\s*pan+u|moodu|மூடு)", "close_application", "vscode"),

        # Media controls
        (r"(paatu|song|music)\s*(podu|play\s*pannu|play\s*panu|chalao)", "media_control", "play_pause"),
        (r"(paatu|song|music)\s*(nirutthu|stop\s*pannu|pause\s*panu|roko)", "media_control", "play_pause"),
        (r"(adutha\s*paatu|next\s*song)", "media_control", "next"),
        (r"(satham\s*kura|volume\s*kura|volume\s*down)", "media_control", "volume_down"),
        (r"(satham\s*yethu|volume\s*yethu|volume\s*up)", "media_control", "volume_up"),

        # Lock Screen
        (r"(laptop|screen|system)\s*(lock\s*pannu|lock\s*panu)", "lock_workstation", ""),
    ]

    TAMIL_UNICODE_RANGE = (0x0B80, 0x0BFF)
    DEVANAGARI_UNICODE_RANGE = (0x0900, 0x097F)

    def detect_language(self, text: str) -> str:
        """
        Detects predominant language of text: 'ta' (Tamil), 'hi' (Hindi), or 'en' (English).
        """
        tamil_chars = sum(1 for c in text if 0x0B80 <= ord(c) <= 0x0BFF)
        hindi_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)

        if tamil_chars > 2:
            return "ta"
        if hindi_chars > 2:
            return "hi"
        return "en"

    def get_preferred_voice_for_language(self, language: str) -> str:
        """Returns appropriate Microsoft Edge Neural voice."""
        if language == "ta":
            return "ta-IN-ValluvarNeural"
        elif language == "hi":
            return "hi-IN-MadhurNeural"
        return "en-US-ChristopherNeural"

    def parse_code_mixed_intent(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Extracts structured intent from code-mixed Tamil/Tanglish/Hindi expressions.

        Returns:
            (action_name, parameters_dict) or None if no direct pattern match.
        """
        t = text.lower().strip()
        for pattern, action, target in self.ACTION_PATTERNS:
            if re.search(pattern, t):
                params = {}
                if "application" in action:
                    params["target"] = target
                elif "media" in action:
                    params["action"] = target
                return action, params

        return None
