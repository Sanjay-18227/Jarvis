"""
Privacy Guard Subsystem
Guards against private data leaks, filters sensitive credentials, and redacts secrets.
"""

import re
from typing import Dict, Any, List, Tuple
from config.constants import SpeakerClassification, PermissionLevel


class PrivacyGuard:
    """
    Dedicated Privacy & Security Guard.
    Scans inputs, outputs, and system actions for sensitive data and leaks.
    """

    # Secret and credential regex patterns
    SECRET_PATTERNS = [
        r"(sk-[a-zA-Z0-9_-]{20,})",                    # OpenAI / ElevenLabs style keys
        r"(gsk_[a-zA-Z0-9_-]{20,})",                   # Groq API keys
        r"(AIza[0-9A-Za-z-_]{35})",                    # Google API keys
        r"(ghp_[a-zA-Z0-9]{36})",                      # GitHub personal tokens
        r"(bearer\s+[a-zA-Z0-9_\-\.]{20,})",           # Bearer tokens
        r"(password\s*[:=]\s*['\"]?[^\s'\"]+['\"]?)",  # Passwords
        r"(\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b)", # Credit card numbers
    ]

    # Sensitive intent keywords
    SENSITIVE_KEYWORDS = {
        "CREDENTIALS": ["password", "api key", "secret", "token", "private key", "credential", "saved credentials"],
        "PRIVATE_FILES": ["bank statement", "tax return", "passport", "private photo", "id card", "my files", "personal data", "private folder", "project data", "private project", "my private"],
        "FINANCIAL": ["pay", "purchase", "transfer money", "credit card", "bank account", "buy"],
        "DELETION": ["delete", "remove folder", "wipe", "format disk", "destroy", "delete this file", "delete all files", "remove this file"],
        "SYSTEM_CHANGE": ["registry", "format", "shutdown", "reboot", "powershell", "cmd", "terminal", "install", "uninstall", "system settings", "change settings", "shell", "run command", "run a powershell"],
        "FILE_MODIFICATION": ["modify file", "edit file", "send file", "upload", "write to file"],
    }

    def redact_secrets(self, text: str) -> str:
        """
        Replaces detected secrets, API keys, tokens, and credentials with [REDACTED].
        """
        if not text:
            return ""

        redacted = text
        for pattern in self.SECRET_PATTERNS:
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
        return redacted

    def assess_privacy_risk(self, user_query: str) -> Tuple[bool, List[str]]:
        """
        Determines whether the user request involves sensitive or private data.

        Returns:
            (is_sensitive, list_of_risk_categories)
        """
        q = user_query.lower()
        matched_categories = []

        for category, keywords in self.SENSITIVE_KEYWORDS.items():
            for kw in keywords:
                if kw in q:
                    matched_categories.append(category)
                    break

        return bool(matched_categories), matched_categories

    def validate_speaker_permission(
        self,
        speaker: SpeakerClassification,
        query: str,
    ) -> Tuple[bool, str]:
        """
        Enforces least-privilege access:
        UNKNOWN speakers cannot access private data, execute privileged commands, or read secrets.

        Returns:
            (is_allowed, reason_or_refusal_message)
        """
        is_sensitive, categories = self.assess_privacy_risk(query)

        if speaker == SpeakerClassification.UNKNOWN:
            if is_sensitive:
                return False, "Access denied. I cannot access or disclose the owner's private information, system settings, or execute commands for an unverified speaker."

        if speaker == SpeakerClassification.UNCERTAIN:
            if is_sensitive:
                return False, "Speaker verification uncertain. Please re-authenticate before accessing sensitive data."

        return True, "Authorized"
