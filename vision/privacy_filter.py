"""
JARVIS Vision Privacy Filter & Secret Redaction Subsystem
Protects user privacy and sensitive credentials before screen analysis or cloud vision upload.
Enforces:
1. UNKNOWN speakers must NEVER access sensitive window contents or private data.
2. Sensitive windows (password managers, banking, incognito, .env) are flagged or blocked.
3. Obvious secrets (API keys, passwords, bearer tokens) are redacted prior to vision logging.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from config.constants import SpeakerClassification


class PrivacyFilter:
    """
    Evaluates window context, queries, and extracted vision text for sensitive data.
    """

    SENSITIVE_APP_KEYWORDS = [
        "keepass", "1password", "bitwarden", "lastpass", "dashlane", "enpass",
        "roboform", "nordpass", "authenticator", "authy", "duo", "yubikey",
        "credential", "credentials", "keychain", "tor browser"
    ]

    SENSITIVE_TITLE_KEYWORDS = [
        "keepass", "1password", "bitwarden", "lastpass", "dashlane",
        "incognito", "inprivate", "private browsing",
        "credential", "credentials", "authenticator", "keychain",
        ".env", "id_rsa", "id_ed25519", "secret", "secrets",
        "password", "passwords", "recovery code", "recovery codes",
        "banking", "online banking", "bank login", "paypal", "credit card",
        "metamask", "crypto wallet"
    ]

    # Secret pattern matchers
    SECRET_PATTERNS = [
        (re.compile(r"\b(sk-[a-zA-Z0-9_-]{20,})\b"), "sk-***REDACTED***"),
        (re.compile(r"\b(AIzaSy[a-zA-Z0-9_-]{33})\b"), "AIzaSy***REDACTED***"),
        (re.compile(r"\b(gsk_[a-zA-Z0-9]{32,})\b"), "gsk_***REDACTED***"),
        (re.compile(r"\bBearer\s+([a-zA-Z0-9_\-\.]{20,})\b", re.IGNORECASE), "Bearer ***REDACTED***"),
        (re.compile(r"(api[_-]?key\s*[:=]\s*)['\"]?([a-zA-Z0-9_\-]{16,})['\"]?", re.IGNORECASE), r"\1'***REDACTED***'"),
        (re.compile(r"(password\s*[:=]\s*)['\"]?([^\s'\"]{6,})['\"]?", re.IGNORECASE), r"\1'***REDACTED***'"),
        (re.compile(r"(secret\s*[:=]\s*)['\"]?([^\s'\"]{6,})['\"]?", re.IGNORECASE), r"\1'***REDACTED***'"),
    ]

    @classmethod
    def is_sensitive_window(cls, title: str, app_name: str = "") -> bool:
        """Determines if a window title or application indicates sensitive content."""
        t_low = (title or "").lower()
        a_low = (app_name or "").lower()

        if any(kw in a_low for kw in cls.SENSITIVE_APP_KEYWORDS):
            return True
        if any(kw in t_low for kw in cls.SENSITIVE_TITLE_KEYWORDS):
            return True
        return False

    @classmethod
    def check_privacy(
        cls,
        window_info: Dict[str, Any],
        speaker: SpeakerClassification = SpeakerClassification.OWNER,
        query: str = "",
    ) -> Tuple[bool, str, bool]:
        """
        Evaluates whether screen inspection is permitted under privacy rules.

        Returns:
            (is_allowed: bool, reason: str, requires_confirmation: bool)
        """
        title = window_info.get("title", "")
        app_name = window_info.get("application", "")
        is_sensitive = cls.is_sensitive_window(title, app_name)

        # Rule 1: UNKNOWN speakers are blocked from sensitive windows
        if is_sensitive and speaker != SpeakerClassification.OWNER:
            return False, "Access denied: The active window contains sensitive private credentials.", False

        # Rule 2: If active window is a dedicated password manager, even OWNER gets a confirmation requirement
        password_managers = ["keepass", "1password", "bitwarden", "lastpass", "dashlane"]
        if any(pm in app_name.lower() or pm in title.lower() for pm in password_managers):
            if speaker == SpeakerClassification.OWNER:
                return True, "Active window is a password manager. Owner confirmation advised.", True

        # Rule 3: General privacy pass
        return True, "Privacy check passed.", False

    @classmethod
    def redact_secrets(cls, text: str) -> str:
        """Redacts sensitive credentials, tokens, and passwords from text."""
        if not text:
            return ""
        result = text
        for pattern, replacement in cls.SECRET_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    @classmethod
    def sanitize_observation(cls, obs_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deeply sanitizes structured observation dictionaries before logging or caching."""
        if not isinstance(obs_data, dict):
            return obs_data

        clean = dict(obs_data)
        if "screen_summary" in clean and isinstance(clean["screen_summary"], str):
            clean["screen_summary"] = cls.redact_secrets(clean["screen_summary"])

        if "visible_text" in clean and isinstance(clean["visible_text"], list):
            clean["visible_text"] = [cls.redact_secrets(str(t)) for t in clean["visible_text"]]

        if "errors" in clean and isinstance(clean["errors"], list):
            sanitized_errors = []
            for err in clean["errors"]:
                if isinstance(err, dict):
                    err_copy = dict(err)
                    if "message" in err_copy:
                        err_copy["message"] = cls.redact_secrets(str(err_copy["message"]))
                    sanitized_errors.append(err_copy)
                else:
                    sanitized_errors.append(cls.redact_secrets(str(err)))
            clean["errors"] = sanitized_errors

        return clean
