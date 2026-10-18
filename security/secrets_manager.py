"""
JARVIS Secrets Manager Subsystem
Guards against accidental exposure of sensitive keys, passwords, authentication tokens,
and financial information across speech, logging, memory, and automation.
"""

import re
from typing import Tuple, Dict, Any, Optional


class SecretsManager:
    """
    Detects, sanitizes, and prevents exposure of credentials and confidential tokens.
    """

    SECRET_PATTERNS = [
        # Generic API keys and bearer tokens
        (r"(?i)(api[_-]?key|apikey|secret|token|auth|password|passwd|pwd)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?", r"\1: ***REDACTED***"),
        # Google API keys
        (r"AIza[0-9A-Za-z-_]{35}", "***GOOGLE_API_KEY_REDACTED***"),
        # ElevenLabs / OpenAI keys
        (r"sk-[a-zA-Z0-9-_]{20,}", "***API_KEY_REDACTED***"),
        # JWT tokens
        (r"eyJ[a-zA-Z0-9-_]+\.eyJ[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+", "***JWT_TOKEN_REDACTED***"),
        # Credit card numbers
        (r"\b(?:\d{4}[ -]?){3}\d{4}\b", "***CREDIT_CARD_REDACTED***"),
        # Private SSH / RSA keys
        (r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----", "***PRIVATE_KEY_REDACTED***"),
    ]

    def contains_secret(self, text: str) -> bool:
        """Returns True if text appears to contain sensitive credentials."""
        if not text:
            return False
        for pattern, _ in self.SECRET_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def sanitize(self, text: str) -> str:
        """Replaces detected credentials with masked placeholders."""
        if not text:
            return ""
        sanitized = text
        for pattern, replacement in self.SECRET_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized)
        return sanitized

    def validate_safe_for_speech(self, text: str) -> Tuple[bool, str]:
        """
        Validates whether text is safe to be spoken aloud by TTS.
        Never speaks raw passwords or keys aloud.
        """
        if self.contains_secret(text):
            safe_text = self.sanitize(text)
            return True, safe_text
        return True, text
