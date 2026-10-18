"""
JARVIS System Settings & Environment Configuration
"""

import os
import sys
import socket
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Prioritize IPv4 resolution to prevent Windows IPv6 DNS timeout latency
_orig_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_ipv4(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses


socket.getaddrinfo = _getaddrinfo_ipv4

# Ensure UTF-8 console output for multilingual characters (Tamil, Hindi, English, emojis)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@dataclass
class Settings:
    # User / Persona
    owner_name: str = "Sanjay"
    assistant_name: str = "JARVIS"
    preferred_title: str = "Sir"

    # AI Provider
    llm_provider: str = "groq"
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_stt_model: str = "whisper-large-v3-turbo"

    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None

    # Voice Input (STT)
    stt_engine: str = "groq_whisper"  # groq_whisper | google | local
    stt_language: str = "en-US"
    stt_timeout_seconds: float = 6.0
    stt_phrase_time_limit: float = 12.0
    energy_threshold: int = 350
    dynamic_energy_threshold: bool = True

    # Voice Output (TTS)
    tts_engine: str = "edge_tts"  # edge_tts | pyttsx3 | elevenlabs
    tts_voice: str = "en-US-ChristopherNeural"
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0%"

    # Wake Word
    wake_word: str = "hey jarvis"
    wake_word_engine: str = "openwakeword"

    # Security
    default_permission_level: str = "OWNER"
    require_confirmation_risk_level: int = 3
    audit_log_enabled: bool = True
    log_level: str = "INFO"

    # Paths
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @classmethod
    def load_from_env(cls, env_path: Optional[str] = None) -> "Settings":
        """Load settings from .env file and environment variables."""
        if env_path:
            load_dotenv(env_path, override=True)
        else:
            default_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
            if os.path.exists(default_env):
                load_dotenv(default_env, override=True)
            else:
                load_dotenv(override=True)

        return cls(
            owner_name=os.getenv("OWNER_NAME", "Sanjay"),
            assistant_name=os.getenv("ASSISTANT_NAME", "JARVIS"),
            preferred_title=os.getenv("PREFERRED_TITLE", "Sir"),
            llm_provider=os.getenv("LLM_PROVIDER", "groq").lower(),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            groq_stt_model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            stt_engine=os.getenv("STT_ENGINE", "groq_whisper").lower(),
            stt_language=os.getenv("STT_LANGUAGE", "en-US"),
            stt_timeout_seconds=float(os.getenv("STT_TIMEOUT_SECONDS", "6.0")),
            stt_phrase_time_limit=float(os.getenv("STT_PHRASE_TIME_LIMIT", "12.0")),
            energy_threshold=int(os.getenv("ENERGY_THRESHOLD", "350")),
            dynamic_energy_threshold=os.getenv("DYNAMIC_ENERGY_THRESHOLD", "true").lower() == "true",
            tts_engine=os.getenv("TTS_ENGINE", "edge_tts").lower(),
            tts_voice=os.getenv("TTS_VOICE", "en-US-ChristopherNeural"),
            tts_rate=os.getenv("TTS_RATE", "+0%"),
            tts_volume=os.getenv("TTS_VOLUME", "+0%"),
            tts_pitch=os.getenv("TTS_PITCH", "+0%"),
            wake_word=os.getenv("WAKE_WORD", "hey jarvis").lower(),
            wake_word_engine=os.getenv("WAKE_WORD_ENGINE", "openwakeword").lower(),
            default_permission_level=os.getenv("DEFAULT_PERMISSION_LEVEL", "OWNER").upper(),
            require_confirmation_risk_level=int(os.getenv("REQUIRE_CONFIRMATION_RISK_LEVEL", "3")),
            audit_log_enabled=os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

    def sanitized_dict(self) -> dict:
        """Returns safe dictionary with secrets masked for logging."""
        data = {}
        for k, v in self.__dict__.items():
            if any(secret_term in k.lower() for secret_term in ["key", "secret", "token", "password"]):
                data[k] = "***REDACTED***" if v else None
            else:
                data[k] = v
        return data


_global_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Singleton getter for system settings."""
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings.load_from_env()
    return _global_settings
