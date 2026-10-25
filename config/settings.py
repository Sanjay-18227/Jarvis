"""
JARVIS System Settings & Environment Configuration
"""

import os
import sys
import socket
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Safe IPv4 prioritization with graceful fallback
_orig_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_ipv4(*args, **kwargs):
    try:
        responses = _orig_getaddrinfo(*args, **kwargs)
        ipv4 = [r for r in responses if r[0] == socket.AF_INET]
        return ipv4 if ipv4 else responses
    except Exception:
        return _orig_getaddrinfo(*args, **kwargs)


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

    # AI Provider (OpenRouter is Primary, Gemini for Multimodal/Vision)
    llm_provider: str = "openrouter"
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_stt_model: str = "whisper-large-v3-turbo"

    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.8-flash"
    screen_vision_enabled: bool = False
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None

    # Voice Layer & LiveKit Voice Session
    voice_provider: str = "livekit"  # livekit | legacy
    livekit_url: Optional[str] = None
    livekit_api_key: Optional[str] = None
    livekit_api_secret: Optional[str] = None

    # Voice Input (STT) & VAD
    stt_engine: str = "groq_whisper"  # groq_whisper | google | local
    stt_language: str = "en-US"
    stt_timeout_seconds: float = 10.0
    stt_phrase_time_limit: float = 60.0  # Generous safety timeout (not normal cutoff)
    silence_threshold_seconds: float = 1.5  # Configurable end-of-speech silence (1.2 - 2.0s)
    max_record_seconds: float = 60.0  # Maximum safety recording duration
    energy_threshold: int = 400
    dynamic_energy_threshold: bool = True

    # Configurable VAD Rejection & Calibration Settings (Part A & B)
    vad_speech_threshold: int = 400
    vad_min_speech_ms: int = 250
    vad_end_silence_ms: int = 1500
    vad_noise_calibration_seconds: float = 0.5

    # Speaker Verification & Biometrics (Part C & E)
    owner_similarity_threshold: float = 0.78
    uncertain_similarity_threshold: float = 0.70

    # Browser Configuration (Default: Microsoft Edge)
    default_browser: str = "edge"

    # Real-time Barge-In / Interruption
    barge_in_enabled: bool = True
    barge_in_cooldown_seconds: float = 0.25

    # Voice Output (TTS)
    tts_engine: str = "edge_tts"  # edge_tts | elevenlabs | pyttsx3
    tts_voice: str = "en-US-AndrewMultilingualNeural"  # Single consistent multilingual voice
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"

    # Wake Word
    wake_word: str = "hey jarvis"
    wake_word_engine: str = "openwakeword"

    # Security (Default permission for text console; Voice strictly verified via biometrics)
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
            llm_provider=os.getenv("LLM_PROVIDER", "openrouter").lower(),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            groq_stt_model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            screen_vision_enabled=os.getenv("SCREEN_VISION_ENABLED", "false").lower() == "true",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_provider=os.getenv("VOICE_PROVIDER", "livekit").lower(),
            livekit_url=os.getenv("LIVEKIT_URL"),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY"),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET"),
            stt_engine=os.getenv("STT_ENGINE", "groq_whisper").lower(),
            stt_language=os.getenv("STT_LANGUAGE", "en-US"),
            stt_timeout_seconds=float(os.getenv("STT_TIMEOUT_SECONDS", "10.0")),
            stt_phrase_time_limit=float(os.getenv("STT_PHRASE_TIME_LIMIT", "60.0")),
            silence_threshold_seconds=float(os.getenv("SILENCE_THRESHOLD_SECONDS", "1.5")),
            max_record_seconds=float(os.getenv("MAX_RECORD_SECONDS", "60.0")),
            energy_threshold=int(os.getenv("VAD_SPEECH_THRESHOLD", os.getenv("ENERGY_THRESHOLD", "400"))),
            dynamic_energy_threshold=os.getenv("DYNAMIC_ENERGY_THRESHOLD", "true").lower() == "true",
            vad_speech_threshold=int(os.getenv("VAD_SPEECH_THRESHOLD", os.getenv("ENERGY_THRESHOLD", "400"))),
            vad_min_speech_ms=int(os.getenv("VAD_MIN_SPEECH_MS", "250")),
            vad_end_silence_ms=int(os.getenv("VAD_END_SILENCE_MS", "1500")),
            vad_noise_calibration_seconds=float(os.getenv("VAD_NOISE_CALIBRATION_SECONDS", "0.5")),
            owner_similarity_threshold=float(os.getenv("OWNER_SIMILARITY_THRESHOLD", "0.78")),
            uncertain_similarity_threshold=float(os.getenv("UNCERTAIN_SIMILARITY_THRESHOLD", "0.70")),
            default_browser=os.getenv("DEFAULT_BROWSER", "edge").lower(),
            barge_in_enabled=os.getenv("BARGE_IN_ENABLED", "true").lower() == "true",
            barge_in_cooldown_seconds=float(os.getenv("BARGE_IN_COOLDOWN_SECONDS", "0.25")),
            tts_engine=os.getenv("TTS_ENGINE", "edge_tts").lower(),
            tts_voice=os.getenv("TTS_VOICE", "en-US-AndrewMultilingualNeural"),
            tts_rate=os.getenv("TTS_RATE", "+0%"),
            tts_volume=os.getenv("TTS_VOLUME", "+0%"),
            tts_pitch=os.getenv("TTS_PITCH", "+0Hz"),
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
