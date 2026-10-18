"""
AI Provider Factory
Instantiates the configured LLM provider with fallback handling.
"""

from typing import Optional
from config.settings import Settings, get_settings
from .base import AIProvider
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider
from .mock_provider import MockProvider


def get_llm_provider(settings: Optional[Settings] = None, force_mock: bool = False) -> AIProvider:
    """
    Returns the appropriate AIProvider based on configuration settings.
    Falls back gracefully if the primary provider lacks credentials.
    """
    if force_mock:
        return MockProvider()

    cfg = settings or get_settings()
    provider_type = cfg.llm_provider.lower()

    if provider_type == "groq":
        provider = GroqProvider(api_key=cfg.groq_api_key, model=cfg.groq_model)
        if provider.is_available():
            return provider
        print("[LLM Factory] Groq API key not set or unavailable. Checking fallbacks...")

    elif provider_type == "gemini":
        provider = GeminiProvider(api_key=cfg.gemini_api_key)
        if provider.is_available():
            return provider
        print("[LLM Factory] Gemini API key not set or unavailable. Checking fallbacks...")

    # Secondary check: if Groq is available as fallback
    if cfg.groq_api_key:
        p = GroqProvider(api_key=cfg.groq_api_key, model=cfg.groq_model)
        if p.is_available():
            return p

    # Secondary check: if Gemini is available as fallback
    if cfg.gemini_api_key:
        p = GeminiProvider(api_key=cfg.gemini_api_key)
        if p.is_available():
            return p

    # If no online provider key is available, return MockProvider so the app remains responsive
    print("[LLM Factory] No valid online API key found. Operating in local Mock/Offline mode.")
    return MockProvider(name="MockJARVIS (Offline Mode - Set GROQ_API_KEY in .env)")
