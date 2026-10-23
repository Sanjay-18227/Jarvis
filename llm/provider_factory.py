"""
AI Provider Factory
Instantiates the configured LLM provider with fallback handling.
Supports OpenRouter, Google Gemini, Groq, Ollama local, and Mock offline.
"""

from typing import Optional
from config.settings import Settings, get_settings
from .base import AIProvider
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider
from .openrouter_provider import OpenRouterProvider
from .ollama_provider import OllamaProvider
from .mock_provider import MockProvider


def get_llm_provider(settings: Optional[Settings] = None, force_mock: bool = False) -> AIProvider:
    """
    Returns the appropriate AIProvider based on configuration settings.
    Falls back gracefully if the primary provider lacks credentials.
    """
    if force_mock:
        return MockProvider()

    cfg = settings or get_settings()
    provider_type = cfg.llm_provider.lower().strip()

    if provider_type == "openrouter":
        provider = OpenRouterProvider(
            api_key=cfg.openrouter_api_key,
            model=cfg.openrouter_model,
            base_url=cfg.openrouter_base_url,
        )
        if provider.is_available():
            return provider
        print("[LLM Factory] OpenRouter API key not configured or unavailable. Checking fallbacks...")

    elif provider_type == "gemini":
        provider = GeminiProvider(api_key=cfg.gemini_api_key, model=cfg.gemini_model)
        if provider.is_available():
            return provider
        print("[LLM Factory] Gemini API key not set or unavailable. Checking fallbacks...")

    elif provider_type == "groq":
        provider = GroqProvider(api_key=cfg.groq_api_key, model=cfg.groq_model)
        if provider.is_available():
            return provider
        print("[LLM Factory] Groq API key not set or unavailable. Checking fallbacks...")

    elif provider_type == "ollama":
        provider = OllamaProvider(base_url=cfg.ollama_base_url, model=cfg.ollama_model)
        if provider.is_available():
            return provider
        print(f"[LLM Factory] Local Ollama daemon not reachable at {cfg.ollama_base_url}. Checking fallbacks...")

    # Secondary check: if OpenRouter is available as fallback
    if cfg.openrouter_api_key:
        p = OpenRouterProvider(
            api_key=cfg.openrouter_api_key,
            model=cfg.openrouter_model,
            base_url=cfg.openrouter_base_url,
        )
        if p.is_available():
            return p

    # Secondary check: if Gemini is available as fallback
    if cfg.gemini_api_key:
        p = GeminiProvider(api_key=cfg.gemini_api_key, model=cfg.gemini_model)
        if p.is_available():
            return p

    # Secondary check: if Groq is available as fallback
    if cfg.groq_api_key:
        p = GroqProvider(api_key=cfg.groq_api_key, model=cfg.groq_model)
        if p.is_available():
            return p

    # Secondary check: if local Ollama is available
    p_ollama = OllamaProvider(base_url=cfg.ollama_base_url, model=cfg.ollama_model)
    if p_ollama.is_available():
        return p_ollama

    # If no online or local LLM is available, return MockProvider so JARVIS operates safely offline
    print("[LLM Factory] No valid online API key or local LLM found. Operating in local Mock/Offline mode.")
    return MockProvider(name="MockJARVIS (Offline Mode - Set OPENROUTER_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY in .env)")
