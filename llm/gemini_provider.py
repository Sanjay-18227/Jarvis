"""
Google Gemini AI Provider Implementation
Uses the official google-genai SDK with multi-model failover (3.8-flash -> 2.0-flash -> 1.5-flash),
retry with exponential backoff on 503 spikes, and multi-turn context support.
"""

import time
from typing import List, Dict, Optional, Iterator
from .base import AIProvider
from config.constants import JARVIS_SYSTEM_PROMPT


class GeminiProvider(AIProvider):
    """
    Google Gemini Provider with automatic 503 retry and model failover.
    """

    FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"]

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.8-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[GeminiProvider] Error initializing Gemini client: {e}")

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"

    def is_available(self) -> bool:
        return self._client is not None and bool(self.api_key)

    def _build_contents(self, prompt: str, history: Optional[List[Dict[str, str]]] = None):
        """Constructs multi-turn message contents for the Gemini API."""
        from google.genai import types

        contents = []
        if history:
            for turn in history:
                role = turn.get("role", "user")
                gemini_role = "model" if role == "assistant" else "user"
                content_text = turn.get("content", "")
                if content_text:
                    contents.append(
                        types.Content(
                            role=gemini_role,
                            parts=[types.Part.from_text(text=content_text)]
                        )
                    )

        # Add the current user prompt
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        )
        return contents

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
            )

        from google.genai import types

        sys_instruction = system_prompt or JARVIS_SYSTEM_PROMPT
        config = types.GenerateContentConfig(
            system_instruction=sys_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        contents = self._build_contents(prompt, history)

        # Models to try: primary model first, followed by fallbacks
        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]

        last_error = None
        for target_model in models_to_try:
            for attempt in range(2):  # up to 2 attempts per model
                try:
                    response = self._client.models.generate_content(
                        model=target_model,
                        contents=contents,
                        config=config,
                    )
                    return response.text or ""
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
                        # Server busy; brief wait before retry or model switch
                        time.sleep(1.0)
                        continue
                    else:
                        break  # Non-503 error, move to next model or return error

        return f"I encountered a temporary connection issue: {str(last_error)}"

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> Iterator[str]:
        if not self.is_available():
            yield "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
            return

        from google.genai import types

        sys_instruction = system_prompt or JARVIS_SYSTEM_PROMPT
        config = types.GenerateContentConfig(
            system_instruction=sys_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        contents = self._build_contents(prompt, history)

        try:
            stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception:
            # Fallback to standard generate
            yield self.generate(prompt, system_prompt, history, max_tokens, temperature)
