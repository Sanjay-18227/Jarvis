"""
Google Gemini AI Provider Implementation
"""

from typing import List, Dict, Optional, Iterator
from .base import AIProvider
from config.constants import JARVIS_SYSTEM_PROMPT


class GeminiProvider(AIProvider):
    """
    Google Gemini Provider using official google-genai SDK.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
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

        try:
            from google.genai import types
            sys_instruction = system_prompt or JARVIS_SYSTEM_PROMPT
            config = types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            return f"I encountered an error connecting to Gemini: {str(e)}"
