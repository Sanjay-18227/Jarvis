"""
Google Gemini AI Provider Implementation
Uses the official google-genai SDK with gemini-3.8-flash and multi-turn context support.
"""

from typing import List, Dict, Optional, Iterator
from .base import AIProvider
from config.constants import JARVIS_SYSTEM_PROMPT


class GeminiProvider(AIProvider):
    """
    Google Gemini Provider using official google-genai SDK.
    """

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
                # Gemini roles are 'user' and 'model'
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

        try:
            from google.genai import types
            sys_instruction = system_prompt or JARVIS_SYSTEM_PROMPT
            config = types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            contents = self._build_contents(prompt, history)
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            return f"I encountered an error connecting to Gemini: {str(e)}"

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

        try:
            from google.genai import types
            sys_instruction = system_prompt or JARVIS_SYSTEM_PROMPT
            config = types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            contents = self._build_contents(prompt, history)
            stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error during streaming: {str(e)}"
