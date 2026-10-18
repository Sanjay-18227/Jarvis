"""
Groq LLM Provider Implementation
Provides ultra-low latency inference using Groq's LPU infrastructure.
"""

from typing import List, Dict, Optional, Iterator
from .base import AIProvider
from config.constants import JARVIS_SYSTEM_PROMPT


class GroqProvider(AIProvider):
    """
    Groq AI Provider supporting Llama-3.3-70b-versatile, Llama-3.1-8b-instant, etc.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self._client = None

        if self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"[GroqProvider] Error initializing Groq client: {e}")

    @property
    def name(self) -> str:
        return f"Groq ({self.model})"

    def is_available(self) -> bool:
        return self._client is not None and bool(self.api_key)

    def _prepare_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        messages = []
        # System prompt
        sys_msg = system_prompt or JARVIS_SYSTEM_PROMPT
        messages.append({"role": "system", "content": sys_msg})

        # History
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})

        # Current prompt
        messages.append({"role": "user", "content": prompt})
        return messages

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
                "Groq API key is not configured. Please set GROQ_API_KEY in your .env file."
            )

        messages = self._prepare_messages(prompt, system_prompt, history)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"I encountered an error connecting to Groq: {str(e)}"

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> Iterator[str]:
        if not self.is_available():
            yield "Groq API key is not configured. Please set GROQ_API_KEY in your .env file."
            return

        messages = self._prepare_messages(prompt, system_prompt, history)
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"Error during streaming: {str(e)}"
