"""
JARVIS AI Provider Abstraction
Decouples JARVIS core reasoning from any single LLM provider.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator


class AIProvider(ABC):
    """
    Abstract Base Class for AI Providers (Groq, Gemini, OpenAI, Local, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the provider is configured and ready to handle requests."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> str:
        """
        Generate a text completion for a given user prompt.

        Args:
            prompt: Current user message.
            system_prompt: System persona/instructions.
            history: List of previous conversation turns [{'role': 'user'|'assistant', 'content': '...'}].
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            String response from the model.
        """
        pass

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> Iterator[str]:
        """
        Streaming token generation for low-latency voice pipeline.
        Default implementation yields the complete generated text.
        """
        yield self.generate(prompt, system_prompt, history, max_tokens, temperature)
