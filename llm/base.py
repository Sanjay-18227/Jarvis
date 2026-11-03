"""
JARVIS AI Provider Abstraction
Decouples JARVIS core reasoning from any single LLM provider.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator


class ReasoningProvider(ABC):
    """
    Abstract Base Class for AI Reasoning Providers (OpenRouter, Gemini, Groq, etc.).
    Decouples JARVIS core reasoning from any single LLM provider.
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

    @property
    def reasoning_provider_available(self) -> bool:
        """Indicates whether reasoning provider is currently available and circuit is healthy."""
        return self.is_available()


class AIProvider(ReasoningProvider):
    """
    Backwards-compatible alias for AIProvider.
    """
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

    def plan_with_tools(
        self,
        goal: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        screen_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reasons about the user's goal with available tools.
        Returns structured plan containing:
        - 'thought': reasoning
        - 'tool_calls': list of {name, parameters}
        - 'response': optional direct answer
        """
        return {"thought": "Tool planning not implemented for this provider.", "tool_calls": [], "response": None}

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

