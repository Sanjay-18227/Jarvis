"""
JARVIS AI Provider Abstraction
Defines common interface for AI models and planners.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator


class AIProvider(ABC):
    """Abstract interface for all AI provider engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider and active model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if provider is configured and reachable."""
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
        """Generates conversational response."""
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
        raise NotImplementedError("Tool planning not implemented for this provider.")
