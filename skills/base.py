"""
JARVIS Skills Subsystem Base
Provides extensible skill plugin architecture for adding new domain capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ai.tool_planner import ToolDefinition
from core.tool_result import ToolResult


class BaseSkill(ABC):
    """Base class for JARVIS extensible skills."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the skill module."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of capabilities provided."""
        pass

    @property
    def supported_actions(self) -> List[str]:
        """List of action names advertised by this skill."""
        return []

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        """
        Determines if this skill can handle the given intent and context.
        Default implementation checks if intent matches any supported action or skill name.
        """
        int_clean = intent.lower().strip()
        return int_clean in [a.lower() for a in self.supported_actions]

    @abstractmethod
    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        """
        Executes a deterministic skill action locally.
        Returns a structured ToolResult contract.
        """
        pass

    def get_tools(self) -> List[ToolDefinition]:
        """Returns tool planner definitions declared by this skill."""
        return []
