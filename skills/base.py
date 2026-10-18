"""
JARVIS Skills Subsystem Base
Provides extensible skill plugin architecture for adding new domain capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ai.tool_planner import ToolDefinition


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

    @abstractmethod
    def get_tools(self) -> List[ToolDefinition]:
        """Returns tools declared by this skill."""
        pass
