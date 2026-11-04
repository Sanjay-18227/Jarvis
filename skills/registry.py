"""
JARVIS Skill Registry
Central dispatcher for all deterministic local skills.
Routes intents and context actions to specialized skills before escalating to LLM.
"""

from typing import Dict, Any, List, Optional, Tuple
from skills.base import BaseSkill
from core.tool_result import ToolResult


class SkillRegistry:
    """
    Extensible Skill Registry managing local deterministic execution.
    """

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        """Registers a skill instance."""
        self._skills[skill.name.lower()] = skill

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Retrieves a skill by name."""
        return self._skills.get(name.lower())

    def list_skills(self) -> List[str]:
        """Returns list of registered skill names."""
        return list(self._skills.keys())

    def can_handle(
        self,
        intent: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None,
    ) -> Optional[Tuple[BaseSkill, str]]:
        """
        Checks if any registered skill can handle the given intent and context.
        Returns (skill_instance, action_name) if matched, else None.
        """
        params = params or {}
        for skill in self._skills.values():
            if skill.can_handle(intent, params, context):
                # If the skill advertised an action or handles the intent
                action = params.get("action") or intent
                return skill, action
        return None

    def execute(
        self,
        skill_name: str,
        action: str,
        params: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> ToolResult:
        """Executes an action on a named skill."""
        skill = self.get_skill(skill_name)
        if not skill:
            return ToolResult.fail(
                action=action,
                error=f"Skill '{skill_name}' not registered in SkillRegistry.",
            )
        return skill.execute(action, params, context)

    def try_execute(
        self,
        intent: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None,
    ) -> Optional[ToolResult]:
        """
        Attempts to resolve and execute intent on any matching skill.
        Returns ToolResult if handled, None otherwise.
        """
        params = params or {}
        match = self.can_handle(intent, params, context)
        if match:
            skill, action = match
            return skill.execute(action, params, context)
        return None


_global_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Singleton getter for default SkillRegistry initialized with all standard skills."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
        _init_default_skills(_global_registry)
    return _global_registry


def _init_default_skills(registry: SkillRegistry):
    """Instantiates and registers all standard skills."""
    from skills.calculator_skill import CalculatorSkill
    from skills.paint_skill import PaintSkill
    from skills.vscode_skill import VSCodeSkill
    from skills.browser_skill import BrowserSkill
    from skills.notepad_skill import NotepadSkill
    from skills.file_skill import FileSkill
    from skills.system_skill import SystemSkill
    from skills.memory_skill import MemorySkill
    from skills.scheduler_skill import SchedulerSkill

    registry.register(CalculatorSkill())
    registry.register(PaintSkill())
    registry.register(VSCodeSkill())
    registry.register(BrowserSkill())
    registry.register(NotepadSkill())
    registry.register(FileSkill())
    registry.register(SystemSkill())
    registry.register(MemorySkill())
    registry.register(SchedulerSkill())
