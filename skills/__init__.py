"""
JARVIS Skills Subsystem
"""

from skills.base import BaseSkill
from skills.registry import SkillRegistry, get_skill_registry
from skills.calculator_skill import CalculatorSkill
from skills.paint_skill import PaintSkill
from skills.vscode_skill import VSCodeSkill
from skills.browser_skill import BrowserSkill
from skills.notepad_skill import NotepadSkill
from skills.file_skill import FileSkill
from skills.system_skill import SystemSkill
from skills.memory_skill import MemorySkill
from skills.scheduler_skill import SchedulerSkill

__all__ = [
    "BaseSkill",
    "SkillRegistry",
    "get_skill_registry",
    "CalculatorSkill",
    "PaintSkill",
    "VSCodeSkill",
    "BrowserSkill",
    "NotepadSkill",
    "FileSkill",
    "SystemSkill",
    "MemorySkill",
    "SchedulerSkill",
]
