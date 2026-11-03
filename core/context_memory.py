"""
Short-Term Task & Entity Context Memory for JARVIS
Maintains conversational continuity across sequential commands:
- Tracks active application (e.g. Calculator, VS Code, Edge, Notepad)
- Tracks active folder (e.g. Kumar, Project, Desktop)
- Tracks active file (e.g. agent.py, main.py)
- Resolves pronoun and implicit references ("it", "the folder", "inside it", "the file", "the app")
"""

import os
import time
from typing import Optional, Dict, Any


class ContextMemory:
    """
    Maintains active task context across consecutive conversational turns.
    """

    def __init__(self, ttl_seconds: float = 300.0):
        self.ttl = ttl_seconds
        self.active_app: Optional[str] = None
        self.active_folder: Optional[str] = None
        self.active_file: Optional[str] = None
        self.active_browser: Optional[str] = None
        self.active_website: Optional[str] = None
        self.last_tool: Optional[str] = None
        self.last_result: Optional[Any] = None
        self.last_goal: Optional[str] = None
        self.last_calculation: Optional[str] = None
        self.last_url: Optional[str] = None
        self.last_observed_error: Optional[str] = None
        self.last_observed_error_app: Optional[str] = None
        self.last_observed_screen_summary: Optional[str] = None
        self.last_observed_time: float = 0.0
        self.last_update_time: float = time.time()

    def update(
        self,
        active_app: Optional[str] = None,
        active_folder: Optional[str] = None,
        active_file: Optional[str] = None,
        active_browser: Optional[str] = None,
        active_website: Optional[str] = None,
        last_tool: Optional[str] = None,
        last_result: Optional[Any] = None,
        last_goal: Optional[str] = None,
        last_calculation: Optional[str] = None,
        last_url: Optional[str] = None,
        last_observed_error: Optional[str] = None,
        last_observed_error_app: Optional[str] = None,
        last_observed_screen_summary: Optional[str] = None,
    ):
        """Updates active context variables and refreshes timestamp."""
        now = time.time()
        self.last_update_time = now

        if active_app:
            self.active_app = active_app
            print(f"[CONTEXT] active_app={active_app}")

        if active_folder:
            self.active_folder = active_folder
            print(f"[CONTEXT] active_folder={active_folder}")

        if active_file:
            self.active_file = active_file
            print(f"[CONTEXT] active_file={active_file}")

        if active_browser:
            self.active_browser = active_browser
            print(f"[CONTEXT] active_browser={active_browser}")

        if active_website:
            self.active_website = active_website
            print(f"[CONTEXT] active_website={active_website}")

        if last_tool:
            self.last_tool = last_tool

        if last_result is not None:
            self.last_result = last_result

        if last_goal:
            self.last_goal = last_goal

        if last_calculation:
            self.last_calculation = last_calculation

        if last_url:
            self.last_url = last_url

        if last_observed_error:
            self.last_observed_error = last_observed_error
            self.last_observed_time = now
            print(f"[CONTEXT] last_observed_error='{last_observed_error}'")

        if last_observed_error_app:
            self.last_observed_error_app = last_observed_error_app

        if last_observed_screen_summary:
            self.last_observed_screen_summary = last_observed_screen_summary

    def get_last_observed_error(self) -> Optional[str]:
        """Returns recent error if within visual context TTL (120s)."""
        if self.last_observed_error and (time.time() - self.last_observed_time) <= 120.0:
            return self.last_observed_error
        return None

    def get_last_observed_error_app(self) -> Optional[str]:
        if self.last_observed_error_app and (time.time() - self.last_observed_time) <= 120.0:
            return self.last_observed_error_app
        return None

    def clear_observed_error(self):
        self.last_observed_error = None
        self.last_observed_error_app = None
        self.last_observed_screen_summary = None
        self.last_observed_time = 0.0

    def is_valid(self) -> bool:
        """Checks if current context has not expired."""
        return (time.time() - self.last_update_time) <= self.ttl

    def resolve_reference(self, text: str) -> str:
        """
        Resolves pronouns like 'it', 'inside it', 'the folder', 'the file'.
        """
        if not isinstance(text, str):
            return text
        t_low = text.lower().strip()

        if t_low in ["it", "inside it", "the folder", "that folder", "the project", "inside the folder", "in it"]:
            if self.active_folder and self.is_valid():
                return self.active_folder

        if t_low in ["the file", "that file", "the script", "the program", "it"]:
            if self.active_file and self.is_valid():
                return self.active_file

        return text

    def get_active_app(self) -> Optional[str]:
        if self.is_valid():
            return self.active_app
        return None

    def get_active_folder(self) -> Optional[str]:
        if self.is_valid():
            return self.active_folder
        return None

    def get_active_file(self) -> Optional[str]:
        if self.is_valid():
            return self.active_file
        return None

    def get_active_browser(self) -> Optional[str]:
        if self.is_valid():
            return self.active_browser
        return None

    def get_active_website(self) -> Optional[str]:
        if self.is_valid():
            return self.active_website
        return None

    def get_last_tool(self) -> Optional[str]:
        if self.is_valid():
            return self.last_tool
        return None

    def get_last_result(self) -> Optional[Any]:
        if self.is_valid():
            return self.last_result
        return None

    def clear(self):
        """Clears all active context variables."""
        self.active_app = None
        self.active_folder = None
        self.active_file = None
        self.active_browser = None
        self.active_website = None
        self.last_tool = None
        self.last_result = None
        self.last_goal = None
        self.last_calculation = None
        self.last_url = None
        self.last_update_time = 0.0


# Global shared context memory
_shared_context = ContextMemory()


def get_context_memory() -> ContextMemory:
    return _shared_context
