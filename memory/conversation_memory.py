"""
Conversation Memory Subsystem
Maintains multi-turn context, role history, and sliding window memory management.
"""

import time
from typing import List, Dict, Optional


class ConversationMemory:
    """
    Manages in-memory dialogue turns with bounded sliding window history.
    """

    def __init__(self, max_turns: int = 15):
        self.max_turns = max_turns
        self._history: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        """Append user input to conversation memory."""
        if content and content.strip():
            self._history.append({"role": "user", "content": content.strip(), "time": str(time.time())})
            self._trim()

    def add_assistant_message(self, content: str):
        """Append assistant response to conversation memory."""
        if content and content.strip():
            self._history.append({"role": "assistant", "content": content.strip(), "time": str(time.time())})
            self._trim()

    def get_history(self) -> List[Dict[str, str]]:
        """Return history turns formatted for LLM context."""
        return [{"role": item["role"], "content": item["content"]} for item in self._history]

    def clear(self):
        """Clear conversation history."""
        self._history.clear()

    def _trim(self):
        """Ensure memory does not exceed max_turns."""
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]
