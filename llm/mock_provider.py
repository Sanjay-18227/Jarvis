"""
Mock / Offline AI Provider for Unit Testing and Offline Operation
"""

from typing import List, Dict, Optional, Iterator
from .base import AIProvider


class MockProvider(AIProvider):
    """
    Simulated AI Provider providing predictable responses for testing and offline fallback.
    """

    def __init__(self, name: str = "MockJARVIS"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> str:
        p = prompt.lower().strip()
        if "who are you" in p or "identity" in p:
            return "I am JARVIS, your personal Windows desktop assistant, Sir."
        elif "hello" in p or "hey jarvis" in p or "hi" in p:
            return "Hello Sanjay, how can I assist you today?"
        elif "status" in p or "check" in p:
            return "All systems are operational, Sir."
        elif "tamil" in p or "வணக்கம்" in p or "panu" in p or "pannu" in p:
            return "வணக்கம் Sanjay, நான் உங்களுக்கு உதவ தயாராக உள்ளேன்."
        elif "password" in p or "api key" in p or "secret" in p:
            return "I cannot provide or expose private credentials or keys."
        else:
            return f"Understood, Sanjay. Processing your request: {prompt}."

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> Iterator[str]:
        words = self.generate(prompt, system_prompt, history).split(" ")
        for w in words:
            yield w + " "
