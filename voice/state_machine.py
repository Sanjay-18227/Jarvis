"""
Voice State Machine
Explicit thread-safe state machine for the JARVIS voice interaction lifecycle.
States: IDLE, LISTENING, PROCESSING, EXECUTING, SPEAKING, INTERRUPTED.
"""

import threading
from enum import Enum
from typing import Optional, Callable, List


class VoiceState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class VoiceStateMachine:
    """
    Thread-safe state manager for voice conversation turns.
    Guarantees thread-safe transitions and standardized development logging.
    """

    def __init__(self, initial_state: VoiceState = VoiceState.IDLE):
        self._lock = threading.RLock()
        self._state = initial_state
        self._listeners: List[Callable[[VoiceState, VoiceState], None]] = []

    @property
    def state(self) -> VoiceState:
        with self._lock:
            return self._state

    def get_state(self) -> VoiceState:
        with self._lock:
            return self._state

    def is_state(self, state: VoiceState) -> bool:
        with self._lock:
            return self._state == state

    def set_state(self, new_state: VoiceState) -> None:
        """
        Transitions to a new state and prints standardized debug log.
        """
        old_state = None
        listeners_to_call = []
        with self._lock:
            if self._state == new_state:
                return
            old_state = self._state
            self._state = new_state
            listeners_to_call = list(self._listeners)
            print(f"[VOICE] State={new_state.value}")

        for callback in listeners_to_call:
            try:
                callback(old_state, new_state)
            except Exception as e:
                print(f"[VOICE] State listener error: {e}")

    def add_state_listener(self, callback: Callable[[VoiceState, VoiceState], None]):
        """Registers a callback for state change events."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_state_listener(self, callback: Callable[[VoiceState, VoiceState], None]):
        """Unregisters a state listener callback."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)
