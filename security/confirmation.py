"""
Confirmation Subsystem
Enforces explicit spoken or typed confirmation for high-risk and sensitive computer operations.
Guarantees that a confirmation applies strictly once to the exact pending action.
"""

import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from config.constants import RiskLevel, SpeakerClassification


@dataclass
class PendingAction:
    action_name: str
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    description: str
    created_at: float
    timeout_seconds: float = 45.0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.timeout_seconds


class ConfirmationManager:
    """
    Manages pending high-risk action confirmation state.
    """

    AFFIRMATIVE_RESPONSES = [
        "yes", "confirm", "proceed", "sure", "do it", "approved", "ok", "go ahead",
        "ஆமாம்", "சரி", "செய்"
    ]

    NEGATIVE_RESPONSES = [
        "no", "cancel", "stop", "abort", "don't", "dont", "வேண்டாம்", "நிறுத்து"
    ]

    def __init__(self):
        self._pending_action: Optional[PendingAction] = None

    @property
    def has_pending_confirmation(self) -> bool:
        if self._pending_action is None:
            return False
        if self._pending_action.is_expired:
            self._pending_action = None
            return False
        return True

    def request_confirmation(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        risk_level: RiskLevel,
        description: str,
    ) -> str:
        """
        Stores pending action and returns verbal prompt requesting confirmation.
        """
        self._pending_action = PendingAction(
            action_name=action_name,
            parameters=parameters,
            risk_level=risk_level,
            description=description,
            created_at=time.time(),
        )
        return f"{description} Do you want me to proceed, Sir? Please say Yes or No."

    def process_confirmation_response(
        self,
        user_response: str,
        speaker: SpeakerClassification,
    ) -> Tuple[bool, Optional[PendingAction], str]:
        """
        Evaluates user response against pending confirmation.

        Returns:
            (is_confirmed, pending_action_or_none, message)
        """
        if not self.has_pending_confirmation:
            return False, None, "No pending action awaiting confirmation."

        if speaker != SpeakerClassification.OWNER:
            self.cancel_pending()
            return False, None, "Only the owner (Sanjay) is authorized to approve sensitive actions."

        clean_resp = user_response.lower().strip()
        pending = self._pending_action

        # Affirmative check
        if any(aff in clean_resp for aff in self.AFFIRMATIVE_RESPONSES):
            # IMMEDIATELY CONSUME AND CLEAR PENDING ACTION
            self._pending_action = None
            return True, pending, "Action confirmed. Executing now."

        # Negative check
        if any(neg in clean_resp for neg in self.NEGATIVE_RESPONSES):
            self.cancel_pending()
            return False, None, "Action cancelled, Sir."

        # Ambiguous response
        return False, None, "Confirmation was unclear. Please say Yes to confirm or No to cancel."

    def cancel_pending(self):
        """Discards pending action."""
        self._pending_action = None
