"""
Permission Access Control Subsystem
Enforces least-privilege policies across speaker classification tiers.
"""

from typing import Tuple, Dict, Any
from config.constants import SpeakerClassification, PermissionLevel, RiskLevel


class PermissionManager:
    """
    Validates whether the current speaker has privileges to execute a given task or tool.
    """

    FORBIDDEN_UNKNOWN_OPERATIONS = [
        "shell", "powershell", "cmd", "terminal", "install", "uninstall",
        "password", "api_key", "secret", "private_file", "email", "upload",
        "delete", "move", "vscode", "coding_agent", "account", "settings",
        "write_file", "click_mouse", "key_press"
    ]

    def check_permission(
        self,
        speaker: SpeakerClassification,
        action_name: str,
        risk_level: RiskLevel,
    ) -> Tuple[bool, str]:
        """
        Determines if a speaker is authorized to execute an action.

        Returns:
            (is_allowed, explanation_message)
        """
        # OWNER can perform actions up to their confirmation limits
        if speaker == SpeakerClassification.OWNER:
            return True, "Owner authorized"

        # UNCERTAIN speaker must re-authenticate for anything above Informational
        if speaker == SpeakerClassification.UNCERTAIN:
            if risk_level > RiskLevel.LEVEL_0_INFORMATIONAL:
                return (
                    False,
                    "Speaker verification was uncertain. Please speak again to verify your identity as Sanjay."
                )
            return True, "Harmless query permitted for uncertain speaker"

        # UNKNOWN speaker
        if speaker == SpeakerClassification.UNKNOWN:
            act_lower = action_name.lower()
            if any(forbidden in act_lower for forbidden in self.FORBIDDEN_UNKNOWN_OPERATIONS) or risk_level > RiskLevel.LEVEL_0_INFORMATIONAL:
                return (
                    False,
                    "Access denied. Privileged computer controls and private data are restricted to the owner (Sanjay)."
                )
            return True, "Informational dialogue allowed for guest"

        return False, "Unknown permission state"
