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
        Determines if a speaker is authorized to execute an action via the Central Security Gate.

        Returns:
            (is_allowed, explanation_message)
        """
        from security.authorization_gate import authorize_action
        allowed, msg, _ = authorize_action(
            speaker=speaker,
            action_name=action_name,
            risk_level=risk_level,
        )
        return allowed, msg

