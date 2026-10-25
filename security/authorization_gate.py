"""
Centralized Security Authorization Gate
Single gatekeeper for all system actions, tools, FastRouter, and Phase 3 Agent Loop.

Enforces:
1. UNKNOWN / GUEST speaker has ZERO computer-control privileges (fail-closed).
2. Words ('I am Sanjay', 'I am your owner') are NEVER authentication.
3. High-risk actions require explicit owner confirmation.
4. Fail-closed: any doubt, error, or unverified state -> DENIED.
5. Invalidate trusted session if speaker changes or verification fails.
"""

import time
from typing import Dict, Any, Optional, Tuple, Set
from config.constants import SpeakerClassification, RiskLevel


# Tools and action categories that require OWNER privileges
PRIVILEGED_COMPUTER_ACTIONS: Set[str] = {
    # Application & Window control
    "open_app", "close_app", "launch_app", "open_application", "window_control", "focus_window",
    "minimize_window", "maximize_window", "close_window",
    # Browser & Web automation
    "open_website", "open_url", "search_youtube", "play_youtube",
    "play_music", "play_latest_music", "search_web", "browser_control",
    # File & Folder operations
    "create_file", "write_file", "read_file", "delete_file", "rename_file",
    "create_folder", "delete_folder", "move_file", "copy_file",
    # System execution & Scripting
    "run_python", "execute_python", "cmd", "powershell", "terminal",
    "shell", "run_command", "install_software", "uninstall_software",
    # Hardware & Desktop interaction
    "mouse_click", "mouse_move", "key_press", "type_text", "volume",
    "media", "lock", "lock_screen", "system_settings",
    # Multi-step & Agent
    "multi_step", "notepad_type_save", "save_file", "screen_vision",
    "inspect_screen", "coding_agent", "agent_loop", "plan_with_tools"
}

# High-risk actions that require explicit confirmation from OWNER
HIGH_RISK_ACTIONS: Set[str] = {
    "delete_file", "delete_folder", "remove_file", "remove_folder", "format",
    "wipe", "destroy", "shutdown", "restart", "reboot", "uninstall",
    "kill_process", "modify_system", "clear_directory", "send_credentials"
}

HIGH_RISK_KEYWORDS: Set[str] = {
    "delete", "remove", "destroy", "format", "wipe", "shutdown", "reboot",
    "restart", "uninstall", "kill"
}


class SpeakerSessionManager:
    """
    Maintains temporary authenticated sessions for verified OWNER.
    Invalidates immediately on any UNKNOWN speech, verification failure, or timeout.
    """

    def __init__(self, session_ttl_seconds: float = 30.0):
        self.session_ttl = session_ttl_seconds
        self._authenticated_speaker: Optional[SpeakerClassification] = None
        self._last_auth_time: float = 0.0

    def authenticate_owner(self):
        """Grants temporary authenticated session to OWNER."""
        self._authenticated_speaker = SpeakerClassification.OWNER
        self._last_auth_time = time.time()
        print("[SECURITY] Owner session authenticated (TTL: 30s)")

    def invalidate(self, reason: str = "Voice verification failed or speaker changed"):
        """Immediately revokes any active session."""
        if self._authenticated_speaker is not None:
            print(f"[SECURITY] Session invalidated: {reason}")
        self._authenticated_speaker = None
        self._last_auth_time = 0.0

    def is_owner_active(self) -> bool:
        """Checks if there is an unexpired OWNER session."""
        if self._authenticated_speaker == SpeakerClassification.OWNER:
            if time.time() - self._last_auth_time <= self.session_ttl:
                return True
            else:
                self.invalidate("Session timed out")
        return False


# Global session manager instance
_global_session = SpeakerSessionManager()


def get_speaker_session() -> SpeakerSessionManager:
    return _global_session


def is_privileged_action(action_name: str, intent: Optional[str] = None) -> bool:
    """Checks if an action or intent represents a computer-control operation."""
    act_clean = (action_name or "").lower().strip()
    int_clean = (intent or "").upper().strip()

    if act_clean in PRIVILEGED_COMPUTER_ACTIONS:
        return True
    if any(p in act_clean for p in ["open", "create", "delete", "write", "run", "click", "type", "shell", "browser"]):
        return True

    privileged_intents = {
        "OPEN_APP", "OPEN_WEBSITE", "SEARCH_YOUTUBE", "PLAY_YOUTUBE",
        "PLAY_LATEST_MUSIC", "SEARCH_WEB", "MULTI_STEP", "SCREEN_VISION",
        "MEDIA_CONTROL", "LOCK_SCREEN"
    }
    if int_clean in privileged_intents:
        return True

    return False


def is_high_risk(action_name: str, parameters: Optional[Dict[str, Any]] = None, risk_level: Optional[RiskLevel] = None) -> bool:
    """Detects whether an action represents a destructive or high-risk computer operation."""
    if risk_level is not None and risk_level >= RiskLevel.LEVEL_3_SENSITIVE:
        return True

    act_clean = (action_name or "").lower().strip()
    if act_clean in HIGH_RISK_ACTIONS:
        return True

    if any(k in act_clean for k in HIGH_RISK_KEYWORDS):
        return True

    if parameters:
        param_str = str(parameters).lower()
        if any(k in param_str for k in HIGH_RISK_KEYWORDS):
            return True

    return False


def authorize_action(
    speaker: SpeakerClassification,
    action_name: str,
    intent: Optional[str] = None,
    risk_level: Optional[RiskLevel] = None,
    parameters: Optional[Dict[str, Any]] = None,
    raw_query: Optional[str] = None,
) -> Tuple[bool, str, bool]:
    """
    Central authorization gate for all system actions.

    Returns:
        (is_allowed: bool, message: str, requires_confirmation: bool)
    """
    act_clean = (action_name or "").lower().strip()
    session = get_speaker_session()

    # Rule 0: Fail-closed verification
    # If speaker is verified OWNER, refresh session
    if speaker == SpeakerClassification.OWNER:
        session.authenticate_owner()
    else:
        # Any UNKNOWN or UNCERTAIN classification strictly invalidates any prior session
        session.invalidate("Speaker classification is not OWNER")

    # Check effective authority: current speaker must be OWNER (or valid unexpired owner session)
    has_owner_authority = (speaker == SpeakerClassification.OWNER) or session.is_owner_active()

    requires_privilege = False
    # Check if raw_query or parameters contain destructive or command keywords
    if raw_query:
        q_lower = raw_query.lower()
        if any(w in q_lower for w in ["delete", "remove", "wipe", "format", "destroy", "powershell", "terminal", "shell", "cmd", "run a command"]):
            requires_privilege = True

    # 1. Non-privileged informational queries (allowed for anyone)
    informational_actions = {
        "chat", "conversation", "informational", "hear_check", "greeting",
        "name_query", "owner_query", "general_ai", "battery", "datetime", "diagnostics"
    }
    if not requires_privilege and act_clean in informational_actions:
        return True, "Informational query authorized", False

    # 2. UNKNOWN / UNCERTAIN speaker -> ZERO computer-control privileges
    if not has_owner_authority:
        print(f"[SECURITY] Action '{act_clean}' BLOCKED for speaker={speaker.value}")
        if speaker == SpeakerClassification.UNCERTAIN:
            return False, "Speaker identity uncertain. Voice verification required before executing.", False
        return False, "Access denied. I don't recognize this voice. Computer controls are restricted and locked.", False

    # 3. OWNER speaker: Check for HIGH RISK confirmation requirement
    if is_high_risk(act_clean, parameters, risk_level):
        print(f"[SECURITY] High-risk action '{act_clean}' requires explicit owner confirmation")
        return True, "Sir, this action can modify or delete data. Shall I continue?", True

    # 4. Standard OWNER privileged action
    return True, "Authorized", False
