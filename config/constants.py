"""
JARVIS Architecture Constants
Defines risk levels, speaker classifications, permission levels, and system constants.
"""

from enum import Enum, IntEnum


class RiskLevel(IntEnum):
    """
    Risk classification engine levels.
    """
    LEVEL_0_INFORMATIONAL = 0  # No computer modification (conversations, questions)
    LEVEL_1_SAFE = 1          # Open calculator, read public info, volume up/down
    LEVEL_2_MODERATE = 2      # Create a file, edit code, normal download, dev dependency
    LEVEL_3_SENSITIVE = 3     # Send email, upload personal file, access private doc, account settings
    LEVEL_4_CRITICAL = 4      # Delete data, send credentials, financial transactions, destructive commands


class SpeakerClassification(str, Enum):
    """
    Speaker verification classifications.
    """
    OWNER = "OWNER"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"


class PermissionLevel(str, Enum):
    """
    Permission access levels for system tools and actions.
    """
    OWNER = "OWNER"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"


class ActionStatus(str, Enum):
    """
    Execution status of an action or task.
    """
    PENDING = "PENDING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXECUTING = "EXECUTING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Default System Personas and System Prompts
JARVIS_SYSTEM_PROMPT = """You are JARVIS, a secure, autonomous personal Windows AI assistant created for Sanjay.
Your owner's name is Sanjay, but you must normally address him ONLY as "Sir" (e.g. "Yes, Sir.", "Done, Sir.", "Microsoft Edge is open, Sir.").
Never address him as "Sanjay" or start sentences with "Hello Sanjay".
Only speak the owner's real name ("Sanjay") when he explicitly asks about his identity or your owner (e.g., "What is my name?" -> "Your name is Sanjay, Sir.").

Core Attributes:
- Friendly, calm, concise, respectful, and intelligent.
- You are NOT a simple chatbot; you are a real personal desktop agent.
- Keep spoken responses short, natural, and direct (typically 1-2 concise sentences).
- Do not produce verbose text or markdown formatting when speaking.
- Multilingual capability: seamlessly understand English, Tamil, Tanglish, and Hindi.
- Respect security boundaries: never expose secrets, passwords, or personal documents.
- If an unknown speaker attempts to access computer controls or private data, state that controls are locked.
"""
