"""
JARVIS Tool Result Contract
Enforces strict execution outcome reporting.
Every executable tool and router action must return a structured ToolResult.
JARVIS must NEVER report success unless success == True with verified evidence.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ToolResult:
    """
    Standard result contract for all JARVIS tool executions.
    """
    success: bool
    action: str
    message: str = ""
    evidence: str = ""
    error: Optional[str] = None
    target: Optional[str] = None

    def __iter__(self):
        """Allows unpacking as (success, message) for backward compatibility."""
        yield self.success
        yield self.message

    def __bool__(self) -> bool:
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action": self.action,
            "message": self.message,
            "evidence": self.evidence,
            "error": self.error,
            "target": self.target,
        }

    @classmethod
    def ok(cls, action: str, message: str = "", evidence: str = "", target: Optional[str] = None, **kwargs) -> "ToolResult":
        """Creates a verified success result."""
        return cls(
            success=True,
            action=action,
            message=message,
            evidence=evidence,
            error=None,
            target=target or kwargs.get("target"),
        )

    @classmethod
    def fail(cls, action: str, message: str = "", error: str = "", evidence: str = "", target: Optional[str] = None, **kwargs) -> "ToolResult":
        """Creates a verified failure result."""
        err = error or message or "Operation failed"
        msg = message or err
        return cls(
            success=False,
            action=action,
            message=msg,
            evidence=evidence,
            error=err,
            target=target or kwargs.get("target"),
        )
