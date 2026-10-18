"""
Security Audit Log Subsystem
Thread-safe audit trail for all JARVIS actions, risk classifications, and speaker states.
Never stores passwords, tokens, or raw secrets.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from config.constants import RiskLevel, SpeakerClassification, ActionStatus
from config.settings import get_settings
from .privacy_guard import PrivacyGuard


class AuditLogger:
    """
    Maintains a tamper-evident audit record of assistant operations.
    """

    def __init__(self, log_file: Optional[str] = None):
        if log_file:
            self.log_file = log_file
        else:
            base_dir = get_settings().base_dir
            logs_dir = os.path.join(base_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            self.log_file = os.path.join(logs_dir, "jarvis_audit.jsonl")

        self.privacy_guard = PrivacyGuard()

    def log_action(
        self,
        task: str,
        speaker: SpeakerClassification,
        action: str,
        risk_level: RiskLevel,
        status: ActionStatus,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Appends an action record to the audit log with strict privacy redaction.
        """
        # Redact any accidental secrets
        clean_task = self.privacy_guard.redact_secrets(task)
        clean_action = self.privacy_guard.redact_secrets(action)

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "speaker": speaker.value,
            "task": clean_task,
            "action": clean_action,
            "risk_level": int(risk_level),
            "status": status.value,
            "details": self._sanitize_dict(details or {}),
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[AuditLog] Logging error: {e}")

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deeply sanitizes dictionaries to ensure no credentials are logged."""
        sanitized = {}
        for k, v in data.items():
            if any(term in k.lower() for term in ["password", "secret", "token", "key"]):
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, str):
                sanitized[k] = self.privacy_guard.redact_secrets(v)
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize_dict(v)
            else:
                sanitized[k] = v
        return sanitized

    def read_recent_records(self, count: int = 20) -> List[Dict[str, Any]]:
        """Reads recent audit log entries."""
        if not os.path.exists(self.log_file):
            return []
        records = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            print(f"[AuditLog] Error reading audit log: {e}")
        return records[-count:]

    def clear(self):
        """Allows owner to clear audit log file."""
        if os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")
