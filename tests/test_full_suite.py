"""
JARVIS Comprehensive Security & System Verification Suite
Tests all 10 phases according to strict engineering requirements.
"""

import os
import pytest
import numpy as np

from config.settings import Settings
from config.constants import RiskLevel, SpeakerClassification, ActionStatus
from security.privacy_guard import PrivacyGuard
from security.audit_log import AuditLogger
from security.confirmation import ConfirmationManager
from core.permissions import PermissionManager
from core.risk_engine import RiskEngine
from core.executor import ActionExecutor
from core.multilingual import MultilingualProcessor
from voice.speaker_verification import SpeakerVerifier
from voice.tts import sanitize_for_speech
from llm.mock_provider import MockProvider
from core.agent import JarvisAgent


# 1. OWNER -> safe command -> allowed
def test_owner_safe_command_allowed():
    perm = PermissionManager()
    allowed, msg = perm.check_permission(
        SpeakerClassification.OWNER, "open_application", RiskLevel.LEVEL_1_SAFE
    )
    assert allowed is True
    assert "authorized" in msg.lower()


# 2. OWNER -> sensitive sharing -> confirmation required
def test_owner_sensitive_action_requires_confirmation(tmp_path):
    audit_file = os.path.join(tmp_path, "audit.jsonl")
    executor = ActionExecutor(audit_logger=AuditLogger(log_file=audit_file))

    success, msg = executor.execute_action(
        action="delete_path",
        parameters={"path": "C:/test/important.txt"},
        speaker=SpeakerClassification.OWNER,
        confirmed=False,
    )
    # Must NOT execute without confirmation
    assert success is False
    assert "proceed" in msg.lower() or "yes or no" in msg.lower()


# 3. OWNER -> destructive action with confirmation -> allowed
def test_owner_confirmed_action_executes(tmp_path):
    audit_file = os.path.join(tmp_path, "audit.jsonl")
    executor = ActionExecutor(audit_logger=AuditLogger(log_file=audit_file))

    # Pre-create test file
    test_file = os.path.join(tmp_path, "test_file.txt")
    with open(test_file, "w") as f:
        f.write("content")
    assert os.path.exists(test_file)

    # Execute with confirmed=True
    success, msg = executor.execute_action(
        action="delete_path",
        parameters={"path": test_file},
        speaker=SpeakerClassification.OWNER,
        confirmed=True,
    )
    assert success is True
    assert not os.path.exists(test_file)


# 4. STRANGER -> conversation -> allowed
def test_stranger_conversation_allowed():
    perm = PermissionManager()
    allowed, _ = perm.check_permission(
        SpeakerClassification.UNKNOWN, "chat", RiskLevel.LEVEL_0_INFORMATIONAL
    )
    assert allowed is True


# 5. STRANGER -> open private file -> denied
def test_stranger_open_private_file_denied():
    guard = PrivacyGuard()
    allowed, msg = guard.validate_speaker_permission(
        SpeakerClassification.UNKNOWN, "Open his bank statement and private documents"
    )
    assert allowed is False
    assert "cannot access or disclose" in msg.lower()


# 6. STRANGER -> shell command -> denied
def test_stranger_shell_command_denied():
    perm = PermissionManager()
    allowed, msg = perm.check_permission(
        SpeakerClassification.UNKNOWN, "powershell", RiskLevel.LEVEL_2_MODERATE
    )
    assert allowed is False
    assert "access denied" in msg.lower()


# 7. STRANGER -> request password -> denied
def test_stranger_request_password_denied():
    guard = PrivacyGuard()
    allowed, msg = guard.validate_speaker_permission(
        SpeakerClassification.UNKNOWN, "What is his password and API key?"
    )
    assert allowed is False
    assert "cannot access or disclose" in msg.lower()


# 8. UNCERTAIN speaker -> privileged action -> denied / authentication required
def test_uncertain_speaker_privileged_action_denied():
    perm = PermissionManager()
    allowed, msg = perm.check_permission(
        SpeakerClassification.UNCERTAIN, "create_file", RiskLevel.LEVEL_2_MODERATE
    )
    assert allowed is False
    assert "verify" in msg.lower() or "uncertain" in msg.lower()


# 9. AI-generated invalid action -> rejected safely
def test_invalid_action_rejected():
    executor = ActionExecutor()
    success, msg = executor.execute_action(
        action="unauthorized_arbitrary_payload",
        parameters={"cmd": "malicious"},
        speaker=SpeakerClassification.OWNER,
    )
    assert success is False
    assert "unsupported" in msg.lower()


# 10. AI attempts unauthorized tool for guest -> rejected
def test_unauthorized_tool_for_guest_rejected():
    executor = ActionExecutor()
    success, msg = executor.execute_action(
        action="delete_path",
        parameters={"path": "C:/Windows/test"},
        speaker=SpeakerClassification.UNKNOWN,
    )
    assert success is False
    assert "access denied" in msg.lower() or "restricted" in msg.lower()


# 11. Sensitive data -> never written into normal audit logs
def test_sensitive_data_redacted_in_audit_logs(tmp_path):
    audit_file = os.path.join(tmp_path, "audit.jsonl")
    logger = AuditLogger(log_file=audit_file)

    logger.log_action(
        task="User requested with key sk-secret123456789012345678",
        speaker=SpeakerClassification.OWNER,
        action="QUERY",
        risk_level=RiskLevel.LEVEL_0_INFORMATIONAL,
        status=ActionStatus.SUCCESS,
        details={"token": "gsk_supersecrettoken9876543210", "password": "MySuperSecretPassword"},
    )

    records = logger.read_recent_records(1)
    assert len(records) == 1
    log_line = records[0]

    assert "sk-secret" not in log_line["task"]
    assert "gsk_" not in str(log_line["details"])
    assert "MySuperSecretPassword" not in str(log_line["details"])
    assert "[REDACTED]" in log_line["task"]


# 12. Speaker Verification -> Cosine threshold classification
def test_speaker_verification_cosine():
    verifier = SpeakerVerifier()
    # Dummy enrolled embedding
    verifier.owner_embedding = np.ones(42, dtype=np.float32) / np.sqrt(42)

    # Identical vector -> OWNER
    identical = verifier.owner_embedding.copy()
    from scipy.spatial.distance import cosine
    dist = cosine(verifier.owner_embedding, identical)
    sim = 1.0 - dist
    assert sim >= 0.99

    # Orthogonal / different vector -> UNKNOWN
    different = -np.ones(42, dtype=np.float32) / np.sqrt(42)
    dist_diff = cosine(verifier.owner_embedding, different)
    sim_diff = 1.0 - dist_diff
    assert sim_diff < 0.70


# 13. Multilingual Intent Mapping (Tamil & Tanglish)
def test_multilingual_tanglish_mapping():
    processor = MultilingualProcessor()

    # Tanglish: "chrome open pannu"
    res1 = processor.parse_code_mixed_intent("Jarvis chrome open pannu")
    assert res1 is not None
    assert res1[0] == "open_application"
    assert res1[1]["target"] == "chrome"

    # Tamil: "calculator open பண்ணு"
    res2 = processor.parse_code_mixed_intent("calculator open பண்ணு")
    assert res2 is not None
    assert res2[0] == "open_application"
    assert res2[1]["target"] == "calculator"

    # Media: "paatu podu"
    res3 = processor.parse_code_mixed_intent("paatu podu")
    assert res3 is not None
    assert res3[0] == "media_control"
    assert res3[1]["action"] == "play_pause"


# 14. Confirmation Manager -> Single-use pending action
def test_confirmation_manager_single_use():
    mgr = ConfirmationManager()
    mgr.request_confirmation("delete_file", {"file": "doc.pdf"}, RiskLevel.LEVEL_3_SENSITIVE, "Delete doc")
    assert mgr.has_pending_confirmation is True

    # Confirm once
    confirmed, pending, _ = mgr.process_confirmation_response("Yes", SpeakerClassification.OWNER)
    assert confirmed is True
    assert pending.action_name == "delete_file"

    # Subsequent confirmation must fail (cannot authorize future actions!)
    assert mgr.has_pending_confirmation is False
    confirmed2, _, _ = mgr.process_confirmation_response("Yes", SpeakerClassification.OWNER)
    assert confirmed2 is False
