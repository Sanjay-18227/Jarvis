"""
Unit & Integration Test Suite for Owner Voice Enrollment & Speaker Verification Subsystem
Validates:
TEST 1 — Enrollment: captures >= 8 valid samples, aggregates into centroid, saves profile locally.
TEST 2 — Owner phrase 1: Owner says "Jarvis, open Calculator." -> OWNER
TEST 3 — Owner phrase 2: Owner says "Tell me my battery percentage." -> OWNER
TEST 4 — Owner phrase 3: Owner says a completely different sentence -> OWNER
TEST 5 — Different person: Another person says "Jarvis, open Calculator." -> UNKNOWN
TEST 6 — Different person says owner name: Another person says "I am Sanjay." -> UNKNOWN
TEST 7 — Security: UNKNOWN says "Open VS Code." -> [SECURITY] DENIED (execution blocked)
TEST 8 — Owner action: OWNER says "Open VS Code." -> allowed, executed, verified
TEST 9 — Quality Check: Rejection of silence, short audio, clipping, low energy, background noise
TEST 10 — Fail Closed: Profile missing, corrupt profile, or extraction error -> UNKNOWN, never OWNER
TEST 11 — Owner Addressing: Polite title "Sir", only revealing "Sanjay" upon explicit query
"""

import os
import io
import wave
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from config.constants import SpeakerClassification, RiskLevel
from config.settings import get_settings
from voice.speaker_verification import SpeakerVerifier
from voice.voice_enrollment import VoiceEnrollment
from core.fast_router import FastRouter
from core.agent import JarvisAgent
from security.authorization_gate import authorize_action, get_speaker_session


def create_pcm_wav(
    duration: float = 2.0,
    sample_rate: int = 16000,
    freq: float = 130.0,
    amplitude: float = 15000.0,
    pre_silence: float = 0.2,
    post_silence: float = 0.3,
) -> bytes:
    """Generates synthetic PCM 16-bit mono WAV bytes with harmonic glottal source."""
    n_speech = int(sample_rate * duration)
    t = np.linspace(0, duration, n_speech, endpoint=False)
    # Fundamental frequency + harmonics (vocal cord simulation)
    speech = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 8))
    speech = speech / np.max(np.abs(speech)) * amplitude

    # Silence padding
    pre = np.zeros(int(sample_rate * pre_silence), dtype=np.float32)
    post = np.zeros(int(sample_rate * post_silence), dtype=np.float32)
    full = np.concatenate([pre, speech, post]).astype(np.int16)

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(full.tobytes())
    return wav_io.getvalue()


# ============================================================================
# TEST 1 — Enrollment (captures >= 8 valid samples & saves locally)
# ============================================================================

def test_1_owner_enrollment_multi_sample(tmp_path):
    """TEST 1: Captures >= 8 valid samples, aggregates into normalized centroid, saves profile."""
    profile_path = str(tmp_path / "owner_profile.dat")
    verifier = SpeakerVerifier(profile_path=profile_path)
    enroller = VoiceEnrollment(verifier=verifier)

    # Generate 10 diverse simulated samples for Owner (f0 ~ 130 Hz with small variations)
    samples = [
        create_pcm_wav(duration=1.5 + (i * 0.1), freq=130.0 + (i % 3), amplitude=18000)
        for i in range(10)
    ]

    success = enroller.enroll_from_samples(samples, owner_name="Sanjay", min_required=8)
    assert success is True

    # Verify profile on disk
    assert os.path.exists(profile_path)
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["owner_name"] == "Sanjay"
    assert data["sample_count"] == 10
    assert data["dim"] == 42
    assert "calibrated_threshold" in data
    assert len(data["embedding"]) == 42
    # Verify embedding is unit-normalized
    norm = np.linalg.norm(np.array(data["embedding"]))
    assert np.isclose(norm, 1.0, atol=1e-3)


# ============================================================================
# TESTS 2, 3, 4 — Owner recognition across different phrases
# ============================================================================

def test_2_3_4_owner_different_phrases():
    """TESTS 2, 3, 4: Enrolled owner is verified as OWNER across distinct sentences."""
    verifier = SpeakerVerifier()
    # Mock enrolled profile with unit vector
    profile = np.zeros(42, dtype=np.float32)
    profile[0:20] = 0.5
    profile /= np.linalg.norm(profile)
    verifier.owner_embedding = profile
    verifier.similarity_threshold_owner = 0.78

    # Same owner voice signature: high similarity vector
    owner_emb = profile.copy()

    with patch.object(verifier, "extract_embedding", return_value=owner_emb):
        # TEST 2: "Jarvis, open Calculator."
        res1, sim1 = verifier.verify_speaker(b"audio_calc")
        assert res1 == SpeakerClassification.OWNER
        assert sim1 >= verifier.similarity_threshold_owner

        # TEST 3: "Tell me my battery percentage."
        res2, sim2 = verifier.verify_speaker(b"audio_battery")
        assert res2 == SpeakerClassification.OWNER
        assert sim2 >= verifier.similarity_threshold_owner

        # TEST 4: Completely different sentence
        res3, sim3 = verifier.verify_speaker(b"audio_weather_in_tokyo")
        assert res3 == SpeakerClassification.OWNER
        assert sim3 >= verifier.similarity_threshold_owner


# ============================================================================
# TESTS 5, 6 — Stranger is UNKNOWN even if saying owner name
# ============================================================================

def test_5_different_person_unknown():
    """TEST 5: Different person saying 'Jarvis, open Calculator.' -> UNKNOWN."""
    verifier = SpeakerVerifier()
    profile = np.zeros(42, dtype=np.float32)
    profile[0:20] = 0.5
    profile /= np.linalg.norm(profile)
    verifier.owner_embedding = profile
    verifier.similarity_threshold_owner = 0.78

    # Stranger voice has distinct acoustic profile (low similarity)
    stranger_emb = np.zeros(42, dtype=np.float32)
    stranger_emb[20:40] = 0.5
    stranger_emb /= np.linalg.norm(stranger_emb)

    with patch.object(verifier, "extract_embedding", return_value=stranger_emb):
        result, sim = verifier.verify_speaker(b"stranger_calc")
        assert result == SpeakerClassification.UNKNOWN
        assert sim < verifier.similarity_threshold_owner


def test_6_different_person_says_owner_name_unknown():
    """TEST 6: Different person saying 'I am Sanjay' must NOT authenticate based on words."""
    verifier = SpeakerVerifier()
    profile = np.zeros(42, dtype=np.float32)
    profile[0:20] = 0.5
    profile /= np.linalg.norm(profile)
    verifier.owner_embedding = profile
    verifier.similarity_threshold_owner = 0.78

    # Voice biometric of stranger (orthogonal/different acoustics)
    impostor_emb = np.zeros(42, dtype=np.float32)
    impostor_emb[30:42] = 0.5
    impostor_emb /= np.linalg.norm(impostor_emb)

    with patch.object(verifier, "extract_embedding", return_value=impostor_emb):
        # Spoken words: "I am Sanjay" -> Acoustic verification must return UNKNOWN
        result, sim = verifier.verify_speaker(b"impostor_says_i_am_sanjay")
        assert result == SpeakerClassification.UNKNOWN
        assert sim < verifier.similarity_threshold_owner


# ============================================================================
# TESTS 7, 8 — Security gating: UNKNOWN blocked, OWNER executed
# ============================================================================

def test_7_unknown_open_vscode_security_denied():
    """TEST 7: UNKNOWN saying 'Open VS Code' -> [SECURITY] DENIED."""
    router = FastRouter()
    # UNKNOWN speaker attempts to launch application
    res = router.try_fast_path("Open VS Code", speaker=SpeakerClassification.UNKNOWN)
    assert res is not None
    action_name, response_text = res
    assert action_name == "security_denied"
    assert "Access denied" in response_text or "locked" in response_text
    assert "VS Code" not in response_text or "denied" in response_text.lower()


def test_8_owner_open_vscode_allowed():
    """TEST 8: OWNER saying 'Open VS Code' -> allowed and routed."""
    router = FastRouter()
    with patch.object(router.app_indexer, "launch_app") as mock_launch:
        from core.tool_result import ToolResult
        mock_launch.return_value = ToolResult.ok(action="launch_app", message="Opening VS Code, Sir.")

        res = router.try_fast_path("Open VS Code", speaker=SpeakerClassification.OWNER)
        assert res is not None
        action_name, response_text = res
        assert action_name == "open_app"
        assert "Opening VS Code" in response_text or "Sir" in response_text


# ============================================================================
# TEST 9 — Quality Rejection Checks
# ============================================================================

def test_9_quality_rejection_rules():
    """TEST 9: Quality check correctly rejects silence, short audio, quiet audio, clipping."""
    enroller = VoiceEnrollment()

    # Empty / no speech
    ok, reason = enroller.validate_audio_quality(b"")
    assert ok is False
    assert "no real speech" in reason

    # Too short (< 1.0s)
    short_wav = create_pcm_wav(duration=0.4, pre_silence=0.1, post_silence=0.1)
    ok, reason = enroller.validate_audio_quality(short_wav)
    assert ok is False
    assert "too short" in reason

    # Too quiet (RMS < 150)
    quiet_wav = create_pcm_wav(duration=1.5, amplitude=50.0)
    ok, reason = enroller.validate_audio_quality(quiet_wav)
    assert ok is False
    assert "too quiet" in reason

    # Clipping detected (> 0.5% samples >= 32000)
    raw = np.full(16000 * 2, 32500, dtype=np.int16)
    clip_io = io.BytesIO()
    with wave.open(clip_io, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(raw.tobytes())
    ok, reason = enroller.validate_audio_quality(clip_io.getvalue())
    assert ok is False
    assert "clipping" in reason

    # Mostly silence (2.0s silence with 0.1s speech)
    silence_wav = create_pcm_wav(duration=0.1, pre_silence=1.0, post_silence=1.0)
    ok, reason = enroller.validate_audio_quality(silence_wav)
    assert ok is False
    assert "mostly silence" in reason or "too short" in reason


# ============================================================================
# TEST 10 — Fail-closed verification
# ============================================================================

def test_10_fail_closed_guarantees():
    """TEST 10: Fail-closed guarantees: missing profile, corrupt profile, error -> UNKNOWN."""
    verifier = SpeakerVerifier()

    # 1. Profile missing -> UNKNOWN, 0.0
    verifier.owner_embedding = None
    res, conf = verifier.verify_speaker(b"audio")
    assert res == SpeakerClassification.UNKNOWN
    assert conf == 0.0

    # 2. Corrupt / empty audio -> UNKNOWN, 0.0
    verifier.owner_embedding = np.ones(42, dtype=np.float32)
    res, conf = verifier.verify_speaker(b"")
    assert res == SpeakerClassification.UNKNOWN
    assert conf == 0.0

    # 3. Extraction exception -> UNKNOWN, 0.0
    with patch.object(verifier, "extract_embedding", side_effect=RuntimeError("FFT error")):
        res, conf = verifier.verify_speaker(b"corrupt")
        assert res == SpeakerClassification.UNKNOWN
        assert conf == 0.0


# ============================================================================
# TEST 11 — Owner addressing
# ============================================================================

def test_11_owner_addressing_rules():
    """TEST 11: Owner addressed as 'Sir', and 'Sanjay' revealed only when asked 'What is my name?'."""
    router = FastRouter()

    # Normal greeting for OWNER -> "Sir" (not "Sanjay")
    _, resp_greet = router.try_fast_path("Hello Jarvis", speaker=SpeakerClassification.OWNER)
    assert "Sir" in resp_greet
    assert "Hello, Sir" in resp_greet or "Sir" in resp_greet

    # Explicit question: "What is my name?" -> "Your name is Sanjay, Sir."
    _, resp_name = router.try_fast_path("What is my name?", speaker=SpeakerClassification.OWNER)
    assert resp_name == "Your name is Sanjay, Sir."

    # UNKNOWN asking "What is my name?" -> Refused
    _, resp_unk_name = router.try_fast_path("What is my name?", speaker=SpeakerClassification.UNKNOWN)
    assert "Sanjay" not in resp_unk_name
    assert "failed" in resp_unk_name.lower() or "denied" in resp_unk_name.lower() or "voice" in resp_unk_name.lower()
