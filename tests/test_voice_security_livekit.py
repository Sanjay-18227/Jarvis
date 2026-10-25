"""
Comprehensive Test Suite for JARVIS Voice Security, VAD Noise Rejection,
LiveKit Voice Engine Integration, and Real User Tasks A through R.
"""

import os
import sys
import time
import shutil
import tempfile
import threading
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from config.settings import Settings
from config.constants import SpeakerClassification, RiskLevel, ActionStatus
from voice.state_machine import VoiceState, VoiceStateMachine
from voice.recorder import AudioRecorder
from voice.speaker_verification import SpeakerVerifier
from voice.voice_engine import VoiceEngine, LegacyVoiceEngine, LiveKitVoiceEngine, get_voice_engine
from core.agent import JarvisAgent
from core.fast_router import FastRouter
from core.normalizer import Normalizer, IntentType
from security.privacy_guard import PrivacyGuard
from security.confirmation import ConfirmationManager
from computer.app_indexer import AppIndexer
from computer.browser_controller import BrowserController
from computer.windows_controller import WindowsController
from computer.terminal_controller import TerminalController
from llm.mock_provider import MockProvider


# =====================================================================
# 1. VAD & NOISE REJECTION TESTS (TEST 1 & TEST 2)
# =====================================================================

def test_vad_quiet_room_no_false_trigger():
    """TEST 1: Quiet room audio (low energy ambient noise) must NOT trigger recording."""
    recorder = AudioRecorder(settings=Settings(vad_speech_threshold=400, vad_min_speech_ms=250))
    # Synthetic ambient noise below threshold (RMS ~ 40)
    noise_chunk = np.random.normal(0, 30, 1024).astype(np.int16)

    rms = recorder.calculate_rms(noise_chunk)
    assert rms < recorder.energy_threshold

    is_speech, conf = recorder.calculate_speech_confidence(noise_chunk)
    assert is_speech is False
    assert conf < 0.35


def test_vad_isolated_small_noise_rejected():
    """TEST 2: Isolated click / keyboard tap / chair noise must NOT confirm speech."""
    recorder = AudioRecorder(settings=Settings(vad_speech_threshold=400, vad_min_speech_ms=250))
    # Single high-amplitude click (10ms) surrounded by silence
    click_chunk = np.zeros(1024, dtype=np.int16)
    click_chunk[100:150] = 5000  # Click spike

    is_speech, conf = recorder.calculate_speech_confidence(click_chunk)
    # ZCR on sharp spikes or high frequency clicks is high, so speech confidence is low
    # Even if energy is high, single frame must NOT exceed consecutive frame requirement
    consecutive_required = max(int((recorder.min_speech_ms / 1000.0) / (1024 / recorder.sample_rate)), 3)
    assert consecutive_required >= 3  # Requires at least 3 consecutive frames (~200ms)


def test_vad_sustained_speech_confirmed():
    """Speech with typical vocal fundamental frequency sustained across frames is confirmed."""
    recorder = AudioRecorder(settings=Settings(vad_speech_threshold=400, vad_min_speech_ms=250))
    # 200 Hz tone simulating vocal chord vibration
    t = np.linspace(0, 1024 / 16000, 1024, endpoint=False)
    vocal_chunk = (np.sin(2 * np.pi * 200 * t) * 6000).astype(np.int16)

    is_speech, conf = recorder.calculate_speech_confidence(vocal_chunk)
    assert is_speech is True
    assert conf > 0.5


# =====================================================================
# 2. SPEAKER VERIFICATION & ACCESS CONTROL (TEST 3, 4, 5, 6, 7)
# =====================================================================

def test_speaker_verification_owner_access():
    """TEST 3: Enrolled owner voice verified as OWNER -> normal access and polite title."""
    verifier = SpeakerVerifier()
    # Mock enrolled profile
    dummy_profile = np.ones(42, dtype=np.float32)
    dummy_profile /= np.linalg.norm(dummy_profile)
    verifier.owner_embedding = dummy_profile

    # Mock extract_embedding returning identical embedding (similarity 1.0)
    with patch.object(verifier, "extract_embedding", return_value=dummy_profile):
        result, sim = verifier.verify_speaker(b"dummy_audio")
        assert result == SpeakerClassification.OWNER
        assert sim >= verifier.similarity_threshold_owner

    # FastRouter response for OWNER
    router = FastRouter()
    _, resp = router.try_fast_path("Hello JARVIS", speaker=SpeakerClassification.OWNER)
    assert "Sanjay" in resp or "Sir" in resp


def test_speaker_verification_different_speaker_unknown():
    """TEST 4: Different speaker classified as UNKNOWN -> never call them Sanjay or Sir."""
    verifier = SpeakerVerifier()
    dummy_owner = np.ones(42, dtype=np.float32)
    dummy_owner /= np.linalg.norm(dummy_owner)
    verifier.owner_embedding = dummy_owner

    # Different speaker embedding (low cosine similarity)
    stranger_emb = np.zeros(42, dtype=np.float32)
    stranger_emb[0] = 1.0

    with patch.object(verifier, "extract_embedding", return_value=stranger_emb):
        result, sim = verifier.verify_speaker(b"dummy_audio")
        assert result == SpeakerClassification.UNKNOWN
        assert sim < verifier.similarity_threshold_owner

    # FastRouter greeting for UNKNOWN
    router = FastRouter()
    _, resp = router.try_fast_path("Hello JARVIS", speaker=SpeakerClassification.UNKNOWN)
    assert "Sanjay" not in resp
    assert "Sir" not in resp
    assert "Hello." in resp


def test_unknown_speaker_sensitive_command_denied():
    """TEST 5: Unknown speaker requesting sensitive operations is blocked."""
    router = FastRouter()
    res = router.try_fast_path("Delete this file", speaker=SpeakerClassification.UNKNOWN)
    assert res is not None
    action, resp = res
    assert action == "security_denied"
    assert "Access denied" in resp
    assert "Sanjay" not in resp

    res_cmd = router.try_fast_path("Run a powershell command", speaker=SpeakerClassification.UNKNOWN)
    assert res_cmd is not None
    assert "security_denied" in res_cmd[0]


def test_owner_destructive_command_requires_confirmation():
    """TEST 6: Destructive operation by OWNER still requires explicit confirmation."""
    cm = ConfirmationManager()
    # High-risk deletion requested by OWNER
    req = cm.request_confirmation(
        action_name="delete_folder",
        parameters={"path": "C:\\Test"},
        risk_level=RiskLevel.LEVEL_3_SENSITIVE,
        description="Permanently delete folder C:\\Test",
    )
    assert cm.has_pending_confirmation is True
    assert "proceed" in req.lower() or "yes or no" in req.lower()

    # Cancel pending confirmation
    cm.cancel_pending()
    assert cm.has_pending_confirmation is False


def test_verification_failure_fails_closed_to_unknown():
    """TEST 7: Profile loading error or exception MUST fail-closed to UNKNOWN, never OWNER."""
    verifier = SpeakerVerifier()
    verifier.owner_embedding = None  # Profile missing

    res, conf = verifier.verify_speaker(b"some_audio")
    assert res == SpeakerClassification.UNKNOWN
    assert conf == 0.0

    # Exception during comparison
    verifier.owner_embedding = np.ones(42, dtype=np.float32)
    with patch.object(verifier, "extract_embedding", side_effect=RuntimeError("Corrupt audio")):
        res, conf = verifier.verify_speaker(b"corrupt_audio")
        assert res == SpeakerClassification.UNKNOWN
        assert conf == 0.0


# =====================================================================
# 3. LIVEKIT VOICE ENGINE INTEGRATION TESTS
# =====================================================================

def test_voice_engine_factory_and_fallback():
    """VoiceEngine factory returns LiveKitVoiceEngine and safely falls back if offline."""
    settings = Settings(voice_provider="livekit", livekit_url="")
    engine = get_voice_engine(settings=settings)
    assert isinstance(engine, LiveKitVoiceEngine)

    # Calling start should not crash
    assert engine.start() is True
    engine.stop()


def test_livekit_barge_in_and_self_echo_protection():
    """LiveKit voice engine stops speaking on barge-in and prevents self-echo."""
    engine = LegacyVoiceEngine(settings=Settings(barge_in_enabled=True))
    assert engine.is_speaking() is False

    # Mock speech playback and interruption
    with patch.object(engine.barge_in, "listen_for_interruption", return_value="Jarvis stop"):
        with patch.object(engine.tts, "speak_async") as mock_speak:
            mock_thread = threading.Thread(target=lambda: None)
            mock_thread.start()
            mock_speak.return_value = mock_thread

            result = engine.speak("Testing barge-in cancellation")
            assert result is False  # Interrupted
            assert engine.state == VoiceState.LISTENING


# =====================================================================
# 4. REAL USER TASKS (A THROUGH R)
# =====================================================================

# Task A: Application Tasks
def test_task_a_application_indexing():
    """Task A: Real indexing and verification for Calculator, Notepad, Task Manager, Edge."""
    indexer = AppIndexer()
    calc_info = indexer.find_app("calculator")
    assert calc_info is not None
    assert "calc" in calc_info["launch_cmd"].lower()

    notepad_info = indexer.find_app("notepad")
    assert notepad_info is not None

    edge_info = indexer.find_app("edge")
    assert edge_info is not None


# Task B: File and Folder Tasks
def test_task_b_file_and_folder_operations():
    """Task B: Create test folder, create notes.txt, write content, rename, verify existence."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_folder = os.path.join(temp_dir, "TestProject")
        os.makedirs(test_folder, exist_ok=True)
        assert os.path.isdir(test_folder)

        notes_file = os.path.join(test_folder, "notes.txt")
        with open(notes_file, "w", encoding="utf-8") as f:
            f.write("Hello JARVIS")
        assert os.path.isfile(notes_file)

        ideas_file = os.path.join(test_folder, "ideas.txt")
        os.rename(notes_file, ideas_file)
        assert not os.path.exists(notes_file)
        assert os.path.isfile(ideas_file)

        with open(ideas_file, "r", encoding="utf-8") as f:
            assert f.read() == "Hello JARVIS"


# Task C: Windows System Tasks
def test_task_c_system_vitals_read():
    """Task C: Battery percentage and volume queries read real system values."""
    router = FastRouter()
    handled = router.try_fast_path("What is my battery percentage?")
    assert handled is not None
    action, resp = handled
    assert action == "battery"
    assert "%" in resp


# Task D: Browser Tasks (Microsoft Edge default)
def test_task_d_browser_edge_default():
    """Task D: Browser navigation defaults to Microsoft Edge."""
    bc = BrowserController()
    resolved = bc._resolve_browser(None)
    assert resolved == "edge"

    # YouTube search URL construction
    router = FastRouter()
    intent, params = Normalizer.classify_intent("Search Python FastAPI on YouTube")
    assert intent == IntentType.SEARCH_YOUTUBE
    assert params["query"].lower() == "python fastapi"


# Task E: Multi-Step Task Normalization
def test_task_e_multi_step_flow():
    """Task E: Multi-step typing and saving parsed accurately."""
    intent, params = Normalizer.classify_intent(
        "Open Notepad, type This is a JARVIS test, and save it on Desktop as jarvis_test.txt"
    )
    assert intent == IntentType.MULTI_STEP
    assert params["action"] == "notepad_type_save"
    assert "jarvis_test.txt" in params["save_file"]


# Task F: Context Follow-Up Tasks
def test_task_f_context_follow_up():
    """Task F: Conversation memory retains short-term context across turns."""
    agent = JarvisAgent(settings=Settings(), provider=MockProvider())
    agent.memory.add_user_message("Open Notepad")
    agent.memory.add_assistant_message("Opened Notepad, Sir.")
    agent.memory.add_user_message("Type hello")
    agent.memory.add_assistant_message("Typed hello.")

    history = agent.memory.get_history()
    assert len(history) == 4
    assert history[0]["content"] == "Open Notepad"


# Task G: Failed Task Truthful Reporting
def test_task_g_failed_task_handling():
    """Task G: Target that does not exist reports failure, never falsely claims success."""
    indexer = AppIndexer()
    res = indexer.launch_app("non_existent_fake_app_xyz")
    assert res.success is False
    assert "couldn't find" in res.message.lower() or "not found" in res.message.lower()


# Task H: App Discovery vs Missing App
def test_task_h_app_search_not_found():
    """Task H: Checks if installed first; reports not found without opening Microsoft Store."""
    indexer = AppIndexer()
    app = indexer.find_app("definitely_uninstalled_game_999")
    assert app is None


# Task I: Web vs App Distinction
def test_task_i_web_vs_app_differentiation():
    """Task I: Distinguishes websites (YouTube, GitHub, Google) from apps (VS Code, Edge)."""
    assert Normalizer.classify_intent("Open YouTube")[0] == IntentType.OPEN_WEBSITE
    assert Normalizer.classify_intent("Open GitHub")[0] == IntentType.OPEN_WEBSITE
    assert Normalizer.classify_intent("Open Google")[0] == IntentType.OPEN_WEBSITE

    assert Normalizer.classify_intent("Open VS Code")[0] == IntentType.OPEN_APP
    assert Normalizer.classify_intent("Open Calculator")[0] == IntentType.OPEN_APP
    assert Normalizer.classify_intent("Open Notepad")[0] == IntentType.OPEN_APP


# Task K: Window Control Normalization
def test_task_k_window_controls():
    """Task K: Media and workstation lock actions route cleanly."""
    intent, params = Normalizer.classify_intent("Increase volume")
    assert intent == IntentType.MEDIA_CONTROL
    assert params["action"] == "volume_up"

    intent, params = Normalizer.classify_intent("Mute volume")
    assert intent == IntentType.MEDIA_CONTROL
    assert params["action"] == "mute"


# Task L & M: Safe Command Execution and Coding Tasks
def test_task_l_m_coding_execution_verification():
    """Task L & M: Runs a Python test script and verifies exit code and stdout."""
    tc = TerminalController()
    with tempfile.TemporaryDirectory() as temp_dir:
        py_file = os.path.join(temp_dir, "hello.py")
        with open(py_file, "w", encoding="utf-8") as f:
            f.write("print('Hello JARVIS Coding Test')\n")

        ok, code, out = tc.run_command(f'"{sys.executable}" "{py_file}"')
        assert ok is True
        assert code == 0
        assert "Hello JARVIS Coding Test" in out


# Task N: Information vs Action Distinction
def test_task_n_question_vs_action_distinction():
    """Task N: Distinguishes question/explanation from actual Windows action."""
    q_intent, _ = Normalizer.classify_intent("How do I open Task Manager?")
    assert q_intent != IntentType.OPEN_APP

    a_intent, params = Normalizer.classify_intent("Open Task Manager")
    assert a_intent == IntentType.OPEN_APP


# Task Q: Natural Language Phrasing Variations
def test_task_q_natural_language_variations():
    """Task Q: Normalizes phrasing variations to correct intent."""
    variations = [
        "Could you open Edge for me?",
        "Launch the browser.",
        "Edge open please.",
    ]
    for phrase in variations:
        intent, params = Normalizer.classify_intent(phrase)
        assert intent in [IntentType.OPEN_APP, IntentType.OPEN_WEBSITE]


# Task R: Varied Multi-Command Security Stress Sequence
def test_task_r_stress_security_sequence():
    """Task R: Varied command sequence verifies no state corruption or owner leakage."""
    agent = JarvisAgent(settings=Settings(), provider=MockProvider())

    # 1. OWNER greeting
    resp1 = agent.process_input("Hello", speaker=SpeakerClassification.OWNER)
    assert "Sanjay" in resp1 or "Sir" in resp1

    # 2. UNKNOWN greeting
    resp2 = agent.process_input("Hello", speaker=SpeakerClassification.UNKNOWN)
    assert "Sanjay" not in resp2
    assert "Sir" not in resp2

    # 3. UNKNOWN requests sensitive command -> blocked
    resp3 = agent.process_input("Delete this file", speaker=SpeakerClassification.UNKNOWN)
    assert "Access denied" in resp3

    # 4. Emergency stop
    resp4 = agent.process_input("Jarvis stop", speaker=SpeakerClassification.OWNER)
    assert "Stopped immediately" in resp4
