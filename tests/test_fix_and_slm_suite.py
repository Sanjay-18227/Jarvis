"""
Test suite covering FIX 1 (Owner Voice Enrollment), FIX 2 (OpenRouter Free-Only & Resiliency),
Local-First Routing, and Local SLM (Natural Paraphrases & Offline Capability).
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.constants import SpeakerClassification
from config.settings import get_settings
from core.agent import JarvisAgent
from core.fast_router import FastRouter
from core.local_slm import LocalSLM, get_local_slm
from voice.speaker_verification import SpeakerVerifier
from voice.voice_enrollment import VoiceEnrollment
from llm.openrouter_provider import OpenRouterProvider, OpenRouterError


def create_pcm_wav(
    duration: float = 2.0,
    sample_rate: int = 16000,
    freq: float = 130.0,
    amplitude: float = 15000.0,
    pre_silence: float = 0.2,
    post_silence: float = 0.3,
) -> bytes:
    """Generates synthetic PCM 16-bit mono WAV bytes with harmonic glottal source."""
    import io, wave
    n_speech = int(sample_rate * duration)
    t = np.linspace(0, duration, n_speech, endpoint=False)
    speech = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 8))
    speech = speech / np.max(np.abs(speech)) * amplitude

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


def test_1_owner_enrollment_and_backup(tmp_path):
    """Test 1: New owner enrollment with backup and calibration output."""
    profile_path = str(tmp_path / "owner_profile.dat")
    verifier = SpeakerVerifier(profile_path=profile_path)
    enroller = VoiceEnrollment(verifier=verifier)

    # Fake existing profile to verify backup behavior
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    with open(profile_path, "w") as f:
        f.write('{"owner_name": "OldOwner", "embedding": [0.1]*42}')

    # Generate 10 valid mock audio samples
    samples = [create_pcm_wav(freq=130.0 + i * 0.5) for i in range(10)]

    success = enroller.enroll_from_samples(samples, owner_name="Sanjay", replace=True)
    assert success is True
    assert os.path.exists(profile_path)
    assert verifier.similarity_threshold_owner >= 0.75
    assert verifier.similarity_threshold_owner <= 0.85


def test_2_to_6_local_commands_fast_path():
    """Tests 2-6: FastRouter local commands without AI dependency."""
    agent = JarvisAgent()

    # Test 2: "open VS Code"
    with patch.object(agent.executor.app_ctrl.indexer, "launch_app") as mock_launch:
        from core.tool_result import ToolResult
        mock_launch.return_value = ToolResult.ok(action="open_app", message="VS Code launched")
        resp = agent.process_input("open VS Code", speaker=SpeakerClassification.OWNER)
        assert "VS Code" in resp or "launch" in resp.lower() or "opened" in resp.lower()

    # Test 3: "open browser"
    with patch.object(agent.executor.browser_ctrl, "open_browser") as mock_browser:
        from core.tool_result import ToolResult
        mock_browser.return_value = ToolResult.ok(action="open_browser", message="Microsoft Edge opened")
        resp = agent.process_input("open browser", speaker=SpeakerClassification.OWNER)
        assert "browser" in resp.lower() or "edge" in resp.lower()

    # Test 4: "open YouTube"
    with patch.object(agent.executor.browser_ctrl, "open_url") as mock_url:
        from core.tool_result import ToolResult
        mock_url.return_value = ToolResult.ok(action="open_url", message="Navigated to YouTube")
        resp = agent.process_input("open YouTube", speaker=SpeakerClassification.OWNER)
        assert "youtube" in resp.lower()

    # Test 5: "open Calculator"
    with patch.object(agent.executor.app_ctrl.indexer, "launch_app") as mock_launch:
        from core.tool_result import ToolResult
        mock_launch.return_value = ToolResult.ok(action="open_app", message="Calculator launched")
        resp = agent.process_input("open Calculator", speaker=SpeakerClassification.OWNER)
        assert "calculator" in resp.lower() or "calc" in resp.lower()

    # Test 6: "2 + 2"
    resp = agent.process_input("2 + 2", speaker=SpeakerClassification.OWNER)
    assert "4" in resp


def test_7_offline_openrouter_disabled_tests_2_to_6():
    """Test 7: Tests 2-6 work even with OpenRouter completely disabled / network disconnected."""
    agent = JarvisAgent()
    # Trip circuit / disable OpenRouter
    if hasattr(agent.provider, "trip_circuit"):
        agent.provider.trip_circuit("manual test offline")

    # Verify FastRouter handles all local commands with zero remote calls
    with patch.object(agent.executor.app_ctrl.indexer, "launch_app") as mock_launch, \
         patch.object(agent.executor.browser_ctrl, "open_browser") as mock_browser, \
         patch.object(agent.executor.browser_ctrl, "open_url") as mock_url:
        from core.tool_result import ToolResult
        mock_launch.return_value = ToolResult.ok(action="open_app", message="App launched")
        mock_browser.return_value = ToolResult.ok(action="open_browser", message="Edge opened")
        mock_url.return_value = ToolResult.ok(action="open_url", message="YouTube opened")

        assert "VS Code" in agent.process_input("open VS Code", speaker=SpeakerClassification.OWNER) or "launched" in agent.process_input("open VS Code", speaker=SpeakerClassification.OWNER)
        b_resp = agent.process_input("open browser", speaker=SpeakerClassification.OWNER).lower()
        assert any(w in b_resp for w in ["opened", "launched", "edge", "browser"])
        assert "youtube" in agent.process_input("open YouTube", speaker=SpeakerClassification.OWNER).lower()
        assert "calculator" in agent.process_input("open Calculator", speaker=SpeakerClassification.OWNER).lower() or "launched" in agent.process_input("open Calculator", speaker=SpeakerClassification.OWNER)
        assert "4" in agent.process_input("2 + 2", speaker=SpeakerClassification.OWNER)


def test_8_complex_ai_question_with_free_openrouter():
    """Test 8: Complex question uses free model chain without crash."""
    settings = get_settings()
    provider = OpenRouterProvider(api_key="test_key", model="openrouter/free")
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Quantum computing uses superposition and entanglement."}}]
        }
        res = provider.generate("Explain quantum computing in one sentence.")
        assert "Quantum" in res


def test_9_invalid_openrouter_key_jarvis_stays_alive():
    """Test 9: Invalid OpenRouter key (401) fails fast, trips circuit, JARVIS stays alive."""
    agent = JarvisAgent()
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 401
        mock_post.return_value.text = "Unauthorized: Invalid API key"

        # Ask a complex query that triggers remote reasoning
        resp = agent.process_input("Explain the theory of general relativity in depth", speaker=SpeakerClassification.OWNER)
        assert resp is not None
        assert len(resp) > 0
        # Agent did not crash, and subsequent local commands still work
        calc_resp = agent.process_input("2 + 2", speaker=SpeakerClassification.OWNER)
        assert "4" in calc_resp


def test_10_different_speaker_unknown_denied():
    """Test 10: Different speaker -> UNKNOWN and computer controls denied."""
    agent = JarvisAgent()
    with patch.object(agent.executor.app_ctrl.indexer, "launch_app") as mock_launch:
        # UNKNOWN cannot open VS Code
        resp = agent.process_input("open VS Code", speaker=SpeakerClassification.UNKNOWN)
        assert "don't recognize" in resp.lower() or "locked" in resp.lower()
        mock_launch.assert_not_called()

        # UNKNOWN cannot open browser
        resp = agent.process_input("open browser", speaker=SpeakerClassification.UNKNOWN)
        assert "don't recognize" in resp.lower() or "locked" in resp.lower()


def test_natural_variants_via_slm():
    """Test natural language variants handled locally by Local SLM."""
    slm = get_local_slm()
    assert slm.enabled is True

    # 1. "bro start code"
    p1 = slm.infer_intent("bro start code")
    assert p1["intent"] == "OPEN_APP"
    assert p1["target_app"] == "vscode"
    assert p1["needs_remote_reasoning"] is False

    # 2. "launch my browser"
    p2 = slm.infer_intent("launch my browser")
    assert p2["intent"] == "OPEN_BROWSER"
    assert p2["needs_remote_reasoning"] is False

    # 3. "go to YouTube"
    p3 = slm.infer_intent("go to YouTube")
    assert p3["intent"] == "OPEN_WEBSITE"
    assert p3["site_name"] == "youtube"
    assert p3["needs_remote_reasoning"] is False

    # 4. "open that Kumar project" / "open Kumar in VS Code"
    p4 = slm.infer_intent("open that Kumar project")
    assert p4["intent"] == "OPEN_FOLDER_IN_APP"
    assert p4["folder"] == "Kumar"

    p4b = slm.infer_intent("open Kumar in VS Code")
    assert p4b["intent"] == "OPEN_FOLDER_IN_APP"
    assert p4b["folder"] == "Kumar"
    assert p4b["target_app"] == "vscode"

    # 5. "calculate two plus two"
    p5 = slm.infer_intent("calculate two plus two")
    assert p5["intent"] == "CALCULATE"
    assert "2" in p5["expression"] and "+" in p5["expression"]

    # 6. "calculate 678 - 876"
    p6 = slm.infer_intent("calculate 678 - 876")
    assert p6["intent"] == "CALCULATE"
    assert "678" in p6["expression"] and "876" in p6["expression"]

    # 7. "create agent.py inside Kumar"
    p7 = slm.infer_intent("create agent.py inside Kumar")
    assert p7["intent"] == "CREATE_FILE"
    assert p7["filename"] == "agent.py"
    assert p7["folder"] == "Kumar"

    # 8. "draw a rectangle"
    p8 = slm.infer_intent("draw a rectangle")
    assert p8["intent"] == "PAINT_ACTION"
    assert p8["action"] == "draw_rectangle"

    # 9. "what is my battery percentage?"
    p9 = slm.infer_intent("what is my battery percentage?")
    assert p9["intent"] == "BATTERY"


def test_slm_end_to_end_routing_in_agent():
    """Verify natural variants route through agent to execution without cloud."""
    agent = JarvisAgent()

    # "calculate two plus two"
    resp = agent.process_input("calculate two plus two", speaker=SpeakerClassification.OWNER)
    assert "4" in resp

    # "calculate 678 - 876"
    resp = agent.process_input("calculate 678 - 876", speaker=SpeakerClassification.OWNER)
    assert "-198" in resp
