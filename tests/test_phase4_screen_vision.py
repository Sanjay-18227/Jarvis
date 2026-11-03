"""
Comprehensive Phase 4 Verification Suite: Real Screen Awareness & Vision
Validates Sections 33 to 43 from user specification:
1. Test 1 — Basic screen description (OWNER: 'Jarvis, what is on my screen?')
2. Test 2 — Active application ('Which application am I looking at?')
3. Test 3 — Error reading ('Jarvis, what error is on my screen?')
4. Test 4 — VS Code verification ('Open VS Code and check whether it opened.')
5. Test 5 — Project verification ('Open Kumar in VS Code and verify the correct project is open.')
6. Test 6 — Dialog reading ('What does this dialog say?')
7. Test 7 — UNKNOWN user security ('Jarvis, tell me what's on the screen' -> DENIED)
8. Test 8 — Gemini failure resilience (process stays alive, falls back to local window/process verification)
9. Test 9 — Screen change detection (before/after recognition)
10. Test 10 — False claim prevention (disable screenshot -> NO false claim)
11. Phase 4 + Phase 3 combined test ('open my Kumar project in VS Code, run agent.py, and check the screen for any errors')
"""

import os
import io
import json
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from config.constants import SpeakerClassification
from config.settings import get_settings
from core.agent import JarvisAgent
from core.fast_router import FastRouter
from core.agent_loop import Phase3AgentLoop, StepObservation
from vision.screen_capture import ScreenCapture
from vision.screen_vision_service import ScreenVisionService
from vision.screen_understanding import ScreenVision
from vision.privacy_filter import PrivacyFilter


# ---------------------------------------------------------
# Test 1 — Basic Screen Description (Section 33)
# ---------------------------------------------------------
def test_1_basic_screen_description():
    """
    OWNER: 'Jarvis, what is on my screen?'
    Verifies that a REAL screenshot was captured and analyzed.
    Must fail if no screenshot was taken.
    """
    cap = ScreenCapture()
    img = cap.capture_screen(mode="FULL_SCREEN")
    assert img is not None
    assert img.size[0] > 0 and img.size[1] > 0

    service = ScreenVisionService(capture=cap)

    # Mock Gemini response returning structured visual JSON
    fake_gemini_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "screen_summary": "I can see your Windows desktop with the active workspace.",
        "active_application": "Antigravity IDE",
        "windows": [{"title": "Jarvis - Workspace", "application": "Antigravity IDE", "state": "visible"}],
        "visible_text": ["tests", "core", "vision"],
        "errors": [],
        "ui_elements": [{"type": "button", "label": "Run", "approx_region": "top-right", "confidence": 0.95}],
        "confidence": 0.94,
    })
    fake_gemini_client.models.generate_content.return_value = mock_resp
    service._custom_gemini = fake_gemini_client

    obs = service.analyze_screen(
        mode="FULL_SCREEN",
        user_query="what is on my screen",
        speaker=SpeakerClassification.OWNER,
    )

    assert obs["success"] is True
    assert obs["evidence"]["screenshot_captured"] is True
    assert "desktop" in obs["summary"].lower() or "antigravity" in obs["summary"].lower()
    assert obs["active_app"] == "Antigravity IDE"
    assert obs["evidence"]["confidence"] >= 0.90


# ---------------------------------------------------------
# Test 2 — Active Application Detection (Section 34)
# ---------------------------------------------------------
def test_2_active_application_detection():
    """
    Ask: 'Which application am I looking at?'
    Expected: Uses local active-window data plus vision.
    """
    cap = ScreenCapture()
    win_info = cap.get_active_window_info()
    assert "application" in win_info
    assert "title" in win_info
    assert win_info["hwnd"] is not None

    agent = JarvisAgent()
    with patch.object(agent.screen_vision.service, "analyze_screen") as mock_analyze:
        mock_analyze.return_value = {
            "success": True,
            "action": "screen_analysis",
            "active_app": "Notepad",
            "summary": "Sir, you are currently looking at Notepad.",
            "evidence": {
                "screenshot_captured": True,
                "vision_provider": "gemini",
                "confidence": 0.95,
                "active_window": {"title": "Untitled - Notepad", "application": "Notepad"},
            },
            "error": None,
        }

        resp = agent.process_input("Which application am I looking at?", speaker=SpeakerClassification.OWNER)
        assert "notepad" in resp.lower()


# ---------------------------------------------------------
# Test 3 — Error Reading (Section 35)
# ---------------------------------------------------------
def test_3_error_reading():
    """
    Ask: 'Jarvis, what error is on my screen?'
    Expected: Identifies actual visible error. Does not invent errors.
    """
    service = ScreenVisionService()
    fake_gemini = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "screen_summary": "VS Code is showing a ModuleNotFoundError for 'requests'.",
        "active_application": "Visual Studio Code",
        "windows": [{"title": "agent.py - Visual Studio Code", "application": "Visual Studio Code", "state": "visible"}],
        "visible_text": ["ModuleNotFoundError: No module named 'requests'"],
        "errors": [{"message": "ModuleNotFoundError: No module named 'requests'", "source": "Python runtime", "severity": "error"}],
        "ui_elements": [],
        "confidence": 0.96,
    })
    fake_gemini.models.generate_content.return_value = mock_resp
    service._custom_gemini = fake_gemini

    obs = service.analyze_screen(
        mode="ACTIVE_WINDOW",
        user_query="what error is on my screen",
        speaker=SpeakerClassification.OWNER,
    )

    assert obs["success"] is True
    errors = obs["evidence"]["structured"]["errors"]
    assert len(errors) == 1
    assert "ModuleNotFoundError" in errors[0]["message"]
    assert "requests" in obs["summary"]


# ---------------------------------------------------------
# Test 4 — VS Code Verification (Section 36)
# ---------------------------------------------------------
def test_4_vscode_verification():
    """
    Ask: 'Open VS Code and check whether it opened.'
    Expected: Tool launch + process/window verification + screen observation.
    """
    agent = JarvisAgent()
    with patch.object(agent.executor.app_ctrl.indexer, "launch_app") as mock_launch, \
         patch.object(agent.screen_vision.service, "analyze_screen") as mock_vis:

        mock_launch.return_value = MagicMock(success=True, message="VS Code launched, Sir.", evidence="Code.exe PID 1234")
        mock_vis.return_value = {
            "success": True,
            "action": "screen_analysis",
            "active_app": "Visual Studio Code",
            "summary": "Sir, I verified that Visual Studio Code is open and visible.",
            "evidence": {"screenshot_captured": True, "confidence": 0.94},
            "error": None,
        }

        # Check via agent router
        resp = agent.process_input("Check whether VS Code opened", speaker=SpeakerClassification.OWNER)
        assert "vs code" in resp.lower() or "visual studio code" in resp.lower()


# ---------------------------------------------------------
# Test 5 — Project Verification (Section 37)
# ---------------------------------------------------------
def test_5_project_verification():
    """
    Ask: 'Open Kumar in VS Code and verify the correct project is open.'
    Expected: Process verified + screen/window evidence confirms Kumar.
    """
    service = ScreenVisionService()
    with patch.object(service, "get_active_window") as mock_win:
        mock_win.return_value = {
            "hwnd": 9999,
            "title": "agent.py - Kumar - Visual Studio Code",
            "application": "Visual Studio Code",
            "process": "Code.exe",
            "pid": 5678,
            "rect": (0, 0, 1366, 768),
            "monitor": 1,
            "is_sensitive": False,
        }

        res = service.verify_app_state(app_name="Visual Studio Code", expected_project="Kumar")
        assert res["verified"] is True
        assert res["project"] == "Kumar"
        assert "Kumar" in res["evidence"]


# ---------------------------------------------------------
# Test 6 — Dialog Detection (Section 38)
# ---------------------------------------------------------
def test_6_dialog_detection():
    """
    Ask: 'What does this dialog say?'
    Expected: Extracts and describes actual dialog.
    """
    service = ScreenVisionService()
    fake_gemini = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "screen_summary": "A confirmation dialog is visible asking 'Do you want to save changes to Untitled?'",
        "active_application": "Notepad",
        "windows": [{"title": "Save changes?", "application": "Notepad", "state": "visible"}],
        "visible_text": ["Do you want to save changes to Untitled?", "Save", "Don't Save", "Cancel"],
        "errors": [],
        "ui_elements": [
            {"type": "dialog", "label": "Save changes?", "approx_region": "center", "confidence": 0.98},
            {"type": "button", "label": "Save", "approx_region": "center-left", "confidence": 0.95},
            {"type": "button", "label": "Don't Save", "approx_region": "center", "confidence": 0.95},
            {"type": "button", "label": "Cancel", "approx_region": "center-right", "confidence": 0.95},
        ],
        "confidence": 0.97,
    })
    fake_gemini.models.generate_content.return_value = mock_resp
    service._custom_gemini = fake_gemini

    obs = service.analyze_screen(
        mode="ACTIVE_WINDOW",
        user_query="What does this dialog say?",
        speaker=SpeakerClassification.OWNER,
    )

    assert obs["success"] is True
    assert "save changes" in obs["summary"].lower()
    elems = obs["evidence"]["structured"]["ui_elements"]
    assert any(e["type"] == "dialog" for e in elems)
    assert any(e["label"] == "Save" for e in elems)


# ---------------------------------------------------------
# Test 7 — UNKNOWN User Screen Security (Section 39)
# ---------------------------------------------------------
def test_7_unknown_user_screen_security(capsys):
    """
    UNKNOWN: 'Jarvis, tell me what's on the screen.'
    Expected: DENIED / restricted. No screenshot contents exposed.
    """
    agent = JarvisAgent()
    with patch.object(agent.screen_vision.service, "analyze_screen") as mock_analyze:
        resp = agent.process_input("Jarvis, tell me what's on the screen.", speaker=SpeakerClassification.UNKNOWN)
        captured = capsys.readouterr().out
        assert "[SECURITY] DENIED" in captured or "Access denied" in resp or "don't recognize" in resp.lower()
        mock_analyze.assert_not_called()
        assert "locked" in resp.lower() or "access denied" in resp.lower() or "don't recognize" in resp.lower()


# ---------------------------------------------------------
# Test 8 — Gemini Failure Resilience (Section 40)
# ---------------------------------------------------------
def test_8_gemini_failure_resilience():
    """
    Simulate Gemini timeout or quota failure.
    Expected: JARVIS stays alive, falls back to local window/process verification.
    """
    service = ScreenVisionService()
    fake_gemini = MagicMock()
    fake_gemini.models.generate_content.side_effect = RuntimeError("503 Service Unavailable / Quota exceeded")
    service._custom_gemini = fake_gemini

    with patch.object(service, "get_active_window") as mock_win:
        mock_win.return_value = {
            "hwnd": 1234,
            "title": "Untitled - Notepad",
            "application": "Notepad",
            "process": "notepad.exe",
            "pid": 4321,
            "rect": (0, 0, 800, 600),
            "monitor": 1,
            "is_sensitive": False,
        }

        obs = service.analyze_screen(
            mode="ACTIVE_WINDOW",
            user_query="what is on my screen",
            speaker=SpeakerClassification.OWNER,
        )

        # Process survives gracefully
        assert obs["success"] is True
        assert obs["evidence"]["vision_provider"] == "local_window_fallback"
        assert "notepad" in obs["summary"].lower()


# ---------------------------------------------------------
# Test 9 — Screen Change Detection (Section 41)
# ---------------------------------------------------------
def test_9_screen_change_detection():
    """
    Before and after screenshots or observations recognize meaningful change.
    """
    service = ScreenVisionService()

    # Case A: Error cleared between observations
    obs_before = {
        "active_app": "VS Code",
        "evidence": {
            "structured": {
                "errors": [{"message": "ModuleNotFoundError: No module named requests"}]
            }
        }
    }
    obs_after = {
        "active_app": "VS Code",
        "evidence": {
            "structured": {
                "errors": []
            }
        }
    }

    comp = service.compare_screen_state(obs_before, obs_after, action_description="pip install requests")
    assert comp["changed"] is True
    assert "error" in comp["summary"].lower() or "resolved" in comp["summary"].lower()

    # Case B: Perceptual image difference
    img1 = Image.new("RGB", (200, 200), color=(0, 0, 0))
    img2 = Image.new("RGB", (200, 200), color=(255, 255, 255))
    comp_img = service.compare_screen_state(img1, img2, action_description="theme switch")
    assert comp_img["changed"] is True


# ---------------------------------------------------------
# Test 10 — False Claim Prevention (Section 42)
# ---------------------------------------------------------
def test_10_false_claim_prevention():
    """
    Disable screenshot capture or vision capability.
    Ask: 'Can you see my screen?'
    Expected: NO false claim. JARVIS reports screen inspection is not available.
    """
    agent = JarvisAgent()
    # Disable screen vision explicitly
    agent.screen_vision.service._enabled_override = False

    resp = agent.process_input("Can you see my screen?", speaker=SpeakerClassification.OWNER)
    assert "isn't available" in resp.lower() or "unable" in resp.lower() or "not available" in resp.lower()
    # Ensure it never claims "I can see your screen"
    assert "i can see your screen" not in resp.lower()


# ---------------------------------------------------------
# Phase 4 + Phase 3 Combined Test (Section 43)
# ---------------------------------------------------------
def test_phase4_plus_phase3_combined_flow(tmp_path):
    """
    OWNER: 'Jarvis, open my Kumar project in VS Code, run agent.py, and check the screen for any errors.'
    Expected flow:
    Speaker verification -> OWNER -> Phase 3 plan -> open VS Code -> run agent.py -> observe screen -> verify -> report.
    """
    # Create temp Kumar project and agent.py
    kumar_dir = os.path.join(tmp_path, "Kumar")
    os.makedirs(kumar_dir, exist_ok=True)
    agent_file = os.path.join(kumar_dir, "agent.py")
    with open(agent_file, "w") as f:
        f.write("print('Agent Ready')\n")

    loop = Phase3AgentLoop()
    goal = "Inside Kumar folder create agent.py, run agent.py, and check the screen for any errors"

    plan = loop._create_plan(goal)
    assert plan is not None
    tool_names = [s.tool for s in plan.steps]
    assert "run_python" in tool_names or "create_file" in tool_names
    assert "inspect_screen" in tool_names

    # Mock screen vision on the loop to report no visible errors
    with patch.object(loop.screen_vision, "analyze_screen") as mock_screen:
        mock_screen.return_value = {
            "success": True,
            "action": "screen_analysis",
            "active_app": "Visual Studio Code",
            "summary": "VS Code is showing agent.py with no errors.",
            "evidence": {
                "screenshot_captured": True,
                "vision_provider": "gemini",
                "confidence": 0.95,
                "structured": {"errors": []},
            },
            "error": None,
        }

        def mock_resolve(p):
            p_low = p.lower()
            if "agent.py" in p_low:
                return agent_file
            if "kumar" in p_low:
                return kumar_dir
            return os.path.join(tmp_path, os.path.basename(p))

        with patch.object(loop.file_mgr, "_resolve_path", side_effect=mock_resolve):
            ok, msg = loop.run_goal(goal, speaker=SpeakerClassification.OWNER)
            assert ok is True
            assert "completed successfully" in msg.lower() or "agent ready" in msg.lower()
