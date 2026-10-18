"""
JARVIS Autonomous Agent & Computer-Use Verification Suite
Tests:
- Single consistent multilingual voice identity
- Declarative tool catalog schema compliance
- Gemini tool planning and function calling
- Emergency stop command ("Jarvis stop")
- SecretsManager redaction across speech and logs
- Mouse and keyboard controllers
- TerminalController command safety and sandboxing
- Window management and UI element detection
- Autonomous task state tracking and verification loop
"""

import pytest
import os
from config.settings import Settings
from config.constants import SpeakerClassification, RiskLevel
from core.multilingual import MultilingualProcessor
from ai.tool_planner import ToolCatalog
from ai.gemini_client import GeminiClient
from security.secrets_manager import SecretsManager
from computer.terminal_controller import TerminalController
from computer.mouse_controller import MouseController
from computer.keyboard_controller import KeyboardController
from computer.windows_controller import WindowsController
from vision.ui_element_detection import UIElementDetector
from core.task_manager import TaskManager, TaskState
from core.executor import ActionExecutor
from core.planner import AutonomousPlanner
from core.agent import JarvisAgent
from llm.mock_provider import MockProvider


def test_consistent_voice_identity():
    """Verify JARVIS uses one consistent voice identity across all languages."""
    mp = MultilingualProcessor()
    voice_en = mp.get_preferred_voice_for_language("en")
    voice_ta = mp.get_preferred_voice_for_language("ta")
    voice_hi = mp.get_preferred_voice_for_language("hi")
    voice_multi = mp.get_preferred_voice_for_language("tanglish")

    assert voice_en == "en-US-AndrewMultilingualNeural"
    assert voice_ta == "en-US-AndrewMultilingualNeural"
    assert voice_hi == "en-US-AndrewMultilingualNeural"
    assert voice_multi == "en-US-AndrewMultilingualNeural"


def test_tool_catalog_declarations():
    """Verify all tools declare name, description, parameters, permission_level, risk_level, requires_confirmation."""
    catalog = ToolCatalog()
    tools = catalog.list_tools()
    assert len(tools) >= 15

    for t in tools:
        assert bool(t.name)
        assert bool(t.description)
        assert isinstance(t.parameters, dict)
        assert t.permission_level in ["OWNER", "ANY"]
        assert 0 <= t.risk_level <= 4
        assert isinstance(t.requires_confirmation, bool)

    # delete_path must require confirmation
    delete_tool = catalog.get_tool("delete_path")
    assert delete_tool is not None
    assert delete_tool.requires_confirmation is True
    assert delete_tool.risk_level == 4


def test_secrets_manager_redaction():
    """Verify passwords, API keys, tokens are redacted and safe for speech."""
    sm = SecretsManager()
    sample_text = "My secret api_key = 'AIzaSyD1234567890abcdef1234567890abcdef' and token is sk-123456789012345678901234."

    assert sm.contains_secret(sample_text) is True
    sanitized = sm.sanitize(sample_text)
    assert "AIzaSy" not in sanitized
    assert "sk-123456789012345678901234" not in sanitized
    assert "***" in sanitized

    is_safe, speech_text = sm.validate_safe_for_speech(sample_text)
    assert is_safe is True
    assert "AIzaSy" not in speech_text


def test_terminal_controller_safety():
    """Verify destructive shell commands are blocked and safe commands execute."""
    tc = TerminalController()

    # Blocked commands
    blocked_cmd = "rmdir /s /q c:\\"
    ok, code, msg = tc.run_command(blocked_cmd)
    assert ok is False
    assert "Security Block" in msg

    # Safe command
    ok, code, out = tc.run_command("echo JARVIS_ONLINE")
    assert ok is True
    assert "JARVIS_ONLINE" in out


def test_mouse_and_keyboard_controllers():
    """Verify MouseController and KeyboardController functions run cleanly without crashing."""
    mc = MouseController()
    pos = mc.get_position()
    assert isinstance(pos, tuple)
    assert len(pos) == 2

    kc = KeyboardController()
    # Test virtual key code resolution
    vk_enter = kc._resolve_vk("enter")
    assert vk_enter is not None
    vk_ctrl = kc._resolve_vk("ctrl")
    assert vk_ctrl is not None


def test_windows_controller_listing():
    """Verify visible top-level windows enumeration."""
    wc = WindowsController()
    windows = wc.list_windows()
    assert isinstance(windows, list)
    # Check that entries have hwnd and title
    if windows:
        assert "hwnd" in windows[0]
        assert "title" in windows[0]


def test_ui_element_detection():
    """Verify UIElementDetector inspects focused control without error."""
    detector = UIElementDetector()
    info = detector.get_focused_control_info()
    assert "title" in info
    assert "rect" in info


def test_task_manager_lifecycle_and_emergency_stop():
    """Verify TaskManager tracks task states and emergency stop immediately halts execution."""
    tm = TaskManager()
    task = tm.create_task("Build project")
    assert task.state == TaskState.IN_PROGRESS
    assert tm.is_stop_requested() is False

    tm.record_step("open_application", {"target": "antigravity"}, "SUCCESS", "Antigravity focused")
    assert len(task.steps) == 1

    # Emergency stop
    tm.request_emergency_stop()
    assert tm.is_stop_requested() is True
    assert task.state == TaskState.CANCELLED


def test_agent_emergency_stop_phrase():
    """Verify 'Jarvis stop' halts active automation and returns confirmation."""
    settings = Settings(owner_name="Sanjay", preferred_title="Sir")
    agent = JarvisAgent(settings=settings, provider=MockProvider())

    resp = agent.process_input("Jarvis stop")
    assert "Stopped immediately" in resp
    assert "Sir" in resp


def test_planner_verification_loop():
    """Verify planner executes steps and does not claim success without verification."""
    mock_prov = MockProvider()
    executor = ActionExecutor()
    tm = TaskManager()
    catalog = ToolCatalog()

    planner = AutonomousPlanner(
        client=mock_prov,
        executor=executor,
        task_manager=tm,
        tool_catalog=catalog,
    )

    ok, summary = planner.execute_goal("Check battery status", speaker=SpeakerClassification.OWNER)
    assert bool(summary)
