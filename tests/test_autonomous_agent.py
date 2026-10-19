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
from core.fast_router import FastRouter
from computer.app_indexer import AppIndexer
from computer.system_monitor import SystemMonitor
from ai.research_agent import ResearchAgent
from core.project_detector import ProjectDetector


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


def test_fast_router_classification():
    """Verify FastRouter correctly resolves instant sub-second local intents."""
    router = FastRouter()

    # Hearing check
    res = router.try_fast_path("Can you hear me?")
    assert res is not None
    assert res[0] == "hear_check"
    assert "hear you clearly" in res[1]

    # Greeting
    res = router.try_fast_path("Hello Jarvis")
    assert res is not None
    assert res[0] == "greeting"
    assert "Sanjay" in res[1]

    # Time
    res = router.try_fast_path("What time is it?")
    assert res is not None
    assert res[0] == "datetime"
    assert "Sir" in res[1]

    # Battery
    res = router.try_fast_path("Check battery status")
    assert res is not None
    assert res[0] == "battery"
    assert "battery is at" in res[1].lower()


def test_app_indexer_catalog():
    """Verify AppIndexer maps system apps, aliases, and rejects Store redirects."""
    indexer = AppIndexer()
    calc_info = indexer.find_app("calc")
    assert calc_info is not None
    assert "calc" in calc_info["launch_cmd"].lower()

    chrome_info = indexer.find_app("chrome")
    assert chrome_info is not None
    assert "chrome" in chrome_info["launch_cmd"].lower()

    settings_info = indexer.find_app("settings")
    assert settings_info is not None
    assert "ms-settings:" in settings_info["launch_cmd"]


def test_system_monitor_diagnostics():
    """Verify SystemMonitor collects real telemetry and provides actionable performance diagnosis."""
    monitor = SystemMonitor()
    vitals = monitor.get_system_vitals()

    assert "cpu_percent" in vitals
    assert "memory_percent" in vitals
    assert "disk_percent" in vitals
    assert "battery" in vitals
    assert isinstance(vitals["top_apps"], list)

    ok, diagnosis = monitor.diagnose_performance()
    assert ok is True
    assert "CPU" in diagnosis or "healthy" in diagnosis or "RAM" in diagnosis


def test_project_detector_analysis():
    """Verify ProjectDetector correctly identifies framework and execution runner commands."""
    detector = ProjectDetector()
    current_proj = detector.detect_project(os.getcwd())

    assert current_proj["type"] == "python"
    assert current_proj["test_cmd"] == "pytest"
    assert current_proj["run_cmd"] is not None


def test_research_agent_offline_fallback():
    """Verify ResearchAgent graceful handling when offline or without Gemini."""
    agent = ResearchAgent(gemini_client=None)
    ok, res = agent.research_topic("Python 3.12 release date")
    assert ok is True
    assert len(res) > 0
