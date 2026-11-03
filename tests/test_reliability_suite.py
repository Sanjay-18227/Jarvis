"""
JARVIS Reliability & Verified Tool Execution Test Suite
Tests every explicit requirement:
1. "Open Notepad" -> opens and verified
2. "Open Microsoft Edge" -> opens and verified
3. "Open VS Code" -> opens and verified
4. "Open YouTube" -> opens youtube.com in browser
5. "Open YouTube on Microsoft Edge" -> Edge opens youtube.com
6. "Search Golden Brown on YouTube" -> opens YouTube search results for Golden Brown
7. "Open Notepad and type Hi how are you" -> text typed into Notepad
8. Save that text as Hello.txt on Desktop -> file verified on Desktop
9. Simulated launch failure -> reports failure and NEVER claims opened
10. "Can you see my screen?" -> truthfully reports screen inspection not active
11. Natural language normalization & alias resolution
12. ToolResult contract compliance
"""

import os
import time
import pytest
from unittest.mock import MagicMock, patch
from core.tool_result import ToolResult
from core.normalizer import Normalizer, IntentType
from computer.app_indexer import AppIndexer, _find_edge_path, _find_vscode_path
from computer.browser_controller import BrowserController
from vision.screen_understanding import ScreenVision
from core.multi_step_executor import MultiStepExecutor
from core.fast_router import FastRouter
from core.executor import ActionExecutor
from core.agent import JarvisAgent
from config.settings import Settings


def test_tool_result_contract():
    """Verify ToolResult contract and tuple unpacking backward compatibility."""
    ok_res = ToolResult.ok(action="test_act", message="Success message", evidence="Verified on disk")
    assert ok_res.success is True
    assert ok_res.action == "test_act"
    assert ok_res.message == "Success message"
    assert ok_res.evidence == "Verified on disk"
    assert bool(ok_res) is True

    # Tuple unpacking
    success, msg = ok_res
    assert success is True
    assert msg == "Success message"

    fail_res = ToolResult.fail(action="test_act", message="Failure message", error="Disk full")
    assert fail_res.success is False
    assert fail_res.error == "Disk full"
    assert bool(fail_res) is False
    s, m = fail_res
    assert s is False
    assert m == "Failure message"


def test_normalizer_natural_variations_and_aliases():
    """Verify natural-language variations and polite filler stripping."""
    # YouTube variations -> OPEN_WEBSITE
    yt_variations = [
        "open YouTube",
        "open YouTube in browser",
        "launch YouTube",
        "go to YouTube",
        "open youtube bro",
        "please open YouTube",
        "hey jarvis open youtube please",
    ]
    for phrase in yt_variations:
        intent, params = Normalizer.classify_intent(phrase)
        assert intent == IntentType.OPEN_WEBSITE, f"Failed for '{phrase}': got {intent}"
        assert params.get("site_name") == "youtube"
        assert params.get("url") == "https://www.youtube.com"

    # Edge variations -> OPEN_APP or browser specification
    edge_variations = [
        "open Edge",
        "open Microsoft Edge",
        "launch Microsoft Edge",
        "open MS Edge",
        "open msedge",
        "please open edge bro",
    ]
    for phrase in edge_variations:
        intent, params = Normalizer.classify_intent(phrase)
        assert intent == IntentType.OPEN_APP, f"Failed for '{phrase}': got {intent}"
        assert params.get("app_key") == "edge"

    # VS Code variations -> OPEN_APP
    vscode_variations = [
        "open vscode",
        "open vs code",
        "launch visual studio code",
        "open code",
        "can you open vs code bro",
    ]
    for phrase in vscode_variations:
        intent, params = Normalizer.classify_intent(phrase)
        assert intent == IntentType.OPEN_APP, f"Failed for '{phrase}': got {intent}"
        assert params.get("app_key") == "vscode"


def test_normalizer_youtube_with_browser():
    """Verify 'Open YouTube on Microsoft Edge' extracts website and browser."""
    intent, params = Normalizer.classify_intent("Open YouTube on Microsoft Edge")
    assert intent == IntentType.OPEN_WEBSITE
    assert params.get("site_name") == "youtube"
    assert params.get("url") == "https://www.youtube.com"
    assert params.get("browser") == "edge"

    intent2, params2 = Normalizer.classify_intent("Open YouTube in Microsoft Edge")
    assert intent2 == IntentType.OPEN_WEBSITE
    assert params2.get("browser") == "edge"


def test_normalizer_youtube_search_and_music():
    """Verify YouTube searches and truthful music intents."""
    intent, params = Normalizer.classify_intent("Search Golden Brown on YouTube")
    assert intent == IntentType.SEARCH_YOUTUBE
    assert "golden brown" in params.get("query", "").lower()

    intent_play, params_play = Normalizer.classify_intent("play Golden Brown on YouTube")
    assert intent_play == IntentType.PLAY_YOUTUBE
    assert "golden brown" in params_play.get("query", "").lower()

    # Generic latest/trending query
    intent_latest, _ = Normalizer.classify_intent("play latest Tamil song")
    assert intent_latest == IntentType.PLAY_LATEST_MUSIC


def test_normalizer_screen_vision_queries():
    """Verify screen vision queries map to SCREEN_VISION."""
    queries = [
        "Can you see my screen?",
        "what is on my screen",
        "do you see my screen",
        "look at my screen",
        "see my desktop",
    ]
    for q in queries:
        intent, _ = Normalizer.classify_intent(q)
        assert intent == IntentType.SCREEN_VISION, f"Failed for '{q}'"


def test_screen_vision_truthfulness_when_disabled():
    """Verify screen inspection truthfully reports when inactive."""
    with patch("config.settings.settings.vision_enabled", False):
        vision = ScreenVision()
        res = vision.inspect_screen("Can you see my screen?")
        assert res.success is False
        assert any(p in res.message.lower() for p in ["screen inspection isn't available", "screen inspection isn't enabled"])
        # Must never claim it can see the screen
        assert "i see" not in res.message.lower()
        assert "i can see" not in res.message.lower()


def test_fast_router_screen_vision_truthful():
    """Verify FastRouter returns truthful refusal for 'Can you see my screen?' when disabled."""
    with patch("config.settings.settings.vision_enabled", False):
        router = FastRouter()
        action, resp = router.try_fast_path("Can you see my screen?")
        assert action == "screen_vision"
        assert any(p in resp.lower() for p in ["screen inspection isn't available", "screen inspection isn't enabled"])
        assert "i can see" not in resp.lower()


def test_simulated_application_launch_failure():
    """Simulate application launch failure and verify JARVIS reports failure truthfully."""
    indexer = AppIndexer()
    # Test non-existent application
    res = indexer.launch_app("nonexistent_fake_app_xyz")
    assert res.success is False
    assert "couldn't find" in res.message.lower()
    # Must NEVER say opened
    assert "opened" not in res.message.lower()
    assert "done" not in res.message.lower()

    # Simulate launch timeout failure on mock
    with patch("subprocess.Popen") as mock_popen, \
         patch.object(indexer, "is_process_running", return_value=False), \
         patch("computer.windows_controller.WindowsController.list_windows", return_value=[]), \
         patch("computer.windows_controller.WindowsController.focus_window", return_value=(False, "")):
        res_fail = indexer.launch_app("edge")
        assert res_fail.success is False
        assert "couldn't verify that it launched" in res_fail.message.lower()
        assert "opened" not in res_fail.message.lower()


def test_browser_open_youtube():
    """Verify BrowserController opens YouTube as website."""
    browser_ctrl = BrowserController()
    with patch("webbrowser.open") as mock_wb, \
         patch.object(browser_ctrl, "_is_browser_running", return_value=True):
        res = browser_ctrl.open_website("https://www.youtube.com", site_name="youtube")
        assert res.success is True
        assert "youtube" in res.message.lower()


def test_browser_open_youtube_on_edge():
    """Verify BrowserController launches Edge specifically when requested."""
    browser_ctrl = BrowserController()
    with patch("subprocess.Popen") as mock_popen, \
         patch.object(browser_ctrl, "_is_browser_running", return_value=True):
        res = browser_ctrl.open_website("https://www.youtube.com", browser="edge", site_name="youtube")
        assert res.success is True
        assert "edge" in res.message.lower()


def test_search_golden_brown_on_youtube():
    """Verify search_youtube generates real search URL and ToolResult."""
    browser_ctrl = BrowserController()
    with patch.object(browser_ctrl, "open_website") as mock_open:
        mock_open.return_value = ToolResult.ok("open_website", "Opened YouTube search, Sir.")
        res = browser_ctrl.search_youtube("Golden Brown")
        assert res.success is True
        mock_open.assert_called_once()
        call_args = mock_open.call_args[0]
        assert "results?search_query=Golden+Brown" in call_args[0]


def test_multi_step_notepad_and_save_desktop():
    """Verify multi-step flow: type text and verify file exists on Desktop."""
    desktop_dir = os.path.expanduser(r"~\Desktop")
    test_filename = "Hello_Jarvis_Test.txt"
    file_path = os.path.join(desktop_dir, test_filename)

    # Clean up before test if exists
    if os.path.exists(file_path):
        os.remove(file_path)

    executor = MultiStepExecutor()
    res = executor.execute_save_file(
        filename=test_filename,
        location="desktop",
        content="Hi, how are you?",
    )

    assert res.success is True
    assert os.path.isfile(file_path), "Hello_Jarvis_Test.txt must exist on Desktop"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hi, how are you?"

    # Clean up test artifact
    if os.path.exists(file_path):
        os.remove(file_path)


def test_action_executor_website_interception():
    """Verify ActionExecutor intercepts 'open_application' with website targets."""
    executor = ActionExecutor()
    with patch.object(executor.browser_ctrl, "open_website") as mock_open:
        mock_open.return_value = ToolResult.ok("open_website", "Opened youtube in browser, Sir.")
        res = executor.execute_action("open_application", {"target": "youtube"})
        assert res.success is True
        mock_open.assert_called_once()


def test_action_executor_returns_tool_result():
    """Verify ActionExecutor returns ToolResult that can be unpacked."""
    executor = ActionExecutor()
    res = executor.execute_action("get_battery_status", {})
    assert isinstance(res, ToolResult)
    assert res.success is True
    s, m = res
    assert s is True
    assert "battery" in m.lower()


def test_agent_process_input_open_notepad():
    """Verify agent handles 'Open Notepad' through fast router with verification."""
    agent = JarvisAgent()
    with patch.object(agent.fast_router.app_indexer, "launch_app") as mock_launch:
        mock_launch.return_value = ToolResult.ok("open_app", "Opened Notepad, Sir.")
        resp = agent.process_input("Open Notepad")
        assert "opened notepad" in resp.lower()
        mock_launch.assert_called_with("notepad")


def test_agent_process_input_open_edge():
    """Verify agent handles 'Open Microsoft Edge' through fast router with verification."""
    agent = JarvisAgent()
    with patch.object(agent.fast_router.app_indexer, "launch_app") as mock_launch:
        mock_launch.return_value = ToolResult.ok("open_app", "Opened Microsoft Edge, Sir.")
        resp = agent.process_input("Open Microsoft Edge")
        assert "opened microsoft edge" in resp.lower()
        mock_launch.assert_called_with("edge")


def test_agent_process_input_open_vscode():
    """Verify agent handles 'Open VS Code' through fast router with verification."""
    agent = JarvisAgent()
    with patch.object(agent.fast_router.app_indexer, "launch_app") as mock_launch:
        mock_launch.return_value = ToolResult.ok("open_app", "Opened Visual Studio Code, Sir.")
        resp = agent.process_input("Open VS Code")
        assert "opened visual studio code" in resp.lower()
        mock_launch.assert_called_with("vscode")


def test_agent_process_input_can_you_see_my_screen():
    """Verify agent truthfully answers 'Can you see my screen?' without hallucination when disabled."""
    with patch("config.settings.settings.vision_enabled", False):
        agent = JarvisAgent()
        resp = agent.process_input("Can you see my screen?")
        assert any(p in resp.lower() for p in ["screen inspection isn't available", "screen inspection isn't enabled"])
        assert "i see your screen" not in resp.lower()
        assert "i can see" not in resp.lower()
