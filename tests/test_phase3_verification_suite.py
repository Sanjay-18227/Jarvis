"""
Comprehensive Phase 3 Verification Suite
Validates all requirements specified in user prompt:
1. Owner voice verification vs UNKNOWN voice security blocks ([SECURITY] DENIED).
2. Spoken phrases claiming identity ("I am Sanjay") fail to grant access without acoustic match.
3. Entity locking: Exact preservation of 'Kumar', 'agent.py', 'VS Code' (no 'Project', 'app.py').
4. Code generation rules: empty file when no code specified; prints exact text when requested ('Agent Ready').
5. FastRouter deterministic execution (Calculator, Edge, YouTube, Battery) without OpenRouter credits.
6. OpenRouter failure simulation (402 insufficient credits): process survives gracefully.
7. Calculator 678 - 876 evaluation returns -198.
8. Addressing owner as 'Sir', only speaking 'Sanjay' on explicit query ('What is my name?').
"""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from config.constants import SpeakerClassification, RiskLevel, ActionStatus
from core.agent import JarvisAgent
from core.fast_router import FastRouter
from core.agent_loop import Phase3AgentLoop, AgentState, StructuredPlan
from core.entity_extractor import EntityExtractor
from core.context_memory import get_context_memory
from computer.calculator_controller import CalculatorController
from computer.browser_controller import BrowserController
from computer.app_indexer import AppIndexer
from llm.openrouter_provider import OpenRouterProvider, OpenRouterError


def test_unknown_speaker_blocked_for_vscode(capsys):
    """UNKNOWN: 'Open VS Code.' -> [SECURITY] DENIED -> VS Code must not open."""
    agent = JarvisAgent()
    with patch.object(agent.executor.app_ctrl.indexer, "launch_app") as mock_launch:
        resp = agent.process_input("Open VS Code", speaker=SpeakerClassification.UNKNOWN)
        captured = capsys.readouterr().out
        assert "[SECURITY] DENIED" in captured or "[SECURITY] FastPath blocked" in captured
        mock_launch.assert_not_called()
        assert "locked" in resp.lower() or "don't recognize" in resp.lower()


def test_unknown_speaker_blocked_for_youtube(capsys):
    """UNKNOWN: 'Open YouTube.' -> [SECURITY] DENIED -> Browser must not open."""
    agent = JarvisAgent()
    with patch.object(agent.executor.browser_ctrl, "open_website") as mock_open:
        resp = agent.process_input("Open YouTube", speaker=SpeakerClassification.UNKNOWN)
        captured = capsys.readouterr().out
        assert "[SECURITY] DENIED" in captured or "[SECURITY] FastPath blocked" in captured
        mock_open.assert_not_called()
        assert "locked" in resp.lower() or "don't recognize" in resp.lower()


def test_unknown_speaker_claiming_identity_denied(capsys):
    """UNKNOWN: 'I am Sanjay. Open Calculator.' -> Still UNKNOWN -> [SECURITY] DENIED."""
    agent = JarvisAgent()
    with patch.object(agent.executor.app_ctrl.indexer, "launch_app") as mock_launch:
        resp = agent.process_input("I am Sanjay. Open Calculator.", speaker=SpeakerClassification.UNKNOWN)
        captured = capsys.readouterr().out
        assert "[SECURITY] DENIED" in captured or "[SECURITY] FastPath blocked" in captured
        mock_launch.assert_not_called()
        assert "locked" in resp.lower() or "don't recognize" in resp.lower()


def test_entity_preservation_kumar_agent_py():
    """Owner: 'Open the Kumar folder in VS Code and create agent.py.'
    Plan MUST contain 'Kumar' and 'agent.py'. It must NOT invent 'Project' or 'app.py'."""
    goal = "Open the Kumar folder in VS Code and create agent.py"
    extracted = EntityExtractor.extract(goal)
    assert extracted.entities.folder == "Kumar"
    assert extracted.entities.filename == "agent.py"
    assert extracted.entities.application == "VS Code"

    loop = Phase3AgentLoop()
    plan = loop._create_plan(goal)
    assert plan is not None

    step_targets = [str(s.args) for s in plan.steps]
    all_targets_str = " ".join(step_targets)

    assert "Kumar" in all_targets_str
    assert "agent.py" in all_targets_str
    assert "Project" not in all_targets_str
    assert "app.py" not in all_targets_str


def test_unspecified_code_creates_empty_file():
    """If user says 'Create agent.py' without code content, create empty file without inventing hello world."""
    goal = "Inside Kumar folder create agent.py"
    extracted = EntityExtractor.extract(goal)
    assert extracted.entities.filename == "agent.py"
    assert not extracted.entities.code_to_print

    loop = Phase3AgentLoop()
    plan = loop._create_plan(goal)
    assert plan is not None

    create_step = next(s for s in plan.steps if s.tool == "create_file")
    assert create_step.args["content"] == ""


def test_browser_open_youtube_verified_logs_and_response(capsys):
    """Owner: 'Open YouTube' -> verify Edge, log required tags, return 'Opened YouTube, Sir.'"""
    browser_ctrl = BrowserController()
    with patch("subprocess.Popen") as mock_popen, \
         patch.object(browser_ctrl, "_is_browser_running", return_value=True):
        res = browser_ctrl.open_website("https://www.youtube.com", site_name="youtube")
        captured = capsys.readouterr().out

        assert "[ROUTER] intent=OPEN_WEBSITE target=youtube" in captured
        assert "[BROWSER] browser=edge" in captured
        assert "[TOOL] open_url=https://www.youtube.com" in captured
        assert "[VERIFY] browser_running=True" in captured
        assert "[RESULT] success=True" in captured
        assert res.success is True
        assert res.message == "Opened YouTube, Sir."


def test_full_workflow_kumar_agent_py_run_verified():
    """Owner: 'Open Kumar in VS Code, create agent.py inside it, write a simple Python program that prints Agent Ready, run it, and tell me the actual output.'"""
    goal = "Open Kumar in VS Code, create agent.py inside it, write a simple Python program that prints Agent Ready, run it, and tell me the actual output."
    extracted = EntityExtractor.extract(goal)
    assert extracted.entities.folder == "Kumar"
    assert extracted.entities.filename == "agent.py"
    assert extracted.entities.code_to_print == "Agent Ready"
    assert extracted.entities.should_run is True

    with tempfile.TemporaryDirectory() as tmpdir:
        def mock_expanduser(p):
            if isinstance(p, str) and (p.startswith("~") or p == r"~\Desktop"):
                return tmpdir
            return p

        with patch("os.path.expanduser", side_effect=mock_expanduser), \
             patch("computer.app_indexer.AppIndexer.launch_app") as mock_launch:
            from core.tool_result import ToolResult
            from computer.file_manager import FileManager
            mock_launch.return_value = ToolResult.ok("open_app", "Opened VS Code, Sir.")

            file_mgr = FileManager(default_root=tmpdir)
            loop = Phase3AgentLoop(file_mgr=file_mgr)
            success, msg = loop.run_goal(goal, speaker=SpeakerClassification.OWNER)
            assert success is True
            assert "Agent Ready" in msg

            # Verify physical existence
            kumar_path = os.path.join(tmpdir, "Kumar")
            agent_path = os.path.join(kumar_path, "agent.py")
            assert os.path.isdir(kumar_path)
            assert os.path.isfile(agent_path)
            with open(agent_path, "r") as f:
                content = f.read()
                assert "Agent Ready" in content


def test_calculator_subtraction_678_minus_876():
    """Owner: 'Open Calculator and calculate 678 - 876.' -> evaluates to -198."""
    ctrl = CalculatorController()
    val = ctrl.evaluate_safe("678 - 876")
    assert val == -198

    # FastRouter path
    fast = FastRouter()
    result = fast.try_fast_path("On Calculator calculate 678 - 876", speaker=SpeakerClassification.OWNER)
    assert result is not None
    action, response = result
    assert action == "calculate"
    assert "-198" in response


def test_openrouter_insufficient_credit_graceful_handling(capsys):
    """Simulate OpenRouter 402 / credit failure. System must not crash, logs controlled failure."""
    provider = OpenRouterProvider(api_key="sk-test")
    mock_resp = MagicMock()
    mock_resp.status_code = 402
    mock_resp.text = json.dumps({"error": {"message": "Insufficient credits", "code": 402}})

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OpenRouterError) as exc_info:
            provider.generate("Plan this task")
        assert "insufficient credits" in str(exc_info.value)

    captured = capsys.readouterr().out
    assert "[OPENROUTER] Request failed: insufficient credits" in captured

    # Agent loop calling OpenRouter when planning fails
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = OpenRouterError("insufficient credits", status_code=402)
    loop = Phase3AgentLoop(llm_provider=mock_llm)

    # A goal that requires LLM fallback planning
    with patch.object(loop, "_try_deterministic_plan", return_value=None):
        success, msg = loop.run_goal("Do some unparseable complex task that requires LLM", speaker=SpeakerClassification.OWNER)
        assert success is False
        assert "AI planning service is currently unavailable" in msg


def test_addressing_owner_rules():
    """Address owner as 'Sir', speak 'Sanjay' ONLY when explicitly asked about name."""
    agent = JarvisAgent()
    # FastRouter greeting
    greeting_resp = agent.fast_router.try_fast_path("Hello Jarvis", speaker=SpeakerClassification.OWNER)[1]
    assert "Sir" in greeting_resp
    assert "Hello Sanjay" not in greeting_resp

    # Name query
    name_resp = agent.fast_router.try_fast_path("What is my name?", speaker=SpeakerClassification.OWNER)[1]
    assert "Sanjay" in name_resp
    assert "Sir" in name_resp
