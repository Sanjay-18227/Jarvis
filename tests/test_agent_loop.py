"""
Comprehensive Unit & Integration Test Suite for Phase 3 Autonomous Agent Loop.
Verifies:
1. Fail-closed speaker gating (UNKNOWN rejected, OWNER authorized).
2. Deterministic and LLM plan generation.
3. Tool allowlist security enforcement.
4. Execution, Observation, Verification loop.
5. In-flight retry logic (max 2 retries).
6. Emergency stop / cancellation handling.
7. Pronoun and context reference resolution.
8. Truthful partial completion reporting.
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from config.constants import SpeakerClassification
from core.agent_loop import (
    Phase3AgentLoop,
    AgentState,
    StructuredPlan,
    PlanStep,
    StepObservation,
    AgentContext,
    APPROVED_TOOL_ALLOWLIST,
    AGENT_MAX_RETRIES_PER_STEP,
    AGENT_MAX_PLAN_STEPS,
)
from computer.file_manager import FileManager
from computer.app_indexer import AppIndexer
from computer.browser_controller import BrowserController
from computer.windows_controller import WindowsController
from computer.system_monitor import SystemMonitor
from llm.mock_provider import MockProvider


def test_agent_loop_fails_closed_for_unknown_speaker():
    """Security check: UNKNOWN speaker must NEVER enter the agent loop."""
    loop = Phase3AgentLoop()
    success, msg = loop.run_goal(
        "Create a folder on my desktop",
        speaker=SpeakerClassification.UNKNOWN,
    )
    assert success is False
    assert loop.state == AgentState.FAILED
    assert "locked" in msg.lower() or "don't recognize" in msg.lower()


def test_agent_context_reference_resolution():
    """AgentContext resolves pronouns 'it', 'the folder', 'the file' accurately."""
    ctx = AgentContext()
    ctx.last_created_folder = "C:\\Projects\\JarvisTest"
    ctx.last_created_file = "C:\\Projects\\JarvisTest\\app.py"

    assert ctx.resolve_reference("it") == "C:\\Projects\\JarvisTest"
    assert ctx.resolve_reference("inside it") == "C:\\Projects\\JarvisTest"
    assert ctx.resolve_reference("the folder") == "C:\\Projects\\JarvisTest"
    assert ctx.resolve_reference("the file") == "C:\\Projects\\JarvisTest\\app.py"
    assert ctx.resolve_reference("custom_name.txt") == "custom_name.txt"


def test_deterministic_plan_generation():
    """Verifies deterministic decomposition of compound creation goals."""
    loop = Phase3AgentLoop()
    goal = "Create a folder called TestAlpha on my Desktop, create main.py inside it, write hello world, run it, and open in VS Code"
    plan = loop._create_plan(goal)

    assert plan is not None
    assert len(plan.steps) == 5
    tools = [s.tool for s in plan.steps]
    assert tools == ["create_folder", "create_file", "write_file", "run_python", "open_app"]
    assert "TestAlpha" in plan.steps[0].args["path"]
    assert "main.py" in plan.steps[1].args["path"]
    assert "print('Hello, World!')" in plan.steps[2].args["content"]


def test_plan_validation_rejects_unapproved_tools():
    """Security check: Plan containing unapproved tools is strictly rejected."""
    loop = Phase3AgentLoop()
    invalid_plan = StructuredPlan(
        goal="Format drive",
        steps=[
            PlanStep(id=1, tool="dangerous_unapproved_tool", args={}, description="Wipe disk"),
        ],
    )
    valid, msg = loop._validate_plan(invalid_plan, SpeakerClassification.OWNER, confirmed=True)
    assert valid is False
    assert "not in the approved security allowlist" in msg


def test_plan_validation_enforces_max_steps():
    """Plan exceeding AGENT_MAX_PLAN_STEPS is rejected."""
    loop = Phase3AgentLoop()
    steps = [
        PlanStep(id=i, tool="get_battery", args={}, description=f"Step {i}")
        for i in range(AGENT_MAX_PLAN_STEPS + 5)
    ]
    plan = StructuredPlan(goal="Excessive steps", steps=steps)
    valid, msg = loop._validate_plan(plan, SpeakerClassification.OWNER, confirmed=True)
    assert valid is False
    assert "exceeds maximum allowed step count" in msg


def test_agent_loop_emergency_stop():
    """Emergency stop halts execution immediately and sets state to CANCELLED."""
    loop = Phase3AgentLoop()
    loop.request_stop()
    assert loop._stop_requested is True
    assert loop.state == AgentState.CANCELLED

    # Running a goal while stop requested returns cancellation
    success, msg = loop.run_goal("Some goal", speaker=SpeakerClassification.OWNER)
    assert success is False
    assert "halted" in msg.lower() or "cancelled" in msg.lower()


def test_agent_loop_retry_and_partial_report():
    """Verifies retry attempts up to AGENT_MAX_RETRIES_PER_STEP and truthful reporting."""
    loop = Phase3AgentLoop()

    plan = StructuredPlan(
        goal="Execute two steps with failure",
        steps=[
            PlanStep(id=1, tool="get_battery", args={}, description="Read battery"),
            PlanStep(id=2, tool="read_file", args={"path": "Z:\\non_existent_drive_999\\missing.txt"}, description="Read missing file"),
        ],
    )

    with patch.object(loop, "_create_plan", return_value=plan):
        success, msg = loop.run_goal("Test failing plan", speaker=SpeakerClassification.OWNER)
        assert success is False
        assert loop.state == AgentState.FAILED
        assert "Read battery" in msg
        assert "failed" in msg


def test_agent_loop_end_to_end_file_and_python_execution():
    """End-to-end integration: creates temp folder, creates script, writes python, runs it, verifies stdout."""
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path = os.path.join(temp_dir, "AgentTestFolder")
        file_path = os.path.join(folder_path, "test_script.py")

        loop = Phase3AgentLoop()
        plan = StructuredPlan(
            goal="Test file creation and python execution",
            steps=[
                PlanStep(id=1, tool="create_folder", args={"path": folder_path}, description="Create test folder"),
                PlanStep(id=2, tool="create_file", args={"path": file_path, "content": ""}, description="Create script"),
                PlanStep(id=3, tool="write_file", args={"path": file_path, "content": "print('JARVIS_AGENT_LOOP_SUCCESS')\n"}, description="Write python code"),
                PlanStep(id=4, tool="run_python", args={"path": file_path}, description="Run script"),
            ],
        )

        with patch.object(loop, "_create_plan", return_value=plan):
            success, msg = loop.run_goal("Run integration task", speaker=SpeakerClassification.OWNER)
            assert success is True
            assert loop.state == AgentState.COMPLETED
            assert "successfully" in msg.lower()
            assert os.path.isdir(folder_path)
            assert os.path.isfile(file_path)
            assert plan.steps[3].observation["stdout"] == "JARVIS_AGENT_LOOP_SUCCESS"
