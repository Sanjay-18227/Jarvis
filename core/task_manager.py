"""
JARVIS Task Manager Subsystem
Tracks autonomous task lifecycle, step execution history, and cancellation states.
"""

import time
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


class TaskState(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskStep:
    step_number: int
    action_name: str
    parameters: Dict[str, Any]
    status: str  # SUCCESS | FAILED | PENDING
    observation: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AutonomousTask:
    task_id: str
    goal: str
    state: TaskState = TaskState.PENDING
    steps: List[TaskStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    verification_summary: str = ""


class TaskManager:
    """
    Manages active computer tasks and enforces emergency stop states.
    """

    def __init__(self):
        self._active_task: Optional[AutonomousTask] = None
        self._stop_requested: bool = False

    @property
    def has_active_task(self) -> bool:
        return self._active_task is not None and self._active_task.state in [
            TaskState.PENDING,
            TaskState.IN_PROGRESS,
            TaskState.VERIFYING,
        ]

    def create_task(self, goal: str) -> AutonomousTask:
        """Initializes a new autonomous goal."""
        task_id = f"task_{int(time.time() * 1000)}"
        self._active_task = AutonomousTask(task_id=task_id, goal=goal, state=TaskState.IN_PROGRESS)
        self._stop_requested = False
        return self._active_task

    def request_emergency_stop(self):
        """Immediately halts active task automation."""
        self._stop_requested = True
        if self._active_task:
            self._active_task.state = TaskState.CANCELLED
            self._active_task.completed_at = time.time()

    def is_stop_requested(self) -> bool:
        return self._stop_requested

    def record_step(
        self,
        action: str,
        parameters: Dict[str, Any],
        status: str,
        observation: str = "",
    ):
        """Records an action step and its observed result."""
        if not self._active_task:
            return
        step_num = len(self._active_task.steps) + 1
        self._active_task.steps.append(
            TaskStep(
                step_number=step_num,
                action_name=action,
                parameters=parameters,
                status=status,
                observation=observation,
            )
        )

    def mark_completed(self, verification_summary: str):
        """Marks active task as verified and completed."""
        if self._active_task:
            self._active_task.state = TaskState.COMPLETED
            self._active_task.completed_at = time.time()
            self._active_task.verification_summary = verification_summary

    def mark_failed(self, error_summary: str):
        """Marks active task as failed."""
        if self._active_task:
            self._active_task.state = TaskState.FAILED
            self._active_task.completed_at = time.time()
            self._active_task.verification_summary = error_summary
