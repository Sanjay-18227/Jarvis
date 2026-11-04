"""
JARVIS Scheduler Skill
Local durable scheduled tasks (only while the laptop is ON; see docs/BACKGROUND_TASKS.md).
  - schedule task <title> at <epoch or now> repeating daily
  - list scheduled tasks
  - cancel job <id>

Jobs dispatch through the SkillRegistry by explicit payload:
  {"skill": "<name>", "action": "<action>", "params": {...}}
"""

import time
from typing import Dict, Any, List, Optional
from skills.base import BaseSkill
from core.tool_result import ToolResult
from core.automation.job_store import get_job_store


class SchedulerSkill(BaseSkill):
    def __init__(self):
        self.store = get_job_store()

    @property
    def name(self) -> str:
        return "scheduler"

    @property
    def description(self) -> str:
        return "Durable local job scheduler (daily/weekly/interval tasks while JARVIS runs)."

    @property
    def supported_actions(self) -> List[str]:
        return ["schedule_task", "list_jobs", "cancel_job"]

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        int_clean = intent.lower().strip()
        return int_clean in self.supported_actions or "schedul" in int_clean

    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        act = (params.get("action") or action).lower().strip()

        if act in ("schedule_task", "schedule"):
            title = params.get("title") or params.get("text") or "scheduled task"
            payload = {
                "skill": params.get("skill"),
                "action": params.get("skill_action"),
                "params": params.get("params", {}),
            }
            if not payload["skill"]:
                return ToolResult.fail(action="schedule_task",
                                       error="Missing 'skill' in schedule payload; refusing to schedule an unknown action.")
            recurrence = params.get("recurrence")  # e.g. "daily", "weekly", "interval_seconds:600"
            next_at = params.get("next_run_at")
            jid = self.store.add_job(title=title, payload=payload, recurrence=recurrence,
                                     next_run_at=float(next_at) if next_at else None,
                                     owner_approved=bool(params.get("owner_approved", False)),
                                     confirmation_required=bool(params.get("confirmation_required", False)))
            if not jid:
                return ToolResult.fail(action="schedule_task", error="job store rejected the job")
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(
                float(next_at))) if next_at else "immediately"
            return ToolResult.ok(action="schedule_task", target=jid,
                                 evidence=f"sqlite job id={jid}, next_run={when}, recurrence={recurrence}",
                                 message=f"Scheduled: {title} ({jid}), Sir.")

        if act in ("list_jobs", "list"):
            jobs = self.store.list_jobs(limit=10)
            if not jobs:
                return ToolResult.ok(action="list_jobs", target="jobs", evidence="0 rows",
                                     message="No scheduled jobs, Sir.")
            summary = "; ".join(f"{j['id']} {j['title']} [{j['status']}]" for j in jobs[:8])
            return ToolResult.ok(action="list_jobs", target="jobs",
                                 evidence=f"sqlite jobs={len(jobs)}", message=f"Jobs: {summary}")

        if act in ("cancel_job", "cancel"):
            jid = params.get("id") or params.get("job_id")
            if not jid:
                return ToolResult.fail(action="cancel_job", error="Job id required.")
            ok = self.store.cancel(str(jid))
            if not ok:
                return ToolResult.fail(action="cancel_job", error=f"Job {jid} not found or not cancellable.")
            return ToolResult.ok(action="cancel_job", target=str(jid), evidence=f"sqlite status=cancelled",
                                 message="Job cancelled, Sir.")

        return ToolResult.fail(action=act, error=f"Unsupported scheduler action '{act}'.")
