"""
JARVIS Background Task Scheduler
Runs a daemon thread that occasionally pops due jobs from the durable JobStore,
dispatches them to the local CapabilityRegistry / AgentLoop, and records history.

HONEST LIMITATION (documented in docs/BACKGROUND_TASKS.md):
 - This scheduler only runs while JARVIS (or this laptop) is powered ON.
 - Scheduled tasks while the laptop is OFF require an external worker (OAuth/cloud),
   which is NOT automatically assumed (user must explicitly configure it).
"""

import threading
import time
import logging
from typing import Callable, Dict, Any, Optional

from core.automation.job_store import get_job_store

log = logging.getLogger("jarvis.scheduler")


class TaskScheduler:
    """
    Polls due jobs every `poll_seconds`. Each job runs in its own worker thread so
    a slow job never blocks future dispatch. Bounded by max_concurrent workers.
    """

    def __init__(self, poll_seconds: float = 15.0, max_concurrent: int = 3, handler=None):
        self.store = get_job_store()
        self.poll_seconds = poll_seconds
        self.max_concurrent = max_concurrent
        self._handler = handler or self._default_handler
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="jarvis-scheduler", daemon=True)
        self._active = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def start(self) -> bool:
        if self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread.start()
        print("[SCHEDULER] background scheduler started")
        return True

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                log.error("[SCHEDULER] tick error: %s", e)
            self._stop.wait(self.poll_seconds)

    def _tick(self) -> None:
        if len(self._active) >= self.max_concurrent:
            return
        for job in self.store.due_jobs():
            if len(self._active) >= self.max_concurrent:
                break
            # High-risk/confirmation-required jobs only run after explicit owner approval.
            if job.get("confirmation_required") and not job.get("owner_approved"):
                log.warning("[SCHEDULER] job %s requires owner approval; skipping", job["id"])
                continue
            with self._lock:
                if job["id"] in self._active:
                    continue
                self._active.add(job["id"])
            self.store.mark_running(job["id"])
            thread = threading.Thread(target=self._run_job, args=(job,), daemon=True,
                                      name=f"job-{job['id']}")
            thread.start()

    def _run_job(self, job: dict) -> None:
        try:
            result = self._handler(job)
            self.store.mark_done(job["id"], str(result)[:400])
        except Exception as e:
            log.error("[SCHEDULER] job %s failed: %s", job["id"], e)
            self.store.mark_failed(job["id"], str(e)[:400])
        finally:
            with self._lock:
                self._active.discard(job["id"])

    # ------------------------------------------------------------------
    @staticmethod
    def _default_handler(job: dict) -> Any:
        """
        Local task dispatch. Payload contract:
          {"skill": "<registry skill name>", "action": "<action>", "params": {...}}
        Jobs without an explicit skill are refused (no model-invented actions).
        """
        payload = job.get("payload", {})
        skill_name = payload.get("skill")
        action = payload.get("action") or skill_name
        params = payload.get("params", {}) or {}
        if not skill_name:
            raise ValueError("job payload has no explicit 'skill'; refusing to run")
        from skills.registry import get_skill_registry
        reg = get_skill_registry()
        res = reg.execute(skill_name, action, params)
        if getattr(res, "success", False):
            return getattr(res, "message", None) or getattr(res, "evidence", "ok")
        raise RuntimeError(getattr(res, "error", "skill execution failed"))


_scheduler = None


def get_scheduler(poll_seconds: float = 15.0, max_concurrent: int = 3, handler=None) -> TaskScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler(poll_seconds=poll_seconds, max_concurrent=max_concurrent, handler=handler)
    return _scheduler
