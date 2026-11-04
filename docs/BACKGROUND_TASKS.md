# Background Tasks and Scheduling

JARVIS has a durable local job queue backed by SQLite. Jobs survive restarts and
run while JARVIS (and the laptop) is powered on.

> **Honest limitation:** there is no cloud worker. Jobs do not run while the
> laptop is off, asleep, or JARVIS is stopped. On next start, overdue jobs run
> once and recurring jobs continue from that point.

---

## 1. Components

| File | Role |
|---|---|
| `core/automation/job_store.py` | SQLite persistence, CRUD, recurrence, retries, history |
| `core/automation/task_scheduler.py` | daemon poller + worker threads |
| `skills/scheduler_skill.py` | voice/text skill interface |
| `data/jobs.db` | database (jobs + job_history) |

The scheduler starts automatically in `main.py` unless `--no-scheduler` is passed.
It polls every 15 seconds and runs at most 3 jobs concurrently.

---

## 2. Job payload contract

A job must name an explicit skill and action. The scheduler refuses unknown
payloads — a language model cannot invent a background action.

```json
{
  "skill": "system",
  "action": "get_battery",
  "params": {}
}
```

Other examples:

```json
{"skill": "file", "action": "list_dir", "params": {"path": "Desktop"}}
{"skill": "memory", "action": "list_memories", "params": {}}
{"skill": "scheduler", "action": "list_jobs", "params": {}}
```

The handler dispatches through `SkillRegistry.execute(skill, action, params)`.
A failed skill raises and is retried with backoff.

---

## 3. Recurrence

| Value | Meaning |
|---|---|
| `null` / omitted | one-shot (runs once, then `done`) |
| `hourly` | every 3600 s after completion |
| `daily` | every 24 h after completion |
| `weekly` | every 7 days after completion |
| `interval_seconds:N` | every N seconds after completion |

Recurrence is scheduled **after** a successful run (`mark_done`), so long jobs
do not overlap themselves.

---

## 4. Retries and failure handling

- Default `max_retries = 3`.
- Retry backoff is exponential (`retry_backoff_seconds × 2^retry`, default 300 s).
- After retries are exhausted, status becomes `failed`.
- Every run/failure/cancellation is recorded in `job_history`.
- A failed job does not crash the scheduler or JARVIS.

---

## 5. High-risk jobs

Jobs can be created with:

```json
{"confirmation_required": true, "owner_approved": false}
```

The scheduler **skips** such jobs until the owner explicitly approves them.
This prevents an unknown speaker (or a misunderstood command) from scheduling a
destructive action that executes later without review.

Ordinary jobs (battery check, list files, memory listing) do not require
approval.

---

## 6. Creating jobs

### From voice/text

```
"schedule a task called morning check ..."
"list scheduled tasks"
"cancel job job_ab12cd34ef"
```

The scheduler skill requires an explicit `skill` field in the payload. If it is
missing, the job is refused.

### From Python (owner console)

```python
from core.automation.job_store import get_job_store

js = get_job_store()
jid = js.add_job(
    title="morning battery check",
    payload={"skill": "system", "action": "get_battery", "params": {}},
    recurrence="daily",
)
print(jid)
```

### From the scheduler skill

```python
from skills.registry import get_skill_registry

reg = get_skill_registry()
res = reg.execute("scheduler", "schedule_task", {
    "title": "battery check",
    "skill": "system",
    "skill_action": "get_battery",
    "recurrence": "interval_seconds:3600",
})
print(res.success, res.evidence)
```

---

## 7. Managing jobs

```python
from core.automation.job_store import get_job_store
js = get_job_store()

for job in js.list_jobs():
    print(job["id"], job["title"], job["status"], job["next_run_at"])

js.cancel("job_ab12cd34ef")
```

Statuses: `pending`, `running`, `done`, `failed`, `cancelled`.

---

## 8. Persistence and restart behavior

- `data/jobs.db` is a normal SQLite file; inspect it with any SQLite tool.
- On restart, pending jobs whose `next_run_at` has passed are picked up at the
  next poll.
- A job that was `running` when the process died remains `running`; it is not
  automatically re-queued (prevents duplicate side effects). Inspect and reset
  manually if needed:
  ```sql
  UPDATE jobs SET status='pending' WHERE status='running';
  ```
- Deleting `data/jobs.db` removes all jobs and history.

---

## 9. Limitations

- No execution while the laptop is off/asleep.
- No cloud queue or remote worker.
- No per-job OS-level sandbox; jobs run with the same permissions as JARVIS.
- Recurrence is interval-based (not full cron); use `interval_seconds:N` for
  custom cadences.
- Job results keep only the last 400 characters.

---

## 10. Tests

`tests/test_modernization_suite.py` covers:

- create/due/run/done/retry/cancel flows
- recurrence rescheduling
- registry dispatch
- refusal of payloads without an explicit skill
- skipping confirmation-required jobs until `owner_approved=true`

Run:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_modernization_suite.py -q
```
