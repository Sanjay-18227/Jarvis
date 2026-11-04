"""
JARVIS Durable Job Store - SQLite-backed task queue for scheduled / recurring jobs.
Supports: one-shot, recurring (cron-lite: daily/hourly/weekly intervals), retry policy,
history, cancellation, and owner confirmation records. WAL mode for concurrent access.
"""

import os
import json
import time
import uuid
import logging
import sqlite3
from typing import Optional, List

log = logging.getLogger("jarvis.jobs")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    payload_json TEXT NOT NULL,           -- {"action": "...", "args": {...}}
    recurrence TEXT,                       -- NULL | 'daily' | 'weekly' | 'hourly' | interval_seconds:N
    next_run_at REAL NOT NULL,
    last_run_at REAL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending|running|done|failed|cancelled
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    retry_backoff_seconds INTEGER DEFAULT 300,
    owner_approved INTEGER DEFAULT 0,
    confirmation_required INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS job_history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    started_at REAL, ended_at REAL,
    status TEXT, result TEXT, error TEXT
);
"""


class JobStore:
    """Durable task queue. Jobs survive process restarts (Next_run_at vs now)."""

    def __init__(self, db_path: Optional[str] = None):
        from config.settings import get_settings
        base = get_settings().base_dir
        self.path = db_path or os.path.join(base, "data", "jobs.db")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA busy_timeout=5000;")
        return con

    def _ensure_schema(self):
        try:
            with self._connect() as con:
                con.executescript(_SCHEMA)
        except Exception as e:
            log.error("[SCHEDULER] job store schema failed: %s", e)

    # ------------------------------------------------------------- create
    def add_job(self, title: str, payload: dict, recurrence: Optional[str] = None,
                next_run_at: Optional[float] = None, owner_approved: bool = False,
                confirmation_required: bool = False, max_retries: int = 3) -> str:
        jid = f"job_{uuid.uuid4().hex[:10]}"
        now = time.time()
        try:
            with self._connect() as con:
                con.execute(
                    "INSERT INTO jobs(id,title,payload_json,recurrence,next_run_at,owner_approved,"
                    "confirmation_required,max_retries,created_at,updated_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (jid, title, json.dumps(payload), recurrence,
                     next_run_at if next_run_at is not None else now,
                     int(owner_approved), int(confirmation_required), max_retries, now, now))
            print(f"[SCHEDULER] job added: {jid} '{title}' recurrence={recurrence}")
            return jid
        except Exception as e:
            log.error("[SCHEDULER] add_job failed: %s", e)
            return ""

    # ------------------------------------------------------------- query
    def due_jobs(self) -> List[dict]:
        now = time.time()
        try:
            with self._connect() as con:
                rows = con.execute(
                    "SELECT id,title,payload_json,recurrence,next_run_at,max_retries,retry_count,"
                    "owner_approved,confirmation_required FROM jobs"
                    " WHERE status='pending' AND next_run_at<=?", (now,)).fetchall()
            out = []
            for r in rows:
                out.append({
                    "id": r[0], "title": r[1], "payload": json.loads(r[2] or "{}"),
                    "recurrence": r[3], "next_run_at": r[4], "max_retries": r[5],
                    "retry_count": r[6], "owner_approved": bool(r[7]),
                    "confirmation_required": bool(r[8]),
                })
            return out
        except Exception as e:
            log.error("[SCHEDULER] due_jobs failed: %s", e)
            return []

    def list_jobs(self, status: Optional[str] = None, limit: int = 30) -> List[dict]:
        sql = "SELECT id,title,status,recurrence,next_run_at,last_run_at,retry_count FROM jobs"
        params: list = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY next_run_at DESC LIMIT ?"
        params.append(limit)
        try:
            with self._connect() as con:
                return [dict(id=r[0], title=r[1], status=r[2], recurrence=r[3],
                             next_run_at=r[4], last_run_at=r[5], retries=r[6])
                        for r in con.execute(sql, params)]
        except Exception as e:
            log.error("[SCHEDULER] list_jobs failed: %s", e)
            return []

    # ------------------------------------------------------------- lifecycle
    def mark_running(self, job_id: str) -> bool:
        with self._connect() as con:
            con.execute("UPDATE jobs SET status='running', last_run_at=?, updated_at=? WHERE id=?",
                        (time.time(), time.time(), job_id))
        return True

    def mark_done(self, job_id: str, result: str = "") -> bool:
        now = time.time()
        try:
            with self._connect() as con:
                row = con.execute("SELECT recurrence FROM jobs WHERE id=?", (job_id,)).fetchone()
                rec = row[0] if row else None
                if rec:
                    # recurring -> schedule next occurrence
                    next_at = self._next_run_after(rec, now)
                    con.execute("UPDATE jobs SET status='pending', next_run_at=?, updated_at=?, retry_count=0 WHERE id=?",
                                (next_at, now, job_id))
                else:
                    con.execute("UPDATE jobs SET status='done', updated_at=? WHERE id=?", (now, job_id))
                con.execute("INSERT INTO job_history(job_id, started_at, ended_at, status, result)"
                            " VALUES(?,?,?,?,?)", (job_id, now - 1, now, "done", result))
            return True
        except Exception as e:
            log.error("[SCHEDULER] mark_done failed: %s", e)
            return False

    def mark_failed(self, job_id: str, error: str) -> bool:
        now = time.time()
        try:
            with self._connect() as con:
                row = con.execute("SELECT retry_count, max_retries, retry_backoff_seconds FROM jobs WHERE id=?",
                                  (job_id,)).fetchone()
                if not row:
                    return False
                retry, maxr, backoff = row
                if retry + 1 <= maxr:
                    next_at = now + backoff * (2 ** retry)
                    con.execute("UPDATE jobs SET status='pending', retry_count=?, next_run_at=?, updated_at=? WHERE id=?",
                                (retry + 1, next_at, now, job_id))
                else:
                    con.execute("UPDATE jobs SET status='failed', updated_at=? WHERE id=?", (now, job_id))
                con.execute("INSERT INTO job_history(job_id, started_at, ended_at, status, error)"
                            " VALUES(?,?,?,?,?)", (job_id, now - 1, now, "failed", error))
            return True
        except Exception as e:
            log.error("[SCHEDULER] mark_failed failed: %s", e)
            return False

    def cancel(self, job_id: str) -> bool:
        try:
            with self._connect() as con:
                cur = con.execute("UPDATE jobs SET status='cancelled', updated_at=? WHERE id=? AND status IN ('pending','running')",
                                  (time.time(), job_id))
                if cur.rowcount:
                    con.execute("INSERT INTO job_history(job_id, ended_at, status) VALUES(?,?,'cancelled')",
                                (job_id, time.time()))
                return bool(cur.rowcount)
        except Exception as e:
            log.error("[SCHEDULER] cancel failed: %s", e)
            return False

    @staticmethod
    def _next_run_after(recurrence: str, now: float) -> float:
        if recurrence.startswith("interval_seconds:"):
            try:
                return now + float(recurrence.split(":", 1)[1])
            except Exception:
                return now + 3600
        if recurrence == "daily":
            return now + 24 * 3600
        if recurrence == "hourly":
            return now + 3600
        if recurrence == "weekly":
            return now + 7 * 24 * 3600
        return now + 24 * 3600


_store = None


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store


if __name__ == "__main__":
    js = get_job_store()
    jid = js.add_job("demo daily report", {"action": "report"}, recurrence="daily")
    print("jobs:", js.list_jobs())
    print("due:", [j["id"] for j in js.due_jobs()])
    js.mark_running(jid)
    js.mark_done(jid, "demo result")
    print("after:", [j["status"] for j in js.list_jobs()])
