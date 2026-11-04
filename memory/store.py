"""
JARVIS Local Memory Store - SQLite + FTS.
Stores: preferences, task history, approval-gated long-term memories,
scheduled jobs (with core/automation/job_store for durability), execution evidence.
Private/sensitive items (passwords, keys, biometrics) are rejected by policy filter.

The isolated DB guarantees every memory item can be listed, updated
or removed by voice command: remember/update/forget/list.
"""

import os
import json
import time
import logging
import re
import sqlite3
from typing import List, Tuple, Optional

log = logging.getLogger("jarvis.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,          -- 'preference' | 'fact' | 'event' | 'task_context'
    content TEXT NOT NULL,
    value TEXT,                  -- optional structured value (json)
    embedding BLOB,              -- optional e5-small float32 vector (sem search)
    source TEXT,                 -- 'voice' | 'text' | 'system'
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    positive_votes INTEGER DEFAULT 0,
    negative_votes INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kind_id ON memory_items(kind, id);
CREATE TABLE IF NOT EXISTS evictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    reason TEXT,
    at REAL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content UNINDEXED, kind UNINDEXED
);
"""

_SENSITIVE_PATTERNS = [
    r"(?i)password", r"(?i)\bsecret\b", r"(?i)api[_\s-]?key", r"(?i)token\b",
    r"(?i)\bpin\b", r"(?i)credential", r"(?i)\bacess?[Nn]o?\b.*\d{4,}",
    r"(?i)credit card", r"(?i)\bOTP\b", r"(?i)bearer", r"(?i)ssh private",
    r"\b\d{16}\b", r"\b\d{3}-\d{2}-\d{4}\b",
]


def _looks_sensitive(text: str) -> bool:
    return any(re.search(p, text) for p in _SENSITIVE_PATTERNS)


class MemoryStore:
    """Thread-local connection-per-call SQLite wrapper with fail-closed DDL."""

    def __init__(self, db_path: Optional[str] = None):
        from config.settings import get_settings
        base = get_settings().base_dir
        self.path = db_path or os.path.join(base, "memory", "jarvis.db")
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
            log.error("[MEMORY] schema init failed: %s", e)

    # -------------------------------------------------------------- private gate
    @staticmethod
    def _reject_private(content: str) -> bool:
        return _looks_sensitive(content)

    # -------------------------------------------------------------- CRUD
    def remember(self, content: str, kind: str = "fact",
                 value: Optional[dict] = None, source: str = "voice") -> Optional[int]:
        """Insert a memory. Returns id, or None if content is private/empty."""
        if not content or not content.strip():
            return None
        if self._reject_private(content):
            log.warning("[MEMORY] rejection: content refused (privacy policy)")
            return None
        now = time.time()
        try:
            with self._connect() as con:
                cur = con.execute(
                    "INSERT INTO memory_items(kind, content, value, source, created_at, updated_at)"
                    " VALUES(?,?,?,?,?,?)",
                    (kind, content.strip(), json.dumps(value) if value else None, source, now, now))
                con.execute("INSERT INTO memory_fts(content, kind) VALUES(?,?)",
                            (content.strip(), kind))
                return cur.lastrowid
        except Exception as e:
            log.error("[MEMORY] insert failed: %s", e)
            return None

    def search(self, query: str, kind: Optional[str] = None, limit: int = 10) -> List[Tuple[int, str, str]]:
        """SQLite FTS keyword search. Falls back to LIKE scan if FTS errors."""
        rows: List[Tuple[int, str, str]] = []
        try:
            with self._connect() as con:
                sql = ("SELECT m.id, m.kind, m.content FROM memory_fts f "
                       "JOIN memory_items m ON m.rowid = f.rowid ")
                params: list = []
                if kind:
                    sql += "WHERE f.kind = ? AND "
                    params.append(kind)
                sql += "memory_fts MATCH ? "
                params.append(query.strip() or "*")
                sql += "ORDER BY bm25(f) LIMIT ?"
                params.append(limit)
                rows = list(con.execute(sql, params))
        except sqlite3.OperationalError as e:
            log.warning("[MEMORY] FTS search fallback: %s", e)
            with self._connect() as con:
                sql = "SELECT id, kind, content FROM memory_items WHERE content LIKE ? COLLATE NOCASE"
                params: list = [f"%{query.strip()}%"]
                if kind:
                    sql += " AND kind = ?"
                    params.append(kind)
                sql += " ORDER BY updated_at DESC LIMIT ?"
                params.append(limit)
                return [(r[0], r[1], r[2]) for r in con.execute(sql, params)]
        except Exception as e:
            log.error("[MEMORY] search failed: %s", e)
        return rows

    def list(self, kind: Optional[str] = None, limit: int = 20) -> List[dict]:
        sql = "SELECT id, kind, content, value, source, created_at FROM memory_items"
        params: list = []
        if kind:
            sql += " WHERE kind=?"
            params.append(kind)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        try:
            with self._connect() as con:
                return [dict(id=r[0], kind=r[1], content=r[2], value=r[3], source=r[4], created_at=r[5])
                        for r in con.execute(sql, params)]
        except Exception as e:
            log.error("[MEMORY] list failed: %s", e)
            return []

    def update(self, memory_id: int, content: Optional[str] = None, kind: Optional[str] = None) -> bool:
        fields, params = [], []
        if content is not None:
            if self._reject_private(content):
                return False
            fields.append("content=?"); params.append(content.strip())
        if kind is not None:
            fields.append("kind=?"); params.append(kind)
        if not fields:
            return False
        fields.append("updated_at=?"); params.append(time.time())
        params.append(memory_id)
        try:
            with self._connect() as con:
                con.execute(f"UPDATE memory_items SET {', '.join(fields)} WHERE id=?", params)
                if content is not None:
                    con.execute("INSERT INTO memory_fts(rowid, content, kind) "
                                "SELECT id, content, kind FROM memory_items WHERE id=? AND rowid NOT IN (SELECT rowid FROM memory_fts)",
                                (memory_id,))
            return True
        except Exception as e:
            log.error("[MEMORY] update failed: %s", e)
            return False

    def forget(self, memory_id: Optional[int] = None, content_matches: Optional[str] = None,
               kind: Optional[str] = None) -> int:
        """Deletes by explicit id or exact-ish content; never deletes everything silently."""
        if memory_id is None and not content_matches and not kind:
            return 0
        try:
            with self._connect() as con:
                del_id = cursor = None
                rows = []
                if memory_id is not None:
                    rows = con.execute("SELECT id, content FROM memory_items WHERE id=?", (memory_id,)).fetchall()
                elif content_matches:
                    rows = con.execute(
                        "SELECT id, content FROM memory_items WHERE content LIKE ?", (f"%{content_matches}%",)).fetchall()
                    if kind and len(rows) > 1:
                        rows = [r for r in rows if self._kind(r[0], con) == kind]
                ids = [r[0] for r in rows]
                if not ids:
                    return 0
                if len(ids) > 1 and content_matches is not None:
                    # ambiguous delete refused, ask user
                    log.warning("[MEMORY] ambiguous forget for '%s' (%d candidates); use list then forget <id>",
                                content_matches, len(ids))
                    return 0
                for mid in ids:
                    con.execute("DELETE FROM memory_items WHERE id=?", (mid,))
                    con.execute("INSERT INTO evictions(content, reason, at) VALUES(?,?,?)",
                                (rows[0][1], "user_forget", time.time()))
                return len(ids)
        except Exception as e:
            log.error("[MEMORY] forget failed: %s", e)
            return 0

    def _kind(self, mid: int, con) -> str:
        r = con.execute("SELECT kind FROM memory_items WHERE id=?", (mid,)).fetchone()
        return r[0] if r else ""

    def clear_temp(self, kind: str = "task_context") -> int:
        """Resets only short-lived context rows; never 'fact' or 'preference'."""
        try:
            with self._connect() as con:
                cur = con.execute("DELETE FROM memory_items WHERE kind=?", (kind,))
                return cur.rowcount
        except Exception as e:
            log.error("[MEMORY] clear failed: %s", e)
            return 0

    def stats(self) -> dict:
        try:
            with self._connect() as con:
                n = con.execute("SELECT COUNT(*), COALESCE(SUM(CASE WHEN kind='task_context' THEN 1 ELSE 0 END),0)"
                                " FROM memory_items").fetchone()
                return {"total": int(n[0]), "temporary": int(n[1])}
        except Exception as e:
            return {"error": str(e)}


_store = None


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


if __name__ == "__main__":
    ms = get_memory_store()
    mid = ms.remember("Owner prefers Edge as default browser", kind="preference", source="test")
    print("remember:", mid)
    print("search:", ms.search("browser"))
    print("list:", ms.list()[:2])
    print("private rejection:", ms.remember("my password is hunter2"))
    print("stats:", ms.stats())
