"""
JARVIS Memory Skill
Voice/text control over the local SQLite memory store:
  - remember <fact>            -> kind=fact
  - remember preference <..>   -> kind=preference
  - what do you remember       -> list
  - search memory for <term>   -> FTS search
  - forget <id> | forget <text>-> delete (refuses ambiguous bulk deletes)
  - clear temporary context    -> short-term only

Privacy policy: passwords, keys, tokens, PINs, card numbers are rejected.
"""

from typing import Dict, Any, List, Optional
from skills.base import BaseSkill
from core.tool_result import ToolResult
from memory.store import get_memory_store


class MemorySkill(BaseSkill):
    def __init__(self):
        self.store = get_memory_store()

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Local memory: remember, list, search, forget preferences and facts."

    @property
    def supported_actions(self) -> List[str]:
        return ["remember", "list_memories", "search_memory", "forget", "clear_temp"]

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        int_clean = intent.lower().strip()
        return int_clean in self.supported_actions or any(
            phrase in int_clean for phrase in ("remember", "what do you remember", "forget", "clear temporary"))

    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        act = (params.get("action") or action).lower().strip()
        text = (params.get("text") or params.get("content") or "").strip()

        if act in ("remember", "remember_preference"):
            kind = "preference" if ("preference" in text.lower() or act == "remember_preference") else "fact"
            content = text.replace("preference", "", 1).strip() if kind == "preference" else text
            mid = self.store.remember(content, kind=kind, source="voice")
            if mid is None:
                return ToolResult.fail(
                    action="remember",
                    error="Memory rejected: empty or sensitive content (credentials/passwords are never stored).",
                )
            return ToolResult.ok(action="remember", target=f"memory:{mid}",
                                 evidence=f"sqlite row id={mid} kind={kind}",
                                 message="I'll remember that, Sir.")

        if act in ("list_memories", "list"):
            items = self.store.list(limit=10)
            if not items:
                return ToolResult.ok(action="list_memories", target="memory", evidence="0 rows",
                                     message="I have no saved memories yet, Sir.")
            summary = "; ".join(f"{i['id']}. {i['content'][:80]}" for i in items[:8])
            return ToolResult.ok(action="list_memories", target="memory",
                                 evidence=f"{len(items)} rows in sqlite",
                                 message=f"Saved memories: {summary}")

        if act in ("search_memory", "search"):
            query = text or params.get("query", "")
            rows = self.store.search(query, limit=8)
            if not rows:
                return ToolResult.ok(action="search_memory", target="memory",
                                     evidence="FTS returned 0", message="Nothing matched, Sir.")
            summary = "; ".join(f"{r[0]}. {r[2][:80]}" for r in rows)
            return ToolResult.ok(action="search_memory", target="memory",
                                 evidence=f"FTS rows={len(rows)}",
                                 message=f"Matches: {summary}")

        if act in ("forget", "forget_memory"):
            mid = params.get("id")
            if mid is not None:
                n = self.store.forget(memory_id=int(mid))
            elif text:
                n = self.store.forget(content_matches=text)
            else:
                return ToolResult.fail(action="forget", error="Specify memory id or exact text; refusing bulk delete.")
            if not n:
                return ToolResult.fail(action="forget",
                                       error="No unique memory matched. Use 'list memories' then 'forget <id>'.")
            return ToolResult.ok(action="forget", target="memory", evidence=f"deleted={n}",
                                 message="Forgotten, Sir.")

        if act in ("clear_temp", "clear_temporary"):
            n = self.store.clear_temp()
            return ToolResult.ok(action="clear_temp", target="memory",
                                 evidence=f"deleted task_context rows={n}",
                                 message="Temporary context cleared, Sir.")

        return ToolResult.fail(action=act, error=f"Unsupported memory action '{act}'.")
