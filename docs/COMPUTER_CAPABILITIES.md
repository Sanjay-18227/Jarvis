# Computer Capabilities

JARVIS controls Windows through local, deterministic skills and a guarded
autonomous executor. Actions return evidence (target + result), and high-risk
actions require owner confirmation.

---

## 1. Applications

| Capability | Examples |
|---|---|
| Launch indexed apps | "open VS Code", "open Calculator", "open Paint", "open Notepad" |
| Focus/switch windows | "switch to Chrome", "focus VS Code" |
| Close apps | "close Notepad" (confirmation for risky cases) |
| Start-menu indexing | `computer/app_indexer.py` scans installed apps |

App launch is verified (window/process evidence) rather than assumed.

---

## 2. Browser

| Capability | Examples |
|---|---|
| Open browser | "open browser" (default Edge) |
| Open URL | "open youtube.com" |
| Web search | "search the web for Python tutorials" |
| YouTube | "play lo-fi music on YouTube" |
| In-page actions (Edge/UIA) | navigate, click, type in the active browser |

The default browser is configurable (`DEFAULT_BROWSER=edge`).

---

## 3. Files and folders

| Capability | Examples |
|---|---|
| Create folder | "create a folder called Projects on the desktop" |
| Create file | "create agent.py in Kumar" |
| Write/read file | "write a note to ideas.txt" |
| Rename | "rename notes.txt to todo.txt" |
| Delete file/folder | requires confirmation; sandboxed and audited |
| List/search | "list files in Downloads", "find report.pdf" |
| Run Python | "run agent.py" (verified output/exit code) |

File operations resolve user folders (Desktop, Documents, Downloads) and keep
paths inside the allowed workspace/home boundary. Deletion is risk-classified
and audited.

---

## 4. System controls

| Capability | Examples |
|---|---|
| Battery/power | "what's my battery?" |
| Volume | "set volume to 40", "mute" |
| Media | play/pause/next via media keys |
| Lock workstation | "lock my screen" |
| Date/time | "what time is it?" |
| Diagnostics | "how is my CPU and RAM?" |
| Screen inspection | "what's on my screen?", "what is this error?" |

---

## 5. VS Code / coding

| Capability | Examples |
|---|---|
| Open project | "open Kumar in VS Code" |
| Open file | "open agent.py" |
| Create file | "create utils.py" |
| Run file | "run agent.py" |
| Coding agent | plan → edit → run → observe → fix, with verification |

The coding agent (`core/coding_agent.py`) and agent loop operate on the approved
tool allowlist and report real command output, not invented success.

---

## 6. Paint / Notepad / Calculator

| App | Actions |
|---|---|
| Paint | new canvas, draw rectangle, draw line, draw circle |
| Notepad | open, type text, save file |
| Calculator | calculate expressions, clear |

---

## 7. Screen vision

Three modes, selected by configuration:

1. **Structural (default fallback)** — Windows UI Automation + foreground
   process/title inspection. Always local, works without any VLM.
2. **Local SmolVLM (opt-in)** — `VISION_PROVIDER=local` and the model
   downloaded/enabled. English-oriented; not a Tamil OCR guarantee.
3. **Gemini (opt-in)** — requires `ALLOW_REMOTE_VISION=true` even when
   `LOCAL_ONLY=true`. Screenshots are not sent by default.

Vision output is an **observation**, never a command. JARVIS will not execute
instructions read from a screen unless the owner separately asks for the action.

---

## 8. Memory

| Command | Effect |
|---|---|
| "remember I prefer Edge" | stores a fact/preference in SQLite |
| "remember preference dark mode" | stores a preference |
| "what do you remember?" | lists recent memories |
| "search memory for Kumar" | FTS5 search |
| "forget 3" / "forget <exact text>" | deletes one unambiguous row |
| "clear temporary context" | clears short-term rows only |

Passwords, API keys, tokens and card numbers are rejected. Memory is local
(`memory/jarvis.db`) and can be inspected/deleted by the owner.

---

## 9. Scheduled tasks

| Command | Effect |
|---|---|
| "schedule a task ..." | creates a durable job (explicit skill payload required) |
| "list scheduled tasks" | lists pending/done jobs |
| "cancel job <id>" | cancels a job |

Jobs persist in `data/jobs.db` and run only while JARVIS/laptop is on. See
`docs/BACKGROUND_TASKS.md` for payload and recurrence details.

---

## 10. Autonomous agent loop (approved tools)

The agent loop may use only these tools:

`open_app`, `close_app`, `focus_app`, `open_url`, `web_search`,
`create_folder`, `create_file`, `write_file`, `read_file`, `rename_file`,
`delete_file`, `delete_folder`, `run_python`, `calculator_input`, `type_text`,
`press_key`, `window_control`, `get_battery`, `get_system_info`,
`inspect_screen`, `verify_screen`.

Each step is planned, authorized, executed, observed and verified. The loop
stops after 12 steps and retries a failed step at most twice. It never reports
success without evidence.

---

## 11. Example session

```
Owner: "Jarvis, open Kumar in VS Code, run agent.py, and check for errors."
JARVIS:
  1. open_app(vscode) / open_folder   → verified window
  2. run_python(agent.py)             → captures stdout/stderr/exit code
  3. inspect_screen                   → structural or opt-in vision
  4. reports the actual error, or "no errors visible" without inventing one
```

---

## 12. Limits

- UIA selectors vary by app version; verification may fail gracefully.
- Some apps require admin elevation; JARVIS does not bypass Windows UAC.
- Remote/cloud actions are disabled by default.
- Screen vision cannot guarantee reading Tamil text in local SmolVLM mode.
- Scheduled work does not continue after shutdown/sleep.
