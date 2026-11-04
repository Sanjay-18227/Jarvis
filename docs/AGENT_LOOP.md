# Agent Loop and Routing

JARVIS uses a layered routing strategy: fast deterministic skills first, a small
local language model for paraphrases, and an autonomous plan–act–observe–verify
loop for multi-step goals.

---

## 1. Turn pipeline (`JarvisAgent.process_input`)

| Step | What happens | Fail-safe |
|---|---|---|
| 0 | Emergency stop check | Stops immediately if triggered |
| 1 | Pending high-risk confirmation | Yes/no is matched to the pending action |
| 2 | Privacy guard + speaker access validation | `UNKNOWN` → no computer control |
| 2.5 | Fast deterministic router (`FastRouter`, `SkillRegistry`) | Sub-second local actions, greetings |
| 2.7 | Active-app context skill | Routes "type this", "save", etc. to the focused app |
| 2.8 | Local SLM intent augmentation (`core/local_slm.py`) | Rules first; optional Qwen3 JSON pass |
| 3 | Direct screen-vision intents | Structural fallback if no vision provider |
| 3.5 | Visual follow-up ("fix it") | Uses last observation; refuses if none |
| 4 | Autonomous planner / agent loop | Multi-step goals with authorization + verification |

Every branch records performance metrics and an audit entry.

---

## 2. Fast router / skills

`skills/registry.py` registers nine deterministic skills:

- calculator, paint, vscode, browser, notepad, file, system, memory, scheduler

The router recognizes intents such as `OPEN_APP`, `OPEN_BROWSER`,
`OPEN_WEBSITE`, `OPEN_FOLDER_IN_APP`, `CREATE_FOLDER`, `CREATE_FILE`,
`RUN_PYTHON`, `PAINT_ACTION`, `NOTEPAD_ACTION`, `BATTERY`, `DIAGNOSTICS`,
`MEDIA_CONTROL`, `CALCULATE`, plus memory/scheduler actions.

Skill execution returns a `ToolResult` with `success`, `message`, `evidence`
and an error when applicable. Nothing is reported as done without evidence.

---

## 3. Local SLM augmentation

`core/local_slm.py` first applies broad deterministic rules. Only when those
fail does it call the local Qwen3-0.6B model:

- Prompt asks for **one JSON object only**, with a fixed intent allowlist.
- `enable_thinking=False` (native Qwen3 non-thinking template) for latency.
- Output is parsed, confidence must be ≥ 0.80, and intent must be allowed.
- Folder/filename entities must appear in the user's words or context; invented
  entities are rejected.
- The model **never executes tools**; it only proposes a structured intent that
  the existing router validates and runs.
- If the model is missing, disabled, or RAM is insufficient, the neural path is
  skipped — deterministic routing continues.

Examples it handles well:

- "bro start my code editor" → `OPEN_APP` (vscode)
- "open Kumar in vscode" → `OPEN_FOLDER_IN_APP`
- "go to youtube" → `OPEN_WEBSITE`

If no rule or valid SLM output matches, the intent becomes `GENERAL_AI` and the
configured reasoning provider handles it — which is the offline mock provider
when `LOCAL_ONLY=true`.

---

## 4. Autonomous agent loop

`core/agent_loop.py` implements:

```
Goal → UNDERSTANDING → PLANNING → AUTHORIZING → EXECUTING
     → OBSERVING → VERIFYING → COMPLETED / RETRYING / FAILED
```

### Planning

- The planner produces a structured plan of at most **12 steps**.
- Only tools in `APPROVED_TOOL_ALLOWLIST` are accepted.
- Tool arguments are validated before execution.
- Invalid/unknown tools mark the plan invalid instead of guessing.

### Authorization

- `authorize_action()` re-checks the speaker and risk.
- `UNKNOWN`/`BORDERLINE` cannot run privileged steps.
- High-risk steps require explicit owner confirmation.

### Execution and observation

- One step at a time; each step returns a `StepObservation` with success,
  evidence and error.
- A failed step is retried at most **2 times**; the planner may replan once.
- Python steps capture real stdout/stderr and exit codes.
- App/window steps verify the window/process actually exists.

### Final verification

- Before declaring success, the loop verifies the world state (window present,
  file exists, command exit code, screen observation).
- Partial success is reported as partial; no invented completion.

---

## 5. Example

```
Goal: "open my Kumar project in VS Code, run agent.py, and check for errors"
Plan:
  1. open_app(vscode) + open_folder(Kumar)   → verify window title
  2. run_python(agent.py)                    → capture output/exit code
  3. inspect_screen(query="errors")          → structural/vision observation
  4. report actual result
```

If `agent.py` exits non-zero, JARVIS reports the actual traceback/error instead
of saying it succeeded.

---

## 6. Performance characteristics

| Path | Typical latency | Notes |
|---|---|---|
| Fast skill (battery, volume, open app) | tens of ms | no model load |
| STT (Whisper-tiny, 2 s audio) | ~0.6 s measured | CPU int8 |
| ECAPA verification | ~0.5 s warm | 192-d embedding |
| SLM intent (Qwen3-0.6B) | ~5.8 s generation + load | on demand, RAM-gated |
| Cloud planner | network-dependent | disabled by default |

The SLM is intentionally not on the hot path for common commands; deterministic
skills keep latency low and work offline.

---

## 7. Failure behavior

| Failure | Behavior |
|---|---|
| STT empty/noise | Turn is dropped; no action |
| Speaker unknown | Access denied; no computer control |
| ECAPA unavailable with profile | Fail closed; PIN recovery only |
| Model missing / low RAM | Deterministic rules continue; no crash |
| Planner invalid | Refuses to execute; reports why |
| Tool failure | Retry ≤ 2, then partial report |
| Vision unavailable | Structural observation; no fabricated description |

---

## 8. Tests

- `tests/test_modernization_suite.py` — model policy, registry, memory, jobs,
  scheduler, recovery PIN, fail-closed ECAPA.
- `tests/test_phase3_verification_suite.py` — multi-step loop and verification.
- `tests/test_agent_loop.py`, `tests/test_autonomous_agent.py` — planning,
  authorization, retries.
- `tests/test_fix_and_slm_suite.py` — local command routing and SLM fallbacks.

See `docs/TEST_RESULTS.md`.
