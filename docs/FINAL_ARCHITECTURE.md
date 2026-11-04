# JARVIS — Final Architecture

**Owner:** Sanjay · **Assistant name:** JARVIS · **Address:** "Sir"
**Platform:** Windows 10/11, CPU-first (no CUDA required)
**Status:** local-first Hugging Face + PyTorch stack integrated on top of the existing assistant

---

## 1. Design principles

1. **Local first.** Voice, speaker verification, speech-to-text, intent routing, memory and scheduling run on this laptop. Cloud reasoning/vision are optional and disabled by default (`LOCAL_ONLY=true`).
2. **Preserve working features.** The original computer-control stack (apps, browser, files, VS Code, screen inspection, security gate) is unchanged; new layers plug into it.
3. **One heavy model at a time.** The `ModelManager` enforces a single resident heavy model and unloads idle models.
4. **Fail closed.** If the enrolled speaker model cannot run, voice authentication does not silently fall back to a weaker method. The owner uses a local recovery PIN.
5. **No hidden downloads at command time.** Models are acquired explicitly with `download_models.py`; the registry pins a verified Hugging Face revision for each model.
6. **Evidence, not claims.** Actions return `ToolResult` objects with target/evidence; screen observations are labeled as observations, never commands.

---

## 2. Component map

```
┌──────────────────────────── INPUT ────────────────────────────┐
│  Microphone (sounddevice)   │   Keyboard / --text mode         │
│  Silero VAD + RMS gate      │                                  │
└───────────────┬───────────────────────────────┬───────────────┘
                │                               │
        ┌───────▼────────┐              ┌───────▼────────┐
        │  Faster-Whisper│              │  Direct text   │
        │  local STT     │              │  input         │
        └───────┬────────┘              └───────┬────────┘
                └───────────────┬───────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Speaker verification  │
                    │  ECAPA 192-d + profile │
                    │  FAIL CLOSED           │
                    └───────────┬────────────┘
                                │ OWNER / BORDERLINE / UNKNOWN
                    ┌───────────▼────────────┐
                    │  Security Gate         │
                    │  risk levels, confirm, │
                    │  privacy filter, audit │
                    └───────────┬────────────┘
                                │
        ┌───────────────────────▼─────────────────────────┐
        │                JarvisAgent                      │
        │  1. Confirmation / privacy pre-checks           │
        │  2. Fast deterministic router (skills)          │
        │  3. Direct screen-vision intents                │
        │  4. Local SLM intent augmentation (on demand)   │
        │  5. Autonomous planner / agent loop             │
        └───────┬───────────────────┬─────────────────────┘
                │                   │
     ┌──────────▼─────────┐  ┌──────▼──────────────────────┐
     │ SkillRegistry       │  │ Planner + ToolCatalog       │
     │ calculator, paint,  │  │ open/close app, type, click,│
     │ vscode, browser,    │  │ files, shell, vision, run   │
     │ notepad, file,      │  │ Python, verify results      │
     │ system, memory,     │  └──────┬──────────────────────┘
     │ scheduler           │         │
     └──────────┬──────────┘         │
                │                    │
        ┌───────▼────────────────────▼────────┐
        │           ActionExecutor            │
        │ app/browser/file/windows/UI/shell   │
        └───────┬─────────────────────────────┘
                │
     ┌──────────▼──────────┐   ┌──────────────────────────┐
     │ Vision              │   │ Memory + Scheduler       │
     │ Gemini (opt-in) or  │   │ SQLite FTS5, durable jobs│
     │ SmolVLM (opt-in) or │   │ background daemon thread │
     │ structural fallback │   │ (laptop must be ON)      │
     └─────────────────────┘   └──────────────────────────┘
```

---

## 3. Request lifecycle

1. **Capture.** `voice/recorder.py` records audio with RMS gating and Silero VAD confirmation (`voice/vad.py`). Silence/noise is rejected before transcription.
2. **Transcribe.** `voice/stt.py` uses local Faster-Whisper (`whisper-tiny` by default, `whisper-base` on demand). Vocabulary correction fixes common misrecognitions ("vs code" → VS Code, "my craft" → Microsoft).
3. **Authenticate.** `voice/speaker_verification.py` extracts a 192-d ECAPA embedding when ECAPA is enrolled; otherwise the legacy 42-d MFCC profile is used. Classification is `OWNER`, `BORDERLINE`, or `UNKNOWN`.
4. **Authorize.** `security/authorization_gate.py` applies risk levels 0–4. UNKNOWN callers get zero computer control. High-risk actions require fresh owner confirmation.
5. **Route.** Deterministic `FastRouter` and `SkillRegistry` handle known commands in milliseconds. Unmatched phrasing goes to `core/local_slm.py`, which combines rules with an optional Qwen3-0.6B JSON intent pass.
6. **Plan/execute.** `core/agent.py` hands general goals to the autonomous planner/agent loop, which produces a plan, executes tools through `ActionExecutor`, observes results, and verifies outcomes.
7. **Observe.** Screen questions go through `vision/screen_vision_service.py`: opt-in local SmolVLM, opt-in Gemini (blocked under `LOCAL_ONLY`), or structural UI Automation/window inspection.
8. **Remember/schedule.** `memory/store.py` stores durable facts/preferences in SQLite FTS5; `core/automation/job_store.py` + `task_scheduler.py` run durable recurring jobs while JARVIS is running.
9. **Respond.** Edge TTS (network) or pyttsx3 (offline) speaks the result; every action is written to the redacted audit log.

---

## 4. Model lifecycle (Hugging Face + PyTorch)

`models/model_registry.yaml` is the single source of truth. Every entry declares:

- `model_id`, `hf_repo`, `task`, `license`
- pinned `model_revision` (verified commit SHA)
- `expected_disk_size_mb`, `estimated_ram_mb`
- `runtime`, quantization, hardware requirements
- `enabled` (default on/off) and `load_policy`
- **explicit `loader_method`** — a method name on `ModelManager`; no fragile model-id string replacement

`core/models/model_manager.py` provides:

- `statuses()`, `entry()`, `is_downloaded()`
- `download(model_id)` with 2× disk-space guard and 5-minute failure backoff
- `get_runtime(model_id)` with live RAM policy checks and one-heavy-model concurrency
- `generate_text()` for Qwen3 (non-thinking chat template, fp32 on CPU)
- `unload()` / idle eviction (`MODEL_UNLOAD_IDLE_SECONDS`, default 300 s)

### RAM policy (CPU laptop)

| Available RAM | SLM | Whisper | VLM | ECAPA |
|---|---|---|---|---|
| < 250 MB | off | off | off | **off** (fail closed → PIN) |
| 250–699 MB | off | off | off | on |
| 700–1499 MB | off | on | off | on |
| 1500–2199 MB | off | on | off | on |
| ≥ 2200 MB | on demand | on | on demand | on |

ECAPA is security-critical and is **not** permanently disabled by normal memory pressure. It is only denied below ~250 MB free, where the process cannot run reliably; in that case voice auth fails closed and the owner uses `--auth-pin`.

### Measured model cost on the target laptop

See `docs/TEST_RESULTS.md` for exact runs. Summary:

- Qwen3-0.6B fp32: ~2.5 GB RSS, ~5.8 s per short JSON generation, load ~11 s when memory is healthy.
- int8 dynamic quantization: ~13.7 s per generation — **slower**, so not used.
- ECAPA: ~1 s load, ~0.6 s per embedding.
- Whisper-tiny: smallest practical CPU ASR; whisper-base available on demand.
- SmolVLM and indic-parler TTS remain disabled by default on 8 GB hardware.

---

## 5. Storage layout

| Path | Contents | Sensitivity |
|---|---|---|
| `data/speaker_profiles/owner_profile.dat` | legacy 42-d MFCC owner profile | biometric — never commit/share |
| `data/speaker_profiles/owner_profile_ecapa.json` | ECAPA 192-d centroid + representatives | biometric — never commit/share |
| `data/speaker_profiles/backups/` | timestamped owner-profile backups | biometric |
| `memory/jarvis.db` | durable facts/preferences, FTS5 index | personal |
| `data/jobs.db` | durable scheduled jobs + history | personal |
| `security/recovery.json` (or configured path) | salted PBKDF2 PIN hash + lockout state | secret |
| `logs/` | redacted audit + runtime logs | operational |
| `models/` | Hugging Face snapshots (pinned revisions) | large, re-downloadable |

---

## 6. Concurrency and resource rules

- Only **one heavy model** (≥ 500 MB estimated) may be resident. Requesting a second returns `None` rather than thrashing RAM.
- The scheduler runs a daemon thread; each job runs in its own worker thread (max 3 concurrent).
- The model manager uses a re-entrant lock; load/unload is serialized.
- Voice capture, TTS and the agent loop remain on the existing single-session model.

---

## 7. What is local vs. optional

| Capability | Default | Local? |
|---|---|---|
| Wake word / VAD | on | VAD local; wake-word engine depends on installed model |
| Speech-to-text | Faster-Whisper local | ✅ |
| Speaker verification | ECAPA / MFCC local | ✅ |
| Intent rules + SLM | Qwen3-0.6B local, on demand | ✅ |
| Computer control skills | local Win32/UIA | ✅ |
| Screen inspection | structural fallback | ✅ |
| Screen description (VLM) | Gemini opt-in / SmolVLM opt-in | local or remote |
| Open-ended reasoning | cloud opt-in; local-only falls back to mock/SLM routing | configurable |
| TTS | Edge TTS (network) default; pyttsx3 offline available | configurable |
| Memory / scheduler | SQLite local | ✅ |

---

## 8. Known architectural limitations

- Background jobs only execute while the laptop and JARVIS are running; there is no cloud worker.
- Fully local open-ended conversation is limited by the 0.6B SLM; deterministic skills are the reliable path.
- Local vision (SmolVLM) is English-oriented and not a substitute for Tamil UI OCR.
- Edge TTS requires network; set `TTS_ENGINE=pyttsx3` for a fully offline voice loop.
- LiveKit is an optional cloud transport; `VOICE_PROVIDER=legacy` keeps the microphone local.

---

## 9. Source map

| Area | Files |
|---|---|
| Entry point / CLI | `main.py`, `run_jarvis.bat`, `setup_jarvis.ps1` |
| Agent orchestration | `core/agent.py`, `core/fast_router.py`, `core/local_slm.py`, `core/agent_loop.py`, `core/planner.py` |
| Models | `models/model_registry.yaml`, `core/models/model_manager.py`, `core/system/hardware_detector.py` |
| Voice | `voice/recorder.py`, `voice/vad.py`, `voice/stt.py`, `voice/ecapa_engine.py`, `voice/speaker_verification.py`, `voice/voice_enrollment.py` |
| Security | `security/authorization_gate.py`, `security/recovery.py`, `security/audit_log.py`, `security/privacy_guard.py` |
| Computer control | `computer/`, `core/executor.py`, `skills/` |
| Vision | `vision/screen_capture.py`, `vision/screen_vision_service.py`, `vision/local_vlm.py` |
| Memory / automation | `memory/store.py`, `core/automation/job_store.py`, `core/automation/task_scheduler.py`, `skills/memory_skill.py`, `skills/scheduler_skill.py` |
| Diagnostics | `doctor.py`, `download_models.py`, `benchmark_models.py`, `scripts/mic_test.py` |
| Tests | `tests/` (see `docs/TEST_RESULTS.md`) |

Related docs: [AUDIT](CURRENT_ARCHITECTURE_AUDIT.md) · [MODEL_CATALOG](MODEL_CATALOG.md) · [HARDWARE](HARDWARE_COMPATIBILITY.md) · [VOICE_ENROLLMENT](VOICE_ENROLLMENT.md) · [SECURITY](SECURITY.md) · [CAPABILITIES](COMPUTER_CAPABILITIES.md) · [AGENT_LOOP](AGENT_LOOP.md) · [OFFLINE](OFFLINE_MODE.md) · [BACKGROUND_TASKS](BACKGROUND_TASKS.md) · [TROUBLESHOOTING](TROUBLESHOOTING.md) · [TEST_RESULTS](TEST_RESULTS.md)
