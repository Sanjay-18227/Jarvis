# JARVIS — Current Architecture Audit

*Audit date: 2026-11-03 (continuation session; original audit + fix work done across commits `7cb7729..fca0930` plus uncommitted repair work)*

## 1. Module inventory (as found in the repository)

| Area | Files | Status |
|------|-------|--------|
| Entry | `main.py`, `enroll_owner.py` | Working (`--text`, `--test`, `--enroll` flags) |
| Orchestrator | `core/agent.py` (JarvisAgent), `core/agent_loop.py` (Phase-3 loop) | Working for fast-path + autonomous loop; autonomous path depends on remote planner |
| Routing | `core/fast_router.py`, `core/normalizer.py` | Working. Deterministic fast paths (open app/website, calculator, greetings, verbatim Q&A) avoid remote calls |
| NLU | `core/normalizer.py`, `core/entity_extractor.py`, `core/local_slm.py` | `local_slm.py` is a **heuristic/pattern engine** (regex + skill registry scoring), not a real neural SLM. `slm_model`, `quantization` fields are cosmetic |
| Context | `core/context_memory.py` | Working. Active app/folder/file tracking, entity carry-over |
| Voice | `voice/recorder.py`, `voice/voice_engine.py`, `voice/tts.py`, `voice/stt.py`, `voice/vad.py`, `voice/speaker_verification.py`, `voice/voice_enrollment.py` | Fixed in repair work: barge-in cancelled cleanly, single playback task, enrollment hardening (sample quality rejection, backup profile, atomic replace) |
| STT | `voice/stt.py` | **Remote-only** (Groq Whisper → Google SpeechRecognition fallback). No local ASR |
| Speaker security | `voice/speaker_verification.py` + `data/speaker_profiles/owner_profile.dat` | MFCC+cosine hand-rolled verifier. Multi-embedding profile with atomic backup/replace now in place. **No ML speaker model** (ECAPA) — brittle against recording variation |
| LLM providers | `llm/base.py`, `llm/openrouter_provider.py`, `llm/provider_factory.py` | OpenRouter/Gemini keys are optional. Failed planning surfaced as "AI planning service unavailable" (relates to OpenRouter 502s seen 31-10). LOCAL_ONLY policy now defaults remote providers OFF |
| Computer control | `computer/app_indexer.py`, `computer/browser_controller.py`, `computer/calculator_controller.py`, `skills/*` (browser, calculator, file, notepad, paint, system, vscode + `skills/registry.py`) | Working. Skill registry pattern with risk levels and ToolResult evidence |
| Security | `security/authorization_gate.py`, `core/risk_engine.py`, `core/permissions.py` | UNKNOWN: computer control denied (verified in `logs/jarvis_audit.jsonl`: `FAST_PATH_SECURITY_DENIED`) |
| Vision | `vision/screen_capture.py`, `vision/screen_vision_service.py`, `vision/screen_understanding.py`, `vision/ui_element_detection.py`, `vision/privacy_filter.py` | Capture + privacy filter local; **understanding is remote (Gemini)** and silent when no API key |
| Memory | `memory/conversation_memory.py`, `memory/preferences.py`, `memory/owner_voice_profile.json` | JSON-based. No SQLite store, no FTS/semantic search, no durable task ledger |
| Scheduling | none | **Missing entirely** |
| Hardware/models | none | **Missing entirely** until this modernization |
| Tests | `tests/test_*.py` (12 suites) | 38 / 76 / 107-test historical suites + new `test_fix_and_slm_suite.py`, `test_owner_enrollment_verification.py` (17 passing), `test_phase3_verification_suite.py`, `test_phase4_screen_vision.py` |
| Logging | `logs/jarvis_audit.jsonl` (append-only decision audit) | Working; no rotation, shared tag set exists |

## 2. Working features

- Fast-path command routing (open app/browser/website, math, status) with no remote AI.
- Speaker gating: UNKNOWN voice cannot trigger computer actions (audited in logs).
- Calculator, Notepad, Paint, VS Code, browser (Edge default), system-info skills.
- Agent loop with step cap, cancellation, ToolResult evidence contract.
- Edge-TTS voice with cancellation/barge-in; SAPI5 offline fallback.
- Enrollment workflow: noise calibration, quality rejection, multi-embedding profile, backup + atomic replace.

## 3. Broken / missing features and root causes

| # | Problem | Root cause | Fix in this modernization |
|---|---------|-----------|---------------------------|
| 1 | Voice pipeline is unusable offline | STT is remote-only (`voice/stt.py` Groq/Google) | Local ASR (faster-whisper/whisper-tiny fallback) added behind `stt_engine` setting |
| 2 | No PyTorch/transformers in the YAKI venv | LiveKit + heavy deps were the goal of that venv; ML models never installed | Dedicated `venv/` (Python 3.13) with CPU torch, transformers, HF hub |
| 3 | One venv (`YAKI`) shared across two projects | Dependency drift/clobber risk; the project is launched from a foreign environment | `venv/` inside Jarvis; `run_jarvis.bat`, `setup_jarvis.ps1` use it; YAKI untouched |
| 4 | No real local SLM | `local_slm.py` is rules-based; fields like "Qwen3-1.7B GGUF" are cosmetic | Neural backend (Qwen3-0.6B via transformers, lazy-loaded, idle-unloaded) added behind the same interface with graceful fallback to the rule engine |
| 5 | No VAD boundary robustness story in docs/tests (RMS-only thresholds historically rejected good speech during enrollment) | RMS-only energy gate | Silero VAD (ONNX/torch) with hysteresis in flow: LISTENING → candidate → confirm → record → pause-decay end (1.2–1.8 s configurable); RMS approach demoted |
| 6 | Speaker verification is not a modern embedding model | No torch → MFCC hand-rolled | ECAPA-TDNN (speechbrain `spkrec-ecapa-voxceleb`) as primary verifier when model present; MFCC verifier retained as fallback; both use same 16 kHz mono normalization |
| 7 | Vision understanding needs Gemini key; silently degrades | Remote-only VLM | Local SmolVLM-256M optional provider + UIA/process-based structural analysis; redaction kept |
| 8 | Memory is JSON, no DURABLE task log | Windows task objects with job history absent | SQLite store (`jarvis.db`) with FTS fallback for preferences/history/jobs |
| 9 | No background scheduling | Never implemented | `core/automation/task_scheduler.py` + `job_store.py` (SQLite-backed, recurring policy). True OFF-laptop execution documented honestly as LIMITED (no free always-on worker assumed) |
| 10 | OpenRouter failures made autonomous tasks fail | Autonomous planner always required remote model | LOCAL_ONLY default; remote only for deliberately opted-in heavy reasoning; fail-fast on 401/402, bounded backoff 429, circuit breaker |
| 11 | No hardware detection / model governance | Not implemented | `core/system/hardware_detector.py`, `core/models/model_manager.py`, `models/model_registry.yaml` |
| 12 | No diagnostics scripts | Not implemented | `doctor.py`, `download_models.py`, `benchmark_models.py`, `--mic-test` |
| 13 | Logs without rotation, no PERF metrics | Not implemented | Rotating app log + `[PERF]` metrics emitted by the pipeline |

## 4. Dependency conflict notes

- YAKI venv (Python 3.11): has `edge-tts`, `psutil`, `sounddevice`, `scipy`; **no torch/transformers/speechbrain**. Keep untouched.
- System pip points to Python 3.14 — torch has **no 3.14 wheels**; dedicated venv must use 3.13 (available via `py -3.13`).
- `requirements.txt` pulls unusable/optional deps on 3.13 (pyaudio, livekit, groq, elevenlabs) — replaced by modular `requirements-core/voice/ml/automation/dev.txt`.

## 5. Migration plan (implemented order)

1. ✅ Audit + backup (`git` commit before modernization; old code preserved by branch tip `fca0930`).
2. ✅ Voice repairs (enrollment/VAD/barge-in/TTS duplication) — done in earlier session, covered by 17 passing tests.
3. ⏳ Hardware detector, model registry, model manager, dedicated venv.
4. ⏳ Local ASR/SLM/VAD/speaker integration.
5. ⏳ Capability/registry cleanup (skills already exist; add controller layer docs).
6. ⏳ Memory SQLite + scheduler.
7. ⏳ Diagnostics, docs, PERF measurements, final test report.

## 6. Risk register

- **8 GB RAM with high current usage** (~6.4 GB in use at audit time): model manager MUST lazy-load and unload; only one heavy model resident at a time.
- **No CUDA** (AMD iGPU): all additions must be CPU-first; DirectML is not assumed.
- **Disk**: 66 GB free — fine for ≈4 GB model cache cap.
- **Remote AI keys**: optional by policy; all local paths must run with none.
