# Test Results and Verification Evidence

**Date:** 2026-11-04
**Machine:** Acer laptop · AMD Ryzen 3 7320U (4C/8T) · 7.3 GB usable RAM ·
Windows 11 Home 64-bit · AMD Radeon integrated (no CUDA) · ~68 GB free disk
**Project Python:** `venv\Scripts\python.exe` · Python 3.13.14
**ML stack:** torch 2.15.0+cpu · transformers 5.22.1 · faster-whisper 1.2.1 ·
speechbrain 1.1.2 · silero-vad 6.2.3 · huggingface_hub 2.4.0

All commands below were run in the project root. Results are from the real
environment, not mocks, unless explicitly labeled.

---

## 1. Test suite

### Full suite (final run)

```powershell
.\venv\Scripts\python.exe -m pytest tests -q --no-header
```

```
167 passed, 2 warnings in 189.21s (0:03:09)
```

**Status: PASS — 167 passed, 0 failed.**

The two warnings are upstream deprecations:

- `torch.jit.load` FutureWarning (speechbrain/silero)
- speechbrain symlink warning on Windows

Neither affects functionality.

### Focused modernization suite

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_modernization_suite.py -q
```

```
22 passed in 17.20s
```

Covers: hardware contract + RAM policy, ECAPA never disabled under normal RAM,
registry integrity and explicit loaders, ASR vocabulary, SQLite memory
roundtrip/privacy/forget/clear, job store recurrence/retry/cancel, scheduler
dispatch/refusal/owner-approval skip, recovery PIN hash/lockout/weak-PIN
rejection, LOCAL_ONLY cloud-reasoning block, LOCAL_ONLY voice-transport block,
isolated custom ECAPA profiles, skill registration, fail-closed ECAPA.

### Phase 4 vision suite

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_phase4_screen_vision.py -q
```

```
11 passed in 38.96s
```

### Owner enrollment suites

```powershell
.\venv\Scripts\python.exe -m pytest `
  tests\test_fix_and_slm_suite.py::test_1_owner_enrollment_and_backup `
  tests\test_owner_enrollment_verification.py::test_1_owner_enrollment_multi_sample -q
```

```
2 passed, 1 warning in 24.16s
```

During this work the tests correctly stopped writing the production ECAPA
profile: custom profiles now keep their ECAPA file beside the custom profile.
The synthetic `owner_profile_ecapa.json` created by an earlier run was deleted;
the owner must enroll with the real voice.

---

## 2. Environment / model health

```powershell
.\venv\Scripts\python.exe doctor.py
```

```
summary: 0 FAIL, 5 WARN
JARVIS environment is healthy.
```

- All required ML imports: PASS
- Hardware detection: PASS (`cuda=False`, `device=cpu`, threads=4)
- Models: silero-vad, whisper-tiny, whisper-base, ECAPA, Qwen3-0.6B: downloaded
- Optional models: SmolVLM, e5-small, indic-parler: disabled by registry policy
- WARN: API keys present while `LOCAL_ONLY=true` (expected; they are not used)
- Audio: 12 input devices detected

`doctor.py` exits with code 1 by design when warnings are present; the summary
is `0 FAIL`.

---

## 3. Benchmarks (measured)

```powershell
.\venv\Scripts\python.exe benchmark_models.py --vad --speaker
.\venv\Scripts\python.exe benchmark_models.py --asr
.\venv\Scripts\python.exe benchmark_models.py --slm
```

### VAD (Silero)

| Metric | Measured |
|---|---|
| Load | 152 ms |
| Mean 512-sample chunk | 5.144 ms |
| Max chunk | 174.563 ms |
| Status | PASS |

### ASR (Faster-Whisper, CTranslate2 int8, CPU)

| Model | Load | Process RSS delta | 2 s synthetic audio |
|---|---|---|---|
| whisper-tiny-local | 5266 ms | ~73 MB | 569.3 ms |
| whisper-base-local | 4645 ms | ~35 MB | 2469.5 ms |

**Status: PASS (mechanical).** The 2-second clip is synthetic test audio, not
speech, so real word accuracy and real-time factor are **MANUAL** — run with a
real microphone recording. Whisper-tiny remains the default for latency.

> Note: the earlier `openai/whisper-*` snapshots were the wrong format for
> Faster-Whisper. The registry now uses the official CTranslate2 conversions:
> `Systran/faster-whisper-tiny` (`d90ca5f…`) and
> `Systran/faster-whisper-base` (`ebe41f7…`), both MIT.

### Speaker verification (ECAPA)

| Metric | Measured |
|---|---|
| Model load | 1316 ms |
| First embedding (includes engine init) | 5265.9 ms |
| Warm embedding | **224.2 ms** |
| Embedding dimension | 192 |
| L2 norm | 1.0 |
| Status | PASS |

### Local SLM (Qwen3-0.6B, fp32 CPU)

| Metric | Measured |
|---|---|
| Real JSON generation verified | PASS (e.g. `open Kumar in vscode` → `OPEN_FOLDER_IN_APP`) |
| Generation, short JSON | ~5.8 s |
| Load, healthy RAM | ~11.2 s |
| Load, under memory pressure | ~54 s (observed once) |
| Resident RSS | ~2.5 GB |
| int8 dynamic quantization | ~13.7 s generation — **slower**, not used |

**Fresh benchmark status: BLOCKED by the RAM policy**, not by a code failure.
At benchmark time the laptop had 1333–1388 MB free; the required gate is
2200 MB. The benchmark correctly returned:

```
"slm": {"status": "unavailable (not downloaded / policy denied)"}
```

Close browsers/IDEs to free ≥ 2200 MB and rerun `benchmark_models.py --slm`.
Functional inference was verified earlier with the real model and prompt.

### Optional models

| Model | Status | Reason |
|---|---|---|
| SmolVLM-256M | BLOCKED / disabled | ~0.9–1.2 GB RAM; 8 GB laptop, English-only caveat |
| multilingual-e5-small | disabled | SQLite FTS5 is the default |
| indic-parler TTS | BLOCKED / disabled | ~1.8 GB RAM; not benchmarked on CPU |

---

## 4. Functional verification performed

| Check | Method | Result |
|---|---|---|
| Local command routing | FastRouter/skills tests | PASS |
| SLM JSON intent (real Qwen3) | `open Kumar in vscode` | PASS (`OPEN_FOLDER_IN_APP`, folder=Kumar, app=VS Code) |
| Memory remember/list/search/forget | SQLite store tests + smoke | PASS |
| Sensitive memory rejection | password/API-key tests | PASS |
| Durable jobs + recurrence + retry | job store tests | PASS |
| Scheduler dispatch through registry | unit test | PASS |
| Recovery PIN hashing + lockout | unit test | PASS |
| Fail-closed ECAPA | unit test | PASS |
| Vision Gemini-mocked flow | 11 Phase 4 tests | PASS |
| LOCAL_ONLY blocks cloud reasoning | provider factory code + doctor WARN | PASS |
| LOCAL_ONLY blocks cloud voice transport | `voice/voice_engine.py` + unit test | PASS |
| Full test suite | pytest | PASS (167/167) |

---

## 5. Fixes made during verification

1. **Faster-Whisper loader** — CTranslate2 expects `cpu_threads`, not `threads`
   or `intra_threads`.
2. **ASR model format** — switched registry to Systran CTranslate2 snapshots;
   `openai/whisper-*` Transformers snapshots cannot be loaded by Faster-Whisper.
3. **ECAPA profile isolation** — custom/test profiles keep ECAPA data beside the
   custom profile, never touching the production biometric file.
4. **Enrollment space separation** — legacy `.dat` profile always uses the
   42-d MFCC space; ECAPA remains in its own 192-d file.
5. **Vision tests** — opt in to remote vision explicitly (mocked client only);
   installed the missing `google-genai` client library.
6. **LOCAL_ONLY enforcement** — cloud reasoning now requires
   `ALLOW_REMOTE_LLM=true` (vision already required `ALLOW_REMOTE_VISION=true`).
7. **Recovery PIN** — constant-time hash comparison; failure window clears on
   success; CLI message corrected.
8. **Benchmark** — ECAPA measured through the production engine (normalized
   embedding) and warm latency reported separately.
9. **Scheduler safety** — confirmation-required jobs are skipped until
   `owner_approved=true`.
10. **Registry truthfulness** — Qwen3 measured RAM/runtime updated (fp32 CPU,
    ~2.5 GB), Whisper disk sizes corrected; Whisper now points at CT2 snapshots.
11. **Interactive ECAPA enrollment** — both enrollment paths build the 192-d
    ECAPA profile from the same clean samples; the legacy `.dat` profile stays
    in the 42-d MFCC space. `--synth` writes only to
    `data/speaker_profiles/synthetic_test/`.
12. **Custom-profile backup isolation** — test/custom profiles back up and
    restore beside themselves, never touching production biometric backups.
13. **LOCAL_ONLY voice transport** — LiveKit (cloud relay) is disabled under
    `LOCAL_ONLY=true`; the local microphone path is used instead.

---

## 6. Status mapping

| Requirement area | Status | Evidence |
|---|---|---|
| Audit + preserve existing features | PASS | `docs/CURRENT_ARCHITECTURE_AUDIT.md`; 167 tests |
| Voice security repairs | PASS | ECAPA tests, fail-closed test, PIN tests |
| Model stack (HF + PyTorch) | PASS | registry, loaders, downloads, benchmarks |
| Computer control | PASS | skills + agent loop tests |
| Routing / SLM / agent loop | PASS | real Qwen3 inference, Phase 3 tests |
| Vision | PASS (mock/structural) | Phase 4 11 tests; local VLM BLOCKED by RAM |
| Memory + scheduling | PASS | SQLite tests + smoke |
| Test suite | PASS | 167 passed, 0 failed |
| Docs | PASS | this docs set |
| Real owner voice enrollment | MANUAL | user must run enrollment |
| Real Tamil/code-mixed ASR | MANUAL | needs real speech |
| Second-speaker UNKNOWN test | MANUAL | needs a second person |
| ECAPA-unavailable live PIN test | MANUAL | needs model rename + live session |
| Fresh SLM benchmark | BLOCKED | RAM gate (needs ≥ 2200 MB free) |
| SmolVLM / indic-parler benchmark | BLOCKED | disabled on 8 GB hardware |
| Background jobs while laptop OFF | BLOCKED (not implemented) | local daemon only |

---

## 7. Manual test checklist (owner)

- [ ] Re-enroll with the real voice: `python enroll_owner.py --re-enroll`
- [ ] Confirm `OWNER` on 3 different phrases
- [ ] Second person → `UNKNOWN`, no computer control
- [ ] Tamil / Tamil-English mixed command accuracy
- [ ] Noisy-room rejection
- [ ] Rename ECAPA model dir → fail-closed + `python main.py --auth-pin`
- [ ] Real microphone STT latency and accuracy (tiny vs base)
- [ ] Fresh SLM benchmark with ≥ 2200 MB free RAM
- [ ] Optional SmolVLM run only if RAM permits

---

## 8. How to reproduce

```powershell
cd C:\Users\acer\Desktop\Jarvis

# Environment
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe doctor.py

# Tests
.\venv\Scripts\python.exe -m pytest tests -q

# Benchmarks
.\venv\Scripts\python.exe benchmark_models.py --vad --asr --speaker
.\venv\Scripts\python.exe benchmark_models.py --slm

# Live text mode (no microphone needed)
.\venv\Scripts\python.exe main.py --text

# Voice mode
.\venv\Scripts\python.exe main.py
```
