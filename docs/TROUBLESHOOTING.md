# Troubleshooting

All commands assume the project root:

```powershell
cd C:\Users\acer\Desktop\Jarvis
.\venv\Scripts\python.exe --version
```

If `venv\Scripts\python.exe` cannot start, see §1.

---

## 1. Python / venv problems

### `did not find executable at ...Python313\python.exe`

The venv's base interpreter files were removed (cleanup/antivirus/uninstall).
Repair Python 3.13 at the same path:

```powershell
$o="$env:TEMP\python-3.13.14-amd64.exe"
Invoke-WebRequest "https://www.python.org/ftp/python/3.13.14/python-3.13.14-amd64.exe" -OutFile $o
& $o /repair /quiet
.\venv\Scripts\python.exe -c "import sys; print(sys.version)"
```

The existing venv then works again (packages live in `venv\Lib\site-packages`).

### Rebuild the venv (last resort)

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not delete `data/speaker_profiles/` or `memory/` when rebuilding.

### `torch` missing / wrong Python

Use the venv Python, never system Python 3.14 (no torch wheels):

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

Expected: `2.15.0+cpu` (or another CPU 2.x build).

---

## 2. Models

### Check state

```powershell
.\venv\Scripts\python.exe doctor.py
.\venv\Scripts\python.exe download_models.py --list
```

### Download required models

```powershell
.\venv\Scripts\python.exe download_models.py --only whisper-tiny-local ecapa-voxceleb-speaker qwen3-0.6b-instruct
```

### `whisper ... Unable to open file 'model.bin'`

You have the Transformers Whisper snapshot, not the CTranslate2 one. The
registry must point at `Systran/faster-whisper-*`. Re-download:

```powershell
.\venv\Scripts\python.exe download_models.py --only whisper-tiny-local whisper-base-local
```

### Model not loading / policy denies

```powershell
.\venv\Scripts\python.exe -c "from core.models.model_manager import get_model_manager as g; import json; print(json.dumps(g().policy, indent=2))"
```

- SLM requires ≥ 2200 MB free RAM.
- Whisper requires ≥ 700 MB.
- ECAPA is denied only below 250 MB (then auth fails closed → PIN).
- Close browsers/IDEs to free RAM.

---

## 3. Microphone / STT

### No audio or empty transcription

```powershell
.\venv\Scripts\python.exe main.py --mic-test
```

- Check Windows microphone privacy settings.
- Select the correct input device in Windows Sound settings.
- Speak 1–3 seconds after the listening cue.
- Avoid fan noise; the RMS gate rejects quiet input.

### STT is slow

- Default is `whisper-tiny` int8; measured ~0.6 s for 2 s audio.
- `whisper-base` is more accurate but ~4× slower on this CPU.
- First command includes model load (~5 s).

### Wrong words ("vs code" → "V S Code")

The vocabulary corrector runs automatically. If a term is still wrong, add it
to `_VOCAB_FIXES` in `voice/stt.py`.

---

## 4. Speaker verification

### Real owner gets UNKNOWN

1. Do not lower thresholds.
2. Run the read-only live diagnostic:
   ```powershell
   .\venv\Scripts\python.exe main.py --speaker-test
   ```
   It prints the MFCC score, ECAPA score, and each threshold separately.
3. Re-enroll with clean samples:
   ```powershell
   .\venv\Scripts\python.exe enroll_owner.py --re-enroll
   ```
4. Test:
   ```powershell
   .\venv\Scripts\python.exe enroll_owner.py --test-verification
   ```
5. Check the ECAPA profile exists:
   ```powershell
   Get-ChildItem data\speaker_profiles
   ```

If the ECAPA score is high but the result is still `UNKNOWN`, check that the
profile was built after the latest enrollment (the loader prints
`ECAPA profile loaded ... threshold=0.xx`). If the score itself is low, the
microphone captured mostly silence/noise: re-enroll in a quiet room and stay
close to the mic. ECAPA and MFCC use separate thresholds; the 0.78 MFCC value
must never be applied to ECAPA scores.

### `ECAPA model unavailable ... FAIL CLOSED`

This is intentional when an ECAPA profile is enrolled but the model cannot run.
Fix the model/RAM, or use the recovery PIN:

```powershell
.\venv\Scripts\python.exe main.py --auth-pin
```

### PIN problems

- Set: `.\venv\Scripts\python.exe main.py --set-pin`
- 5 failures → 300 s lockout; wait and retry.
- No PIN enrolled → `--set-pin` first.
- `--auth-pin` does not unlock an already-running JARVIS; use spoken
  `pin <digits>` in the live session.

---

## 5. Vision

### "couldn't visually inspect the screen"

- `LOCAL_ONLY=true` blocks Gemini unless `ALLOW_REMOTE_VISION=true`.
- Structural fallback only reports window/process facts; it cannot read UI text
  like a VLM.
- For local VLM:
  ```powershell
  .\venv\Scripts\python.exe download_models.py --only smolvlm-256m-instruct
  ```
  then set `VISION_PROVIDER=local` and enable the registry entry. Expect
  ~1 GB RAM usage and English-oriented output.

### Screenshot privacy

`VISION_SAVE_SCREENSHOTS=false` by default. Captures are in memory only.

---

## 6. TTS / audio output

- `TTS_ENGINE=edge_tts` requires internet. Offline: `TTS_ENGINE=pyttsx3`.
- No sound: check Windows output device and volume.
- Edge TTS errors: verify network/DNS; fall back to pyttsx3.
- Barge-in issues: see `BARGE_IN_ENABLED` and `BARGE_IN_COOLDOWN_SECONDS`.

---

## 7. Local-only / cloud confusion

Symptom: general questions get a generic/mock answer even though API keys exist.

Cause: `LOCAL_ONLY=true` + `ALLOW_REMOTE_LLM=false` intentionally disables cloud
reasoning. This is the default.

Options:

```ini
# Keep local privacy (default)
LOCAL_ONLY=true
ALLOW_REMOTE_LLM=false

# Explicitly allow cloud reasoning
ALLOW_REMOTE_LLM=true
# or
LOCAL_ONLY=false
```

Deterministic commands, memory, scheduling and computer control still work.

---

## 8. Scheduler

### Jobs don't run

- JARVIS must be running (laptop on, not asleep).
- Check `data/jobs.db`:
  ```powershell
  .\venv\Scripts\python.exe -c "from core.automation.job_store import get_job_store as g; print(g().list_jobs())"
  ```
- Jobs requiring confirmation are skipped until `owner_approved=true`.
- A job stuck in `running` after a crash must be reset manually (see
  `docs/BACKGROUND_TASKS.md`).

### Job failed repeatedly

Inspect `job_history` in `data/jobs.db` and the `[SCHEDULER]` log lines. Ensure
the payload names a real skill/action.

---

## 9. Tests

Run the focused modernization suite first:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_modernization_suite.py -q
```

Full suite:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

Known test-order note: some legacy tests assume a running desktop session and
screen capture. They pass in the normal full-suite order; run individual tests
if diagnosing.

---

## 10. Logs and diagnostics

| File | Contents |
|---|---|
| `logs/jarvis_audit.jsonl` | redacted action audit |
| `logs/benchmark_latest.json` | latest benchmark numbers |
| console output | `[MODEL]`, `[SPEAKER]`, `[VISION]`, `[SLM]`, `[SCHEDULER]` tags |

Useful commands:

```powershell
.\venv\Scripts\python.exe doctor.py
.\venv\Scripts\python.exe benchmark_models.py --vad --asr --speaker
.\venv\Scripts\python.exe -m core.models.model_manager
```

When reporting a bug, include the command, the exact error, and the relevant
log lines. Never paste `.env`, biometric files, or PINs.
