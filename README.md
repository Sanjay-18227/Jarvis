# JARVIS — Local-First Windows AI Computer Agent

JARVIS is a voice-first autonomous desktop assistant for **Sanjay** on Windows
11. It listens, recognizes the owner's voice, understands English/Tamil/Tanglish
commands, controls applications and files, inspects the screen, remembers
preferences, and schedules durable local tasks.

The modernized stack runs on **Hugging Face models + PyTorch CPU**. Cloud
providers are optional and **disabled by default** (`LOCAL_ONLY=true`).

---

## Quick start

```powershell
cd C:\Users\acer\Desktop\Jarvis

# 1. One-time setup (Python 3.13 venv + CPU torch + dependencies)
powershell -ExecutionPolicy Bypass -File setup_jarvis.ps1

# 2. Download the required local models
.\venv\Scripts\python.exe download_models.py --only whisper-tiny-local ecapa-voxceleb-speaker qwen3-0.6b-instruct

# 3. Health check
.\venv\Scripts\python.exe doctor.py

# 4. Enroll the owner voice (real microphone)
.\venv\Scripts\python.exe enroll_owner.py --re-enroll

# 5. Verify the live microphone against the owner profile (read-only)
.\venv\Scripts\python.exe main.py --speaker-test

# 6. Run
.\run_jarvis.bat
# or text-only mode:
.\venv\Scripts\python.exe main.py --text
```

If `venv\Scripts\python.exe` cannot start, see
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) §1.

---

## What runs locally

| Layer | Implementation |
|---|---|
| VAD | Silero VAD + RMS gate |
| Speech-to-text | Faster-Whisper tiny/base (CTranslate2 int8, CPU) |
| Speaker verification | ECAPA-TDNN 192-d + legacy MFCC fallback |
| Intent routing | deterministic rules + Qwen3-0.6B JSON intent |
| Computer control | Win32/UIA skills + guarded agent loop |
| Screen inspection | structural fallback; local SmolVLM opt-in |
| Memory | SQLite FTS5 |
| Scheduling | SQLite durable jobs + daemon scheduler |
| TTS | Edge TTS (network) or pyttsx3 (offline) |

Measured on this laptop: Whisper-tiny ~0.57 s for 2 s audio, ECAPA warm
embedding ~0.22 s, Qwen3-0.6B ~5.8 s per short JSON generation at ~2.5 GB RSS.
See [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md).

---

## Models

| Model | Purpose | Default |
|---|---|---|
| `Systran/faster-whisper-tiny` | local ASR | ✅ on demand |
| `Systran/faster-whisper-base` | higher-accuracy ASR | on demand |
| `speechbrain/spkrec-ecapa-voxceleb` | owner voice biometrics | ✅ on demand |
| `Qwen/Qwen3-0.6B` | local intent/NLU | ✅ on demand, RAM-gated |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | local screen description | disabled |
| `intfloat/multilingual-e5-small` | semantic memory | disabled |
| `ai4bharat/indic-parler-tts` | Tamil/English TTS | disabled |

All revisions are pinned in `models/model_registry.yaml`; every entry names an
explicit loader method. See [docs/MODEL_CATALOG.md](docs/MODEL_CATALOG.md).

---

## Security defaults

- `UNKNOWN` speaker → **zero computer control**.
- ECAPA-model failure with an enrolled profile → **fail closed**; owner uses a
  local recovery PIN.
- Thresholds are never lowered automatically.
- High-risk actions (risk level 3–4) require explicit owner confirmation.
- Memory rejects passwords/API keys/tokens; audit logs redact secrets.
- Cloud reasoning requires `ALLOW_REMOTE_LLM=true`; remote screenshots require
  `ALLOW_REMOTE_VISION=true`.

Details: [docs/SECURITY.md](docs/SECURITY.md) ·
[docs/VOICE_ENROLLMENT.md](docs/VOICE_ENROLLMENT.md).

---

## Common commands

```powershell
# Voice / text
.\venv\Scripts\python.exe main.py
.\venv\Scripts\python.exe main.py --text

# Owner enrollment and verification
.\venv\Scripts\python.exe enroll_owner.py --re-enroll
.\venv\Scripts\python.exe enroll_owner.py --test-verification
.\venv\Scripts\python.exe main.py --set-pin
.\venv\Scripts\python.exe main.py --auth-pin

# Diagnostics
.\venv\Scripts\python.exe main.py --mic-test
.\venv\Scripts\python.exe doctor.py
.\venv\Scripts\python.exe benchmark_models.py --vad --asr --speaker

# Tests
.\venv\Scripts\python.exe -m pytest tests -q
```

---

## Voice examples

```
"Jarvis, open VS Code"
"open Kumar in VS Code and run agent.py"
"what's my battery?"
"set volume to 40"
"remember I prefer Edge as the default browser"
"what do you remember?"
"what is this error on my screen?"
"schedule a battery check every hour"
"lock my screen"
```

Tamil/Tanglish commands are supported by the existing multilingual parser and
the local SLM; real-speech accuracy is a manual test.

---

## Optional cloud mode

To explicitly allow cloud reasoning or vision:

```ini
ALLOW_REMOTE_LLM=true
ALLOW_REMOTE_VISION=true
```

With `LOCAL_ONLY=true` and no opt-in, the provider factory returns the offline
provider, screenshots never leave the laptop, and LiveKit cloud audio transport
is disabled in favor of the local microphone. See
[docs/OFFLINE_MODE.md](docs/OFFLINE_MODE.md).

---

## Documentation

- [Architecture audit](docs/CURRENT_ARCHITECTURE_AUDIT.md)
- [Final architecture](docs/FINAL_ARCHITECTURE.md)
- [Model catalog](docs/MODEL_CATALOG.md)
- [Hardware compatibility](docs/HARDWARE_COMPATIBILITY.md)
- [Voice enrollment](docs/VOICE_ENROLLMENT.md)
- [Security model](docs/SECURITY.md)
- [Computer capabilities](docs/COMPUTER_CAPABILITIES.md)
- [Agent loop](docs/AGENT_LOOP.md)
- [Offline mode](docs/OFFLINE_MODE.md)
- [Background tasks](docs/BACKGROUND_TASKS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Test results](docs/TEST_RESULTS.md)

---

## Known limitations

- Background jobs run only while the laptop and JARVIS are on; there is no cloud
  worker.
- Qwen3-0.6B requires ≥ 2200 MB free RAM and is unloaded after idle.
- Local SmolVLM vision is English-oriented and disabled on 8 GB by default.
- Open-ended offline conversation is limited; deterministic skills are the
  reliable local path.
- Edge TTS needs network; use `TTS_ENGINE=pyttsx3` for a fully offline voice loop.
- Voice verification is not a defense against a high-quality voice clone; use
  OS-level login protection for high-value accounts.
