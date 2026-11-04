# Offline Mode

JARVIS is designed to keep working without internet for the core voice and
computer-control loop. This page lists exactly what is local, what needs the
network, and how to configure a fully offline setup.

---

## 1. Fully offline capabilities

These run on this laptop with no cloud calls:

| Capability | Component |
|---|---|
| Microphone capture + noise gate | `voice/recorder.py` |
| Voice activity detection | Silero VAD (`voice/vad.py`) |
| Speech-to-text | Faster-Whisper tiny/base (`voice/stt.py`) |
| Speaker verification | ECAPA 192-d or MFCC (`voice/`, `core/models/`) |
| Intent routing | deterministic rules + local Qwen3-0.6B (`core/local_slm.py`) |
| App/browser/file/system control | `skills/`, `computer/` |
| Autonomous agent loop | `core/agent_loop.py` (with a local planner provider) |
| Screen inspection fallback | Windows UIA/foreground process (`vision/`) |
| Memory | SQLite FTS5 (`memory/store.py`) |
| Scheduled jobs | SQLite + daemon scheduler (`core/automation/`) |
| Audit logging | `logs/jarvis_audit.jsonl` |

Offline speech output is available through **pyttsx3** (Windows SAPI). Set:

```ini
TTS_ENGINE=pyttsx3
```

Edge TTS is the higher-quality default but requires network.

---

## 2. Network-dependent (optional) features

| Feature | Default | Offline alternative |
|---|---|---|
| Edge TTS | on (`TTS_ENGINE=edge_tts`) | `TTS_ENGINE=pyttsx3` |
| Gemini screen vision | opt-in only | structural UIA fallback |
| OpenRouter/Gemini/Groq reasoning | opt-in only | deterministic skills + SLM routing |
| Groq/Google STT | off (`STT_ENGINE=local`) | local Faster-Whisper |
| LiveKit transport | configurable | `VOICE_PROVIDER=legacy` (local mic) |

> With `LOCAL_ONLY=true`, JARVIS forces the local microphone transport even if
> `VOICE_PROVIDER=livekit` is set, because LiveKit is a cloud relay.
| Model downloads | one-time | pre-download, then offline |

With the default `.env.example` (`LOCAL_ONLY=true`, local STT, local models),
JARVIS makes no cloud request unless you explicitly opt in.

---

## 3. Recommended fully offline `.env`

```ini
LOCAL_ONLY=true
ALLOW_REMOTE_LLM=false
ALLOW_REMOTE_VISION=false

STT_ENGINE=local
ASR_LOCAL_BACKEND=faster-whisper
ASR_LOCAL_MODEL=tiny

TTS_ENGINE=pyttsx3
VOICE_PROVIDER=legacy

VISION_ENABLED=true
VISION_PROVIDER=local       # only if SmolVLM downloaded+enabled; otherwise structural fallback

SLM_ENABLED=true
SLM_NEURAL_BACKEND=          # auto: local Qwen3 when RAM allows
```

> `VISION_PROVIDER=local` without the SmolVLM model falls back to structural
> inspection; it does not download anything at runtime.

---

## 4. One-time setup (requires network)

```powershell
# Python + dependencies
.\setup_jarvis.ps1

# Required local models
.\venv\Scripts\python.exe download_models.py --only whisper-tiny-local ecapa-voxceleb-speaker qwen3-0.6b-instruct

# Verify
.\venv\Scripts\python.exe doctor.py
```

After this, disconnect the network and start JARVIS normally:

```powershell
.\run_jarvis.bat
# or
.\venv\Scripts\python.exe main.py
```

---

## 5. What to expect offline

- **Works:** wake/listen (mic), local transcription, owner verification,
  open apps, browser URLs already known, files, system controls, memory,
  scheduler, structural screen checks, pyttsx3 speech.
- **Degraded:** open-ended general knowledge (0.6B SLM is limited; cloud is
  disabled), free-form screen description (structural only), TTS quality
  (pyttsx3 vs Edge).
- **Unavailable:** Gemini vision, OpenRouter/Groq reasoning, LiveKit cloud
  transport, cloud STT, model downloads.

JARVIS reports unavailability honestly; it does not claim a cloud action
succeeded while offline.

---

## 6. Offline failure behavior

| Situation | Behavior |
|---|---|
| Model not downloaded | `ModelManager` returns `None`; feature reports unavailable |
| RAM below gate | SLM/vision skipped; deterministic skills continue |
| ECAPA unavailable with profile | Fail closed; owner uses PIN |
| Remote provider unavailable | Offline provider/fallback; no crash |
| Download attempted offline | 5-minute failure backoff, no retry storm |
| TTS network failure | Falls back per TTS engine configuration |

---

## 7. Verifying true offline operation

1. Disconnect Wi-Fi/Ethernet.
2. Start JARVIS and run a local command ("open calculator", "what's my battery").
3. Ask a screen question; confirm structural fallback answers truthfully.
4. Check `logs/` for unexpected network errors.
5. Confirm `LOCAL_ONLY=true` and `ALLOW_REMOTE_LLM=false` in `.env`.

If a cloud call is attempted with `LOCAL_ONLY=true`, that is a bug; report it
with the relevant log line.
