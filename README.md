# JARVIS — Personal Windows Autonomous Voice AI

JARVIS is a secure, voice-first autonomous personal AI assistant engineered for **Sanjay** on Windows 10/11. Built with modularity, defense-in-depth security, and real-time speech processing, JARVIS can hear, speak, and intelligently manage tasks while protecting sensitive personal data.

---

## Architecture Overview

```
jarvis/
├── main.py                     # Primary entry point (Voice, Text, Test modes)
├── config/
│   ├── settings.py             # System settings, IPv4 prioritization, UTF-8 console
│   └── constants.py            # Risk levels, speaker classifications, prompts
├── core/
│   └── agent.py                # JarvisAgent orchestrator
├── llm/
│   ├── base.py                 # Abstract AIProvider interface
│   ├── groq_provider.py        # Groq ultra-fast LPU inference (Llama 3.3 / 3.1)
│   ├── gemini_provider.py      # Google Gemini API integration
│   ├── mock_provider.py        # Offline simulated provider for tests/offline usage
│   └── provider_factory.py     # Resilient provider selection with failovers
├── voice/
│   ├── recorder.py             # Audio recorder with acoustic RMS energy VAD
│   ├── stt.py                  # Groq Whisper API + Google Speech Recognition fallback
│   └── tts.py                  # Microsoft Edge Neural TTS with text sanitizer & barge-in
├── memory/
│   ├── conversation_memory.py  # Sliding window conversational context
│   └── preferences.py          # Persistent owner preferences (JSON)
├── security/
│   ├── privacy_guard.py        # Redacts API keys, passwords, and enforces access control
│   └── audit_log.py            # Audit logger recording tasks and risk levels without secrets
├── tests/
│   └── test_phase1.py          # Comprehensive Phase 1 verification test suite
├── .env.example                # Configuration template
├── requirements.txt            # Dependency specification
└── README.md
```

---

## Key Features Implemented (Phase 1)

1. **Provider Independence (`AIProvider`)**:
   - Primary: **Groq API** (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) for ultra-low latency inference.
   - Built-in graceful fallbacks to **Google Gemini** or **Offline Mock mode**.
2. **Audio Recorder & Voice Activity Detection (VAD)**:
   - Dynamic acoustic noise calibration and RMS energy thresholding.
   - Silence cutoff detection (stops recording when you finish speaking).
   - Barge-in / interruption capability.
3. **Speech-to-Text (STT)**:
   - Whisper transcription via Groq API (`whisper-large-v3-turbo`).
   - Automatic fallback to Google Speech Recognition with multilingual support (English, Tamil, Hindi).
4. **Natural Neural Speech (TTS)**:
   - Microsoft Edge Neural voices (`en-US-ChristopherNeural`, `ta-IN-ValluvarNeural`, `hi-IN-MadhurNeural`).
   - In-memory decoding and playback via `miniaudio` and `sounddevice` with instant interruption support.
   - Text sanitizer strips markdown, URLs, and code blocks before speaking.
   - Offline fallback to Windows SAPI5 (`pyttsx3`).
5. **Privacy Guard & Defense-in-Depth**:
   - Detects and redacts credentials (API keys, passwords, tokens, credit cards).
   - Unknown speakers cannot access private documents or credentials.
   - Unauthenticated sensitive commands are blocked.
6. **Audit Logger**:
   - Thread-safe JSONL audit trail recording every task, speaker class, risk level, and execution status.
   - Guaranteed secret redaction in all logs.

---

## Quick Start

### 1. Configuration

Copy `.env.example` to `.env` and insert your **Groq API Key**:
```bash
cp .env.example .env
```
Edit `.env`:
```env
OWNER_NAME=Sanjay
PREFERRED_TITLE=Sir
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

*(If `GROQ_API_KEY` is not provided, JARVIS automatically boots in offline mock mode so you can test all features safely).*

### 2. Run Verification Tests
```powershell
python main.py --test
```
or
```powershell
pytest tests/test_phase1.py -v
```

### 3. Interactive Text Mode (Development / Headless)
```powershell
python main.py --text
```

### 4. Continuous Voice Mode (Microphone to Speaker)
```powershell
python main.py
```
