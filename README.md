# JARVIS — Personal Windows Autonomous Voice AI

**JARVIS** is a production-grade, voice-first autonomous desktop AI assistant built specifically for **Sanjay** on Windows 10/11. JARVIS operates via natural speech-to-speech interaction, recognizes your enrolled voice signature, perceives on-screen context, safely controls Windows applications and files, acts as an autonomous coding assistant in VS Code, and enforces defense-in-depth security with mandatory spoken confirmation for sensitive actions.

---

## 🏗 System Architecture

```
jarvis/
├── main.py                         # Unified entry point (Voice, Text, Test, Enroll)
├── enroll_owner.py                 # Owner biometric voice calibration protocol
├── config/
│   ├── settings.py                 # System configuration, LiveKit transport, VAD thresholds
│   └── constants.py                # Risk levels (0-4), speaker classifications, system personas
│
├── core/
│   ├── agent.py                    # Central JarvisAgent orchestrator
│   ├── agent_loop.py               # Phase 3 Autonomous Agent Loop (Understand-Plan-Auth-Exec-Verify)
│   ├── fast_router.py              # Sub-10ms Fast Path deterministic intent router
│   ├── normalizer.py               # Intent extraction and conversational prefix stripper
│   ├── autonomous_planner.py       # Autonomous goal & tool decomposition planner
│   ├── permissions.py              # Permission matrix delegating to Security Gate
│   ├── coding_agent.py             # VS Code coding agent (Plan-Act-Observe-Verify-Fix)
│   └── multilingual.py             # Code-mixed Tanglish/Tamil/Hindi intent parser
│
├── voice/
│   ├── voice_engine.py             # LiveKit Voice Engine with seamless Legacy fallback
│   ├── recorder.py                 # Audio recorder with acoustic RMS energy & VAD noise rejection
│   ├── stt.py                      # Speech-to-text (Groq Whisper / Google STT)
│   ├── tts.py                      # Microsoft Edge Neural TTS with text sanitizer & barge-in
│   ├── barge_in.py                 # Real-time barge-in interruption detection
│   ├── state_machine.py            # Voice states (IDLE, LISTENING, PROCESSING, EXECUTING, SPEAKING)
│   ├── speaker_verification.py     # MFCC feature extraction, .dat/.json profile loader & Cosine similarity
│   └── voice_enrollment.py         # Multi-sample voice signature enrollment workflow
│
├── computer/
│   ├── app_indexer.py              # Dynamic Windows Start Menu & system app indexer
│   ├── browser_controller.py       # Edge & browser automation (URL, YouTube search, web query)
│   ├── file_manager.py             # Verified directory and file operations with sandboxing
│   ├── windows_controller.py       # Master volume, media playback, battery, lock screen
│   ├── system_monitor.py           # Real-time CPU, RAM, battery, and process diagnostics
│   └── ui_automation.py            # Win32 mouse clicks, wheel scrolling, typing, and hotkeys
│
├── vision/
│   ├── screen_capture.py           # Display capture and active window detection
│   └── screen_understanding.py     # Multimodal Gemini 3.8 Flash UI & error inspection (truthful fail-safe)
│
├── security/
│   ├── authorization_gate.py       # Centralized Security Gate (fail-closed UNKNOWN blocking)
│   ├── privacy_guard.py            # Secret redaction, credential protection, and keyword filtering
│   ├── confirmation.py             # Ephemeral single-use confirmation manager for high-risk actions
│   └── audit_log.py                # Redacted JSONL action audit trail
│
├── llm/
│   ├── provider_factory.py         # Multi-provider factory (OpenRouter, Gemini, Groq, Mock)
│   ├── openrouter_provider.py      # Primary cloud brain (Llama 3.3 70B via OpenRouter)
│   ├── gemini_provider.py          # Multimodal vision & reasoning brain (Gemini 3.8 Flash)
│   ├── groq_provider.py            # Ultra-low-latency Groq inference
│   └── mock_provider.py            # Safe offline mock provider for automated testing
│
├── memory/
│   ├── conversation_memory.py      # Bounded sliding window dialogue context
│   ├── preferences.py              # Persistent user preferences (JSON)
│   └── owner_voice_profile.json    # Calibrated owner acoustic MFCC embedding
│
├── scripts/
│   ├── start_jarvis.bat            # Launches JARVIS silently in background via pythonw.exe
│   ├── stop_jarvis.bat             # Gracefully terminates running background instance
│   ├── register_startup.bat        # Registers JARVIS to run automatically at Windows logon
│   └── unregister_startup.bat      # Removes JARVIS from Windows Startup
│
├── tests/
│   ├── test_agent_loop.py          # Phase 3 Agent Loop verification
│   ├── test_autonomous_agent.py    # Autonomous tool execution and planning tests
│   ├── test_full_suite.py          # Comprehensive 14-point security and behavior test suite
│   ├── test_llm_providers.py       # LLM provider factory and mock tests
│   ├── test_phase1.py              # Phase 1 baseline unit tests
│   ├── test_reliability_suite.py   # Multi-step, failover, and crash recovery tests
│   ├── test_vad_barge_in.py        # VAD calibration, noise rejection, and barge-in tests
│   └── test_voice_security_livekit.py # Voice security, biometrics, LiveKit transport & Tasks A-R
│
├── .env.example                    # Environment variable template
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete documentation
```

---

## 🚀 All 10 Phases Implemented

| Phase | Subsystem | Status | Features |
|---|---|---|---|
| **Phase 1** | **Core Architecture & Speech** | ✅ Complete | Modular architecture, OpenRouter/Gemini brain, Edge Neural TTS, VAD audio recording, and privacy redaction. |
| **Phase 2** | **Wake Word Engine** | ✅ Complete | Local keyword detector for "Hey Jarvis" and "Jarvis" with enable/disable controls. |
| **Phase 3** | **Owner Voice & Agent Loop** | ✅ Complete | Biometric voice verification (MFCC Cosine similarity), fail-closed speaker gating, and Phase 3 Autonomous Agent Loop (`Goal -> Understand -> Plan -> Authorize -> Execute -> Observe -> Verify -> Report`). |
| **Phase 4** | **Risk & Security Gate** | ✅ Complete | Centralized Security Authorization Gate, 30s owner session TTL, high-risk spoken confirmation, and complete lockdown for unverified guests. |
| **Phase 5** | **Computer Control** | ✅ Complete | Dynamic app indexing (Edge, Chrome, Calc, VS Code, Notepad, Spotify), file creation/editing/sandboxing, volume, media keys, screen locking. |
| **Phase 6** | **Screen Understanding** | ✅ Complete | Active window context inspection, sensitive window shielding, truthful fail-safe when vision API is inactive, and multimodal Gemini 3.8 Flash UI analysis. |
| **Phase 7** | **VS Code Coding Agent** | ✅ Complete | Autonomous multi-step development loop with file creation, python execution, error handling, and test verification. |
| **Phase 8** | **Multilingual Voice** | ✅ Complete | Code-mixed Tanglish/Tamil/Hindi intent extraction (e.g. *"Jarvis chrome open பண்ணு"* or *"paatu podu"*) and dynamic neural voice switching (`ta-IN-ValluvarNeural`, `hi-IN-MadhurNeural`). |
| **Phase 9** | **Background Startup** | ✅ Complete | Windows Startup integration, background execution via `pythonw.exe`, and one-click start/stop/restart batch scripts. |
| **Phase 10** | **Testing & Hardening** | ✅ Complete | 107 automated tests verifying least privilege, credential redaction, emergency stop, barge-in, LiveKit abstraction, and confirmation enforcement. |

---

## ⚡ Quick Start Guide

### 1. Configure Credentials
Edit your [`.env`](file:///c:/Users/acer/Desktop/Jarvis/.env) file:
```env
OWNER_NAME=Sanjay
ASSISTANT_NAME=JARVIS
PREFERRED_TITLE=Sir

# AI Brain Options: openrouter (Primary) | gemini | groq
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct

# Multimodal & Vision Brain
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
SCREEN_VISION_ENABLED=false

# Real-Time Voice Transport: livekit | legacy
VOICE_PROVIDER=livekit
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Speech Engines (Single Consistent Multilingual Voice)
STT_ENGINE=groq_whisper
TTS_ENGINE=edge_tts
TTS_VOICE=en-US-AndrewMultilingualNeural

# VAD Calibration & Biometric Thresholds
VAD_SPEECH_THRESHOLD=400
VAD_MIN_SPEECH_MS=250
VAD_END_SILENCE_MS=1500
OWNER_SIMILARITY_THRESHOLD=0.78
```

### 2. Enroll Your Voice Signature (Owner Calibration)
Register your unique acoustic MFCC voice embedding (already completed for Sanjay in `data/speaker_profiles/owner_profile.dat` and `memory/owner_voice_profile.json`):
```powershell
python enroll_owner.py
```
Or via main:
```powershell
python main.py --enroll
```

### 3. Run Autonomous Verification Test Suite
Verify all 107 unit, security, LLM provider, reliability, agent loop, and LiveKit voice tests:
```powershell
python main.py --test
```
Or directly with pytest:
```powershell
pytest -v
```

### 4. Interactive Development Modes
- **Continuous Voice Mode** (Hands-Free Wake Word "Hey Jarvis"):
  ```powershell
  python main.py
  ```
- **Interactive Text Mode**:
  ```powershell
  python main.py --text
  ```

### 5. Windows Background Production Operation
- **Start in Background** (Silent execution via `pythonw.exe`): Double-click [`scripts/start_jarvis.bat`](file:///c:/Users/acer/Desktop/Jarvis/scripts/start_jarvis.bat).
- **Stop Background Process**: Double-click [`scripts/stop_jarvis.bat`](file:///c:/Users/acer/Desktop/Jarvis/scripts/stop_jarvis.bat).
- **Emergency Stop Phrase**: Say **"Jarvis stop"** at any time to instantly freeze automation.
- **Auto-Start on Windows Boot**: Right-click [`scripts/register_startup.bat`](file:///c:/Users/acer/Desktop/Jarvis/scripts/register_startup.bat) and run as Administrator.
