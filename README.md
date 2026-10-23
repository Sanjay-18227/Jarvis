# JARVIS — Personal Windows Autonomous Voice AI

**JARVIS** is a production-grade, voice-first autonomous desktop AI assistant built specifically for **Sanjay** on Windows 10/11. JARVIS operates via natural speech-to-speech interaction, recognizes your enrolled voice signature, perceives on-screen context, safely controls Windows applications and files, acts as an autonomous coding assistant in VS Code, and enforces defense-in-depth security with mandatory spoken confirmation for sensitive actions.

---

## 🏗 System Architecture

```
jarvis/
├── main.py                         # Unified entry point (Voice, Text, Test, Enroll)
├── config/
│   ├── settings.py                 # System configuration, IPv4 prioritization, UTF-8 console
│   └── constants.py                # Risk levels (0-4), speaker classifications, system personas
│
├── core/
│   ├── agent.py                    # Central JarvisAgent orchestrator
│   ├── executor.py                 # Trusted ActionExecutor gatekeeper
│   ├── permissions.py              # Permission matrix (OWNER, UNKNOWN, UNCERTAIN)
│   ├── risk_engine.py              # Risk classification engine (Levels 0 to 4)
│   ├── coding_agent.py             # VS Code coding agent (Plan-Act-Observe-Verify-Fix)
│   └── multilingual.py             # Code-mixed Tanglish/Tamil/Hindi intent parser
│
├── voice/
│   ├── wake_word.py                # Local low-power wake-word detector ("Hey Jarvis")
│   ├── recorder.py                 # Audio recorder with acoustic RMS energy VAD
│   ├── stt.py                      # Multi-engine speech-to-text (Google STT / Groq Whisper)
│   ├── tts.py                      # Microsoft Edge Neural TTS with text sanitizer & barge-in
│   ├── speaker_verification.py     # MFCC feature extraction & Cosine similarity scoring
│   └── voice_enrollment.py         # Multi-sample voice signature enrollment workflow
│
├── computer/
│   ├── app_controller.py           # Native launcher for Chrome, Calc, VS Code, Spotify, Settings
│   ├── file_manager.py             # Verified directory and file operations with sandboxing
│   ├── windows_controller.py       # Master volume, media playback, battery, lock screen, Google search
│   ├── vscode_controller.py        # VS Code workspace and file opener
│   └── ui_automation.py            # Win32 mouse clicks, wheel scrolling, typing, and hotkeys
│
├── vision/
│   ├── screen_capture.py           # Display capture and active window detection
│   └── screen_understanding.py     # Multimodal Gemini 3.8 Flash UI & error inspection
│
├── security/
│   ├── privacy_guard.py            # Secret redaction and credential protection
│   ├── confirmation.py             # Ephemeral single-use confirmation manager
│   └── audit_log.py                # Redacted JSONL action audit trail
│
├── memory/
│   ├── conversation_memory.py      # Bounded sliding window dialogue context
│   └── preferences.py              # Persistent user preferences (JSON)
│
├── scripts/
│   ├── start_jarvis.bat            # Launches JARVIS silently in background via pythonw.exe
│   ├── stop_jarvis.bat             # Gracefully terminates running background instance
│   ├── register_startup.bat        # Registers JARVIS to run automatically at Windows logon
│   └── unregister_startup.bat      # Removes JARVIS from Windows Startup
│
├── tests/
│   ├── test_phase1.py              # Phase 1 unit test suite
│   └── test_full_suite.py          # Comprehensive 14-point security and behavior test suite
│
├── .env.example                    # Environment variable template
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete documentation
```

---

## 🚀 All 10 Phases Implemented

| Phase | Subsystem | Status | Features |
|---|---|---|---|
| **Phase 1** | **Core Architecture & Speech** | ✅ Complete | Modular architecture, Google Gemini 3.8 Flash integration, Edge Neural TTS, VAD audio recording, memory, and privacy redaction. |
| **Phase 2** | **Wake Word Engine** | ✅ Complete | Local keyword detector for "Hey Jarvis" and "Jarvis" with enable/disable controls. |
| **Phase 3** | **Owner Voice Recognition** | ✅ Complete | Multi-sample voice enrollment, MFCC cepstral feature extraction, and Cosine similarity scoring into `OWNER`, `UNKNOWN`, and `UNCERTAIN`. |
| **Phase 4** | **Risk & Permissions** | ✅ Complete | Tiered risk classification (Level 0-4), mandatory spoken confirmation for destructive actions, and strict guest access blocking. |
| **Phase 5** | **Computer Control** | ✅ Complete | Verified app launching/closing (Calculator, Chrome, VS Code, Notepad, Spotify, Settings), file management, volume, media keys, screen locking. |
| **Phase 6** | **Screen Understanding** | ✅ Complete | Active window context inspection, sensitive window shielding, and multimodal Gemini 3.8 Flash vision analysis for screen errors and UI. |
| **Phase 7** | **VS Code Coding Agent** | ✅ Complete | Autonomous `PLAN -> ACT -> OBSERVE -> VERIFY -> FIX -> VERIFY -> REPORT` development loop with automated test execution. |
| **Phase 8** | **Multilingual Voice** | ✅ Complete | Code-mixed Tanglish/Tamil/Hindi intent extraction (e.g. *"Jarvis chrome open பண்ணு"* or *"paatu podu"*) and dynamic neural voice switching (`ta-IN-ValluvarNeural`, `hi-IN-MadhurNeural`). |
| **Phase 9** | **Background Startup** | ✅ Complete | Windows Startup integration, background execution via `pythonw.exe`, and one-click start/stop/restart batch scripts. |
| **Phase 10** | **Testing & Hardening** | ✅ Complete | 23 automated tests verifying least privilege, credential redaction, emergency stop, and confirmation enforcement. |

---

## ⚡ Quick Start Guide

### 1. Configure Credentials
Edit your [`.env`](file:///c:/Users/acer/Desktop/Jarvis/.env) file:
```env
OWNER_NAME=Sanjay
ASSISTANT_NAME=JARVIS
PREFERRED_TITLE=Sir

# AI Brain Options: openrouter | gemini | groq | ollama
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct

# Or Google Gemini:
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_gemini_api_key_here
# GEMINI_MODEL=gemini-3.8-flash

# Or Local Offline Ollama:
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3:latest

# Speech Engines (Single Consistent Multilingual Voice)
STT_ENGINE=groq_whisper
TTS_ENGINE=edge_tts
TTS_VOICE=en-US-AndrewMultilingualNeural
```

### 2. Enroll Your Voice Signature (Owner Calibration)
Register your unique acoustic MFCC voice embedding (already completed for Sanjay in `memory/owner_voice_profile.json`):
```powershell
python main.py --enroll
```

### 3. Run Autonomous Verification Test Suite
Verify all 76 unit, security, LLM provider, reliability, and barge-in tests:
```powershell
python main.py --test
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
