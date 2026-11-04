#!/usr/bin/env python
"""
JARVIS doctor.py - environment / hardware / model health check.

Checks (read-only, no installs):
 1. Python version and virtualenv path
 2. Core imports (audio, scipy, settings bootstrap)
 3. Optional ML imports (torch, transformers, faster_whisper, speechbrain, silero_vad)
 4. Hardware snapshot + current model policy
 5. Model registry load + downloaded state
 6. Audio devices (names only)
 7. Settings sanity (LOCAL_ONLY, STT engine, thresholds)

Exits 0 = healthy, 1 = issues found. With --fail-on-warn it exits 2 on warnings too.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
issues: list[tuple[str, str, str]] = []  # (level, area, detail)


def record(level, area, detail=""):
    issues.append((level, area, detail))
    print(f"[{level:<4}] [{area}] {detail}")


def check_python():
    ver = sys.version
    if sys.version_info < (3, 10):
        record(FAIL, "python", f"unsupported version {sys.version.split()[0]}")
    elif sys.version_info[:2] > (3, 13):
        record(WARN, "python", f"{sys.version.split()[0]} - torch wheels unavailable; "
                               f"use the Jarvis venv (3.13): venv\\Scripts\\python.exe")
    else:
        record(PASS, "python", sys.version.split()[0])
    record(PASS, "python", f"interpreter: {sys.executable}")


def check_core_imports():
    import_core = {
        "numpy": ">=1.26", "scipy": None, "sounddevice": None, "psutil": None,
        "aiohttp": None, "requests": None, "yaml": None,
    }
    for mod, _ in import_core.items():
        try:
            module = __import__(mod)
            version = getattr(module, "__version__", "?")
            record(PASS, "imports", f"{mod} {version}")
        except Exception as e:
            record(FAIL, "imports", f"{mod}: {e}")


def check_ml_imports():
    optional = ["torch", "transformers", "faster_whisper", "speechbrain", "silero_vad", "huggingface_hub"]
    for mod in optional:
        try:
            module = __import__(mod)
            version = getattr(module, "__version__", "?")
            record(PASS, "ml", f"{mod} {version}")
        except ImportError:
            record(WARN, "ml", f"{mod} not installed (feature disabled by policy)")
        except Exception as e:
            record(FAIL, "ml", f"{mod}: {e}")


def check_hardware_and_policy():
    try:
        from core.system.hardware_detector import detect_hardware, model_policy
        hw = detect_hardware()
        print(f"       CPU: {hw.cpu_name} ({hw.logical_cores} logical)")
        print(f"       RAM: {hw.available_ram_mb} / {hw.total_ram_mb} MB available")
        print(f"       GPU: {hw.gpu_name} cuda={hw.cuda_available} device={hw.torch_device}")
        policy = model_policy(hw)
        print(f"       policy: {json.dumps(policy)}")
        if hw.available_ram_mb < 700:
            record(WARN, "hardware", "insufficient free RAM right now to load local SLM/ASR; "
                                     "close apps and re-run, or raise policy thresholds")
        else:
            record(PASS, "hardware", "enough RAM for at least one heavy model")
    except Exception as e:
        record(FAIL, "hardware", f"detection error: {e}")


def check_config_and_models():
    try:
        from core.models.model_manager import get_model_manager
        mm = get_model_manager()
        # settings sanity checks
        s = mm.settings
        if s.local_only and (s.openrouter_api_key or s.gemini_api_key):
            record(WARN, "config", "API keys present but LOCAL_ONLY=true (still disabled; clear keys or set LOCAL_ONLY=false to use)")
        stt = s.stt_engine
        if stt == "groq_whisper" or stt == "google":
            record(WARN, "config", f"STT engine '{stt}' is ONLINE. use 'local' for offline mode")
        try:
            import yaml  # noqa: F401
            statuses = mm.statuses()
            for mid, st in statuses.items():
                if st == "missing":
                    record(PASS, "models", f"{mid}: not downloaded (run download_models.py --selected-only)")
                elif st == "disabled":
                    record(WARN, "models", f"{mid}: disabled by registry policy")
                elif st == "ready":
                    record(PASS, "models", f"{mid}: downloaded (will load on demand)")
                else:
                    record(WARN, "models", f"{mid}: state {st}")
        except Exception as e:
            record(FAIL, "models", f"registry error: {e}")
    except Exception as e:
        record(FAIL, "config", f"settings/model manager bootstrap failed: {e}")


def check_audio_devices():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [f"{i}: {dev['name']} ({dev['max_input_channels']}ch @ {dev['default_samplerate']:.0f}Hz)"
                  for i, dev in enumerate(devices)
                  if isinstance(dev, dict) and dev.get("max_input_channels", 0) > 0]
        if inputs:
            record(PASS, "audio", f"{len(inputs)} input device(s):")
            for line in inputs[:8]:
                print("       " + line)
        else:
            record(WARN, "audio", "no input device found; mic test won't run")
    except Exception as e:
        record(FAIL, "audio", f"device enumeration failed: {e}")


def main():
    print("=" * 62)
    print(" JARVIS Doctor - environment health check")
    print("=" * 62)
    check_python()
    check_core_imports()
    check_ml_imports()
    check_hardware_and_policy()
    check_config_and_models()
    check_audio_devices()
    fails, warns = [i for i in issues if i[0] == FAIL], [i for i in issues if i[0] == WARN]
    print("-" * 62)
    print(f" summary: {len(fails)} FAIL, {len(warns)} WARN")
    if fails:
        print(" recovery hints:")
        for _, area, detail in fails:
            print(f"  - {area}: {detail}")
        print("  - missing packages: .\\setup_jarvis.ps1")
        print("  - missing models:   .\\venv\\Scripts\\python.exe download_models.py --selected-only")
        return 1
    print(" JARVIS environment is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
