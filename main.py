"""
JARVIS — Personal Windows Autonomous Voice AI
Main Entry Point
"""

import sys
import os

# Ensure current working directory is added to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import get_settings
from core.agent import JarvisAgent


def print_banner(agent: JarvisAgent):
    settings = agent.settings
    mode = ("Local-Only" if settings.local_only and not settings.allow_remote_llm
            else "Hybrid/Online")
    print("=" * 55)
    print("      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗")
    print("      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝")
    print("      ██║███████║██████╔╝██║   ██║██║███████╗")
    print(" ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║")
    print(" ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║")
    print("  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝")
    print(f"    Windows Autonomous Personal AI — {mode}")
    print("=" * 55)
    print(f" Owner:         {settings.owner_name} ({settings.preferred_title})")
    print(f" AI Brain:      {agent.provider.name}")
    print(f" STT Engine:    {settings.stt_engine} ({settings.stt_language})")
    print(f" Unified Voice: {settings.tts_voice}")
    print(f" Speaker Mode:  {agent.speaker_state.value}")
    print("=" * 55 + "\n")


def main():
    settings = get_settings()

    # diagnostics & owner recovery that must not initialize heavy subsystems
    if "--mic-test" in sys.argv:
        from scripts.mic_test import run_mic_test
        sys.exit(run_mic_test())

    if "--speaker-test" in sys.argv:
        from scripts.speaker_test import run_speaker_test
        sys.exit(run_speaker_test())

    if "--set-pin" in sys.argv:
        from security.recovery import _interactive_set_pin
        _interactive_set_pin()

    if "--auth-pin" in sys.argv:
        from security.recovery import _interactive_auth
        _interactive_auth()

    agent = JarvisAgent(settings=settings)
    print_banner(agent)

    # Durable local scheduler (only runs while JARVIS/laptop is ON - documented limitation)
    if "--no-scheduler" not in sys.argv:
        try:
            from core.automation.task_scheduler import get_scheduler
            get_scheduler().start()
        except Exception as e:
            print(f"[SCHEDULER] not started: {e}")

    # Check CLI flags
    if "--enroll" in sys.argv:
        print("[JARVIS] Launching Voice Enrollment Protocol for Sanjay...")
        from voice.voice_enrollment import VoiceEnrollment
        enroller = VoiceEnrollment(recorder=agent.recorder)
        enroller.enroll_owner(speaker_name=settings.owner_name)
        sys.exit(0)

    if "--test" in sys.argv:
        print("[JARVIS] Executing Comprehensive Autonomous Verification Suite...\n")
        # pyrefly: ignore[missing-import]
        import pytest
        exit_code = pytest.main(["-v", os.path.join(PROJECT_ROOT, "tests")])
        sys.exit(exit_code)

    if "--text" in sys.argv or "-t" in sys.argv:
        agent.run_text_interactive()
    else:
        agent.run_continuous_voice()


if __name__ == "__main__":
    main()
