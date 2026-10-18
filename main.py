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
    print("=" * 55)
    print("      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗")
    print("      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝")
    print("      ██║███████║██████╔╝██║   ██║██║███████╗")
    print(" ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║")
    print(" ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║")
    print("  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝")
    print("    Windows Autonomous Personal AI — Online")
    print("=" * 55)
    print(f" Owner:         {settings.owner_name} ({settings.preferred_title})")
    print(f" AI Brain:      {agent.provider.name}")
    print(f" STT Engine:    {settings.stt_engine} ({settings.stt_language})")
    print(f" Unified Voice: {settings.tts_voice}")
    print(f" Speaker Mode:  {agent.speaker_state.value}")
    print("=" * 55 + "\n")


def main():
    settings = get_settings()
    agent = JarvisAgent(settings=settings)
    print_banner(agent)

    # Check CLI flags
    if "--enroll" in sys.argv:
        print("[JARVIS] Launching Voice Enrollment Protocol for Sanjay...")
        from voice.voice_enrollment import VoiceEnrollment
        enroller = VoiceEnrollment(recorder=agent.recorder)
        enroller.enroll_owner(speaker_name=settings.owner_name)
        sys.exit(0)

    if "--test" in sys.argv:
        print("[JARVIS] Executing Comprehensive Autonomous Verification Suite...\n")
        import pytest
        exit_code = pytest.main(["-v", os.path.join(PROJECT_ROOT, "tests")])
        sys.exit(exit_code)

    if "--text" in sys.argv or "-t" in sys.argv:
        agent.run_text_interactive()
    else:
        agent.run_continuous_voice()


if __name__ == "__main__":
    main()
