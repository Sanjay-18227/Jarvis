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
    print("\n" + "=" * 55)
    print("      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗")
    print("      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝")
    print("      ██║███████║██████╔╝██║   ██║██║███████╗")
    print(" ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║")
    print(" ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║")
    print("  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝")
    print("    Windows Autonomous Voice AI — Phase 1 Online")
    print("=" * 55)
    print(f" Owner:         {settings.owner_name} ({settings.preferred_title})")
    print(f" LLM Provider:  {agent.provider.name}")
    print(f" STT Engine:    {settings.stt_engine} ({settings.stt_language})")
    print(f" TTS Engine:    {settings.tts_engine} ({settings.tts_voice})")
    print(f" Speaker Mode:  {agent.speaker_state.value}")
    print("=" * 55 + "\n")


def main():
    settings = get_settings()
    agent = JarvisAgent(settings=settings)
    print_banner(agent)

    # Check CLI flags
    if "--test" in sys.argv:
        print("[JARVIS] Executing Phase 1 Verification Suite...\n")
        import pytest
        exit_code = pytest.main(["-v", os.path.join(PROJECT_ROOT, "tests", "test_phase1.py")])
        sys.exit(exit_code)

    if "--text" in sys.argv or "-t" in sys.argv:
        agent.run_text_interactive()
    else:
        agent.run_continuous_voice()


if __name__ == "__main__":
    main()
