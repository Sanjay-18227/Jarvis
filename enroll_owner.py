"""
JARVIS Biometric Voice Enrollment Protocol
Enrolls the voice of Sanjay (Owner) with quality validation and consistency calibration.
Usage:
    python enroll_owner.py
    python enroll_owner.py --name Sanjay
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import get_settings
from voice.voice_enrollment import VoiceEnrollment
from voice.speaker_verification import SpeakerVerifier
from voice.recorder import AudioRecorder


def main():
    owner_name = "Sanjay"
    if "--name" in sys.argv:
        try:
            idx = sys.argv.index("--name")
            owner_name = sys.argv[idx + 1]
        except (IndexError, ValueError):
            pass

    settings = get_settings()
    recorder = AudioRecorder(settings=settings)
    verifier = SpeakerVerifier()

    enroller = VoiceEnrollment(verifier=verifier, recorder=recorder)
    success = enroller.run_interactive_enrollment(owner_name=owner_name)

    if success:
        print("\n" + "=" * 55)
        print(f"✓ OWNER VOICE PROFILE ACTIVATED FOR {owner_name.upper()}")
        print(f"Profile saved to: data/speaker_profiles/owner_profile.dat")
        print(f"Biometric threshold: {verifier.similarity_threshold_owner}")
        print("Computer control security: LOCKED to verified owner voice signature.")
        print("=" * 55 + "\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 55)
        print("❌ VOICE ENROLLMENT INCOMPLETE")
        print("Please rerun in a quiet environment and speak clearly.")
        print("=" * 55 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
