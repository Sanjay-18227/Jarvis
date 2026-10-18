"""
Voice Enrollment Subsystem
Collects multiple speaker voice samples and computes an averaged enrollment embedding.
"""

from typing import List, Optional
import numpy as np
from voice.recorder import AudioRecorder
from voice.speaker_verification import SpeakerVerifier
from config.settings import get_settings


class VoiceEnrollment:
    """
    Manages the multi-sample voice enrollment workflow for the owner (Sanjay).
    """

    ENROLLMENT_PROMPTS = [
        "Hey Jarvis, my name is Sanjay.",
        "Jarvis, this is my authorized voice signature.",
        "Authorize my personal desktop assistant, Sir.",
        "Voice calibration sample four, all systems operational.",
    ]

    def __init__(self, verifier: Optional[SpeakerVerifier] = None):
        self.verifier = verifier or SpeakerVerifier()
        self.recorder = AudioRecorder(settings=get_settings())

    def enroll_from_samples(self, audio_samples: List[bytes], owner_name: str = "Sanjay") -> bool:
        """
        Enrolls owner profile from a list of audio sample bytes.
        """
        embeddings = []
        for sample in audio_samples:
            emb = self.verifier.extract_embedding(sample)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            print("[VoiceEnrollment] Error: No valid embeddings could be extracted.")
            return False

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        self.verifier.save_profile(avg_embedding, owner_name=owner_name)
        print(f"[VoiceEnrollment] Successfully enrolled voice with {len(embeddings)} samples.")
        return True

    def run_interactive_enrollment(self, owner_name: str = "Sanjay") -> bool:
        """
        Interactive terminal guide for recording enrollment samples.
        """
        print("\n" + "=" * 50)
        print(f"      JARVIS VOICE ENROLLMENT: {owner_name.upper()}")
        print("=" * 50)
        print("You will be asked to speak 4 phrases clearly into your microphone.\n")

        collected_samples: List[bytes] = []

        for idx, prompt in enumerate(self.ENROLLMENT_PROMPTS, 1):
            print(f"[{idx}/4] Please speak the following phrase:")
            print(f"      \"{prompt}\"")
            print("Listening in 1 second...")
            self.recorder.calibrate_ambient_noise(duration=0.5)

            audio = self.recorder.record_phrase(timeout=6.0, phrase_time_limit=8.0)
            if audio:
                emb = self.verifier.extract_embedding(audio)
                if emb is not None:
                    collected_samples.append(audio)
                    print("✓ Sample captured successfully!\n")
                else:
                    print("⚠ Sample audio was too short or unclear. Retrying next phrase...\n")
            else:
                print("⚠ No speech detected. Moving to next phrase...\n")

        if len(collected_samples) >= 2:
            return self.enroll_from_samples(collected_samples, owner_name=owner_name)
        else:
            print("[VoiceEnrollment] Failed: Could not collect enough clear samples.")
            return False
