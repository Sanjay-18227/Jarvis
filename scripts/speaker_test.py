"""
Live speaker-verification diagnostic.

Records one phrase and reports both embedding spaces separately so a failed
owner login can be diagnosed without exposing biometric vectors:

  python main.py --speaker-test

The profile is read-only; this command never enrolls or changes thresholds.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _best_score(centroid, reps, sample):
    scores = []
    if centroid is not None:
        scores.append(float(np.dot(centroid, sample)))
    for rep in reps or []:
        scores.append(float(np.dot(rep, sample)))
    return max(scores) if scores else 0.0


def run_speaker_test() -> int:
    from config.settings import get_settings
    from voice.recorder import AudioRecorder
    from voice.speaker_verification import SpeakerVerifier
    from config.constants import SpeakerClassification

    settings = get_settings()
    verifier = SpeakerVerifier()
    recorder = AudioRecorder(settings=settings)

    print("=" * 60)
    print("           JARVIS SPEAKER VERIFICATION TEST")
    print("=" * 60)
    print(f"MFCC profile:      {'loaded' if verifier.owner_embedding is not None else 'missing'}")
    print(f"ECAPA profile:     {'loaded' if verifier.ecapa_centroid is not None else 'missing'}")
    print(f"MFCC threshold:    {verifier.similarity_threshold_owner:.2f}")
    print(f"ECAPA threshold:   {verifier.ecapa_threshold:.2f}")
    print("\nSpeak normally after the prompt. The test must not be run while")
    print("music, fans, or JARVIS speech are playing.\n")

    try:
        print("[TEST] Calibrating ambient noise (stay quiet)...")
        recorder.calibrate_ambient_noise(
            duration=getattr(settings, "vad_noise_calibration_seconds", 0.5))
        print("[TEST] Now speak: \"Jarvis, verify my voice for computer control.\"")
        audio = recorder.record_phrase(timeout=10.0, phrase_time_limit=12.0)
    except KeyboardInterrupt:
        print("\n[TEST] Cancelled.")
        return 130

    if not audio:
        print("[TEST] FAIL: no speech captured. Check the microphone and try again.")
        return 2

    mfcc = verifier.extract_mfcc_embedding(audio)
    ecapa = verifier._ecapa.embedding(audio) if verifier._ecapa.available() else None

    print("\n" + "-" * 60)
    if mfcc is not None and verifier.owner_embedding is not None:
        score = _best_score(verifier.owner_embedding,
                            verifier.representative_embeddings, mfcc)
        print(f"[MFCC]  score={score:.3f} threshold="
              f"{verifier.similarity_threshold_owner:.2f} "
              f"({'PASS' if score >= verifier.similarity_threshold_owner else 'BELOW'})")
    else:
        print("[MFCC]  unavailable")

    if ecapa is not None and verifier.ecapa_centroid is not None:
        score = _best_score(verifier.ecapa_centroid,
                            verifier.ecapa_representatives, ecapa)
        print(f"[ECAPA] score={score:.3f} threshold="
              f"{verifier.ecapa_threshold:.2f} "
              f"({'PASS' if score >= verifier.ecapa_threshold else 'BELOW'})")
    else:
        print("[ECAPA] unavailable")

    result, confidence = verifier.verify_speaker(audio)
    print("-" * 60)
    print(f"[RESULT] {result.value} (confidence={confidence:.3f})")

    if result == SpeakerClassification.OWNER:
        print("\nPASS: this microphone/room accepts your enrolled voice.")
        return 0

    print("\nFAIL: the owner was not accepted.")
    print("Try these in order:")
    print("  1. Move closer to the microphone and repeat in a quiet room.")
    print("  2. Run: .\\venv\\Scripts\\python.exe enroll_owner.py --re-enroll")
    print("  3. Re-run this test; send the printed scores if it still fails.")
    return 3


if __name__ == "__main__":
    sys.exit(run_speaker_test())
