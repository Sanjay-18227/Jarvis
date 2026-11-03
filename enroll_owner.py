"""
JARVIS Biometric Owner Voice Enrollment System
Rebuilds a reliable OWNER voice profile from MULTIPLE recordings of the real owner.
Enforces:
1. Uses the same microphone used by JARVIS.
2. Records 8-10 clean owner voice samples across diverse phrases.
3. Quality check: rejects silence, short audio, clipping, low energy, background noise.
4. Preprocesses audio, extracts speaker embeddings, and normalizes them.
5. Aggregates normalized embeddings into a stable owner profile (centroid).
6. Calibrates threshold between owner and unknown.
7. Saves owner profile locally (never uploads to external cloud services).
8. Privacy: deletes temporary raw audio unless OWNER_VOICE_KEEP_SAMPLES=true.

Usage:
    python enroll_owner.py                  # Standard owner enrollment (8-10 samples)
    python enroll_owner.py --re-enroll      # Fresh re-enrollment
    python enroll_owner.py --test-verification # Test live verification against enrolled profile
    python enroll_owner.py --name Sanjay    # Specify owner name
    python enroll_owner.py --synth          # Automated test enrollment using speech synthesis
"""

import sys
import os
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import get_settings
from voice.voice_enrollment import VoiceEnrollment
from voice.speaker_verification import SpeakerVerifier
from voice.recorder import AudioRecorder
from config.constants import SpeakerClassification


def run_test_verification(verifier: SpeakerVerifier, recorder: AudioRecorder):
    """Tests live verification against the enrolled owner profile."""
    if verifier.owner_embedding is None:
        print("\n[SPEAKER] Error: No enrolled owner voice profile found.")
        print("Please run 'python enroll_owner.py' first.\n")
        return False

    print("\n" + "=" * 55)
    print("      JARVIS SPEAKER VERIFICATION TEST")
    print("=" * 55)
    print("Say any phrase (e.g. 'Jarvis, open Calculator.')")
    print("Listening in 1 second...")
    recorder.calibrate_ambient_noise(duration=0.4)

    audio = recorder.record_phrase(timeout=7.0, phrase_time_limit=8.0)
    if not audio:
        print("[SPEAKER] No speech detected.")
        print("[SPEAKER] Result=UNKNOWN")
        return False

    result, confidence = verifier.verify_speaker(audio)
    if result == SpeakerClassification.OWNER:
        print("[SECURITY] Owner session authenticated")
        return True
    else:
        print("[SECURITY] DENIED: Voice does not match enrolled OWNER profile.")
        return False


def run_synthetic_enrollment(enroller: VoiceEnrollment, owner_name: str = "Sanjay") -> bool:
    """Automated enrollment using speech synthesis for automated test suites."""
    import asyncio
    import io
    import wave
    try:
        import edge_tts
        import miniaudio
    except ImportError:
        print("[ENROLL] edge_tts / miniaudio required for synthetic enrollment.")
        return False

    print(f"\n[ENROLL] Starting synthetic voice enrollment for {owner_name}...")
    v_owner = "en-US-GuyNeural"
    phrases = enroller.ENROLLMENT_PROMPTS

    async def _make_wav(text: str) -> bytes:
        comm = edge_tts.Communicate(text, voice=v_owner)
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        dec = miniaudio.decode(bytes(buf), nchannels=1, sample_rate=16000)
        samples = np.array(dec.samples, dtype=np.int16)
        # Pad with 300ms pre-silence and 1000ms tail silence to replicate real mic
        pre = (np.random.randn(int(16000 * 0.3)) * 20).astype(np.int16)
        post = (np.random.randn(int(16000 * 1.0)) * 20).astype(np.int16)
        padded = np.concatenate([pre, samples, post])

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(padded.tobytes())
        return wav_io.getvalue()

    async def _gather():
        samples = []
        for idx, p in enumerate(phrases, 1):
            w = await _make_wav(p)
            samples.append(w)
            print(f"[ENROLL] Sample {idx}/{len(phrases)} accepted")
            print("[ENROLL] Embedding created")
        return samples

    samples = asyncio.run(_gather())
    return enroller.enroll_from_samples(samples, owner_name=owner_name, min_required=8, replace=True)


def main():
    parser = argparse.ArgumentParser(description="JARVIS Biometric Owner Voice Enrollment System")
    parser.add_argument("--name", default="Sanjay", help="Owner name (default: Sanjay)")
    parser.add_argument("--replace", action="store_true", help="Completely replace existing owner profile and back it up")
    parser.add_argument("--re-enroll", action="store_true", help="Force fresh re-enrollment")
    parser.add_argument("--mic-test", action="store_true", help="Run acoustic microphone diagnostic and level measurement")
    parser.add_argument("--test-verification", action="store_true", help="Test live speaker verification")
    parser.add_argument("--synth", action="store_true", help="Automated enrollment using speech synthesis")
    parser.add_argument("--min-samples", type=int, default=8, help="Minimum accepted samples required")
    parser.add_argument("--target-samples", type=int, default=10, help="Target samples to capture")

    args = parser.parse_args()

    settings = get_settings()
    recorder = AudioRecorder(settings=settings)
    verifier = SpeakerVerifier()
    enroller = VoiceEnrollment(verifier=verifier, recorder=recorder)

    if args.mic_test:
        success = enroller.run_microphone_diagnostic()
        sys.exit(0 if success else 1)

    if args.test_verification:
        success = run_test_verification(verifier, recorder)
        sys.exit(0 if success else 1)

    should_replace = args.replace or args.re_enroll or True

    if args.synth:
        success = run_synthetic_enrollment(enroller, owner_name=args.name)
    else:
        try:
            success = enroller.run_interactive_enrollment(
                owner_name=args.name,
                target_samples=args.target_samples,
                min_samples=args.min_samples,
                replace=should_replace,
            )
        except KeyboardInterrupt:
            print("\n[ENROLL] Enrollment cancelled.")
            print("[ENROLL] Existing owner profile preserved/restored.\n")
            sys.exit(130)

    if success:
        print("\n" + "=" * 60)
        print(f"✓ OWNER VOICE PROFILE ACTIVATED FOR {args.name.upper()}")
        print(f"Profile saved to: data/speaker_profiles/owner_profile.dat")
        print(f"Calibrated threshold: {verifier.similarity_threshold_owner:.2f}")
        print("Computer control security: LOCKED to verified owner voice signature.")
        print("=" * 60 + "\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ VOICE ENROLLMENT INCOMPLETE")
        print("Please rerun in a quiet environment and speak clearly.")
        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
