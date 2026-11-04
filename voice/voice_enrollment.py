"""
Voice Enrollment Subsystem
Rebuilds the OWNER voice profile from MULTIPLE recordings of the real owner.
Enforces:
1. 8-10 diverse phonetic phrases covering owner's acoustic signature.
2. Quality filtering (rejection of silence, short audio, clipping, low energy, noise).
3. Extraction and normalization of speaker embeddings.
4. Aggregation into one stable centroid owner profile.
5. Automated threshold calibration and local storage.
6. Optional temporary raw recording cleanup for privacy.
"""

import os
import io
import wave
import time
import shutil
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cosine

from voice.recorder import AudioRecorder
from voice.speaker_verification import SpeakerVerifier
from config.settings import get_settings


class VoiceEnrollment:
    """
    Manages the multi-sample voice enrollment workflow for the owner (Sanjay).
    Creates a reliable, aggregated speaker profile from 8-10 validated recordings.
    """

    ENROLLMENT_PROMPTS = [
        "Hey Jarvis, check my battery percentage.",
        "Open Microsoft Edge and navigate to YouTube.",
        "Create a new project folder on Desktop.",
        "Open VS Code and load the workspace.",
        "Calculate six hundred and seventy-eight minus eight hundred.",
        "Open Paint and draw a rectangle.",
        "What is the current time and system status?",
        "Jarvis, stop all background tasks immediately.",
        "Open Calculator and verify the result.",
        "Computer controls are verified and ready.",
    ]

    def __init__(
        self,
        verifier: Optional[SpeakerVerifier] = None,
        recorder: Optional[AudioRecorder] = None,
    ):
        self.verifier = verifier or SpeakerVerifier()
        self.recorder = recorder or AudioRecorder(settings=get_settings())
        self.last_audio_stats = {
            "snr_db": 0.0,
            "voiced_duration": 0.0,
            "peak_rms": 0.0,
        }

    def validate_audio_quality(self, audio_data: bytes) -> Tuple[bool, str]:
        """
        Validates audio sample quality against criteria:
        - Rejects if empty or corrupt -> "no real speech detected"
        - Total duration < 0.8s or voiced duration < 0.35s -> "too short"
        - RMS energy < 80.0 or peak energy < 400.0 -> "too quiet"
        - Clipping (> 0.5% samples >= 32000) -> "clipping detected"
        - SNR estimation (< 3.0 dB) -> "Microphone signal-to-noise ratio is too low."
        - Mostly silence (active speech ratio < 8%) -> "mostly silence"
        """
        if not audio_data or len(audio_data) < 200:
            return False, "no real speech detected"

        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

            duration = len(audio) / sample_rate
            if duration < 0.8:
                return False, "too short"

            # RMS energy check
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 80.0:
                return False, "too quiet"

            # Clipping check
            clipped_ratio = np.sum(np.abs(audio) >= 32000) / len(audio)
            if clipped_ratio > 0.005:
                return False, "clipping detected"

            # Frame-level energy analysis for active speech vs silence
            frame_len = int(sample_rate * 0.025)
            hop_len = int(sample_rate * 0.010)
            num_frames = (len(audio) - frame_len) // hop_len
            if num_frames < 8:
                return False, "too short"

            frame_energies = np.array([
                np.mean(audio[i * hop_len : i * hop_len + frame_len] ** 2)
                for i in range(num_frames)
            ])
            peak_energy = np.max(frame_energies)
            if peak_energy < 400.0:
                return False, "too quiet"

            # Count frames with speech activity
            speech_threshold = peak_energy * 0.03
            active_frames = np.sum(frame_energies > speech_threshold)
            speech_ratio = active_frames / num_frames
            speech_duration = active_frames * (hop_len / sample_rate)

            # SNR estimation: active speech energy vs quiet noise floor
            sorted_energies = np.sort(frame_energies)
            noise_floor = np.mean(sorted_energies[: max(1, num_frames // 5)]) + 1e-6
            signal_peak = np.mean(sorted_energies[-max(1, num_frames // 4) :]) + 1e-6
            snr_db = 10.0 * np.log10(signal_peak / noise_floor)

            self.last_audio_stats = {
                "snr_db": float(snr_db),
                "voiced_duration": float(speech_duration),
                "peak_rms": float(np.sqrt(peak_energy)),
            }

            if speech_ratio < 0.08 or speech_duration < 0.35:
                return False, "mostly silence"

            if snr_db < 3.0:
                return False, "Microphone signal-to-noise ratio is too low."

            return True, "Quality verified."

        except Exception as e:
            return False, f"audio validation error: {e}"

    def enroll_owner(self, speaker_name: str = "Sanjay") -> bool:
        """Alias for interactive owner voice enrollment."""
        return self.run_interactive_enrollment(owner_name=speaker_name)

    def filter_consistent_samples(
        self, embeddings: List[np.ndarray], min_consistency: float = 0.55
    ) -> List[np.ndarray]:
        """
        Computes pairwise cosine similarity matrix across sample embeddings.
        Discards outlier samples whose average similarity to other samples is below threshold.
        """
        n = len(embeddings)
        if n < 3:
            return embeddings

        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    dist = cosine(embeddings[i], embeddings[j])
                    sim_matrix[i, j] = 1.0 - dist

        mean_sims = []
        for i in range(n):
            others = [sim_matrix[i, j] for j in range(n) if j != i]
            mean_sims.append(float(np.mean(others)))

        consistent = [embeddings[i] for i in range(n) if mean_sims[i] >= min_consistency]
        if len(consistent) < n:
            print(f"[ENROLL] Consistency filter: retained {len(consistent)}/{n} samples (dropped {n - len(consistent)} outliers)")
        return consistent

    def calibrate_threshold(self, embeddings: List[np.ndarray]) -> Tuple[float, List[float], List[np.ndarray]]:
        """
        Calibrates the OWNER similarity threshold from multiple valid recordings.
        Evaluates similarity of all samples against their aggregated centroid.
        Extracts representative diverse embeddings.
        Outputs:
            (calibrated_threshold, sample_scores, representative_embeddings)
        """
        if not embeddings:
            return 0.78, [], []

        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        scores = [float(1.0 - cosine(centroid, emb)) for emb in embeddings]
        min_score = min(scores)
        mean_score = sum(scores) / len(scores)
        max_score = max(scores)

        # Set proposed threshold safely below owner minimum score with safety buffer
        calibrated = max(0.72, min(0.85, min_score - 0.05))
        calibrated = round(float(calibrated), 2)

        print("\n" + "=" * 55)
        print("      OWNER SCORE DISTRIBUTION & CALIBRATION")
        print("=" * 55)
        print(f"[CALIBRATION] Owner score distribution: min={min_score:.2f}, mean={mean_score:.2f}, max={max_score:.2f}")
        print(f"[CALIBRATION] Proposed threshold: {calibrated:.2f}")
        print("[CALIBRATION] MANUAL NON-OWNER TEST REQUIRED")
        print("=" * 55)

        # Select up to 6 representative embeddings (highest similarity to centroid and diverse)
        sorted_indices = np.argsort(scores)[::-1]
        rep_indices = sorted_indices[: min(6, len(embeddings))]
        representatives = [embeddings[i] for i in rep_indices]

        return calibrated, scores, representatives

    def backup_existing_profile(self) -> Optional[str]:
        """Creates a timestamped backup of existing owner profile before replacement."""
        # Custom/test profiles are isolated: back them up beside themselves and never
        # touch the production biometric profile or its backup directory.
        if getattr(self.verifier, "custom_profile", False):
            target_file = getattr(self.verifier, "profile_path", None)
            if not target_file or not os.path.exists(target_file):
                return None
            backup_dir = os.path.join(os.path.dirname(target_file), "backups")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(
                backup_dir, f"owner_profile_backup_{time.strftime('%Y%m%d_%H%M%S')}.dat")
            try:
                shutil.copy2(target_file, backup_path)
                print(f"[ENROLL] Previous profile backed up to: {backup_path}")
                return backup_path
            except Exception as e:
                print(f"[ENROLL] Backup failed: {e}")
                return None

        base_dir = get_settings().base_dir
        dat_path = os.path.join(base_dir, "data", "speaker_profiles", "owner_profile.dat")
        json_path = os.path.join(base_dir, "memory", "owner_voice_profile.json")

        target_file = None
        if os.path.exists(dat_path):
            target_file = dat_path
        elif os.path.exists(json_path):
            target_file = json_path

        if not target_file:
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(base_dir, "data", "speaker_profiles", "backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"owner_profile_backup_{timestamp}.dat")
        default_bak = os.path.join(base_dir, "data", "speaker_profiles", "owner_profile.dat.bak")

        shutil.copy2(target_file, backup_path)
        shutil.copy2(target_file, default_bak)
        print(f"[ENROLL] Previous owner profile backed up to: {backup_path}")
        return backup_path

    def restore_backup(self, backup_path: Optional[str] = None) -> bool:
        """Restores previous owner profile from backup to guarantee fail-safe replacement."""
        if getattr(self.verifier, "custom_profile", False):
            target_file = getattr(self.verifier, "profile_path", None)
            source = backup_path if (backup_path and os.path.exists(backup_path)) else None
            if source and target_file:
                try:
                    shutil.copy2(source, target_file)
                    print(f"[ENROLL] Existing profile preserved/restored from: {source}")
                    return True
                except Exception as e:
                    print(f"[ENROLL] Failed restoring backup profile: {e}")
            return False

        base_dir = get_settings().base_dir
        dat_path = os.path.join(base_dir, "data", "speaker_profiles", "owner_profile.dat")
        default_bak = os.path.join(base_dir, "data", "speaker_profiles", "owner_profile.dat.bak")

        source = backup_path if (backup_path and os.path.exists(backup_path)) else default_bak
        if os.path.exists(source):
            try:
                shutil.copy2(source, dat_path)
                print(f"[ENROLL] Existing owner profile preserved/restored from: {source}")
                return True
            except Exception as e:
                print(f"[ENROLL] Failed restoring backup profile: {e}")
                return False
        return False

    def run_microphone_diagnostic(self) -> bool:
        """
        Runs microphone acoustic diagnostic mode (--mic-test) without modifying owner profile.
        Measures ambient noise distribution and normal speech levels.
        """
        import sounddevice as sd
        mic_name = "Default System Microphone"
        try:
            dev_info = sd.query_devices(kind="input")
            if dev_info and "name" in dev_info:
                mic_name = dev_info["name"]
        except Exception:
            pass

        print("\n" + "=" * 60)
        print("      JARVIS MICROPHONE ACOUSTIC DIAGNOSTIC")
        print("=" * 60)
        print(f"Device Name:                   {mic_name}")
        print(f"Sample Rate:                   {self.recorder.sample_rate} Hz")
        print(f"Channels:                      {self.recorder.channels}")
        print(f"Data Type:                     int16")
        print("-" * 60)

        # 1. Ambient noise measurement (2.5 seconds of silence)
        print("\n[PHASE 1/2: AMBIENT NOISE CALIBRATION]")
        print("Please stay silent while I calibrate the microphone (2.5 seconds)...")
        time.sleep(0.3)
        ambient_samples = int(self.recorder.sample_rate * 2.5)
        rec_ambient = sd.rec(ambient_samples, samplerate=self.recorder.sample_rate, channels=self.recorder.channels, dtype="int16")
        sd.wait()
        raw_amb = rec_ambient.flatten()
        frame_len = max(int(self.recorder.sample_rate * 0.025), 128)
        amb_frames = [self.recorder.calculate_rms(raw_amb[i * frame_len : (i + 1) * frame_len]) for i in range(len(raw_amb) // frame_len)]
        amb_median = float(np.median(amb_frames)) if amb_frames else self.recorder.calculate_rms(raw_amb)
        amb_peak = float(np.max(amb_frames)) if amb_frames else amb_median
        print(f"[CALIBRATION] completed (noise_floor_rms={amb_median:.1f})")

        # 2. Speech measurement (3.5 seconds of normal speech)
        print("\n[PHASE 2/2: OWNER SPEECH MEASUREMENT]")
        print("Now speak normally:\n\"Hey Jarvis, check my battery percentage.\"")
        time.sleep(0.3)
        speech_samples = int(self.recorder.sample_rate * 3.5)
        rec_speech = sd.rec(speech_samples, samplerate=self.recorder.sample_rate, channels=self.recorder.channels, dtype="int16")
        sd.wait()
        raw_speech = rec_speech.flatten()
        sp_frames = [self.recorder.calculate_rms(raw_speech[i * frame_len : (i + 1) * frame_len]) for i in range(len(raw_speech) // frame_len)]
        sorted_sp = np.sort(sp_frames)
        top_frames = sorted_sp[-max(1, len(sorted_sp) * 2 // 5):]
        sp_median = float(np.median(top_frames)) if len(top_frames) > 0 else float(np.median(sp_frames))
        sp_peak = float(np.max(sp_frames)) if len(sp_frames) > 0 else sp_median

        # SNR calculation
        snr_ratio = (sp_median + 1e-6) / (amb_median + 1e-6)
        snr_db = float(10.0 * np.log10(max(snr_ratio ** 2, 1.0)))

        # Suggested thresholds
        mult = float(getattr(self.recorder.settings, "vad_enroll_threshold_multiplier", 1.40))
        suggested_onset = max(250.0, min(amb_median * mult, 3200.0))
        suggested_continue = max(180.0, min(amb_median * 1.15, suggested_onset * 0.75))

        print("\n" + "=" * 60)
        print("      MICROPHONE DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"Ambient median RMS:            {amb_median:.1f}")
        print(f"Normal speech median RMS:      {sp_median:.1f}")
        print(f"Speech peak RMS:               {sp_peak:.1f}")
        print(f"Signal-to-Noise Ratio (SNR):   {snr_db:.1f} dB")
        print(f"Suggested onset threshold:     {suggested_onset:.1f}")
        print(f"Suggested continuation thresh: {suggested_continue:.1f}")

        if snr_db < 3.0 or sp_median <= amb_median * 1.1:
            print("\n[WARNING] Microphone signal-to-noise ratio is too low.")
        else:
            print("\n✓ Acoustic levels healthy for biometric owner voice enrollment.")
        print("=" * 60 + "\n")
        return True

    def run_calibration_verification(
        self,
        embeddings: List[np.ndarray],
        centroid: np.ndarray,
        threshold: float,
    ) -> bool:
        """
        Runs post-enrollment calibration tests across owner samples and displays:
        - owner sample scores
        - threshold
        - OWNER/UNKNOWN result
        """
        print("\n" + "=" * 55)
        print("      JARVIS OWNER VOICE CALIBRATION RESULTS")
        print("=" * 55)
        print(f"[CALIBRATION] Calibrated Threshold: {threshold:.2f}")

        all_passed = True
        scores = []
        for idx, emb in enumerate(embeddings, 1):
            dist = float(cosine(centroid, emb))
            score = float(1.0 - dist)
            scores.append(score)
            is_owner = score >= threshold
            if not is_owner:
                all_passed = False
            result_label = "OWNER" if is_owner else "UNKNOWN"
            print(f"[CALIBRATION] Sample {idx:02d}: score={score:.2f} (threshold={threshold:.2f}) -> Result={result_label}")

        min_s = min(scores) if scores else 0.0
        mean_s = sum(scores) / len(scores) if scores else 0.0
        print(f"[CALIBRATION] Sample Statistics: min={min_s:.2f}, mean={mean_s:.2f}")
        overall = "OWNER" if all_passed else "UNKNOWN"
        print(f"[CALIBRATION] Overall Verification: {overall}")
        print("=" * 55 + "\n")
        return all_passed

    def enroll_from_samples(
        self,
        audio_samples: List[bytes],
        owner_name: str = "Sanjay",
        min_required: int = 8,
        replace: bool = True,
    ) -> bool:
        """
        Enrolls owner profile from a list of validated audio samples.
        Extracts embeddings, aggregates into stable centroid, and saves profile.
        """
        backup_path = None
        if replace:
            backup_path = self.backup_existing_profile()

        valid_embeddings: List[np.ndarray] = []
        for idx, sample in enumerate(audio_samples, 1):
            is_valid, reason = self.validate_audio_quality(sample)
            if not is_valid:
                print(f"[ENROLL] Sample {idx} rejected: {reason}")
                continue

            emb = self.verifier.extract_mfcc_embedding(sample)
            if emb is not None:
                valid_embeddings.append(emb)

        req = min(min_required, len(audio_samples))
        if len(valid_embeddings) < req:
            if len(valid_embeddings) < 2:
                print(f"[ENROLL] Error: Only {len(valid_embeddings)} valid samples. Need at least {req}.")
                if replace and backup_path:
                    self.restore_backup(backup_path)
                return False

        consistent = self.filter_consistent_samples(valid_embeddings)
        if len(consistent) < min(2, req):
            print("[ENROLL] Error: Samples too inconsistent across recordings.")
            if replace and backup_path:
                self.restore_backup(backup_path)
            return False

        centroid = np.mean(consistent, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        calibrated_threshold, intra_sims, representatives = self.calibrate_threshold(consistent)

        self.verifier.save_profile(
            embedding=centroid,
            owner_name=owner_name,
            sample_count=len(consistent),
            calibrated_threshold=calibrated_threshold,
            representative_embeddings=representatives,
            intra_similarities=intra_sims,
        )

        print(f"[ENROLL] Valid samples={len(consistent)}")
        print("[ENROLL] Owner profile created")
        print("[ENROLL] Profile saved")

        # Optional ECAPA (PyTorch) biometric profile for the same samples.
        try:
            if self.verifier._ecapa.available():
                self._enroll_ecapa(audio_samples)
        except Exception as e:
            print(f"[ENROLL] ECAPA enrollment skipped: {e}")

        self.run_calibration_verification(consistent, centroid, calibrated_threshold)
        return True

    def _enroll_ecapa(self, audio_samples: List[bytes]) -> bool:
        """
        Builds ECAPA 192-d embeddings from the already captured samples.
        Keeps previous profile if validation fails (fail-safe).
        """
        verifier = self.verifier
        vecs = []
        for s in audio_samples:
            v = verifier._ecapa.embedding(s)
            if v is not None:
                vecs.append(v)
        if len(vecs) < 2:
            print("[ENROLL] ECAPA: too few valid embeddings; keeping current profile")
            return False
        cent = np.mean(vecs, axis=0)
        norm = np.linalg.norm(cent)
        if norm <= 0:
            return False
        cent = cent / norm
        v = np.stack(vecs)
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        intra = [float(np.dot(cent, x)) for x in v]
        # representatives = samples close to centroid (>=92% of mean intra similarity)
        reps = [vec for vec, s in zip(v, intra) if s >= float(np.mean(intra)) * 0.92][:8]
        ok = verifier.save_ecapa_profile(cent, reps, sample_count=len(vecs), intra_similarities=intra)
        if ok:
            print(f"[ENROLL] ECAPA profile saved (samples={len(vecs)}, dim=192)")
        return ok

    def run_interactive_enrollment(
        self,
        owner_name: str = "Sanjay",
        target_samples: int = 10,
        min_samples: int = 8,
        replace: bool = True,
    ) -> bool:
        """
        Interactive terminal workflow for recording 8-10 clean owner voice samples.
        Prompts user, enforces separate calibration and speech phases,
        limits failed attempts to max 3 per sample, extracts embeddings,
        aggregates into a stable centroid + representative profile,
        and atomically replaces owner_profile.dat upon full validation.
        """
        backup_path = None
        if replace:
            backup_path = self.backup_existing_profile()

        total = min(target_samples, len(self.ENROLLMENT_PROMPTS))

        mic_name = "Default System Microphone"
        try:
            import sounddevice as sd
            dev_info = sd.query_devices(kind="input")
            if dev_info and "name" in dev_info:
                mic_name = dev_info["name"]
        except Exception:
            pass

        print("\n" + "=" * 60)
        print(f"      JARVIS BIOMETRIC OWNER ENROLLMENT: {owner_name.upper()}")
        print("=" * 60)
        print(f"[ENROLL] Using microphone: {mic_name}")
        print(f"[ENROLL] Sample Rate:       {self.recorder.sample_rate} Hz")
        print(f"[ENROLL] Channels:          {self.recorder.channels}")
        print(f"[ENROLL] Audio Format:      int16")
        print(f"Target: {total} clean voice samples (minimum {min_samples} required).")
        print("Please speak each phrase naturally at normal conversational volume.\n")

        temp_dir = os.path.join(get_settings().base_dir, "data", "speaker_profiles", "enrollment_temp")
        os.makedirs(temp_dir, exist_ok=True)

        accepted_samples: List[bytes] = []
        accepted_embeddings: List[np.ndarray] = []
        temp_files: List[str] = []

        prompt_idx = 0
        try:
            while len(accepted_embeddings) < total and prompt_idx < len(self.ENROLLMENT_PROMPTS):
                sample_num = len(accepted_embeddings) + 1
                prompt = self.ENROLLMENT_PROMPTS[prompt_idx]

                attempt = 1
                max_attempts = 3
                sample_accepted = False

                while attempt <= max_attempts and not sample_accepted:
                    print("-" * 55)
                    print(f"Sample {sample_num}/{total} (Attempt {attempt}/{max_attempts})")

                    # Phase 1: Separate Ambient Calibration Phase
                    print("Please stay silent while I calibrate the microphone.")
                    time.sleep(0.2)
                    self.recorder.calibrate_ambient_noise(duration=0.6)
                    print("[CALIBRATION] completed")

                    # Phase 2: Speech Phase
                    print(f"\nNow speak:\n\"{prompt}\"")
                    audio = self.recorder.record_phrase(timeout=7.0, phrase_time_limit=8.0)

                    if not audio:
                        print(f"[ENROLL] Sample {sample_num} rejected: no real speech detected\n")
                        attempt += 1
                        continue

                    print("[SPEECH] detected")
                    print("[SPEECH] completed")

                    # Validate quality
                    is_valid, reason = self.validate_audio_quality(audio)
                    stats = self.last_audio_stats
                    print(f"[QUALITY] SNR={stats['snr_db']:.1f}dB, voiced_duration={stats['voiced_duration']:.2f}s")

                    if not is_valid:
                        print(f"[ENROLL] Sample {sample_num} rejected: {reason}\n")
                        attempt += 1
                        continue

                    # Extract legacy MFCC embeddings for the .dat profile. ECAPA is
                    # enrolled separately (192-d) so the two spaces never mix.
                    emb = self.verifier.extract_mfcc_embedding(audio)
                    if emb is None:
                        print(f"[ENROLL] Sample {sample_num} rejected: embedding extraction failed\n")
                        attempt += 1
                        continue

                    # Save temporary raw sample
                    temp_path = os.path.join(temp_dir, f"sample_{sample_num:02d}.wav")
                    try:
                        with open(temp_path, "wb") as f:
                            f.write(audio)
                        temp_files.append(temp_path)
                    except Exception:
                        pass

                    accepted_samples.append(audio)
                    accepted_embeddings.append(emb)
                    sample_accepted = True
                    print(f"[ENROLL] Sample {sample_num}/{total} accepted")
                    print("[ENROLL] Embedding created\n")

                if not sample_accepted:
                    print(f"[ENROLL] Sample {sample_num} failed {max_attempts} attempts.")
                    print("[ENROLL] Microphone signal-to-noise ratio is too low.")
                    print("[ENROLL] Terminating enrollment safely without endless loop.")
                    self._cleanup_temp_samples(temp_dir, temp_files)
                    if replace and backup_path:
                        self.restore_backup(backup_path)
                    return False

                prompt_idx += 1

            # Check if minimum samples collected
            if len(accepted_embeddings) < min_samples:
                print(f"[ENROLL] Incomplete: Only {len(accepted_embeddings)} samples captured. Need at least {min_samples}.")
                self._cleanup_temp_samples(temp_dir, temp_files)
                if replace and backup_path:
                    self.restore_backup(backup_path)
                return False

            # Apply consistency filter
            consistent_embs = self.filter_consistent_samples(accepted_embeddings)
            if len(consistent_embs) < min_samples:
                print(f"[ENROLL] Error: Retained only {len(consistent_embs)} consistent samples. Need {min_samples}.")
                self._cleanup_temp_samples(temp_dir, temp_files)
                if replace and backup_path:
                    self.restore_backup(backup_path)
                return False

            # Normalize all valid embeddings & aggregate into centroid
            centroid = np.mean(consistent_embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            # Calibrate threshold & select representative embeddings
            calibrated_threshold, intra_sims, representatives = self.calibrate_threshold(consistent_embs)

            # Atomic file replacement: save to temporary file first, validate, then replace
            base_dir = get_settings().base_dir
            dat_path = os.path.join(base_dir, "data", "speaker_profiles", "owner_profile.dat")
            temp_profile_path = os.path.join(base_dir, "data", "speaker_profiles", "owner_profile.tmp.dat")

            temp_data = {
                "owner_name": owner_name,
                "embedding": centroid.tolist(),
                "dim": len(centroid),
                "calibrated_threshold": calibrated_threshold,
                "representative_embeddings": [r.tolist() for r in representatives],
                "intra_similarities": intra_sims,
                "sample_count": len(consistent_embs),
                "updated_at": time.time(),
            }

            import json
            with open(temp_profile_path, "w", encoding="utf-8") as f:
                json.dump(temp_data, f, indent=2)

            # Validate temporary profile before committing
            with open(temp_profile_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                loaded_emb = np.array(loaded.get("embedding", []))
                if len(loaded_emb) != 42 or np.any(np.isnan(loaded_emb)) or np.any(np.isinf(loaded_emb)):
                    raise ValueError("Temporary owner profile validation failed: corrupt embedding")
                if loaded.get("sample_count", 0) < min_samples:
                    raise ValueError("Temporary owner profile validation failed: insufficient samples")

            # Perform atomic replacement
            os.replace(temp_profile_path, dat_path)
            json_path = os.path.join(base_dir, "memory", "owner_voice_profile.json")
            try:
                shutil.copy2(dat_path, json_path)
            except Exception:
                pass

            self.verifier.load_profile()

            print(f"[ENROLL] Valid samples={len(consistent_embs)}")
            print("[ENROLL] Owner profile created")
            print("[ENROLL] Profile saved")

            # Build the ECAPA biometric profile from the same clean samples. If the
            # model is unavailable, the MFCC profile still works and verification
            # only fails closed when an ECAPA profile is actually present.
            try:
                if self.verifier._ecapa.available():
                    self._enroll_ecapa(accepted_samples)
            except Exception as e:
                print(f"[ENROLL] ECAPA enrollment skipped: {e}")

            self.run_calibration_verification(consistent_embs, centroid, calibrated_threshold)
            self._handle_raw_storage(temp_dir, temp_files)
            return True

        except KeyboardInterrupt:
            print("\n[ENROLL] Enrollment cancelled.")
            print("[ENROLL] Existing owner profile preserved/restored.")
            self._cleanup_temp_samples(temp_dir, temp_files)
            if replace and backup_path:
                self.restore_backup(backup_path)
            return False
        except Exception as e:
            print(f"\n[ENROLL] Unexpected error during enrollment: {e}")
            self._cleanup_temp_samples(temp_dir, temp_files)
            if replace and backup_path:
                self.restore_backup(backup_path)
            return False

    def _handle_raw_storage(self, temp_dir: str, temp_files: List[str]):
        """
        Manages raw sample storage adhering to user privacy rule:
        Default deletes raw samples after successful embedding generation.
        Only keeps samples if OWNER_VOICE_KEEP_SAMPLES=true.
        """
        keep_samples = os.getenv("OWNER_VOICE_KEEP_SAMPLES", "false").lower() == "true"
        if keep_samples:
            dest_dir = os.path.join(get_settings().base_dir, "data", "speaker_profiles", "enrollment_samples")
            os.makedirs(dest_dir, exist_ok=True)
            for f in temp_files:
                if os.path.exists(f):
                    shutil.copy2(f, os.path.join(dest_dir, os.path.basename(f)))
            print(f"[ENROLL] Raw enrollment WAV samples preserved at: {dest_dir}")

        # Always remove temp directory
        self._cleanup_temp_samples(temp_dir, temp_files)

    def _cleanup_temp_samples(self, temp_dir: str, temp_files: List[str]):
        """Removes temporary raw sample files."""
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
