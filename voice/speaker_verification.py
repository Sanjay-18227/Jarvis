"""
Speaker Verification Subsystem
Extracts acoustic MFCC embeddings and classifies speakers into OWNER, UNKNOWN, or UNCERTAIN.
Guarantees that wake word presence alone never grants owner privileges.
"""

import io
import wave
import json
import os
import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial.distance import cosine
from scipy.signal import spectrogram
from scipy.fft import dct

from config.constants import SpeakerClassification
from config.settings import get_settings



def _mel_filterbank(num_filters: int, nfft: int, sample_rate: int) -> np.ndarray:
    """Computes Mel-scale triangular filterbank matrix."""
    low_freq = 100
    high_freq = sample_rate / 2
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_filters, int(nfft / 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            filters[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1] + 1e-6)
        for k in range(f_m, f_m_plus):
            filters[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m] + 1e-6)
    return filters


class SpeakerVerifier:
    """
    Acoustic speaker verification engine using Mel-Frequency feature embeddings.
    Strictly adheres to security principle: verification failure -> UNKNOWN.
    """

    def __init__(self, profile_path: Optional[str] = None):
        base_dir = get_settings().base_dir
        self.dat_path = os.path.join(base_dir, "data", "speaker_profiles", "owner_profile.dat")
        self.json_path = os.path.join(base_dir, "memory", "owner_voice_profile.json")

        self.custom_profile = bool(profile_path)
        if profile_path:
            self.profile_path = profile_path
        elif os.path.exists(self.dat_path):
            self.profile_path = self.dat_path
        elif os.path.exists(self.json_path):
            self.profile_path = self.json_path
        else:
            self.profile_path = self.dat_path

        self.owner_embedding: Optional[np.ndarray] = None
        self.representative_embeddings: List[np.ndarray] = []
        self.owner_name = "Sanjay"
        settings = get_settings()
        self.similarity_threshold_owner = getattr(settings, "owner_similarity_threshold", 0.78)
        self.similarity_threshold_uncertain = getattr(settings, "uncertain_similarity_threshold", 0.70)
        self._mel_filters = _mel_filterbank(26, 512, 16000)

        self.load_profile()

    def load_profile(self) -> bool:
        """Loads enrolled owner voice embedding and representative embeddings from disk with fail-closed validation."""
        paths_to_try = [self.profile_path] if self.custom_profile else [self.profile_path, self.dat_path, self.json_path]
        for p in paths_to_try:
            if not p or not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.owner_name = data.get("owner_name", "Sanjay")
                    if "calibrated_threshold" in data:
                        try:
                            thresh = float(data["calibrated_threshold"])
                            if 0.50 <= thresh <= 0.95:
                                self.similarity_threshold_owner = thresh
                        except (ValueError, TypeError):
                            pass
                    raw_emb = data.get("embedding", [])
                    if raw_emb and len(raw_emb) == 42:
                        arr = np.array(raw_emb, dtype=np.float32)
                        # Reject corrupt embeddings (NaN, Inf, or zero-norm)
                        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                            continue
                        norm = np.linalg.norm(arr)
                        if norm <= 1e-4:
                            continue
                        self.owner_embedding = arr
                        self.profile_path = p

                        # Load representative embeddings if available
                        self.representative_embeddings = []
                        raw_reps = data.get("representative_embeddings", [])
                        if isinstance(raw_reps, list):
                            for item in raw_reps:
                                if isinstance(item, list) and len(item) == 42:
                                    arr_rep = np.array(item, dtype=np.float32)
                                    if not (np.any(np.isnan(arr_rep)) or np.any(np.isinf(arr_rep))):
                                        if np.linalg.norm(arr_rep) > 1e-4:
                                            self.representative_embeddings.append(arr_rep)

                        print("[SPEAKER] Owner profile loaded")
                        print(f"[SPEAKER] Threshold={self.similarity_threshold_owner:.2f}")
                        if self.representative_embeddings:
                            print(f"[SPEAKER] Loaded {len(self.representative_embeddings)} representative owner embeddings")
                        return True
            except Exception as e:
                print(f"[SpeakerVerifier] Error loading voice profile from {p}: {e}")
        self.owner_embedding = None
        self.representative_embeddings = []
        return False

    def save_profile(
        self,
        embedding: np.ndarray,
        owner_name: str = "Sanjay",
        sample_count: int = 1,
        calibrated_threshold: Optional[float] = None,
        representative_embeddings: Optional[List[np.ndarray]] = None,
        intra_similarities: Optional[List[float]] = None,
    ):
        """Saves enrolled owner voice profile to disk."""
        import time
        self.owner_embedding = embedding
        self.owner_name = owner_name
        if representative_embeddings:
            self.representative_embeddings = representative_embeddings
        if calibrated_threshold is not None:
            self.similarity_threshold_owner = calibrated_threshold

        data = {
            "owner_name": owner_name,
            "embedding": embedding.tolist(),
            "dim": len(embedding),
            "calibrated_threshold": self.similarity_threshold_owner,
            "representative_embeddings": [r.tolist() for r in (representative_embeddings or self.representative_embeddings)],
            "intra_similarities": intra_similarities or [],
            "sample_count": sample_count,
            "updated_at": time.time(),
        }

        paths_to_save = [self.profile_path] if self.custom_profile else set([self.profile_path, self.dat_path, self.json_path])
        for path in paths_to_save:
            if not path:
                continue
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[SpeakerVerifier] Error writing profile to {path}: {e}")

        print(f"[SpeakerVerifier] Owner voice profile saved for {owner_name}.")


    def extract_embedding(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Extracts a normalized acoustic Mel-frequency feature embedding (42-dim).
        Performs robust preprocessing:
        1. Decodes 16-bit PCM WAV.
        2. Normalizes peak amplitude.
        3. Energy-based VAD trimming to eliminate pre-buffer and trailing silence.
        4. Pre-emphasis filtering to flatten spectral tilt.
        5. Mel filterbank energies (26 filters) over active speech frames.
        6. Discrete Cosine Transform (DCT) for MFCCs (dropping c0 energy DC bias).
        7. Mean and standard deviation across active speech frames (19 + 19 = 38 dims).
        8. Spectral centroid (mean, std), spectral rolloff, and short-time voiced F0 pitch (4 dims).
        9. L2 normalization into a 42-dimensional unit vector.
        """
        if not audio_data:
            return None

        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

            if len(audio) < sample_rate * 0.3:  # Need at least 0.3 sec of audio
                return None

            # Amplitude normalization
            max_amp = np.max(np.abs(audio))
            if max_amp < 1e-4:
                return None
            audio = audio / max_amp

            # VAD: Trim leading and trailing silence
            frame_len = int(sample_rate * 0.025)
            hop_len = int(sample_rate * 0.010)
            if len(audio) >= frame_len:
                num_frames = (len(audio) - frame_len) // hop_len
                if num_frames >= 5:
                    energies = np.array([
                        np.mean(audio[i * hop_len : i * hop_len + frame_len] ** 2)
                        for i in range(num_frames)
                    ])
                    peak_e = np.max(energies)
                    thr = peak_e * 0.03
                    active = np.where(energies > thr)[0]
                    if len(active) >= 3:
                        start_idx = max(0, active[0] * hop_len - int(sample_rate * 0.05))
                        end_idx = min(len(audio), active[-1] * hop_len + frame_len + int(sample_rate * 0.05))
                        audio = audio[start_idx:end_idx]

            if len(audio) < sample_rate * 0.3:
                return None

            # Pre-emphasis filter (alpha = 0.97)
            emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

            # Spectrogram: 25ms window, 10ms hop, nfft=512
            nperseg = min(512, max(256, int(sample_rate * 0.025)))
            noverlap = int(nperseg * 0.6)
            f, t, Sxx = spectrogram(emphasized, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, nfft=512)

            # Active speech frame selection
            fe = np.sum(Sxx, axis=0)
            sp_mask = fe > (np.max(fe) * 0.04)
            if np.sum(sp_mask) < 3:
                sp_mask = fe > (np.max(fe) * 0.01)
            Sxx_speech = Sxx[:, sp_mask] if np.sum(sp_mask) >= 2 else Sxx

            # Mel filterbank energies
            filters = _mel_filterbank(26, 512, sample_rate) if sample_rate != 16000 else self._mel_filters
            mel_spec = np.dot(filters, Sxx_speech)
            log_mel = np.log(mel_spec + 1e-6)

            # DCT across Mel bins to obtain MFCCs (drop c0 to avoid global energy bias)
            mfcc = dct(log_mel, type=2, axis=0, norm="ortho")
            mfcc_coeffs = mfcc[1:20, :]  # 19 coefficients

            # Mean and std across speech frames
            mfcc_mean = np.mean(mfcc_coeffs, axis=1)  # 19 dims
            mfcc_std = np.std(mfcc_coeffs, axis=1)    # 19 dims

            # Spectral statistics (centroid, rolloff)
            sc = np.sum(f[:, None] * Sxx_speech, axis=0) / (np.sum(Sxx_speech, axis=0) + 1e-6)
            c_mean = np.mean(sc) / 4000.0
            c_std = np.std(sc) / 4000.0

            cum_energy = np.cumsum(Sxx_speech, axis=0)
            total_energy = cum_energy[-1, :] + 1e-6
            rolloff_bins = np.argmax(cum_energy >= 0.85 * total_energy, axis=0)
            rolloff_mean = np.mean(f[rolloff_bins]) / 8000.0

            # Short-time F0 estimation on voiced frames
            frame_len_pitch = int(sample_rate * 0.03)
            hop_len_pitch = int(sample_rate * 0.015)
            f0_list = []
            min_lag = max(1, int(sample_rate / 350))
            max_lag = min(frame_len_pitch - 1, int(sample_rate / 75))
            if len(audio) >= frame_len_pitch:
                for i in range(0, len(audio) - frame_len_pitch, hop_len_pitch):
                    seg = audio[i : i + frame_len_pitch]
                    if np.mean(seg ** 2) > 0.01:
                        ac = np.correlate(seg, seg, mode="full")
                        ac = ac[len(ac) // 2 :]
                        if max_lag < len(ac):
                            peak = min_lag + np.argmax(ac[min_lag:max_lag])
                            if ac[peak] > 0.3 * ac[0]:
                                f0_list.append(sample_rate / peak)
            f0 = float(np.median(f0_list)) if len(f0_list) >= 3 else 150.0
            f0_norm = f0 / 300.0

            # Concatenate to 42 dimensions: 19 + 19 + 4 = 42
            embedding = np.concatenate([mfcc_mean, mfcc_std, [c_mean, c_std, rolloff_mean, f0_norm]])

            # Mean-center and L2-normalize
            embedding = embedding - np.mean(embedding)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            print(f"[SpeakerVerifier] Embedding extraction error: {e}")
            return None

    def verify_speaker(
        self,
        audio_data: bytes,
        allow_session_continuity: bool = True,
        is_high_risk: bool = False,
    ) -> Tuple[SpeakerClassification, float]:
        """
        Compares input audio against enrolled owner voice embedding using a three-tier confidence policy:
        - HIGH CONFIDENCE (>= threshold): OWNER
        - CLEARLY LOW CONFIDENCE (< 0.65): UNKNOWN
        - BORDERLINE (0.65 <= score < threshold): Perform secondary verification via recent
          authenticated-session continuity for low-risk actions. High-risk actions always require fresh high confidence.
        """
        print("[SPEAKER] Verification started")
        if self.owner_embedding is None or len(self.owner_embedding) == 0:
            print("[SPEAKER] Similarity=0.00")
            print(f"[SPEAKER] Threshold={self.similarity_threshold_owner:.2f}")
            print("[SPEAKER] Result=UNKNOWN")
            return SpeakerClassification.UNKNOWN, 0.0

        try:
            sample_embedding = self.extract_embedding(audio_data)
            if sample_embedding is None:
                print("[SPEAKER] Similarity=0.00")
                print(f"[SPEAKER] Threshold={self.similarity_threshold_owner:.2f}")
                print("[SPEAKER] Result=UNKNOWN")
                return SpeakerClassification.UNKNOWN, 0.0

            # Cosine similarity against centroid: 1 - cosine_distance
            dist = cosine(self.owner_embedding, sample_embedding)
            if np.isnan(dist) or np.isinf(dist):
                print("[SPEAKER] Similarity=0.00")
                print(f"[SPEAKER] Threshold={self.similarity_threshold_owner:.2f}")
                print("[SPEAKER] Result=UNKNOWN")
                return SpeakerClassification.UNKNOWN, 0.0

            sim_centroid = float(1.0 - dist)
            top_rep_sim = 0.0
            if self.representative_embeddings:
                rep_scores = []
                for rep_emb in self.representative_embeddings:
                    d = cosine(rep_emb, sample_embedding)
                    if not (np.isnan(d) or np.isinf(d)):
                        rep_scores.append(float(1.0 - d))
                if rep_scores:
                    top_rep_sim = max(rep_scores)

            # Robust score aggregation: combination of centroid and top representative match
            if top_rep_sim > 0.0:
                similarity = max(sim_centroid, 0.65 * sim_centroid + 0.35 * top_rep_sim)
            else:
                similarity = sim_centroid

            confidence = max(0.0, min(1.0, similarity))
            borderline_lower = getattr(self, "similarity_threshold_uncertain", 0.65)

            if confidence >= self.similarity_threshold_owner:
                # Tier 1: HIGH CONFIDENCE -> OWNER
                result = SpeakerClassification.OWNER
            elif confidence < borderline_lower:
                # Tier 2: CLEARLY LOW CONFIDENCE -> UNKNOWN
                result = SpeakerClassification.UNKNOWN
                try:
                    from security.authorization_gate import get_speaker_session
                    get_speaker_session().invalidate("Speaker confidence clearly below threshold")
                except Exception:
                    pass
            else:
                # Tier 3: BORDERLINE -> Secondary verification
                # Check top representative embedding or recent authenticated session continuity
                try:
                    from security.authorization_gate import get_speaker_session
                    session = get_speaker_session()
                    if top_rep_sim >= self.similarity_threshold_owner:
                        result = SpeakerClassification.OWNER
                        confidence = top_rep_sim
                        print(f"[SPEAKER] Secondary verification: Borderline score accepted via representative match ({top_rep_sim:.2f})")
                    elif allow_session_continuity and not is_high_risk and session.is_owner_active():
                        result = SpeakerClassification.OWNER
                        print(f"[SPEAKER] Secondary verification: Borderline score {confidence:.2f} accepted via active session continuity")
                    else:
                        result = SpeakerClassification.UNKNOWN
                        print(f"[SPEAKER] Secondary verification: Borderline score {confidence:.2f} rejected (fresh authentication required)")
                except Exception:
                    result = SpeakerClassification.UNKNOWN

            print(f"[SPEAKER] Similarity={confidence:.2f}")
            print(f"[SPEAKER] Threshold={self.similarity_threshold_owner:.2f}")
            print(f"[SPEAKER] Result={result.value}")
            return result, confidence

        except Exception as e:
            print(f"[SpeakerVerifier] Verification error: {e}")
            print("[SPEAKER] Similarity=0.00")
            print(f"[SPEAKER] Threshold={self.similarity_threshold_owner:.2f}")
            print("[SPEAKER] Result=UNKNOWN")
            return SpeakerClassification.UNKNOWN, 0.0
