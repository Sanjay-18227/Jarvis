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
        if profile_path:
            self.profile_path = profile_path
        else:
            base_dir = get_settings().base_dir
            self.profile_path = os.path.join(base_dir, "memory", "owner_voice_profile.json")

        self.owner_embedding: Optional[np.ndarray] = None
        self.owner_name = "Sanjay"
        self.similarity_threshold_owner = 0.82
        self.similarity_threshold_uncertain = 0.70
        self._mel_filters = _mel_filterbank(26, 512, 16000)

        self.load_profile()

    def load_profile(self) -> bool:
        """Loads enrolled owner voice embedding from disk."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.owner_name = data.get("owner_name", "Sanjay")
                    raw_emb = data.get("embedding", [])
                    if raw_emb:
                        self.owner_embedding = np.array(raw_emb, dtype=np.float32)
                        return len(self.owner_embedding) > 0
            except Exception as e:
                print(f"[SpeakerVerifier] Error loading voice profile: {e}")
        return False

    def save_profile(self, embedding: np.ndarray, owner_name: str = "Sanjay"):
        """Saves enrolled owner voice profile."""
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        self.owner_embedding = embedding
        self.owner_name = owner_name
        data = {
            "owner_name": owner_name,
            "embedding": embedding.tolist(),
            "dim": len(embedding),
        }
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[SpeakerVerifier] Owner voice profile saved for {owner_name}.")

    def extract_embedding(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Extracts a normalized acoustic Mel-frequency feature embedding (42-dim).
        Drops coefficient 0 (energy DC offset) to prevent false-positive matching.
        Normalizes and includes spectral statistics and fundamental pitch.
        """
        if not audio_data:
            return None

        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

            if len(audio) < sample_rate * 0.4:  # Need at least 0.4 sec of audio
                return None

            # Normalize audio signal
            audio = audio / (np.max(np.abs(audio)) + 1e-6)

            # Compute Spectrogram
            f, t, Sxx = spectrogram(audio, fs=sample_rate, nperseg=512, noverlap=256)

            # Mel filterbank energies
            filters = _mel_filterbank(26, 512, sample_rate) if sample_rate != 16000 else self._mel_filters
            mel_spec = np.dot(filters, Sxx)
            log_mel = np.log(mel_spec + 1e-6)

            # DCT across Mel bins to obtain MFCCs (drop c0 to avoid global energy bias)
            mfcc = dct(log_mel, type=2, axis=0, norm="ortho")
            mfcc_coeffs = mfcc[1:20, :]  # 19 coefficients

            # Mean and std across time frames
            mfcc_mean = np.mean(mfcc_coeffs, axis=1)  # 19 dims
            mfcc_std = np.std(mfcc_coeffs, axis=1)    # 19 dims

            # Spectral statistics (centroid, rolloff)
            sc = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-6)
            c_mean = np.mean(sc) / 4000.0
            c_std = np.std(sc) / 4000.0

            cum_energy = np.cumsum(Sxx, axis=0)
            total_energy = cum_energy[-1, :] + 1e-6
            rolloff_bins = np.argmax(cum_energy >= 0.85 * total_energy, axis=0)
            rolloff_mean = np.mean(f[rolloff_bins]) / 8000.0

            # Pitch / F0 estimation via autocorrelation
            autocorr = np.correlate(audio, audio, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            min_lag = max(1, int(sample_rate / 350))
            max_lag = min(len(autocorr) - 1, int(sample_rate / 80))
            if max_lag > min_lag:
                peak_lag = min_lag + np.argmax(autocorr[min_lag:max_lag])
                f0 = sample_rate / peak_lag if peak_lag > 0 else 150.0
            else:
                f0 = 150.0
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

    def verify_speaker(self, audio_data: bytes) -> Tuple[SpeakerClassification, float]:
        """
        Compares input audio against enrolled owner voice embedding.
        MANDATORY SECURITY RULE: Any failure, missing profile, or exception MUST
        return (SpeakerClassification.UNKNOWN, 0.0). NEVER fallback to OWNER.

        Returns:
            (classification: OWNER | UNKNOWN | UNCERTAIN, confidence_score: 0.0 - 1.0)
        """
        if self.owner_embedding is None or len(self.owner_embedding) == 0:
            print("[SPEAKER] similarity=0.00")
            print("[SPEAKER] result=UNKNOWN")
            return SpeakerClassification.UNKNOWN, 0.0

        sample_embedding = self.extract_embedding(audio_data)
        if sample_embedding is None:
            print("[SPEAKER] similarity=0.00")
            print("[SPEAKER] result=UNKNOWN")
            return SpeakerClassification.UNKNOWN, 0.0

        try:
            # Cosine similarity: 1 - cosine_distance
            dist = cosine(self.owner_embedding, sample_embedding)
            similarity = float(1.0 - dist)
            confidence = max(0.0, min(1.0, similarity))

            if confidence >= self.similarity_threshold_owner:
                result = SpeakerClassification.OWNER
            elif confidence >= self.similarity_threshold_uncertain:
                result = SpeakerClassification.UNCERTAIN
            else:
                result = SpeakerClassification.UNKNOWN

            print(f"[SPEAKER] similarity={confidence:.2f}")
            print(f"[SPEAKER] result={result.value}")
            return result, confidence

        except Exception as e:
            print(f"[SpeakerVerifier] Verification error: {e}")
            print("[SPEAKER] similarity=0.00")
            print("[SPEAKER] result=UNKNOWN")
            return SpeakerClassification.UNKNOWN, 0.0
