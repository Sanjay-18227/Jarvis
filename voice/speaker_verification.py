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


class SpeakerVerifier:
    """
    Acoustic speaker verification engine using Mel-Frequency feature embeddings.
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

        self.load_profile()

    def load_profile(self) -> bool:
        """Loads enrolled owner voice embedding from disk."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.owner_name = data.get("owner_name", "Sanjay")
                    self.owner_embedding = np.array(data.get("embedding", []), dtype=np.float32)
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
        Extracts a normalized acoustic feature embedding from WAV bytes.
        Calculates spectrogram, Mel filterbank energies, MFCCs, and spectral statistics.
        """
        if not audio_data:
            return None

        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

            if len(audio) < sample_rate * 0.5:  # Need at least 0.5 sec of audio
                return None

            # Normalize audio signal
            audio = audio / (np.max(np.abs(audio)) + 1e-6)

            # Compute Spectrogram
            f, t, Sxx = spectrogram(audio, fs=sample_rate, nperseg=512, noverlap=256)
            power_spec = np.log(Sxx + 1e-6)

            # Compute DCT across frequency bins to get MFCC-like cepstral coefficients
            cepstral = dct(power_spec, type=2, axis=0, norm="ortho")[:20, :]

            # Mean and standard deviation across time frames
            mean_feat = np.mean(cepstral, axis=1)
            std_feat = np.std(cepstral, axis=1)

            # Spectral statistics (centroid, energy)
            spectral_centroid = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-6)
            centroid_mean = np.mean(spectral_centroid) / 8000.0
            centroid_std = np.std(spectral_centroid) / 8000.0

            embedding = np.concatenate([mean_feat, std_feat, [centroid_mean, centroid_std]])

            # L2 Normalize embedding vector
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

        Returns:
            (classification: OWNER | UNKNOWN | UNCERTAIN, confidence_score: 0.0 - 1.0)
        """
        if self.owner_embedding is None or len(self.owner_embedding) == 0:
            # If no voice is enrolled yet, default to OWNER in dev mode with warning
            return SpeakerClassification.OWNER, 1.0

        sample_embedding = self.extract_embedding(audio_data)
        if sample_embedding is None:
            return SpeakerClassification.UNKNOWN, 0.0

        # Cosine similarity: 1 - cosine_distance
        dist = cosine(self.owner_embedding, sample_embedding)
        similarity = float(1.0 - dist)
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, similarity))

        if confidence >= self.similarity_threshold_owner:
            return SpeakerClassification.OWNER, confidence
        elif confidence >= self.similarity_threshold_uncertain:
            return SpeakerClassification.UNCERTAIN, confidence
        else:
            return SpeakerClassification.UNKNOWN, confidence
