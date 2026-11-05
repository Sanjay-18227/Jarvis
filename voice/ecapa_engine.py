"""
ECAPA-TDNN Speaker Embedding Engine (PyTorch + SpeechBrain)
Owner-voice biometrics with a 192-dimensional embedding model
(speechbrain/spkrec-ecapa-voxceleb, Apache-2.0, rev pinned in model registry).

Contract:
 - 16 kHz mono float32 input (resampled if needed)
 - normalized wave amplitude before encode
 - model runtime lives in the ModelManager cache (lazy download, external)
 - full embeddings never logged (biometric data policy)
"""

import io
import wave
import logging
import numpy as np
from typing import Optional

log = logging.getLogger("jarvis.speaker")

_MODEL_ID = "ecapa-voxceleb-speaker"


class EcapaEmbeddingEngine:
    """Lazy ECAPA-TDNN embedding extractor. `available()` never raises."""

    def __init__(self):
        self._cls = None

    def available(self) -> bool:
        if self._cls is not None:
            return True
        try:
            import torch  # noqa: F401  (must exist to even try)
            from core.models.model_manager import get_model_manager
            runtime = get_model_manager().get_runtime(_MODEL_ID)
            if runtime is None:
                log.warning("[SPEAKER] ECAPA model unavailable (not downloaded / RAM policy). "
                            "Falling back to MFCC verifier")
                return False
            self._cls = runtime
            return True
        except Exception as e:
            log.warning("[SPEAKER] ECAPA engine init failed: %s", e)
            return False

    @staticmethod
    def _trim_speech(audio: np.ndarray, sr: int = 16000,
                     max_seconds: float = 8.0) -> Optional[np.ndarray]:
        """
        Remove leading/trailing silence and keep the strongest natural speech
        cluster. Long idle/noise recordings otherwise dominate the ECAPA mean
        embedding and collapse the owner score.

        This is deliberately conservative: pauses up to 0.6 s are kept inside a
        cluster, and up to 8 s of actual speech is retained.
        """
        if audio is None or len(audio) < int(sr * 0.3):
            return audio

        frame_len = max(int(sr * 0.025), 128)
        hop = max(int(sr * 0.010), 64)
        if len(audio) < frame_len:
            return audio

        n_frames = 1 + (len(audio) - frame_len) // hop
        idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = audio[idx]
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
        peak = float(np.max(rms)) if len(rms) else 0.0
        if peak <= 1e-6:
            return None

        # Robust noise estimate; a single click must not define the floor.
        noise = float(np.percentile(rms, 20))
        threshold = max(noise * 2.5, peak * 0.035, 1e-4)
        active_idx = np.flatnonzero(rms >= threshold)
        if len(active_idx) == 0:
            return audio if len(audio) <= int(max_seconds * sr) else None

        # Group active frames into natural speech clusters, bridging short pauses.
        gap_frames = max(1, int(0.6 * sr / hop))
        clusters = []
        start = prev = int(active_idx[0])
        score = 0.0
        for i in active_idx:
            i = int(i)
            if i - prev > gap_frames:
                clusters.append((start, prev, score))
                start = i
                score = 0.0
            score += float(rms[i])
            prev = i
        clusters.append((start, prev, score))

        # Keep the cluster with the most speech energy (longest wins on ties).
        s, e, _ = max(clusters, key=lambda c: (c[2], c[1] - c[0]))
        start_sample = max(0, s * hop - int(0.20 * sr))
        end_sample = min(len(audio),
                         e * hop + frame_len + int(0.20 * sr))
        segment = audio[start_sample:end_sample]

        max_samples = int(max_seconds * sr)
        if len(segment) > max_samples:
            # Take the highest-energy window inside the selected speech cluster.
            win = max_samples
            csum = np.concatenate(
                [[0.0], np.cumsum(np.square(segment.astype(np.float64)))])
            starts = np.arange(0, len(segment) - win + 1,
                               max(1, int(0.10 * sr)))
            energies = csum[starts + win] - csum[starts]
            best = int(starts[int(np.argmax(energies))])
            segment = segment[best:best + win]

        return segment.astype("float32")

    @staticmethod
    def _decode_wav(audio_data: bytes) -> Optional[np.ndarray]:
        try:
            import scipy.signal as signal
            if audio_data[:4] == b"RIFF":
                with io.BytesIO(audio_data) as buf, wave.open(buf, "rb") as wav:
                    sr = wav.getframerate()
                    n_ch = wav.getnchannels()
                    width = wav.getsampwidth()
                    frames = wav.readframes(wav.getnframes())
            else:
                # raw 16-bit PCM mono @16k (recorder's internal representation)
                sr, n_ch, width = 16000, 1, 2
                frames = audio_data
            if width == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0
            elif width == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype("float32") / 2147483648.0
            elif width == 1:
                audio = (np.frombuffer(frames, dtype=np.uint8).astype("float32") - 128.0) / 128.0
            else:
                return None
            if n_ch > 1:
                audio = audio.reshape(-1, n_ch).mean(axis=1)
            # resample to 16kHz mono (whisper-style)
            if sr != 16000:
                from math import gcd
                g = gcd(int(sr), 16000)
                audio = signal.resample_poly(audio, 16000 // g, int(sr) // g)
            # Drop idle/silence padding before computing the utterance embedding.
            # Without this, a 30-60 s VAD turn is dominated by background audio
            # and a genuine owner can score like an impostor.
            audio = EcapaEmbeddingEngine._trim_speech(audio, 16000)
            if audio is None or len(audio) == 0:
                return None

            # normalize amplitude
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak <= 1e-6:
                return None
            audio = audio / peak * 0.9
            return audio.astype("float32")
        except Exception as e:
            log.error("[SPEAKER] WAV decode error: %s", e)
            return None

    def embedding(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Returns L2-normalized 192-dim ECAPA embedding or None."""
        if not self.available():
            return None
        audio = self._decode_wav(audio_data)
        if audio is None or len(audio) < int(16000 * 0.4):
            return None
        try:
            import torch
            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            with torch.no_grad():
                emb = self._cls.encode_batch(tensor)
            vec = emb.squeeze(0).squeeze(0).cpu().detach().numpy().astype("float32")
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm <= 1e-6:
                return None
            return vec / norm
        except Exception as e:
            log.error("[SPEAKER] ECAPA encode error: %s", e)
            return None


_engine: EcapaEmbeddingEngine | None = None


def get_ecapa_engine() -> EcapaEmbeddingEngine:
    global _engine
    if _engine is None:
        _engine = EcapaEmbeddingEngine()
    return _engine
