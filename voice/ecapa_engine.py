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
