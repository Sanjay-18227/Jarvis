"""
Voice Activity Detection (VAD) Subsystem
Silero VAD (official v5/v6) as the primary speech detector with graceful
fallback to the RMS/ZCR energy gate. Provides hysteresis, pre-roll buffering
support helpers, and a single shared model instance (lazy, thread-safe).

Voice flow contract:
  LISTENING -> speech candidate (RMS or VAD prob) -> speech confirmation
  (Silero prob >= ON threshold sustained) -> record phrase -> natural pause
  detection (prob < OFF threshold for end_silence_ms) -> end of utterance.
"""

import threading
import logging
import numpy as np
from typing import Optional

log = logging.getLogger("jarvis.vad")

_VAD_THRESH_ON = 0.5    # probability to declare speech confirmed
_VAD_THRESH_OFF = 0.35  # below this for end_silence_ms -> end of utterance
_MIN_SPEECH_FRAMES = 2  # >= 2 consecutive frames -> 'weak frame does not end recording'


class SileroVAD:
    """Lazy singleton wrapper around the official `silero-vad` pip package."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._model = None
                cls._instance._failed = False
        return cls._instance

    def available(self) -> bool:
        if self._failed:
            return False
        if self._model is not None:
            return True
        with SileroVAD._lock:
            if self._model is not None:
                return True
            try:
                from silero_vad import load_silero_vad
                t0_ms = None
                import time
                t0 = time.perf_counter()
                self._model = load_silero_vad()
                print(f"[PERF] vad_model_load_ms={int((time.perf_counter() - t0) * 1000)}")
                return True
            except Exception as e:
                self._failed = True
                log.warning("[VAD] silero unavailable, fallback to RMS/ZCR: %s", e)
                return False

    def speech_probability(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[float]:
        """Returns speech probability (0..1) for a 16 kHz mono f32 frame, or None if unavailable."""
        if not self.available():
            return None
        try:
            import torch
            from silero_vad import get_speech_timestamps  # noqa: F401 (models loaded lazily)
            model = self._model
            if not isinstance(audio, np.ndarray):
                return None
            c = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
            if c.dim() == 2:
                c = c.mean(dim=1)
            size = 512 if sample_rate == 16000 else 256
            if c.numel() < size:
                pad = torch.zeros(size - c.numel())
                c = torch.cat([c, pad])
            with torch.no_grad():
                prob = float(model(c, sample_rate).item())
            return prob
        except Exception as e:
            log.debug("[VAD] prob error (using fallback): %s", e)
            return None


class HybridVADStats:
    """
    Combines legacy RMS/ZCR candidate detection with Silero confirmation.
    - Candidate: RMS >= threshold (cheap, every chunk)
    - Confirm:   Silero probability >= ON threshold
    A single weak frame never ends recording (hysteresis handled by caller).
    """

    def __init__(self, silero: Optional[SileroVAD] = None):
        self.silero = silero or SileroVAD()

    def evaluate(
        self,
        audio_chunk: np.ndarray,
        rms_passed: bool,
        confidence: float,
        sample_rate: int = 16000,
    ) -> tuple[bool, float]:
        """
        Given recorder's RMS verdict, optionally confirm/refute with Silero.
        Returns (is_speech, confidence).
        """
        if not rms_passed:
            # RMS-gated already; no VAD needed for cheap rejection
            return False, 0.0
        prob = self.silero.speech_probability(audio_chunk, sample_rate)
        if prob is None:
            # legacy behavior when silero unavailable
            return rms_passed, confidence
        if prob >= _VAD_THRESH_ON:
            blended = max(confidence, prob)
            return True, float(blended)
        # RMS cross-border but VAD says noise (clicks, fans, keyboard): reject
        return False, float(prob)
