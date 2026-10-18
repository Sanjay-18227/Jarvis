"""
Audio Recorder & Voice Activity Detection (VAD)
Captures microphone audio using sounddevice/PyAudio with real-time acoustic energy monitoring.
"""

import io
import time
import wave
import numpy as np
from typing import Optional, Tuple
from config.settings import Settings, get_settings


class AudioRecorder:
    """
    High-performance audio recorder with energy-based Voice Activity Detection (VAD).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        settings: Optional[Settings] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.settings = settings or get_settings()

        self.energy_threshold = self.settings.energy_threshold
        self.dynamic_threshold = self.settings.dynamic_energy_threshold
        self.is_recording = False
        self._interrupted = False

    def interrupt(self):
        """Immediately signals recording loop to stop."""
        self._interrupted = True

    def calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculates Root Mean Square (RMS) energy of an audio chunk."""
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32)))))

    def calibrate_ambient_noise(self, duration: float = 1.0) -> float:
        """
        Samples ambient background noise to dynamically adapt the energy threshold.
        """
        import sounddevice as sd

        try:
            samples_to_read = int(self.sample_rate * duration)
            recording = sd.rec(
                samples_to_read,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            sd.wait()
            rms = self.calculate_rms(recording)
            if self.dynamic_threshold and rms > 0:
                self.energy_threshold = max(int(rms * 1.5), 250)
            return rms
        except Exception as e:
            print(f"[Recorder] Noise calibration notice: {e}")
            return float(self.energy_threshold)

    def record_phrase(
        self,
        timeout: float = 6.0,
        phrase_time_limit: float = 12.0,
        silence_duration: float = 0.9,
    ) -> Optional[bytes]:
        """
        Listens until speech is detected, then records until silence persists.

        Args:
            timeout: Max seconds to wait for speech to begin.
            phrase_time_limit: Max seconds of speech to record.
            silence_duration: Seconds of trailing silence before ending phrase.

        Returns:
            WAV-encoded bytes or None if timed out without speech.
        """
        import sounddevice as sd

        self._interrupted = False
        self.is_recording = True

        frames = []
        speech_started = False
        silence_start_time = None
        start_time = time.time()
        phrase_start_time = None

        chunk_duration = self.chunk_size / self.sample_rate
        silence_chunks_needed = int(silence_duration / chunk_duration)
        silence_counter = 0

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.chunk_size,
            ) as stream:
                while not self._interrupted:
                    now = time.time()
                    data, _ = stream.read(self.chunk_size)
                    chunk = data.flatten()
                    rms = self.calculate_rms(chunk)

                    if not speech_started:
                        # Waiting for speech to begin
                        if rms > self.energy_threshold:
                            speech_started = True
                            phrase_start_time = now
                            frames.append(chunk)
                            silence_counter = 0
                        elif now - start_time > timeout:
                            # Timed out waiting for speech
                            break
                    else:
                        # Recording speech phrase
                        frames.append(chunk)

                        if rms < self.energy_threshold:
                            silence_counter += 1
                            if silence_counter >= silence_chunks_needed:
                                # Silence threshold reached, end phrase
                                break
                        else:
                            silence_counter = 0

                        # Check maximum phrase limit
                        if phrase_start_time and (now - phrase_start_time > phrase_time_limit):
                            break

        except Exception as e:
            print(f"[Recorder] Recording error: {e}")
            return None
        finally:
            self.is_recording = False

        if not frames or not speech_started:
            return None

        # Convert recorded frames into standard WAV bytes
        raw_audio = np.concatenate(frames).tobytes()
        return self._create_wav_bytes(raw_audio)

    def _create_wav_bytes(self, raw_data: bytes) -> bytes:
        """Encodes raw PCM 16-bit mono audio into WAV format."""
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(raw_data)
        return wav_io.getvalue()
