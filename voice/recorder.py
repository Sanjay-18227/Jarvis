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
        self.min_speech_ms = getattr(self.settings, "vad_min_speech_ms", 250)
        self.is_recording = False
        self._interrupted = False

        # Robust adaptive noise floor & hysteresis settings
        self.noise_floor_rms: float = 120.0
        self.onset_threshold: float = float(self.energy_threshold)
        self.continue_threshold: float = float(self.energy_threshold) * 0.75
        self.enroll_threshold_multiplier = float(getattr(self.settings, "vad_enroll_threshold_multiplier", 1.40))
        self.min_threshold = float(getattr(self.settings, "vad_min_threshold", 250.0))
        self.max_threshold = float(getattr(self.settings, "vad_max_threshold", 3200.0))

    def interrupt(self):
        """Immediately signals recording loop to stop."""
        self._interrupted = True

    def calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculates Root Mean Square (RMS) energy of an audio chunk."""
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32)))))

    def calculate_zero_crossing_rate(self, audio_chunk: np.ndarray) -> float:
        """Calculates normalized zero-crossing rate of an audio chunk."""
        if len(audio_chunk) <= 1:
            return 0.0
        signs = np.sign(audio_chunk)
        signs[signs == 0] = 1
        crossings = np.sum(np.abs(np.diff(signs))) / 2.0
        return float(crossings / (len(audio_chunk) - 1))

    def calculate_speech_confidence(
        self, audio_chunk: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Evaluates speech confidence using acoustic RMS energy and Zero-Crossing Rate (ZCR).
        Rejects impulsive clicks, keyboard taps, fan whines, and low-energy room noise.
        """
        rms = self.calculate_rms(audio_chunk)
        effective_threshold = threshold if threshold is not None else self.energy_threshold
        if rms < effective_threshold:
            return False, 0.0

        zcr = self.calculate_zero_crossing_rate(audio_chunk)
        # Sharp high-frequency clicks and static hiss exhibit very high ZCR (>0.45)
        if zcr > 0.45 and rms < effective_threshold * 2.0:
            return False, 0.2

        margin = min(1.0, (rms - effective_threshold) / (effective_threshold + 1e-6))
        confidence = float(0.5 + 0.5 * margin)
        return True, confidence

    def calibrate_ambient_noise(self, duration: Optional[float] = None) -> float:
        """
        Samples ambient background noise to dynamically adapt the speech threshold.
        Uses a robust statistic (median RMS over calibration frames) to ignore spikes.
        Calculates onset and continuation thresholds with safety caps and hysteresis.
        """
        import sounddevice as sd

        cal_dur = duration if duration is not None else getattr(self.settings, "vad_noise_calibration_seconds", 0.6)
        try:
            samples_to_read = int(self.sample_rate * cal_dur)
            recording = sd.rec(
                samples_to_read,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            sd.wait()

            raw_audio = recording.flatten()
            if len(raw_audio) == 0:
                return float(self.energy_threshold)

            # Split into ~25ms calibration frames to compute robust frame-level statistics
            frame_len = max(int(self.sample_rate * 0.025), 128)
            num_frames = len(raw_audio) // frame_len
            if num_frames >= 3:
                frame_rms_values = [
                    self.calculate_rms(raw_audio[i * frame_len : (i + 1) * frame_len])
                    for i in range(num_frames)
                ]
                # Discard extreme transient spikes (> 2.5x raw median)
                raw_median = float(np.median(frame_rms_values))
                valid_frames = [v for v in frame_rms_values if v <= max(raw_median * 2.5, 300.0)]
                noise_floor = float(np.median(valid_frames)) if valid_frames else raw_median
            else:
                noise_floor = self.calculate_rms(raw_audio)

            baseline = max(noise_floor, 80.0)
            self.noise_floor_rms = baseline

            if self.dynamic_threshold:
                # Adaptive speech thresholding: noise_floor * configurable_ratio (1.35 - 1.5)
                # Bounded by min_threshold and max_threshold cap so noisy calibration doesn't lock out owner
                mult = self.enroll_threshold_multiplier
                calculated_onset = baseline * mult
                self.onset_threshold = max(self.min_threshold, min(calculated_onset, self.max_threshold))

                # Hysteresis: continuation threshold is lower than onset threshold
                self.continue_threshold = max(
                    self.min_threshold * 0.75,
                    min(baseline * 1.15, self.onset_threshold * 0.75),
                )
                self.energy_threshold = int(self.onset_threshold)

                print(
                    f"[VAD] Ambient noise calibrated: noise_floor_rms={baseline:.1f}, "
                    f"threshold_rms={self.onset_threshold:.1f} (continue={self.continue_threshold:.1f})"
                )
            return baseline
        except Exception as e:
            fallback = getattr(self.settings, "vad_speech_threshold", 400)
            self.energy_threshold = fallback
            self.onset_threshold = float(fallback)
            self.continue_threshold = float(fallback) * 0.75
            return float(self.energy_threshold)

    def record_phrase(
        self,
        timeout: Optional[float] = None,
        phrase_time_limit: Optional[float] = None,
        silence_duration: Optional[float] = None,
    ) -> Optional[bytes]:
        """
        Listens until sustained speech is detected (multi-frame confirmation),
        then dynamically records while speech continues, tolerating natural pauses,
        and ending upon sustained silence.

        Args:
            timeout: Max seconds to wait for initial speech to begin.
            phrase_time_limit: Maximum safety limit (not normal cutoff).
            silence_duration: Configurable seconds of sustained silence to end utterance.

        Returns:
            WAV-encoded bytes or None if timed out without speech.
        """
        from collections import deque
        import sounddevice as sd

        # Resolve configurable thresholds from settings
        wait_timeout = timeout if timeout is not None else getattr(self.settings, "stt_timeout_seconds", 10.0)
        safety_limit = phrase_time_limit if phrase_time_limit is not None else getattr(self.settings, "max_record_seconds", 60.0)
        silence_thresh = silence_duration if silence_duration is not None else getattr(self.settings, "silence_threshold_seconds", 1.5)
        min_speech_ms = getattr(self.settings, "vad_min_speech_ms", 250)

        chunk_duration = self.chunk_size / self.sample_rate
        chunk_duration_ms = chunk_duration * 1000.0
        consecutive_required = max(2, int(min_speech_ms / chunk_duration_ms))

        self._interrupted = False
        self.is_recording = True

        frames = []
        speech_started = False
        silence_active = False
        candidate_frames = []
        last_noise_log_time = 0.0

        start_time = time.time()
        phrase_start_time = None
        last_speech_time = None
        last_continuing_log_time = None
        end_time = None

        # Pre-speech rolling buffer: ~400ms of audio before speech threshold
        pre_buffer_chunks = max(int(0.4 / chunk_duration), 6)
        pre_buffer = deque(maxlen=pre_buffer_chunks)

        self.last_latency_stats = {
            "vad_wait_seconds": 0.0,
            "speech_duration_seconds": 0.0,
            "silence_tail_seconds": 0.0,
            "total_record_seconds": 0.0,
        }

        speech_frames_count = 0
        silence_frames_count = 0
        speech_rms_list = []
        peak_rms = 0.0

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
                    if rms > peak_rms:
                        peak_rms = rms

                    if not speech_started:
                        pre_buffer.append(chunk)
                        # Onset detection using onset_threshold
                        is_speech, conf = self.calculate_speech_confidence(chunk, threshold=self.onset_threshold)

                        if is_speech or rms >= self.onset_threshold:
                            candidate_frames.append(chunk)
                            speech_rms_list.append(rms)
                            if len(candidate_frames) == 1:
                                print("[VAD] Speech candidate detected")
                            if len(candidate_frames) >= consecutive_required:
                                speech_started = True
                                phrase_start_time = now
                                last_speech_time = now
                                last_continuing_log_time = now
                                silence_active = False
                                speech_frames_count += len(candidate_frames)
                                self.last_latency_stats["vad_wait_seconds"] = round(now - start_time, 3)
                                # Prepend pre-speech buffer and candidate frames so syllables are never clipped
                                frames.extend(list(pre_buffer))
                                frames.extend(candidate_frames)
                                print("[VAD] Speech confirmed")
                                print("[VAD] Speech started")
                                print("[TURN] User turn started")
                        else:
                            # Allow candidate frames a single low frame grace before dropping
                            if len(candidate_frames) > 1:
                                # Keep candidate buffer intact for 1 low-energy transition consonant
                                pass
                            elif candidate_frames:
                                print("[VAD] Ignored background noise")
                                candidate_frames.clear()
                            else:
                                if rms < self.onset_threshold and (now - last_noise_log_time) >= 3.0:
                                    print("[VAD] Energy below speech threshold")
                                    last_noise_log_time = now

                            if (now - start_time) > wait_timeout:
                                break
                    else:
                        # Recording active speech utterance with continuation hysteresis
                        frames.append(chunk)
                        is_active_speech, _ = self.calculate_speech_confidence(chunk, threshold=self.continue_threshold)

                        if is_active_speech or rms >= self.continue_threshold:
                            speech_frames_count += 1
                            speech_rms_list.append(rms)
                            last_speech_time = now
                            if silence_active:
                                silence_active = False
                                # Resumed speaking after a short natural pause
                                print("[VAD] Speech continuing")
                                last_continuing_log_time = now
                            elif last_continuing_log_time and (now - last_continuing_log_time) >= 2.0:
                                # Periodic log during continuous speech
                                print("[VAD] Speech continuing")
                                last_continuing_log_time = now
                        else:
                            # Below continue_threshold -> potential pause / silence
                            silence_frames_count += 1
                            if not silence_active:
                                silence_active = True
                                print("[VAD] Silence detected")
                            else:
                                silence_elapsed = now - (last_speech_time or now)
                                if silence_elapsed >= silence_thresh:
                                    # Sustained continuous silence reached threshold
                                    print("[VAD] End of utterance")
                                    print("[TURN] User turn ended")
                                    end_time = now
                                    break

                        # Safety maximum duration check (NOT the normal way an utterance ends)
                        if phrase_start_time and (now - phrase_start_time > safety_limit):
                            print("[VAD] Safety limit reached (maximum duration)")
                            end_time = now
                            break

        except Exception as e:
            print(f"[Recorder] Recording error: {e}")
            return None
        finally:
            self.is_recording = False

        total_now = time.time()
        if phrase_start_time:
            self.last_latency_stats["speech_duration_seconds"] = round((end_time or total_now) - phrase_start_time, 3)
        self.last_latency_stats["total_record_seconds"] = round(total_now - start_time, 3)

        if not frames or not speech_started:
            return None

        avg_speech_rms = float(np.mean(speech_rms_list)) if speech_rms_list else 0.0
        # Summarized diagnostic log showing real audio levels and frame counts
        print(
            f"[VAD] noise_floor_rms={self.noise_floor_rms:.1f} threshold_rms={self.onset_threshold:.1f} "
            f"frame_rms={avg_speech_rms:.1f} peak_rms={peak_rms:.1f} "
            f"speech_frames={speech_frames_count} silence_frames={silence_frames_count}"
        )

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
