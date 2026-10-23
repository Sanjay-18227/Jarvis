"""
Barge-In & Real-time Interruption Subsystem
Maintains a lightweight interruption listener while TTS is speaking.
Provides self-echo protection, confidence thresholds, and immediate playback cancellation.
"""

import re
import time
import io
import wave
import numpy as np
from typing import Optional, List
from config.settings import Settings, get_settings
from voice.stt import SpeechToText
from voice.speaker_verification import SpeakerVerifier


INTERRUPTION_PHRASES = [
    "stop",
    "jarvis stop",
    "stop jarvis",
    "wait",
    "jarvis wait",
    "hold on",
    "cancel",
    "enough",
    "shut up",
    "please stop",
    "stop please",
    "stop it",
    "stop talking",
    "quiet",
    "be quiet",
    "pause",
    "halt",
]

CORE_INTERRUPTION_WORDS = {
    "stop",
    "wait",
    "cancel",
    "enough",
    "pause",
    "halt",
    "quiet",
}


def normalize_text(text: str) -> str:
    """Removes punctuation and normalizes whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


class BargeInDetector:
    """
    Real-time interruption monitor that runs during TTS playback.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        stt: Optional[SpeechToText] = None,
        verifier: Optional[SpeakerVerifier] = None,
    ):
        self.settings = settings or get_settings()
        self.stt = stt or SpeechToText(settings=self.settings)
        self.verifier = verifier

        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024

    def is_interruption(
        self,
        text: str,
        current_speaking_text: Optional[str] = None,
    ) -> Optional[str]:
        """
        Evaluates transcribed text to determine if it is a genuine interruption command.
        Applies self-echo protection against the text JARVIS is currently speaking.

        Returns:
            Matched interruption phrase or None.
        """
        if not text:
            return None

        clean_user = normalize_text(text)
        if not clean_user:
            return None

        user_words = clean_user.split()

        # SELF-ECHO PROTECTION LAYER 1:
        # Check if the transcribed text matches words from JARVIS's own current output
        if current_speaking_text:
            clean_tts = normalize_text(current_speaking_text)
            tts_words = set(clean_tts.split())

            # If user utterance is an exact subset of words from current TTS sentence
            # AND does not explicitly contain a core interruption word:
            has_core_interruption = any(w in CORE_INTERRUPTION_WORDS for w in user_words)
            if not has_core_interruption:
                # If all words detected are in the active TTS text, it is pure acoustic feedback
                if all(w in tts_words for w in user_words):
                    return None

        # Check exact matches first
        for phrase in INTERRUPTION_PHRASES:
            if clean_user == phrase:
                return phrase

        # Check multi-word phrases in descending order of length
        sorted_phrases = sorted(INTERRUPTION_PHRASES, key=lambda p: len(p.split()), reverse=True)
        for phrase in sorted_phrases:
            if (
                clean_user.startswith(phrase + " ")
                or clean_user.endswith(" " + phrase)
                or f" {phrase} " in f" {clean_user} "
            ):
                return phrase

        # Check two-word combinations like "shut up", "hold on"
        if "shut up" in clean_user:
            return "shut up"
        if "hold on" in clean_user:
            return "hold on"

        # Check single core interruption words for brief commands (<= 4 words total)
        if len(user_words) <= 4:
            for word in user_words:
                if word in CORE_INTERRUPTION_WORDS:
                    return f"jarvis {word}" if "jarvis" in user_words else word

        return None

    def calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculates RMS energy of an audio chunk."""
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32)))))

    def _create_wav_bytes(self, raw_data: bytes) -> bytes:
        """Encodes raw PCM 16-bit mono audio into WAV format."""
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(raw_data)
        return wav_io.getvalue()

    def listen_for_interruption(
        self,
        tts,
        current_speaking_text: str = "",
        poll_interval: float = 0.05,
    ) -> Optional[str]:
        """
        Monitors microphone during active TTS playback.

        Args:
            tts: Active TextToSpeech instance (provides is_speaking and stop()).
            current_speaking_text: The text being spoken for self-echo shielding.
            poll_interval: Sleep interval for polling.

        Returns:
            Detected interruption phrase or None if playback completes uninterrupted.
        """
        import sounddevice as sd

        # SELF-ECHO PROTECTION LAYER 2: Initial cooldown
        # Ignore mic for the first ~250ms of playback to avoid initial speaker pop / transient
        cooldown_seconds = getattr(self.settings, "barge_in_cooldown_seconds", 0.25)
        start_wait = time.time()
        while tts.is_speaking and (time.time() - start_wait < cooldown_seconds):
            time.sleep(0.04)

        if not tts.is_speaking:
            return None

        # SELF-ECHO PROTECTION LAYER 3: Boosted energy threshold during playback
        # Set threshold higher than background noise to require assertive user voice over speaker
        barge_threshold = max(int(self.settings.energy_threshold * 1.4), 450)

        burst_frames = []
        burst_collecting = False
        burst_start_time = 0.0
        max_burst_seconds = 1.2
        silence_burst_chunks = 0
        chunks_for_burst_silence = 4  # ~250ms of trailing silence in burst

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.chunk_size,
            ) as stream:
                while tts.is_speaking:
                    data, _ = stream.read(self.chunk_size)
                    chunk = data.flatten()
                    rms = self.calculate_rms(chunk)
                    now = time.time()

                    if not burst_collecting:
                        if rms >= barge_threshold:
                            burst_collecting = True
                            burst_start_time = now
                            burst_frames = [chunk]
                            silence_burst_chunks = 0
                    else:
                        burst_frames.append(chunk)
                        if rms < barge_threshold:
                            silence_burst_chunks += 1
                        else:
                            silence_burst_chunks = 0

                        burst_duration = now - burst_start_time
                        # Finished collecting short utterance burst
                        if silence_burst_chunks >= chunks_for_burst_silence or burst_duration >= max_burst_seconds:
                            burst_collecting = False
                            if len(burst_frames) >= 6:  # At least ~380ms of audio
                                raw_bytes = np.concatenate(burst_frames).tobytes()
                                wav_data = self._create_wav_bytes(raw_bytes)

                                # Fast STT transcription
                                text = self.stt.transcribe(
                                    wav_data,
                                    prompt="stop, wait, cancel, hold on, Jarvis stop, enough, shut up",
                                )

                                if text:
                                    matched = self.is_interruption(text, current_speaking_text)
                                    if matched:
                                        # Genuine interruption verified!
                                        return matched

                            burst_frames = []
                            silence_burst_chunks = 0

        except Exception as e:
            # Device error or stream interruption
            return None

        return None
