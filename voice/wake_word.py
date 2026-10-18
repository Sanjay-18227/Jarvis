"""
JARVIS Wake Word Detection Subsystem
Listens locally for "Hey Jarvis" or "Jarvis" with low-power acoustic monitoring.
"""

import time
import threading
from typing import Callable, Optional
from config.settings import Settings, get_settings
from voice.recorder import AudioRecorder
from voice.stt import SpeechToText


class WakeWordDetector:
    """
    Continuous local wake-word listener.
    Fires callback upon hearing "hey jarvis" or "jarvis".
    """

    WAKE_PHRASES = ["hey jarvis", "jarvis", "ok jarvis", "hi jarvis", "ஜாவிஸ்"]

    def __init__(
        self,
        on_wake: Optional[Callable[[], None]] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.on_wake = on_wake
        self.recorder = AudioRecorder(settings=self.settings)
        self.stt = SpeechToText(settings=self.settings)

        self._enabled = True
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self):
        """Enables wake-word detection."""
        self._enabled = True
        print("[WakeWord] Detection enabled.")

    def disable(self):
        """Disables wake-word detection completely."""
        self._enabled = False
        print("[WakeWord] Detection disabled.")

    def start_listening(self):
        """Starts background wake-word listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[WakeWord] Background listener started.")

    def stop_listening(self):
        """Stops background wake-word listener thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[WakeWord] Background listener stopped.")

    def check_audio_for_wake_word(self, audio_data: bytes) -> bool:
        """
        Transcribes a short audio chunk to check for wake phrases.
        """
        if not audio_data or not self._enabled:
            return False

        try:
            # Quick transcription using STT
            text = self.stt.transcribe(audio_data, language="en-US", prompt="Jarvis, Hey Jarvis")
            if not text:
                return False

            text_clean = text.lower().strip()
            for phrase in self.WAKE_PHRASES:
                if phrase in text_clean:
                    return True
            return False
        except Exception as e:
            print(f"[WakeWord] Error evaluating audio: {e}")
            return False

    def _listen_loop(self):
        """Continuous background listening loop."""
        while self._running:
            if not self._enabled:
                time.sleep(0.5)
                continue

            try:
                # Record short phrase (1.5 - 3 seconds)
                audio = self.recorder.record_phrase(timeout=2.0, phrase_time_limit=3.0, silence_duration=0.6)
                if audio and self.check_audio_for_wake_word(audio):
                    print("[WakeWord] Wake word detected!")
                    if self.on_wake:
                        self.on_wake()
            except Exception as e:
                time.sleep(0.5)
