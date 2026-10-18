"""
Text-to-Speech (TTS) Subsystem
High-quality neural speech synthesis with Microsoft Edge TTS, in-memory decoding,
Windows SAPI5 fallback, and speech text sanitization.
"""

import re
import asyncio
import threading
from typing import Optional
from config.settings import Settings, get_settings


def sanitize_for_speech(text: str) -> str:
    """
    Cleans response text of markdown formatting, code snippets, URLs, and symbols
    so JARVIS sounds natural and articulate when speaking.
    """
    if not text:
        return ""

    # Remove code blocks entirely or summarize them
    text = re.sub(r"```[\s\S]*?```", " [code snippet omitted] ", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove markdown links: [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove headers (### Header) with optional leading spaces
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove markdown bold/italics (**, *, __, _)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove bullet points and list markers (*, -, 1.)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove blockquotes
    text = re.sub(r"^\s*>\s+", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^-{3,}|^\*{3,}", "", text, flags=re.MULTILINE)
    # Remove XML / HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


class TextToSpeech:
    """
    Neural Text-to-Speech engine supporting Edge TTS and offline SAPI5 fallback.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.voice = self.settings.tts_voice
        self.rate = self.settings.tts_rate
        self.volume = self.settings.tts_volume
        self.pitch = self.settings.tts_pitch

        self._is_speaking = False
        self._stop_event = threading.Event()
        self._pyttsx3_engine = None

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def stop(self):
        """Interrupts and stops active audio playback immediately."""
        self._stop_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._is_speaking = False

    def speak(self, text: str, voice: Optional[str] = None) -> bool:
        """
        Synthesizes and speaks text synchronously.

        Args:
            text: Text to speak.
            voice: Optional override voice ID.

        Returns:
            True if spoken successfully, False otherwise.
        """
        clean_text = sanitize_for_speech(text)
        if not clean_text:
            return False

        self._stop_event.clear()
        target_voice = voice or self.voice

        # Attempt Edge TTS first if selected
        if self.settings.tts_engine == "edge_tts":
            success = self._speak_edge_tts(clean_text, target_voice)
            if success:
                return True
            print("[TTS] Edge TTS synthesis failed. Falling back to SAPI5.")

        # Fallback to local SAPI5 (pyttsx3)
        return self._speak_pyttsx3(clean_text)

    def _speak_edge_tts(self, text: str, voice: str) -> bool:
        """Synthesize using Edge TTS and play via miniaudio + sounddevice."""
        try:
            import edge_tts
            import miniaudio
            import sounddevice as sd

            async def _synthesize():
                communicate = edge_tts.Communicate(
                    text,
                    voice=voice,
                    rate=self.rate,
                    volume=self.volume,
                    pitch=self.pitch,
                )
                audio_buffer = bytearray()
                async for chunk in communicate.stream():
                    if self._stop_event.is_set():
                        return None
                    if chunk["type"] == "audio":
                        audio_buffer.extend(chunk["data"])
                return bytes(audio_buffer)

            # Run async synthesis loop
            audio_bytes = asyncio.run(_synthesize())
            if not audio_bytes or self._stop_event.is_set():
                return False

            # Decode in memory
            decoded = miniaudio.decode(audio_bytes, nchannels=1, sample_rate=24000)

            # Play via sounddevice
            self._is_speaking = True
            sd.play(decoded.samples, samplerate=decoded.sample_rate)
            while sd.get_stream().active:
                if self._stop_event.is_set():
                    sd.stop()
                    break
                sd.sleep(50)

            self._is_speaking = False
            return True

        except Exception as e:
            print(f"[TTS] Edge TTS playback notice: {e}")
            self._is_speaking = False
            return False

    def _speak_pyttsx3(self, text: str) -> bool:
        """Offline fallback using Windows SAPI5 via pyttsx3."""
        try:
            import pyttsx3

            self._is_speaking = True
            engine = pyttsx3.init()
            engine.setProperty("rate", 185)
            engine.say(text)
            engine.runAndWait()
            self._is_speaking = False
            return True
        except Exception as e:
            print(f"[TTS] Pyttsx3 error: {e}")
            self._is_speaking = False
            return False
