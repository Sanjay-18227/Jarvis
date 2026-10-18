"""
Speech-to-Text (STT) Subsystem
Transcribes audio using Groq Whisper API with automatic fallback to Google Speech Recognition.
"""

import io
import tempfile
import os
from typing import Optional
from config.settings import Settings, get_settings


class SpeechToText:
    """
    Unified STT engine supporting Groq Whisper and Google Speech Recognition.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._groq_client = None

        if self.settings.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.settings.groq_api_key)
            except Exception as e:
                print(f"[STT] Groq client initialization note: {e}")

    def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        prompt: str = "JARVIS, Sanjay, Sir, open, close, VS Code, Chrome, Terminal, pannu",
    ) -> Optional[str]:
        """
        Transcribes WAV-encoded audio bytes into text.

        Args:
            audio_data: WAV file bytes.
            language: Language code (e.g. 'en', 'ta', 'hi', 'en-US').
            prompt: Context hints for Whisper models.

        Returns:
            Transcribed text string or None if unparseable.
        """
        if not audio_data:
            return None

        lang = language or self.settings.stt_language

        # Try Groq Whisper if available
        if self._groq_client and self.settings.stt_engine in ["groq_whisper", "groq"]:
            res = self._transcribe_groq(audio_data, lang, prompt)
            if res:
                return res

        # Fallback to Google SpeechRecognition
        return self._transcribe_google(audio_data, lang)

    def _transcribe_groq(self, audio_data: bytes, language: str, prompt: str) -> Optional[str]:
        """Transcribe via Groq Whisper API."""
        try:
            # Short ISO-639-1 code for whisper (e.g. 'en-US' -> 'en', 'ta-IN' -> 'ta')
            whisper_lang = language.split("-")[0].lower() if "-" in language else language

            # Groq audio API accepts a tuple (filename, bytes, content_type)
            audio_file = ("audio.wav", audio_data, "audio/wav")

            transcription = self._groq_client.audio.transcriptions.create(
                model=self.settings.groq_stt_model,
                file=audio_file,
                language=whisper_lang,
                prompt=prompt,
                response_format="text",
                temperature=0.0,
            )
            text = str(transcription).strip()
            return text if text else None
        except Exception as e:
            print(f"[STT] Groq Whisper failed: {e}. Falling back to Google STT.")
            return None

    def _transcribe_google(self, audio_data: bytes, language: str) -> Optional[str]:
        """Transcribe via Google Web Speech API."""
        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            with io.BytesIO(audio_data) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)

            # Map language code for Google STT (default 'en-US')
            google_lang = language if "-" in language else f"{language}-IN" if language in ["ta", "hi"] else "en-US"

            text = recognizer.recognize_google(audio, language=google_lang)
            return text.strip() if text else None
        except sr.UnknownValueError:
            # Speech was unintelligible
            return None
        except sr.RequestError as e:
            print(f"[STT] Google STT service request error: {e}")
            return None
        except Exception as e:
            print(f"[STT] Google STT error: {e}")
            return None
