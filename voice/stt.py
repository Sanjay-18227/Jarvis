"""
Speech-to-Text (STT) Subsystem
Unified transcription engine:
  1. LOCAL whisper (faster-whisper int8 CPU, transformers fallback)
  2. Groq Whisper (remote, disabled under LOCAL_ONLY=true)
  3. Google Web Speech (remote fallback)

Command vocabulary correction runs locally after every backend so proper nouns
like "VS Code", "Paint", "notepad" survive imperfect ASR.
"""

import io
import re
import time
import logging
from typing import Optional
from config.settings import Settings, get_settings

log = logging.getLogger("jarvis.stt")

# Command vocabulary correction table (post-ASR, offline, deterministic)
_VOCAB_FIXES = [
    ("v s code", "vscode"), ("vs code", "vscode"), ("visual studio code", "vscode"),
    ("visual studio", "vscode"), ("more core", "vscode"), ("every code", "vscode"),
    ("v s", "vscode"), ("jarbus", "jarvis"),
    ("micro soft", "microsoft"), ("my craft", "microsoft"), ("my quote", "Microsoft"),
    ("agent dot py", "agent.py"), ("agent py", "agent.py"),
    ("paint brush", "paint"), ("pain", "paint"),
    ("not bad", "notepad"), ("no pad", "notepad"),
    ("kumar", "Kumar"),
]


def apply_vocab_fixes(text: Optional[str]) -> Optional[str]:
    """Deterministic, word-boundary-aware command vocabulary correction."""
    if not text or not text.strip():
        return None
    out = text
    # longest first so "paint brush" is handled before standalone "pain"
    for wrong, right in sorted(_VOCAB_FIXES, key=lambda p: -len(p[0])):
        if wrong == right:
            continue
        pattern = r"\b" + r"\s+".join(map(_esc, wrong.split())) + r"\b"
        out = re.sub(pattern, right, out, flags=re.IGNORECASE)
    return out.strip() or None


def _esc(word: str) -> str:
    return re.escape(word)


class SpeechToText:
    """
    Unified STT engine supporting LOCAL whisper, Groq Whisper, Google SpeechRecognition.

    Selection policy (settings.stt_engine):
      "local"        -> local whisper primary; remote is NOT used on failure (fail fast offline)
      "groq_whisper" -> groq primary; local whisper secondary if installed; google last
      "google"       -> google primary; local whisper secondary if installed
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._groq_client = None
        self._local_warm = False

        if self.settings.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.settings.groq_api_key)
            except Exception as e:
                print(f"[STT] Groq client initialization note: {e}")

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        prompt: str = "JARVIS, Sanjay, Sir, open, close, VS Code, Chrome, Edge, Terminal, pannu",
    ) -> Optional[str]:
        """Transcribes WAV-encoded audio bytes into normalized text (or None)."""
        if not audio_data:
            return None

        engine = self.settings.stt_engine
        lang = language or self.settings.stt_language
        whisper_lang = lang.split("-")[0].lower() if "-" in lang else lang

        result = None
        t0 = time.perf_counter()

        # 1) Engine per settings
        if engine == "local":
            result = self._transcribe_local(audio_data, whisper_lang)
            text = apply_vocab_fixes(result)
            if text:
                self._perf(t0, "local")
            return text  # fail fast offline: no_remote_fallback

        if engine in ("groq_whisper", "groq") and self._groq_client:
            result = self._transcribe_groq(audio_data, lang, prompt) or \
                     self._transcribe_local(audio_data, whisper_lang)
            text = apply_vocab_fixes(result)
            if text:
                self._perf(t0, "groq")
            return text

        if engine == "google":
            result = self._transcribe_google(audio_data, lang) or \
                     self._transcribe_local(audio_data, whisper_lang)
            text = apply_vocab_fixes(result)
            if text:
                self._perf(t0, engine)
            return text

        # unknown engine -> local
        text = apply_vocab_fixes(self._transcribe_local(audio_data, whisper_lang))
        return text

    def _perf(self, t0, backend):
        ms = int((time.perf_counter() - t0) * 1000)
        print(f"[PERF] stt_ms={ms} backend={backend}")

    # ------------------------------------------------------------------
    def _decode_wav_bytes(self, audio_data: bytes):
        """Returns (sr, float32 mono numpy array) or None."""
        try:
            import numpy as np
            import wave
            with io.BytesIO(audio_data) as buf, wave.open(buf, "rb") as wav:
                sr = wav.getframerate()
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())
            if width == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0
            elif width == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype("float32") / 2147483648.0
            elif width == 1:
                audio = (np.frombuffer(frames, dtype=np.uint8).astype("float32") - 128.0) / 128.0
            else:
                return None
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            return sr, audio.astype("float32")
        except Exception as e:
            log.error("[STT] wav decode error: %s", e)
            return None

    def _transcribe_local(self, audio_data: bytes, whisper_lang: str) -> Optional[str]:
        """Local whisper via faster-whisper (from cache/model manager)."""
        try:
            from core.models.model_manager import get_model_manager
            mm = get_model_manager()
            preferred = "whisper-base-local" if getattr(
                self.settings, "asr_local_model", "tiny") == "base" else "whisper-tiny-local"
            fallback = "whisper-tiny-local" if preferred == "whisper-base-local" else "whisper-base-local"
            runtime = mm.get_runtime(preferred) or mm.get_runtime(fallback)
            if runtime is None:
                log.warning("[STT] local whisper unavailable (not downloaded / RAM policy). "
                            "Run download_models.py --selected-only")
                return None
            decoded = self._decode_wav_bytes(audio_data)
            if not decoded:
                return None
            sr, audio = decoded
            segments, info = runtime.transcribe(
                audio, language=("ta" if whisper_lang == "ta" else whisper_lang),
                beam_size=1, vad_filter=False, condition_on_previous_text=False,
            )
            text = " ".join((seg.text or "").strip() for seg in segments).strip()
            return text or None
        except Exception as e:
            log.error("[STT] local whisper failed: %s", e)
            return None

    def _transcribe_groq(self, audio_data: bytes, language: str, prompt: str) -> Optional[str]:
        """Transcribe via Groq Whisper API."""
        try:
            whisper_lang = language.split("-")[0].lower() if "-" in language else language
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
            print(f"[STT] Groq Whisper failed: {e}. Falling back to next engine.")
            return None

    def _transcribe_google(self, audio_data: bytes, language: str) -> Optional[str]:
        """Transcribe via Google Web Speech API."""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with io.BytesIO(audio_data) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
            google_lang = language if "-" in language else \
                f"{language}-IN" if language in ["ta", "hi"] else "en-US"
            text = recognizer.recognize_google(audio, language=google_lang)
            return text.strip() if text else None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[STT] Google STT service request error: {e}")
            return None
        except Exception as e:
            print(f"[STT] Google STT error: {e}")
            return None
