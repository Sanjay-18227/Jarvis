"""
JARVIS Voice Subsystem Package
"""

from .recorder import AudioRecorder
from .stt import SpeechToText
from .tts import TextToSpeech, sanitize_for_speech
from .wake_word import WakeWordDetector
from .speaker_verification import SpeakerVerifier
from .voice_enrollment import VoiceEnrollment
from .state_machine import VoiceState, VoiceStateMachine
from .barge_in import BargeInDetector, INTERRUPTION_PHRASES

__all__ = [
    "AudioRecorder",
    "SpeechToText",
    "TextToSpeech",
    "sanitize_for_speech",
    "WakeWordDetector",
    "SpeakerVerifier",
    "VoiceEnrollment",
    "VoiceState",
    "VoiceStateMachine",
    "BargeInDetector",
    "INTERRUPTION_PHRASES",
]
