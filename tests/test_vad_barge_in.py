"""
Unit and Integration Tests for Dynamic VAD, Barge-In Interruption,
Voice State Machine, and Self-Echo Protection in JARVIS.
"""

import time
import threading
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from config.settings import Settings
from config.constants import SpeakerClassification
from voice.state_machine import VoiceState, VoiceStateMachine
from voice.barge_in import BargeInDetector, normalize_text, INTERRUPTION_PHRASES
from voice.recorder import AudioRecorder
from voice.tts import TextToSpeech
from core.agent import JarvisAgent
from llm.mock_provider import MockProvider


# =====================================================================
# 1. Voice State Machine Tests
# =====================================================================

def test_voice_state_machine_transitions():
    sm = VoiceStateMachine(initial_state=VoiceState.IDLE)
    assert sm.state == VoiceState.IDLE
    assert sm.is_state(VoiceState.IDLE)

    sm.set_state(VoiceState.LISTENING)
    assert sm.state == VoiceState.LISTENING

    sm.set_state(VoiceState.PROCESSING)
    assert sm.state == VoiceState.PROCESSING

    sm.set_state(VoiceState.EXECUTING)
    assert sm.state == VoiceState.EXECUTING

    sm.set_state(VoiceState.SPEAKING)
    assert sm.state == VoiceState.SPEAKING

    sm.set_state(VoiceState.INTERRUPTED)
    assert sm.state == VoiceState.INTERRUPTED

    sm.set_state(VoiceState.LISTENING)
    assert sm.state == VoiceState.LISTENING


def test_voice_state_machine_listener():
    sm = VoiceStateMachine(initial_state=VoiceState.IDLE)
    history = []

    def on_change(old_state, new_state):
        history.append((old_state, new_state))

    sm.add_state_listener(on_change)
    sm.set_state(VoiceState.LISTENING)
    sm.set_state(VoiceState.SPEAKING)

    assert len(history) == 2
    assert history[0] == (VoiceState.IDLE, VoiceState.LISTENING)
    assert history[1] == (VoiceState.LISTENING, VoiceState.SPEAKING)


def test_voice_state_machine_thread_safety():
    sm = VoiceStateMachine(initial_state=VoiceState.IDLE)

    def worker(target_state):
        for _ in range(50):
            sm.set_state(target_state)

    t1 = threading.Thread(target=worker, args=(VoiceState.LISTENING,))
    t2 = threading.Thread(target=worker, args=(VoiceState.SPEAKING,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sm.state in [VoiceState.LISTENING, VoiceState.SPEAKING]


# =====================================================================
# 2. Barge-In Phrase Matching & Self-Echo Protection Tests
# =====================================================================

def test_barge_in_phrase_recognition():
    detector = BargeInDetector()

    # Required triggers from specification:
    required_phrases = [
        ("stop", "stop"),
        ("Jarvis stop", "jarvis stop"),
        ("stop Jarvis", "stop jarvis"),
        ("wait", "wait"),
        ("Jarvis wait", "jarvis wait"),
        ("hold on", "hold on"),
        ("cancel", "cancel"),
        ("enough", "enough"),
        ("shut up", "shut up"),
    ]

    for input_text, expected_core in required_phrases:
        detected = detector.is_interruption(input_text)
        assert detected is not None, f"Failed to detect required phrase: {input_text}"
        assert expected_core in detected.lower(), f"Unexpected match for {input_text}: {detected}"


def test_barge_in_variations_recognition():
    detector = BargeInDetector()

    variations = [
        "Hey Jarvis, please stop now",
        "Stop talking",
        "Jarvis, hold on a second",
        "Just wait",
        "Cancel that",
        "Please shut up",
    ]

    for phrase in variations:
        detected = detector.is_interruption(phrase)
        assert detected is not None, f"Failed to detect variation: {phrase}"


def test_barge_in_non_interruption_queries():
    detector = BargeInDetector()

    non_interruptions = [
        "Can you open VS Code for me?",
        "What is the weather today?",
        "Create a file named test.py",
        "Hello Jarvis how are you doing",
        "Tell me a story about artificial intelligence",
    ]

    for phrase in non_interruptions:
        detected = detector.is_interruption(phrase)
        assert detected is None, f"Incorrectly flagged non-interruption: {phrase} -> {detected}"


def test_barge_in_self_echo_protection():
    detector = BargeInDetector()

    # Scenario: JARVIS is speaking about a topic
    tts_output = "Sir, here is a detailed explanation of your Python script architecture."

    # The microphone hears JARVIS's own speaker playing parts of the sentence:
    echo_chunks = [
        "here is a detailed explanation",
        "explanation of your Python script",
        "architecture",
        "Sir here is",
    ]

    for echo in echo_chunks:
        # Self-echo shield must reject these chunks
        detected = detector.is_interruption(echo, current_speaking_text=tts_output)
        assert detected is None, f"Self-echo falsely detected as interruption: {echo}"

    # But if the user says "Jarvis stop" while JARVIS speaks that sentence:
    user_interruption = "Jarvis stop"
    detected = detector.is_interruption(user_interruption, current_speaking_text=tts_output)
    assert detected is not None
    assert "stop" in detected.lower()


# =====================================================================
# 3. Dynamic VAD Logic & Natural Pause Tolerance Tests
# =====================================================================

def test_audio_recorder_rms_calculation():
    recorder = AudioRecorder()

    silence = np.zeros(1024, dtype=np.int16)
    assert recorder.calculate_rms(silence) == 0.0

    # Test sine wave with known RMS
    samples = 16000
    t = np.linspace(0, 1, samples, endpoint=False)
    amplitude = 10000
    sine = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    rms = recorder.calculate_rms(sine)
    # Expected RMS of sine wave is amplitude / sqrt(2) ~ 7071
    assert 6800 < rms < 7300


def test_vad_natural_pause_and_silence_logic():
    """
    Simulates audio stream chunks through VAD to verify:
    1. Speech starts on active chunk
    2. Short pause (< silence threshold) does NOT terminate recording
    3. Resumed speech resets pause counter
    4. Sustained silence (>= silence threshold) triggers End of Utterance
    """
    settings = Settings(
        silence_threshold_seconds=1.5,
        max_record_seconds=30.0,
        energy_threshold=300,
    )
    recorder = AudioRecorder(settings=settings)

    # 1024 samples at 16kHz = 0.064 seconds per chunk
    active_chunk = np.full(1024, fill_value=2000, dtype=np.int16)
    silence_chunk = np.zeros(1024, dtype=np.int16)

    # Ensure RMS expectations
    assert recorder.calculate_rms(active_chunk) > settings.energy_threshold
    assert recorder.calculate_rms(silence_chunk) < settings.energy_threshold

    chunk_duration = 1024 / 16000.0  # 0.064s

    # Build a simulated stream generator:
    # - 0.2s pre-silence (3 chunks)
    # - 1.0s speech: "Jarvis, open..." (16 chunks)
    # - 0.6s natural pause (9 chunks) -> must NOT terminate!
    # - 1.0s speech: "...VS Code" (16 chunks)
    # - 1.8s sustained silence (28 chunks) -> MUST terminate on chunk reaching 1.5s!
    stream_chunks = (
        [silence_chunk] * 3 +
        [active_chunk] * 16 +
        [silence_chunk] * 9 +
        [active_chunk] * 16 +
        [silence_chunk] * 28
    )

    events = []

    # Mock InputStream to yield our synthetic chunks
    class MockStream:
        def __init__(self, *args, **kwargs):
            self.index = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def read(self, size):
            if self.index < len(stream_chunks):
                chunk = stream_chunks[self.index]
                self.index += 1
                return chunk, False
            time.sleep(0.01)
            return silence_chunk, False

    with patch("sounddevice.InputStream", MockStream):
        # We also patch print to capture VAD logs
        original_print = print
        logs = []

        def mock_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            logs.append(msg)
            original_print(*args, **kwargs)

        with patch("builtins.print", side_effect=mock_print):
            wav_bytes = recorder.record_phrase(
                timeout=5.0,
                phrase_time_limit=30.0,
                silence_duration=1.5,
            )

    assert wav_bytes is not None
    assert len(wav_bytes) > 44  # WAV header + audio payload

    # Verify log sequence
    log_text = "\n".join(logs)
    assert "[VAD] Speech started" in log_text
    assert "[VAD] Silence detected" in log_text
    assert "[VAD] Speech continuing" in log_text
    assert "[VAD] End of utterance" in log_text


# =====================================================================
# 4. Real TTS Cancellation Tests
# =====================================================================

def test_tts_cancellation_lifecycle():
    tts = TextToSpeech()
    assert not tts.is_speaking

    # Test stop() sets flags and clears active state
    tts.stop()
    assert not tts.is_speaking


def test_tts_single_playback_enforcement():
    tts = TextToSpeech()

    with patch("sounddevice.play"), patch("sounddevice.get_stream") as mock_stream:
        mock_active = MagicMock()
        mock_active.active = True
        mock_stream.return_value = mock_active

        # Calling stop terminates playback
        tts.stop()
        assert not tts.is_speaking


# =====================================================================
# 5. Agent Voice Turn & State Flow Tests
# =====================================================================

def test_agent_voice_turn_flow():
    settings = Settings(owner_name="Sanjay", preferred_title="Sir")
    mock_provider = MockProvider()
    agent = JarvisAgent(settings=settings, provider=mock_provider)

    # Initial state
    assert agent.state_machine.state == VoiceState.IDLE

    # Mock recording returning audio bytes
    t = np.linspace(0, 0.5, 8000)
    audio_pcm = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16).tobytes()
    wav_bytes = agent.recorder._create_wav_bytes(audio_pcm)

    with patch.object(agent.recorder, "record_phrase", return_value=wav_bytes):
        with patch.object(agent.verifier, "verify_speaker", return_value=(SpeakerClassification.OWNER, 0.95)):
            with patch.object(agent.stt, "transcribe", return_value="hello jarvis"):
                with patch.object(agent.tts, "speak_async") as mock_speak_async:
                    mock_thread = threading.Thread(target=lambda: None)
                    mock_thread.start()
                    mock_speak_async.return_value = mock_thread

                    response = agent.run_turn()
                    assert response is not None
                    assert "Sanjay" in response or "Sir" in response
                # State must return to LISTENING
                assert agent.state_machine.state == VoiceState.LISTENING


def test_agent_barge_in_interruption_flow():
    settings = Settings(owner_name="Sanjay", preferred_title="Sir")
    mock_provider = MockProvider()
    agent = JarvisAgent(settings=settings, provider=mock_provider)

    # Simulate barge-in trigger during speech
    with patch.object(agent.barge_in, "listen_for_interruption", return_value="Jarvis stop"):
        with patch.object(agent.tts, "stop") as mock_tts_stop:
            mock_thread = threading.Thread(target=lambda: None)
            mock_thread.start()
            with patch.object(agent.tts, "speak_async", return_value=mock_thread):
                result = agent.speak_response("Here is a very long response that will be interrupted.")
                assert result is False  # False indicates interrupted
                mock_tts_stop.assert_called_once()
                # Must return to LISTENING state immediately
                assert agent.state_machine.state == VoiceState.LISTENING
