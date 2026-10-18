"""
JARVIS Core Agent
Orchestrates voice interaction, speaker verification, privacy evaluation,
reasoning via AIProvider, memory management, and speech synthesis.
"""

import sys
import time
from typing import Optional, Dict, Any, List

from config.settings import Settings, get_settings
from config.constants import (
    JARVIS_SYSTEM_PROMPT,
    SpeakerClassification,
    RiskLevel,
    ActionStatus,
)
from llm.base import AIProvider
from llm.provider_factory import get_llm_provider
from voice.recorder import AudioRecorder
from voice.stt import SpeechToText
from voice.tts import TextToSpeech
from memory.conversation_memory import ConversationMemory
from memory.preferences import PreferencesManager
from security.privacy_guard import PrivacyGuard
from security.audit_log import AuditLogger


class JarvisAgent:
    """
    Central JARVIS Orchestrator.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[AIProvider] = None,
    ):
        self.settings = settings or get_settings()
        self.provider = provider or get_llm_provider(self.settings)

        # Subsystems
        self.recorder = AudioRecorder(settings=self.settings)
        self.stt = SpeechToText(settings=self.settings)
        self.tts = TextToSpeech(settings=self.settings)
        self.memory = ConversationMemory()
        self.preferences = PreferencesManager()
        self.privacy_guard = PrivacyGuard()
        self.audit_logger = AuditLogger()

        self._running = False
        self._current_speaker = SpeakerClassification(self.settings.default_permission_level)

    @property
    def speaker_state(self) -> SpeakerClassification:
        return self._current_speaker

    def set_speaker_state(self, speaker: SpeakerClassification):
        self._current_speaker = speaker

    def build_system_prompt(self) -> str:
        """Constructs persona system prompt tailored to Sanjay."""
        owner = self.preferences.get("owner_name", "Sanjay")
        title = self.preferences.get("preferred_title", "Sir")

        prompt = (
            f"You are JARVIS, an autonomous personal Windows desktop AI assistant created for {owner}.\n"
            f"Always address the user naturally as '{owner}' or '{title}'.\n"
            "Key personality traits:\n"
            "- Friendly, calm, concise, respectful, and intelligent.\n"
            "- Never sound robotic. Do not constantly explain what you are doing unless asked.\n"
            "- Spoken responses should usually be short and direct (1-2 sentences).\n"
            "- Understand multilingual input seamlessly (English, Tamil, Tanglish, Hindi).\n"
            "- If asked for private credentials, passwords, or personal documents by an unverified person, politely refuse.\n"
        )
        return prompt

    def process_input(
        self,
        user_text: str,
        speaker: Optional[SpeakerClassification] = None,
    ) -> str:
        """
        Processes a text prompt through security checks, LLM reasoning, memory, and auditing.

        Args:
            user_text: The user's input.
            speaker: Speaker classification (defaults to current agent speaker state).

        Returns:
            Assistant's response text.
        """
        active_speaker = speaker or self._current_speaker
        clean_input = user_text.strip()
        if not clean_input:
            return ""

        # Step 1: Security & Privacy Guard Check
        is_allowed, refusal_message = self.privacy_guard.validate_speaker_permission(
            active_speaker, clean_input
        )
        if not is_allowed:
            self.audit_logger.log_action(
                task=clean_input,
                speaker=active_speaker,
                action="PRIVACY_BLOCKED",
                risk_level=RiskLevel.LEVEL_3_SENSITIVE,
                status=ActionStatus.DENIED,
            )
            return refusal_message

        # Step 2: Retrieve context & build prompt
        history = self.memory.get_history()
        system_prompt = self.build_system_prompt()

        # Step 3: LLM Inference
        response = self.provider.generate(
            prompt=clean_input,
            system_prompt=system_prompt,
            history=history,
        )

        # Step 4: Update Conversation Memory
        self.memory.add_user_message(clean_input)
        self.memory.add_assistant_message(response)

        # Step 5: Audit Log
        is_sensitive, _ = self.privacy_guard.assess_privacy_risk(clean_input)
        risk = RiskLevel.LEVEL_2_MODERATE if is_sensitive else RiskLevel.LEVEL_0_INFORMATIONAL

        self.audit_logger.log_action(
            task=clean_input,
            speaker=active_speaker,
            action="CHAT_COMPLETION",
            risk_level=risk,
            status=ActionStatus.SUCCESS,
        )

        return response

    def listen_and_transcribe(self, timeout: float = 6.0) -> Optional[str]:
        """Records voice from microphone and transcribes to text."""
        audio_data = self.recorder.record_phrase(timeout=timeout)
        if not audio_data:
            return None
        return self.stt.transcribe(audio_data)

    def speak_response(self, text: str):
        """Speaks the response using neural TTS."""
        if text:
            self.tts.speak(text)

    def run_turn(self, speaker: Optional[SpeakerClassification] = None) -> Optional[str]:
        """
        Executes a single voice turn: Listen -> STT -> Process -> TTS.
        """
        print("[JARVIS] Listening...")
        transcription = self.listen_and_transcribe()
        if not transcription:
            return None

        print(f"User: {transcription}")
        response = self.process_input(transcription, speaker=speaker)
        print(f"JARVIS: {response}")

        self.speak_response(response)
        return response

    def run_continuous_voice(self):
        """Continuous voice loop."""
        self._running = True
        print(f"\n[JARVIS] Voice mode active. Provider: {self.provider.name}")
        print("Say something to JARVIS (Press Ctrl+C to stop)...")

        # Initial greeting
        greeting = f"JARVIS online. At your service, {self.preferences.get('preferred_title', 'Sir')}."
        print(f"JARVIS: {greeting}")
        self.speak_response(greeting)

        while self._running:
            try:
                self.run_turn()
            except KeyboardInterrupt:
                print("\n[JARVIS] Stopping voice mode.")
                break
            except Exception as e:
                print(f"[JARVIS] Turn error: {e}")
                time.sleep(1)

    def run_text_interactive(self):
        """Interactive text mode for development and testing."""
        print(f"\n=== JARVIS Interactive Text Console ===")
        print(f"Owner: {self.preferences.get('owner_name', 'Sanjay')} | Provider: {self.provider.name}")
        print("Type your message (or 'exit' / 'quit' to stop):\n")

        while True:
            try:
                user_msg = input("Sanjay: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit", "q"]:
                    print("JARVIS: Goodbye, Sir.")
                    break

                response = self.process_input(user_msg)
                print(f"JARVIS: {response}\n")

            except (KeyboardInterrupt, EOFError):
                print("\nJARVIS: Goodbye, Sir.")
                break
