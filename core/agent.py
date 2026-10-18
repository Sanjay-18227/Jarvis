"""
JARVIS Core Agent
Central Autonomous Desktop Orchestrator integrating:
- Hands-free wake word listener ("Hey Jarvis" / "Jarvis")
- Acoustic speaker verification (Owner / Unknown / Uncertain)
- Dedicated Privacy Guard & Secrets Manager
- Autonomous Gemini tool planner & goal executor (NOT a fixed command system)
- Trusted Action Executor with least-privilege security & confirmation gates
- Vision & Screen understanding loop
- Unified single consistent multilingual neural voice
"""

import sys
import time
from typing import Optional, Dict, Any, Tuple

from config.settings import Settings, get_settings
from config.constants import (
    JARVIS_SYSTEM_PROMPT,
    SpeakerClassification,
    RiskLevel,
    ActionStatus,
)
from ai.provider import AIProvider
from ai.gemini_client import GeminiClient
from ai.tool_planner import ToolCatalog
from core.planner import AutonomousPlanner
from core.task_manager import TaskManager
from core.executor import ActionExecutor
from core.multilingual import MultilingualProcessor
from vision.screen_understanding import ScreenVision
from core.coding_agent import CodingAgent
from voice.recorder import AudioRecorder
from voice.stt import SpeechToText
from voice.tts import TextToSpeech
from voice.wake_word import WakeWordDetector
from voice.speaker_verification import SpeakerVerifier
from memory.conversation_memory import ConversationMemory
from memory.preferences import PreferencesManager
from security.privacy_guard import PrivacyGuard
from security.secrets_manager import SecretsManager
from security.audit_log import AuditLogger
from security.confirmation import ConfirmationManager


class JarvisAgent:
    """
    JARVIS Central Autonomous Desktop AI Agent for Sanjay.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[AIProvider] = None,
    ):
        self.settings = settings or get_settings()

        # AI & Gemini Client
        if provider:
            self.provider = provider
            if isinstance(provider, GeminiClient):
                self.gemini_client = provider
            else:
                self.gemini_client = GeminiClient(
                    api_key=self.settings.gemini_api_key,
                    model=self.settings.gemini_model or "gemini-3.8-flash",
                )
        else:
            self.gemini_client = GeminiClient(
                api_key=self.settings.gemini_api_key,
                model=self.settings.gemini_model or "gemini-3.8-flash",
            )
            self.provider = self.gemini_client

        # Voice Subsystems (Single Consistent Voice Identity)
        self.recorder = AudioRecorder(settings=self.settings)
        self.stt = SpeechToText(settings=self.settings)
        self.tts = TextToSpeech(settings=self.settings)
        self.verifier = SpeakerVerifier()
        self.wake_word = WakeWordDetector(on_wake=self._on_wake_word_triggered, settings=self.settings)

        # Memory & Preferences
        self.memory = ConversationMemory()
        self.preferences = PreferencesManager()

        # Security, Privacy & Audit
        self.privacy_guard = PrivacyGuard()
        self.secrets_mgr = SecretsManager()
        self.audit_logger = AuditLogger()
        self.confirmation_mgr = ConfirmationManager()

        # Computer Control & Automation
        self.executor = ActionExecutor(
            confirmation_mgr=self.confirmation_mgr,
            audit_logger=self.audit_logger,
            secrets_mgr=self.secrets_mgr,
        )
        self.multilingual = MultilingualProcessor()
        self.screen_vision = ScreenVision()
        self.coding_agent = CodingAgent(provider=self.provider)

        # Autonomous Planner & Tools
        self.tool_catalog = ToolCatalog()
        self.task_manager = TaskManager()
        self.planner = AutonomousPlanner(
            client=self.provider,
            executor=self.executor,
            task_manager=self.task_manager,
            screen_vision=self.screen_vision,
            tool_catalog=self.tool_catalog,
        )

        self._running = False
        self._current_speaker = SpeakerClassification(self.settings.default_permission_level)

    @property
    def speaker_state(self) -> SpeakerClassification:
        return self._current_speaker

    def set_speaker_state(self, speaker: SpeakerClassification):
        self._current_speaker = speaker

    def _on_wake_word_triggered(self):
        """Called by WakeWordDetector upon hearing wake word."""
        title = self.preferences.get("preferred_title", "Sir")
        print(f"\n[JARVIS] Wake word detected. Online for {title}.")
        self.speak_response(f"Yes, {title}?")

    def emergency_stop(self):
        """Global emergency stop phrase 'Jarvis stop' immediately halts active automation."""
        self.tts.stop()
        self.recorder.interrupt()
        self.task_manager.request_emergency_stop()
        self.confirmation_mgr.cancel_pending()
        print("[JARVIS] Emergency stop executed. All active automation halted.")

    def build_system_prompt(self) -> str:
        """Constructs persona system prompt tailored to Sanjay."""
        owner = self.preferences.get("owner_name", "Sanjay")
        title = self.preferences.get("preferred_title", "Sir")

        prompt = (
            f"You are JARVIS, an autonomous personal Windows desktop AI assistant and computer-use agent created for {owner}.\n"
            f"Always address the user naturally as '{owner}' or '{title}'.\n"
            "Key personality and behavioral rules:\n"
            "- Friendly, calm, respectful, intelligent, articulate personal assistant.\n"
            "- Speak naturally and conversationally, like a real AI assistant, never mechanical or robotic.\n"
            "- Avoid repetitive phrases like 'Command received'.\n"
            "- Give short, direct responses when a short response is enough (1-2 sentences for speech).\n"
            "- Understand multilingual speech seamlessly across English, Tamil, Tanglish, and Hindi.\n"
            "- When executing computer tasks, reason dynamically using your tools (open apps, switch windows, inspect screen, run commands, create files, mouse, keyboard).\n"
            "- Never fabricate execution: only report what was actually observed and verified.\n"
            "- Never expose or speak passwords, API keys, or private credentials aloud.\n"
        )
        return prompt

    def process_input(
        self,
        user_text: str,
        speaker: Optional[SpeakerClassification] = None,
    ) -> str:
        """
        Processes text through Confirmation -> Privacy Guard -> Autonomous Planner & Tools -> Speech.
        """
        active_speaker = speaker or self._current_speaker
        clean_input = user_text.strip()
        if not clean_input:
            return ""

        # Step 0: Emergency Stop Check
        if clean_input.lower() in ["jarvis stop", "stop", "abort", "cancel"]:
            self.emergency_stop()
            return f"Stopped immediately, {self.preferences.get('preferred_title', 'Sir')}."

        # Step 1: Check Pending High-Risk Confirmation
        if self.confirmation_mgr.has_pending_confirmation:
            confirmed, pending, msg = self.confirmation_mgr.process_confirmation_response(
                clean_input, active_speaker
            )
            if confirmed and pending:
                # Execute the confirmed action
                _, exec_msg = self.executor.execute_action(
                    action=pending.action_name,
                    parameters=pending.parameters,
                    speaker=active_speaker,
                    confirmed=True,
                )
                return f"{msg} {exec_msg}"
            return msg

        # Step 2: Privacy Guard & Speaker Access Validation
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

        # Step 3: Screen Vision Direct Check
        if any(term in clean_input.lower() for term in ["what is this error", "look at my screen", "look at screen", "what's wrong here", "what is on my screen"]):
            _, vision_resp = self.screen_vision.inspect_screen(clean_input, active_speaker)
            self.memory.add_user_message(clean_input)
            self.memory.add_assistant_message(vision_resp)
            return vision_resp

        # Step 4: Autonomous Goal-Oriented Computer & Tool Planning
        system_prompt = self.build_system_prompt()
        success, final_response = self.planner.execute_goal(
            goal=clean_input,
            speaker=active_speaker,
            system_prompt=system_prompt,
        )

        # Update Conversation Memory
        self.memory.add_user_message(clean_input)
        self.memory.add_assistant_message(final_response)

        # Audit Logging
        status = ActionStatus.SUCCESS if success else ActionStatus.FAILED
        self.audit_logger.log_action(
            task=clean_input,
            speaker=active_speaker,
            action="AUTONOMOUS_GOAL",
            risk_level=RiskLevel.LEVEL_1_SAFE,
            status=status,
            details={"response": final_response[:400]},
        )

        return final_response

    def speak_response(self, text: str):
        """
        Speaks the response using ONE consistent neural voice identity
        regardless of whether text is English, Tamil, Tanglish, or Hindi.
        """
        if not text:
            return
        # Sanitizes any accidental credentials or markdown
        safe_to_speak, safe_text = self.secrets_mgr.validate_safe_for_speech(text)
        # Use single unified multilingual voice identity
        voice = self.multilingual.get_preferred_voice_for_language("multilingual")
        self.tts.speak(safe_text, voice=voice)

    def run_turn(self) -> Optional[str]:
        """
        Executes one full voice turn:
        Listen -> Acoustic Speaker Verification -> STT -> Autonomous Agent Reasoning -> Multilingual TTS.
        """
        print("\n[JARVIS] Listening...")
        audio_data = self.recorder.record_phrase(timeout=6.0)
        if not audio_data:
            return None

        # Speaker Verification
        speaker, confidence = self.verifier.verify_speaker(audio_data)
        print(f"[JARVIS] Speaker Verified: {speaker.value} (Confidence: {confidence:.2f})")

        # Transcribe
        transcription = self.stt.transcribe(audio_data)
        if not transcription:
            return None

        print(f"User ({speaker.value}): {transcription}")
        response = self.process_input(transcription, speaker=speaker)
        print(f"JARVIS: {response}")

        self.speak_response(response)
        return response

    def run_continuous_voice(self):
        """Continuous voice loop."""
        self._running = True
        print(f"\n[JARVIS] Voice mode active. Provider: {self.provider.name}")
        title = self.preferences.get("preferred_title", "Sir")
        greeting = f"JARVIS online and fully operational. At your service, {title}."
        print(f"JARVIS: {greeting}")
        self.speak_response(greeting)

        while self._running:
            try:
                self.run_turn()
            except KeyboardInterrupt:
                print("\n[JARVIS] Voice mode terminated by user.")
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

                response = self.process_input(user_msg, speaker=SpeakerClassification.OWNER)
                print(f"JARVIS: {response}\n")

            except (KeyboardInterrupt, EOFError):
                print("\nJARVIS: Goodbye, Sir.")
                break
