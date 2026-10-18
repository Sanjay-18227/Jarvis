"""
JARVIS Core Agent
Central Orchestrator integrating:
- Low-power wake word listener
- Speaker verification (Owner / Unknown / Uncertain)
- Privacy Guard & Audit Logger
- Multilingual & Code-mixed intent processing (English, Tamil, Tanglish, Hindi)
- Confirmation Manager for high-risk actions
- Trusted Action Executor (Apps, Files, System Controls, ScreenVision, Coding Agent)
- Speech synthesis with dynamic language voice switching
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
from llm.base import AIProvider
from llm.provider_factory import get_llm_provider
from voice.recorder import AudioRecorder
from voice.stt import SpeechToText
from voice.tts import TextToSpeech
from voice.wake_word import WakeWordDetector
from voice.speaker_verification import SpeakerVerifier
from memory.conversation_memory import ConversationMemory
from memory.preferences import PreferencesManager
from security.privacy_guard import PrivacyGuard
from security.audit_log import AuditLogger
from security.confirmation import ConfirmationManager
from core.executor import ActionExecutor
from core.multilingual import MultilingualProcessor
from vision.screen_understanding import ScreenVision
from core.coding_agent import CodingAgent


class JarvisAgent:
    """
    JARVIS Central Intelligence & Desktop Controller.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[AIProvider] = None,
    ):
        self.settings = settings or get_settings()
        self.provider = provider or get_llm_provider(self.settings)

        # Voice Subsystems
        self.recorder = AudioRecorder(settings=self.settings)
        self.stt = SpeechToText(settings=self.settings)
        self.tts = TextToSpeech(settings=self.settings)
        self.verifier = SpeakerVerifier()
        self.wake_word = WakeWordDetector(on_wake=self._on_wake_word_triggered, settings=self.settings)

        # Memory & Preferences
        self.memory = ConversationMemory()
        self.preferences = PreferencesManager()

        # Security & Audit
        self.privacy_guard = PrivacyGuard()
        self.audit_logger = AuditLogger()
        self.confirmation_mgr = ConfirmationManager()

        # Computer Control & Agents
        self.executor = ActionExecutor(
            confirmation_mgr=self.confirmation_mgr,
            audit_logger=self.audit_logger,
        )
        self.multilingual = MultilingualProcessor()
        self.screen_vision = ScreenVision()
        self.coding_agent = CodingAgent(provider=self.provider)

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
        """Global emergency stop."""
        self.tts.stop()
        self.recorder.interrupt()
        self.confirmation_mgr.cancel_pending()
        print("[JARVIS] Emergency stop executed.")

    def build_system_prompt(self) -> str:
        """Constructs persona system prompt tailored to Sanjay."""
        owner = self.preferences.get("owner_name", "Sanjay")
        title = self.preferences.get("preferred_title", "Sir")

        prompt = (
            f"You are JARVIS, an autonomous personal Windows desktop AI assistant created for {owner}.\n"
            f"Always address the user naturally as '{owner}' or '{title}'.\n"
            "Key personality traits:\n"
            "- Friendly, calm, concise, respectful, and intelligent.\n"
            "- Never sound robotic. Do not constantly explain routine steps unless asked.\n"
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
        Processes text through Confirmation -> Multilingual/Action Parser -> Screen/Code -> LLM.
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

        # Step 3: Screen Vision Check
        if any(term in clean_input.lower() for term in ["what is this error", "look at screen", "inspect screen", "what is on my screen"]):
            _, vision_resp = self.screen_vision.inspect_screen(clean_input, active_speaker)
            return vision_resp

        # Step 4: Coding Agent Check (VS Code project creation)
        if "create" in clean_input.lower() and any(k in clean_input.lower() for k in ["project", "application", "app in vs code", "weather app"]):
            _, code_resp = self.coding_agent.create_project(
                project_name="jarvis_generated_app",
                description=clean_input,
            )
            return code_resp

        # Step 5: Code-Mixed or Natural Action Parsing
        direct_action = self.multilingual.parse_code_mixed_intent(clean_input)
        if direct_action:
            act_name, params = direct_action
            _, act_res = self.executor.execute_action(act_name, params, active_speaker)
            return act_res

        # English command parsing
        cmd_action = self._parse_english_command(clean_input)
        if cmd_action:
            act_name, params = cmd_action
            _, act_res = self.executor.execute_action(act_name, params, active_speaker)
            return act_res

        # Step 6: General LLM Reasoning & Dialogue
        history = self.memory.get_history()
        system_prompt = self.build_system_prompt()

        response = self.provider.generate(
            prompt=clean_input,
            system_prompt=system_prompt,
            history=history,
        )

        # Update Conversation Memory
        self.memory.add_user_message(clean_input)
        self.memory.add_assistant_message(response)

        # Audit Logging
        self.audit_logger.log_action(
            task=clean_input,
            speaker=active_speaker,
            action="CHAT_COMPLETION",
            risk_level=RiskLevel.LEVEL_0_INFORMATIONAL,
            status=ActionStatus.SUCCESS,
        )

        return response

    def _parse_english_command(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Maps standard English voice commands to computer actions."""
        t = text.lower().strip()
        # Strip conversational fillers and wake words
        for prefix in ["hey jarvis ", "jarvis ", "ok jarvis ", "okay jarvis ", "ok ", "okay ", "please ", "can you ", "could you "]:
            if t.startswith(prefix):
                t = t[len(prefix):].strip()

        if t.startswith("open "):
            target = t[5:].replace("application", "").replace("app", "").strip()
            return "open_application", {"target": target}
        elif t.startswith("close "):
            target = t[6:].replace("application", "").replace("app", "").strip()
            return "close_application", {"target": target}
        elif t.startswith("search ") and "on google" in t:
            q = t.replace("search", "").replace("on google", "").strip()
            return "search_google", {"query": q}
        elif "lock screen" in t or "lock workstation" in t:
            return "lock_workstation", {}
        elif t in ["play music", "pause music", "next song", "volume up", "volume down", "mute"]:
            action_map = {
                "play music": "play_pause",
                "pause music": "play_pause",
                "next song": "next",
                "volume up": "volume_up",
                "volume down": "volume_down",
                "mute": "mute",
            }
            return "media_control", {"action": action_map.get(t, "play_pause")}
        elif t.startswith("delete "):
            target = t[7:].replace("file", "").replace("folder", "").strip()
            return "delete_path", {"path": target}
        return None

    def speak_response(self, text: str):
        """Speaks the response using neural TTS with dynamic language detection."""
        if not text:
            return
        lang = self.multilingual.detect_language(text)
        voice = self.multilingual.get_preferred_voice_for_language(lang)
        self.tts.speak(text, voice=voice)

    def run_turn(self) -> Optional[str]:
        """
        Executes one full voice turn:
        Listen -> Acoustic Speaker Verification -> STT -> Agent Reasoning -> Multilingual TTS.
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
