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
-Very Intelligent System and extremely advanced AI
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
from voice.state_machine import VoiceState, VoiceStateMachine
from voice.barge_in import BargeInDetector
from voice.voice_engine import get_voice_engine, VoiceEngine
from memory.conversation_memory import ConversationMemory
from memory.preferences import PreferencesManager
from security.privacy_guard import PrivacyGuard
from security.secrets_manager import SecretsManager
from security.audit_log import AuditLogger
from security.confirmation import ConfirmationManager
from security.authorization_gate import get_speaker_session
from core.fast_router import FastRouter
from core.context_memory import get_context_memory


class JarvisAgent:
    """
    JARVIS Central Autonomous Desktop AI Agent.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[AIProvider] = None,
    ):
        self.settings = settings or get_settings()

        # AI & LLM Provider (OpenRouter, Gemini, Groq, Ollama, Mock)
        if provider:
            self.provider = provider
        else:
            from llm.provider_factory import get_llm_provider
            self.provider = get_llm_provider(settings=self.settings)

        if isinstance(self.provider, GeminiClient):
            self.gemini_client = self.provider
        else:
            self.gemini_client = GeminiClient(
                api_key=self.settings.gemini_api_key,
                model=self.settings.gemini_model or "gemini-3.8-flash",
            )

        # Voice Subsystems (Single Consistent Voice Identity) & State Machine
        self.state_machine = VoiceStateMachine(initial_state=VoiceState.IDLE)
        self.verifier = SpeakerVerifier()
        self.voice_engine: VoiceEngine = get_voice_engine(settings=self.settings, state_machine=self.state_machine)
        self.recorder = getattr(self.voice_engine, "recorder", None) or getattr(getattr(self.voice_engine, "_fallback_engine", None), "recorder", None) or AudioRecorder(settings=self.settings)
        self.stt = getattr(self.voice_engine, "stt", None) or getattr(getattr(self.voice_engine, "_fallback_engine", None), "stt", None) or SpeechToText(settings=self.settings)
        self.tts = getattr(self.voice_engine, "tts", None) or getattr(getattr(self.voice_engine, "_fallback_engine", None), "tts", None) or TextToSpeech(settings=self.settings)
        self.barge_in = getattr(self.voice_engine, "barge_in", None) or getattr(getattr(self.voice_engine, "_fallback_engine", None), "barge_in", None) or BargeInDetector(
            settings=self.settings,
            stt=self.stt,
            verifier=self.verifier,
        )
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

        # Wire research agent to active AI provider / Gemini client
        if hasattr(self.executor, "research_agent"):
            self.executor.research_agent.gemini = self.provider if self.provider.is_available() else self.gemini_client

        # Hybrid Fast Router (Sub-second local OS actions & conversational greetings)
        self.fast_router = FastRouter(
            app_indexer=self.executor.app_ctrl.indexer,
            win_ctrl=self.executor.win_ctrl,
            sys_monitor=self.executor.sys_monitor,
            browser_ctrl=self.executor.browser_ctrl,
            screen_vision=self.screen_vision,
            multi_step_executor=self.executor.multi_step_executor,
        )

        # Phase 3 Verified Agent Loop (Goal -> Plan -> Authorize -> Execute -> Observe -> Verify -> Report)
        from core.agent_loop import Phase3AgentLoop
        self.agent_loop = Phase3AgentLoop(
            llm_provider=self.provider,
            file_mgr=self.executor.file_mgr,
            app_indexer=self.executor.app_ctrl.indexer,
            browser_ctrl=self.executor.browser_ctrl,
            win_ctrl=self.executor.win_ctrl,
            sys_monitor=self.executor.sys_monitor,
            screen_vision=self.screen_vision.service,
        )

        # Local Small Language Model (SLM) for local-first understanding & paraphrases
        from core.local_slm import LocalSLM
        self.local_slm = LocalSLM(
            settings=self.settings,
            skill_registry=self.fast_router.skill_registry,
        )

        self._running = False
        self._current_speaker = SpeakerClassification(self.settings.default_permission_level)
        self.perf_metrics: Dict[str, float] = {
            "vad_end_ms": 0.0,
            "stt_ms": 0.0,
            "speaker_verify_ms": 0.0,
            "routing_ms": 0.0,
            "llm_ms": 0.0,
            "tool_ms": 0.0,
            "verification_ms": 0.0,
            "tts_start_ms": 0.0,
            "total_response_ms": 0.0,
        }

    def log_perf_metrics(self):
        """Logs performance timing metrics conforming to benchmark instrumentation."""
        print(f"[PERF] vad_end_ms={self.perf_metrics.get('vad_end_ms', 0.0):.1f}")
        print(f"[PERF] stt_ms={self.perf_metrics.get('stt_ms', 0.0):.1f}")
        print(f"[PERF] speaker_verify_ms={self.perf_metrics.get('speaker_verify_ms', 0.0):.1f}")
        print(f"[PERF] routing_ms={self.perf_metrics.get('routing_ms', 0.0):.1f}")
        print(f"[PERF] llm_ms={self.perf_metrics.get('llm_ms', 0.0):.1f}")
        print(f"[PERF] tool_ms={self.perf_metrics.get('tool_ms', 0.0):.1f}")
        print(f"[PERF] verification_ms={self.perf_metrics.get('verification_ms', 0.0):.1f}")
        print(f"[PERF] tts_start_ms={self.perf_metrics.get('tts_start_ms', 0.0):.1f}")
        print(f"[PERF] total_response_ms={self.perf_metrics.get('total_response_ms', 0.0):.1f}")

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

    def emergency_stop(self, speaker: Optional[SpeakerClassification] = None) -> str:
        """Global emergency stop phrase 'Jarvis stop' immediately halts active automation."""
        active_speaker = speaker or self._current_speaker
        if hasattr(self, "agent_loop") and self.agent_loop:
            self.agent_loop.request_stop()
        self.state_machine.set_state(VoiceState.INTERRUPTED)
        self.voice_engine.stop_speaking()
        self.recorder.interrupt()
        self.task_manager.request_emergency_stop()
        self.confirmation_mgr.cancel_pending()
        print("[JARVIS] Emergency stop executed. All active automation halted.")
        self.state_machine.set_state(VoiceState.LISTENING)
        if active_speaker == SpeakerClassification.OWNER:
            return f"Stopped immediately, {self.preferences.get('preferred_title', 'Sir')}."
        return "Stopped immediately."

    def build_system_prompt(self, speaker: Optional[SpeakerClassification] = None) -> str:
        """Constructs persona system prompt tailored to speaker identity."""
        active_speaker = speaker or self._current_speaker
        owner = self.preferences.get("owner_name", "Sanjay")
        title = self.preferences.get("preferred_title", "Sir")

        if active_speaker == SpeakerClassification.OWNER:
            identity_rule = (
                f"The verified speaker is the owner.\n"
                f"CRITICAL NAMING RULE: Always address the owner ONLY as '{title}'.\n"
                f"NEVER address the owner as '{owner}'. Do NOT say 'Hello, {owner}' or 'Yes, {owner}'.\n"
                f"The owner's real name is {owner}, but you must speak the name '{owner}' ONLY when the user explicitly asks about their identity (e.g. 'What is my name?' -> 'Your name is {owner}, {title}.' or 'Who is your owner?' -> 'My owner is {owner}, {title}.').\n"
                f"Under all other circumstances, address the owner strictly as '{title}'."
            )
        else:
            identity_rule = (
                "The current speaker is an UNKNOWN / GUEST speaker (NOT Sanjay).\n"
                "Do NOT address them as Sanjay or Sir.\n"
                "Use neutral, polite phrasing like 'Hello'.\n"
                "Do NOT disclose passwords, API keys, private documents, or project data.\n"
                "Do NOT execute file deletion, system configuration changes, or shell/PowerShell commands."
            )

        prompt = (
            f"You are JARVIS, an autonomous personal Windows desktop AI assistant and computer-use agent created for {owner}.\n"
            f"{identity_rule}\n"
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

    def _is_agent_loop_goal(self, text: str) -> bool:
        """Determines if a command should be orchestrated via Phase 3 Agent Loop."""
        t_low = text.lower()
        if any(marker in t_low for marker in [" and inside ", " and create ", " and run ", " then ", " and write ", " inside it ", " in it "]):
            return True
        if "folder" in t_low and any(w in t_low for w in ["create", "make", "delete", "remove"]):
            return True
        if "file" in t_low and any(w in t_low for w in ["create", "write", "delete", "rename"]):
            return True
        if any(ext in t_low for ext in [".py", ".txt", ".js", ".json", ".html", ".md"]) and any(w in t_low for w in ["create", "write", "run", "make"]):
            return True
        if any(p in t_low for p in ["run python", "execute python", "run it"]):
            return True
        return False

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
        stop_phrases = ["jarvis stop", "stop", "abort", "cancel", "cancel task", "stop this", "wait"]
        if clean_input.lower() in stop_phrases or any(clean_input.lower().startswith(p) for p in ["jarvis stop", "cancel task", "stop this"]):
            return self.emergency_stop(active_speaker)

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

        # Step 2.5: Fast Path Hybrid Intelligence (Sub-second local OS actions & conversational greetings)
        t_route_start = time.perf_counter()
        fast_result = self.fast_router.try_fast_path(clean_input, active_speaker)
        if fast_result is not None:
            self.perf_metrics["routing_ms"] = getattr(self.fast_router, "last_routing_ms", (time.perf_counter() - t_route_start) * 1000.0)
            self.perf_metrics["tool_ms"] = getattr(self.fast_router, "last_tool_ms", 0.0)
            self.perf_metrics["verification_ms"] = getattr(self.fast_router, "last_verify_ms", 0.0)
            self.perf_metrics["llm_ms"] = 0.0
            action_name, fast_resp = fast_result
            if action_name not in ["chat", "greeting"]:
                self.state_machine.set_state(VoiceState.EXECUTING)
            self.memory.add_user_message(clean_input)
            self.memory.add_assistant_message(fast_resp)
            self.audit_logger.log_action(
                task=clean_input,
                speaker=active_speaker,
                action=f"FAST_PATH_{action_name.upper()}",
                risk_level=RiskLevel.LEVEL_0_INFORMATIONAL,
                status=ActionStatus.SUCCESS,
                details={"response": fast_resp[:400]},
            )
            return fast_resp

        # Step 2.7: Active-Context Skill Resolution (When app context dictates command handling)
        ctx = get_context_memory()
        active_app = ctx.get_active_app()
        if active_app:
            active_skill = self.fast_router.skill_registry.get_skill(active_app.lower())
            if active_skill and active_skill.can_handle(clean_input, {}, ctx):
                res = active_skill.execute(clean_input, {}, ctx)
                if res.success:
                    self.perf_metrics["routing_ms"] = (time.perf_counter() - t_route_start) * 1000.0
                    self.perf_metrics["tool_ms"] = 0.0
                    self.perf_metrics["verification_ms"] = 0.0
                    self.perf_metrics["llm_ms"] = 0.0
                    self.state_machine.set_state(VoiceState.EXECUTING)
                    self.memory.add_user_message(clean_input)
                    self.memory.add_assistant_message(res.message)
                    self.audit_logger.log_action(
                        task=clean_input,
                        speaker=active_speaker,
                        action=f"CONTEXT_SKILL_{active_app.upper()}",
                        risk_level=RiskLevel.LEVEL_0_INFORMATIONAL,
                        status=ActionStatus.SUCCESS,
                        details={"response": res.message[:400]},
                    )
                    return res.message

        # Step 2.8: Local SLM Layer (Natural-language understanding, paraphrases, entity extraction, local tools)
        if hasattr(self, "local_slm") and self.local_slm.enabled:
            slm_result = self.local_slm.try_slm_routing(clean_input, active_speaker, context=ctx)
            if slm_result is not None:
                action_name, slm_resp = slm_result
                self.perf_metrics["routing_ms"] = getattr(self.local_slm, "last_routing_latency_ms", (time.perf_counter() - t_route_start) * 1000.0)
                self.perf_metrics["tool_ms"] = 0.0
                self.perf_metrics["verification_ms"] = 0.0
                self.perf_metrics["llm_ms"] = 0.0
                if action_name not in ["chat", "greeting", "security_denied"]:
                    self.state_machine.set_state(VoiceState.EXECUTING)
                self.memory.add_user_message(clean_input)
                self.memory.add_assistant_message(slm_resp)
                self.audit_logger.log_action(
                    task=clean_input,
                    speaker=active_speaker,
                    action=f"SLM_{action_name.upper()}",
                    risk_level=RiskLevel.LEVEL_0_INFORMATIONAL,
                    status=ActionStatus.SUCCESS if action_name != "security_denied" else ActionStatus.DENIED,
                    details={"response": slm_resp[:400]},
                )
                return slm_resp

        # Step 3: Screen Vision Direct Check
        if any(term in clean_input.lower() for term in ["what is this error", "look at my screen", "look at screen", "what's wrong here", "what is on my screen"]):
            t_vision_start = time.perf_counter()
            self.perf_metrics["routing_ms"] = (t_vision_start - t_route_start) * 1000.0
            self.state_machine.set_state(VoiceState.EXECUTING)
            _, vision_resp = self.screen_vision.inspect_screen(clean_input, active_speaker)
            t_vision_end = time.perf_counter()
            self.perf_metrics["tool_ms"] = 0.0
            self.perf_metrics["verification_ms"] = (t_vision_end - t_vision_start) * 1000.0
            self.perf_metrics["llm_ms"] = 0.0
            self.memory.add_user_message(clean_input)
            self.memory.add_assistant_message(vision_resp)
            return vision_resp

        # Step 3.5: Visual Contextual Follow-up ("Fix it") (Section 19)
        if clean_input.lower() in ["fix it", "fix this", "fix this error", "resolve it", "fix the error", "fix that"]:
            ctx = get_context_memory()
            recent_err = ctx.get_last_observed_error()
            recent_app = ctx.get_last_observed_error_app()
            if recent_err:
                clean_input = f"Fix the error '{recent_err}' in {recent_app or 'current project'}"
                print(f"[CONTEXT] Resolved 'fix it' to: '{clean_input}'")
            else:
                resp = "Sir, I don't have a record of an observed error to fix. Would you like me to inspect your screen first?"
                self.memory.add_user_message(clean_input)
                self.memory.add_assistant_message(resp)
                return resp

        # Step 4: Autonomous Goal-Oriented Computer & Tool Planning
        self.state_machine.set_state(VoiceState.EXECUTING)

        # Fail-closed security check for UNKNOWN speakers attempting computer tasks
        if active_speaker != SpeakerClassification.OWNER:
            print(f"[SECURITY] DENIED: Goal execution blocked for speaker={active_speaker.value}")
            fail_resp = "I don't recognize this voice. Computer controls are locked."
            self.memory.add_user_message(clean_input)
            self.memory.add_assistant_message(fail_resp)
            return fail_resp

        # Route complex / multi-step or file/code tasks to Phase 3 Agent Loop
        t_llm_start = time.perf_counter()
        self.perf_metrics["routing_ms"] = (t_llm_start - t_route_start) * 1000.0
        success = False
        final_response = ""
        try:
            if self._is_agent_loop_goal(clean_input):
                success, final_response = self.agent_loop.run_goal(
                    goal=clean_input,
                    speaker=active_speaker,
                    confirmed=False,
                )
            else:
                system_prompt = self.build_system_prompt(active_speaker)
                success, final_response = self.planner.execute_goal(
                    goal=clean_input,
                    speaker=active_speaker,
                    system_prompt=system_prompt,
                )
        except Exception as e:
            print(f"[AI] Reasoning provider unavailable: {e}")
            success = False
            final_response = "Sir, the cloud reasoning service is currently unavailable, but all local capabilities and computer controls remain fully operational."
        t_llm_end = time.perf_counter()
        self.perf_metrics["llm_ms"] = (t_llm_end - t_llm_start) * 1000.0
        self.perf_metrics["tool_ms"] = 0.0
        self.perf_metrics["verification_ms"] = 0.0

        # Update Conversation Memory
        self.memory.add_user_message(clean_input)
        self.memory.add_assistant_message(final_response)

        # Model resource hygiene: unload heavy idle models after the configured idle window
        try:
            from core.models.model_manager import get_model_manager
            get_model_manager().unload_idle()
        except Exception:
            pass

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

    def speak_response(self, text: str) -> bool:
        """
        Speaks the response using ONE consistent neural voice identity.
        Monitors concurrently for user barge-in / interruption phrases while speaking.
        """
        if not text:
            return False
        # Sanitizes any accidental credentials or markdown
        safe_to_speak, safe_text = self.secrets_mgr.validate_safe_for_speech(text)
        # Use single unified multilingual voice identity
        voice = self.multilingual.get_preferred_voice_for_language("multilingual")

        return self.voice_engine.speak(safe_text, voice=voice)

    def run_turn(self) -> Optional[str]:
        """
        Executes one full voice turn:
        LISTENING -> Speech Detection (VAD) -> End-of-Speech -> PROCESSING -> EXECUTING (if required) -> SPEAKING -> LISTENING.
        Runs STT and Speaker Verification concurrently on captured audio turn.
        Instruments and logs timing metrics.
        """
        import concurrent.futures

        t_turn_start = time.perf_counter()
        self.state_machine.set_state(VoiceState.LISTENING)
        print("\n[JARVIS] Listening...")

        t_vad_start = time.perf_counter()
        audio_data = None
        if hasattr(self.voice_engine, "record_audio"):
            audio_data = self.voice_engine.record_audio()
        if not audio_data:
            rec_res = self.voice_engine.listen()
            audio_data = rec_res[0] if rec_res else None

        if not audio_data:
            return None

        t_vad_end = time.perf_counter()
        vad_end_ms = (t_vad_end - t_vad_start) * 1000.0
        self.perf_metrics["vad_end_ms"] = vad_end_ms

        # State transition to PROCESSING
        self.state_machine.set_state(VoiceState.PROCESSING)

        # Run STT and Speaker Verification CONCURRENTLY
        stt_ms = 0.0
        speaker_verify_ms = 0.0

        def _do_stt():
            nonlocal stt_ms
            t_s = time.perf_counter()
            res = self.stt.transcribe(audio_data)
            stt_ms = (time.perf_counter() - t_s) * 1000.0
            return res

        def _do_verify():
            nonlocal speaker_verify_ms
            t_v = time.perf_counter()
            spk, conf = self.verifier.verify_speaker(audio_data)
            speaker_verify_ms = (time.perf_counter() - t_v) * 1000.0
            return spk, conf

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_stt = executor.submit(_do_stt)
            fut_verify = executor.submit(_do_verify)
            transcription = fut_stt.result()
            speaker, confidence = fut_verify.result()

        self.perf_metrics["stt_ms"] = stt_ms
        self.perf_metrics["speaker_verify_ms"] = speaker_verify_ms

        if transcription:
            print(f"[STT] Transcript received: {transcription}")

        # Speaker Verification & session gate
        self._current_speaker = speaker
        if speaker == SpeakerClassification.OWNER:
            print(f"[JARVIS] Speaker Verified: OWNER (Confidence: {confidence:.2f})")
            print("[SECURITY] Owner session authenticated")
            get_speaker_session().authenticate_owner()
        else:
            print(f"[JARVIS] Speaker Verification: UNKNOWN / GUEST (Confidence: {confidence:.2f})")
            print("[SECURITY] DENIED: Speaker is not OWNER")
            get_speaker_session().invalidate("Speaker is not OWNER")

        # Voice PIN recovery fallback (last resort; typing is preferred: python main.py --auth-pin)
        if not transcription:
            self.state_machine.set_state(VoiceState.LISTENING)
            return None

        try:
            from security.recovery import authenticate_with_pin, has_recovery_pin
            if (speaker != SpeakerClassification.OWNER
                    and has_recovery_pin()
                    and not get_speaker_session().is_owner_active()):
                import re
                m = re.search(r'\bpin\s*(?:is\s+)?(\d{4,12})\b', transcription, re.IGNORECASE)
                if m and authenticate_with_pin(m.group(1)):
                    print("[SECURITY] Owner session recovered via PIN (voice fallback)")
                    self._current_speaker = SpeakerClassification.OWNER
                    speaker = SpeakerClassification.OWNER
                elif m:
                    print("[SECURITY] PIN fallback rejected (denied / lockout)")
                    return "PIN verification failed. Access remains locked, Sir."
        except Exception:
            pass

        print(f"User ({speaker.value}): {transcription}")
        response = self.process_input(transcription, speaker=speaker)
        print(f"JARVIS: {response}")

        t_before_tts = time.perf_counter()
        self.perf_metrics["tts_start_ms"] = (t_before_tts - t_turn_start) * 1000.0
        self.perf_metrics["total_response_ms"] = (t_before_tts - t_vad_end) * 1000.0

        # Log performance timing metrics
        self.log_perf_metrics()

        self.speak_response(response)
        return response

    def run_continuous_voice(self):
        """Continuous voice loop."""
        self._running = True
        self.voice_engine.start()

        # Fix the microphone threshold to the actual room before listening.
        # Without this the default 400 RMS floor triggers on fans/keyboard noise
        # and produces 30-60 s recordings dominated by silence (which destroys
        # ECAPA speaker verification).
        try:
            if getattr(self.settings, "dynamic_energy_threshold", True):
                print("[VOICE] Calibrating ambient noise (please stay quiet)...")
                self.recorder.calibrate_ambient_noise(
                    duration=getattr(self.settings, "vad_noise_calibration_seconds", 0.5))
        except Exception as e:
            print(f"[VOICE] Ambient calibration skipped: {e}")

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
                self.state_machine.set_state(VoiceState.LISTENING)
                time.sleep(1)

        self.voice_engine.stop()


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
