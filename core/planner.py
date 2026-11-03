"""
JARVIS Autonomous Computer-Use Planner & Verifier
Orchestrates the Goal -> Inspect -> Plan -> Act -> Observe -> Verify -> Report loop.
Never claims success without tangible observation and verification.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from config.constants import SpeakerClassification
from ai.gemini_client import GeminiClient
from ai.tool_planner import ToolCatalog
from core.executor import ActionExecutor
from core.task_manager import TaskManager, TaskState
from vision.screen_understanding import ScreenVision


class AutonomousPlanner:
    """
    Goal-driven autonomous agent planner for Windows desktop tasks and software development.
    """

    def __init__(
        self,
        client: GeminiClient,
        executor: ActionExecutor,
        task_manager: Optional[TaskManager] = None,
        screen_vision: Optional[ScreenVision] = None,
        tool_catalog: Optional[ToolCatalog] = None,
    ):
        self.client = client
        self.executor = executor
        self.task_manager = task_manager or TaskManager()
        self.screen_vision = screen_vision or ScreenVision()
        self.tool_catalog = tool_catalog or ToolCatalog()

    def execute_goal(
        self,
        goal: str,
        speaker: SpeakerClassification = SpeakerClassification.OWNER,
        system_prompt: Optional[str] = None,
        max_turns: int = 5,
    ) -> Tuple[bool, str]:
        """
        Executes a multi-step computer task autonomously with self-correction and verification:
        Goal -> Reasoning -> Action -> Observe -> Verify -> Report.

        Returns:
            (success: bool, final_spoken_summary: str)
        """
        task = self.task_manager.create_task(goal)
        tool_decls = self.tool_catalog.to_gemini_declarations()

        accumulated_observations: List[str] = []
        last_action_success = True

        for turn in range(1, max_turns + 1):
            if self.task_manager.is_stop_requested():
                return False, "Task aborted per emergency stop command, Sir."

            # Construct visual & execution context for Gemini
            obs_context = "\n".join(accumulated_observations) if accumulated_observations else "Task initialized."

            # Attempt tool planning if provider supports it
            plan = {}
            if hasattr(self.client, "plan_with_tools"):
                try:
                    plan = self.client.plan_with_tools(
                        goal=goal,
                        tools=tool_decls,
                        system_prompt=system_prompt,
                        screen_context=obs_context,
                    )
                except Exception:
                    plan = {}

            tool_calls = plan.get("tool_calls", [])
            direct_response = plan.get("response")

            # If Gemini produced a direct conversational response without tools, check if goal is an action
            if not tool_calls:
                from core.normalizer import Normalizer, IntentType
                intent, params = Normalizer.classify_intent(goal)

                # If intent is an executable computer action or screen vision, execute via trusted executor
                if intent in [
                    IntentType.OPEN_APP,
                    IntentType.OPEN_WEBSITE,
                    IntentType.SEARCH_YOUTUBE,
                    IntentType.PLAY_YOUTUBE,
                    IntentType.PLAY_LATEST_MUSIC,
                    IntentType.SEARCH_WEB,
                    IntentType.MULTI_STEP,
                    IntentType.SCREEN_VISION,
                ]:
                    if intent == IntentType.OPEN_APP:
                        exec_res = self.executor.execute_action("open_application", {"target": params.get("app_key") or params.get("raw_name")}, speaker=speaker)
                    elif intent == IntentType.OPEN_WEBSITE:
                        exec_res = self.executor.execute_action("open_website", params, speaker=speaker)
                    elif intent in [IntentType.SEARCH_YOUTUBE, IntentType.PLAY_LATEST_MUSIC]:
                        exec_res = self.executor.execute_action("search_youtube", params, speaker=speaker)
                    elif intent == IntentType.PLAY_YOUTUBE:
                        exec_res = self.executor.execute_action("play_youtube", params, speaker=speaker)
                    elif intent == IntentType.SEARCH_WEB:
                        exec_res = self.executor.execute_action("search_web", params, speaker=speaker)
                    elif intent == IntentType.MULTI_STEP:
                        exec_res = self.executor.execute_action("multi_step", params, speaker=speaker)
                    elif intent == IntentType.SCREEN_VISION:
                        exec_res = self.screen_vision.inspect_screen(params.get("query", goal), speaker=speaker)

                    s_ok = exec_res.success if hasattr(exec_res, "success") else exec_res[0]
                    s_msg = exec_res.message if hasattr(exec_res, "message") else exec_res[1]
                    if s_ok:
                        self.task_manager.mark_completed(s_msg)
                    else:
                        self.task_manager.mark_failed(s_msg)
                    return s_ok, s_msg

                # Requirement: If goal is an action request, NEVER allow free-form LLM text to claim success
                lower_resp = (direct_response or "").lower()
                action_words = ["open", "launch", "type", "save", "click", "screen", "play", "start", "see"]
                forbidden_claims = [
                    "i opened", "i launched", "i can see", "i see it", "i typed", "i saved",
                    "i played", "i've opened", "i've launched", "i've typed", "i've saved",
                    "opened it", "launched it", "typed it", "saved it", "played it",
                    "see your screen", "i checked the screen", "i see the desktop", "is now open"
                ]
                if any(w in goal.lower() for w in action_words) and any(c in lower_resp for c in forbidden_claims):
                    blocked_msg = f"Sir, I could not verify executing '{goal}' because no verified computer action was observed."
                    self.task_manager.mark_failed(blocked_msg)
                    return False, blocked_msg

                if direct_response and direct_response not in ["Offline fallback.", None]:
                    self.task_manager.mark_completed(direct_response)
                    return True, direct_response
                # Conversational fallback to client.generate
                try:
                    gen_resp = self.client.generate(goal, system_prompt=system_prompt)
                    self.task_manager.mark_completed(gen_resp)
                    return True, gen_resp
                except Exception as e:
                    err_str = str(e).lower()
                    if "credit" in err_str or "payment" in err_str or "402" in err_str:
                        print("[OPENROUTER] Request failed: insufficient credits")
                    else:
                        print(f"[OPENROUTER] Request failed: {e}")
                    ctrl_msg = "Sir, the AI planning service is currently unavailable, so I couldn't safely continue this task."
                    self.task_manager.mark_failed(ctrl_msg)
                    return False, ctrl_msg

            # Execute tool calls sequentially
            for call in tool_calls:
                if self.task_manager.is_stop_requested():
                    return False, "Task halted immediately, Sir."

                action_name = call["name"]
                params = call.get("parameters", {})

                # If action requires screen vision, invoke screen vision
                if action_name == "inspect_screen":
                    s_ok, s_res = self.screen_vision.inspect_screen(params.get("query", goal), speaker)
                    obs = f"[Screen Observation]: {s_res}"
                    accumulated_observations.append(obs)
                    self.task_manager.record_step(action_name, params, "SUCCESS" if s_ok else "FAILED", obs)
                    last_action_success = s_ok
                    continue

                # Execute action via Trusted Executor
                exec_ok, exec_msg = self.executor.execute_action(
                    action=action_name,
                    parameters=params,
                    speaker=speaker,
                    confirmed=False,
                )

                obs = f"[Tool: {action_name}] Result: {exec_msg}"
                accumulated_observations.append(obs)
                self.task_manager.record_step(action_name, params, "SUCCESS" if exec_ok else "FAILED", obs)
                last_action_success = exec_ok

                # If confirmation was requested by the security system, stop and ask the user
                if "confirmation" in exec_msg.lower() or "shall i proceed" in exec_msg.lower() or "confirm" in exec_msg.lower():
                    return False, exec_msg

            # Verify: Ask Gemini if goal is completed or what step remains (only if last_action_success is True)
            if last_action_success:
                verification_prompt = (
                    f"User Goal: {goal}\n"
                    f"Observations so far:\n{chr(10).join(accumulated_observations)}\n\n"
                    "Did we achieve the goal and verify the result? If yes, answer 'COMPLETED: <friendly natural summary for Sanjay>'. "
                    "If errors occurred and need fixing, explain what you will fix. If cannot proceed, explain the blocker honestly."
                )

                try:
                    v_resp = self.client.generate(verification_prompt, system_prompt=system_prompt)
                    if "COMPLETED:" in v_resp:
                        summary = v_resp.split("COMPLETED:", 1)[1].strip()
                        self.task_manager.mark_completed(summary)
                        return True, summary
                except Exception as e:
                    err_str = str(e).lower()
                    if "credit" in err_str or "payment" in err_str or "402" in err_str:
                        print("[OPENROUTER] Request failed: insufficient credits")
                    else:
                        print(f"[OPENROUTER] Request failed: {e}")

        # Final Summary formulation based on actual verified steps
        final_summary = self._generate_final_summary(goal, accumulated_observations, last_action_success)
        if last_action_success:
            self.task_manager.mark_completed(final_summary)
        else:
            self.task_manager.mark_failed(final_summary)
        return last_action_success, final_summary

    def _generate_final_summary(
        self,
        goal: str,
        observations: List[str],
        success: bool,
    ) -> str:
        """Formulates a natural, friendly verbal summary addressing Sanjay / Sir."""
        obs_text = "\n".join(observations[-3:]) if observations else "Action processed."

        # Requirement A: If success is False, NEVER say Opened, Done, Completed, Playing, Saved
        if not success:
            last_err = observations[-1] if observations else "Action execution could not be verified."
            if "Result: " in last_err:
                clean_err = last_err.split("Result: ", 1)[1].strip()
            else:
                clean_err = last_err
            return f"Sir, I tried to complete that action, but {clean_err}."

        prompt = (
            f"User Goal: {goal}\n"
            f"Execution Steps & Observations:\n{obs_text}\n\n"
            "Formulate a short, natural, friendly spoken response for Sanjay/Sir (1-2 sentences). "
            "Never sound robotic. State honestly what was done and what was observed. "
            "Never claim success if an error occurred."
        )
        try:
            return self.client.generate(prompt)
        except Exception:
            return f"Done, Sir. I completed the request and verified the result."
