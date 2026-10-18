"""
JARVIS Google Gemini Client & Autonomous Reasoning Engine
Implements multi-model failover, exponential backoff, function/tool calling,
and multimodal vision integration using the official google-genai SDK.
"""

import time
import json
import warnings
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", message=".*automatic function calling.*")
from ai.provider import AIProvider
from ai.tool_planner import ToolCatalog, ToolDefinition


class GeminiClient(AIProvider):
    """
    High-performance Google Gemini client with tool calling and model failover.
    """

    FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"]

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.8-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[GeminiClient] Initialization notice: {e}")

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"

    def is_available(self) -> bool:
        return self._client is not None and bool(self.api_key)

    def _build_contents(self, prompt: str, history: Optional[List[Dict[str, str]]] = None):
        """Constructs multi-turn message contents for Gemini."""
        from google.genai import types

        contents = []
        if history:
            for turn in history:
                role = turn.get("role", "user")
                gemini_role = "model" if role == "assistant" else "user"
                content_text = turn.get("content", "")
                if content_text:
                    contents.append(
                        types.Content(
                            role=gemini_role,
                            parts=[types.Part.from_text(text=content_text)]
                        )
                    )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        )
        return contents

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> str:
        """Generates conversational response with model failover."""
        if not self.is_available():
            return "I am currently running in offline fallback mode, Sir. Please configure your GEMINI_API_KEY."

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        contents = self._build_contents(prompt, history)

        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        last_error = None

        for target_model in models_to_try:
            for attempt in range(2):
                try:
                    response = self._client.models.generate_content(
                        model=target_model,
                        contents=contents,
                        config=config,
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if "503" in err_str or "unavailable" in err_str or "overloaded" in err_str:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    break

        return f"I encountered a temporary service glitch, Sir. ({str(last_error)})"

    def plan_with_tools(
        self,
        goal: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        screen_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reasons about the user's goal and returns structured tool calls or conversation.

        Returns:
            {
                "thought": str,
                "tool_calls": List[{"name": str, "parameters": dict}],
                "response": Optional[str]
            }
        """
        if not self.is_available():
            return {"thought": "Offline mode", "tool_calls": [], "response": "Offline fallback."}

        from google.genai import types

        # Build tools for Gemini Function Calling
        gemini_tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t.get("parameters"),
                    )
                    for t in tools
                ]
            )
        ]

        full_prompt = goal
        if screen_context:
            full_prompt = f"[Visual / Screen Context]: {screen_context}\n\n[User Goal]: {goal}"

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=gemini_tools,
            temperature=0.3,
        )
        contents = self._build_contents(full_prompt, history)

        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]

        for target_model in models_to_try:
            for attempt in range(2):
                try:
                    response = self._client.models.generate_content(
                        model=target_model,
                        contents=contents,
                        config=config,
                    )
                    tool_calls = []
                    text_resp = ""

                    # Inspect function calls from candidate parts
                    if response.candidates:
                        first_cand = response.candidates[0]
                        if first_cand.content and first_cand.content.parts:
                            for part in first_cand.content.parts:
                                if hasattr(part, "function_call") and part.function_call:
                                    fc = part.function_call
                                    args = dict(fc.args) if fc.args else {}
                                    tool_calls.append({"name": fc.name, "parameters": args})
                                elif hasattr(part, "text") and part.text:
                                    text_resp += part.text

                    return {
                        "thought": "Gemini tool reasoning complete.",
                        "tool_calls": tool_calls,
                        "response": text_resp.strip() if text_resp else None,
                    }

                except Exception as e:
                    err_str = str(e).lower()
                    if "503" in err_str or "unavailable" in err_str:
                        time.sleep(1.0)
                        continue
                    break

        return {"thought": "Failed calling tools", "tool_calls": [], "response": None}
