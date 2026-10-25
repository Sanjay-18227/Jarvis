"""
JARVIS OpenRouter AI Provider
Enables OpenRouter as the primary online LLM provider using OpenAI-compatible HTTP REST API.
Reads OPENROUTER_API_KEY from environment/.env.
Never exposes or logs API keys.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Iterator
from .base import AIProvider


class OpenRouterProvider(AIProvider):
    """
    High-performance OpenRouter provider for online AI reasoning and tool planning.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-3.3-70b-instruct",
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "https://github.com/sanjay/jarvis",
        app_name: str = "JARVIS",
        timeout: float = 25.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_name = app_name
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"OpenRouter ({self.model})"

    def is_available(self) -> bool:
        """Returns True if a non-empty OpenRouter API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"
        return headers

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> str:
        """
        Generates conversational text completion via OpenRouter.
        Raises RuntimeError on API failure so that the caller/gateway can fall back.
        """
        if not self.is_available():
            raise RuntimeError("OpenRouter API key is not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        messages = self._build_messages(prompt, system_prompt, history)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter returned status {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("No choices returned from OpenRouter response.")

            message = choices[0].get("message", {})
            content = message.get("content", "")
            return content.strip() if content else ""

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"OpenRouter request failed: {e}") from e

    def plan_with_tools(
        self,
        goal: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        screen_context: Optional[str] = None,
        max_tokens: int = 800,
    ) -> Dict[str, Any]:
        """
        Executes autonomous goal reasoning with tools via OpenRouter function calling.
        """
        if not self.is_available():
            raise RuntimeError("OpenRouter API key is not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()

        full_prompt = goal
        if screen_context:
            full_prompt = f"[Visual / Screen Context]: {screen_context}\n\n[User Goal]: {goal}"

        messages = self._build_messages(full_prompt, system_prompt, history)

        # Convert tool catalog declarations to standard OpenAI tool format
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                }
            })

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter status {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})

            parsed_tool_calls = []
            raw_tool_calls = msg.get("tool_calls", [])
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args_raw = fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except Exception:
                        fn_args = {}
                    if fn_name:
                        parsed_tool_calls.append({"name": fn_name, "parameters": fn_args})

            direct_text = msg.get("content")
            return {
                "thought": "OpenRouter reasoning complete.",
                "tool_calls": parsed_tool_calls,
                "response": direct_text.strip() if direct_text else None,
            }

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"OpenRouter tool planning failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> Iterator[str]:
        """
        Streams conversational tokens via OpenRouter SSE API for rapid speech generation.
        """
        if not self.is_available():
            raise RuntimeError("OpenRouter API key is not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        messages = self._build_messages(prompt, system_prompt, history)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout) as resp:
                if resp.status_code != 200:
                    yield self.generate(prompt, system_prompt, history, max_tokens, temperature)
                    return

                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    yield token
                            except Exception:
                                continue
        except Exception:
            yield self.generate(prompt, system_prompt, history, max_tokens, temperature)

