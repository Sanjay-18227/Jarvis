"""
JARVIS Ollama Local AI Provider
Enables local, private offline LLM inference via Ollama's REST API.
Does not require any cloud API key.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Iterator
from .base import AIProvider


class OllamaProvider(AIProvider):
    """
    Local AI Provider for running Llama 3, Mistral, Qwen, etc. completely offline via Ollama.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3:latest",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"Ollama Local ({self.model})"

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is running and reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1.0)
            return resp.status_code == 200
        except Exception:
            return False

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

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> str:
        """Generates text completion via Ollama."""
        if not self.is_available():
            raise RuntimeError(f"Ollama daemon not reachable at {self.base_url}. Please start Ollama.")

        url = f"{self.base_url}/api/chat"
        messages = self._build_messages(prompt, system_prompt, history)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama returned status {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def plan_with_tools(
        self,
        goal: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        screen_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes autonomous goal reasoning with tools via Ollama OpenAI-compatible endpoint.
        """
        if not self.is_available():
            raise RuntimeError(f"Ollama daemon not reachable at {self.base_url}.")

        url = f"{self.base_url}/v1/chat/completions"
        full_prompt = goal
        if screen_context:
            full_prompt = f"[Visual / Screen Context]: {screen_context}\n\n[User Goal]: {goal}"

        messages = self._build_messages(full_prompt, system_prompt, history)
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": openai_tools,
            "temperature": 0.2,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                parsed_calls = []
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args_raw = fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except Exception:
                        fn_args = {}
                    if fn_name:
                        parsed_calls.append({"name": fn_name, "parameters": fn_args})

                return {
                    "thought": "Ollama local reasoning complete.",
                    "tool_calls": parsed_calls,
                    "response": msg.get("content"),
                }
        except Exception:
            pass

        # Fallback to standard generate
        text_resp = self.generate(full_prompt, system_prompt, history)
        return {"thought": "Ollama text response.", "tool_calls": [], "response": text_resp}

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ) -> Iterator[str]:
        """Streams tokens from local Ollama daemon for instant speech output."""
        if not self.is_available():
            yield f"Local Ollama is not running at {self.base_url}."
            return

        url = f"{self.base_url}/api/chat"
        messages = self._build_messages(prompt, system_prompt, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=self.timeout) as resp:
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
        except Exception:
            yield self.generate(prompt, system_prompt, history, max_tokens, temperature)
