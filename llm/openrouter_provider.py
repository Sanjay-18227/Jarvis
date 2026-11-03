"""
JARVIS OpenRouter AI Provider
Enables OpenRouter as the primary online LLM provider using OpenAI-compatible HTTP REST API.
Reads OPENROUTER_API_KEY from environment/.env.
Never exposes or logs API keys.
Includes circuit-breaker with cooldown for fast local fallback on 401/402 errors.
"""

import json
import os
import time
import requests
from typing import List, Dict, Any, Optional, Iterator
from .base import AIProvider, ReasoningProvider


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class _CreditExhausted(OpenRouterError):
    """Internal sentinel: account credit exhaustion (402) on the primary model.
    Handled by the free-model fallback logic rather than failing the request."""


class OpenRouterProvider(ReasoningProvider):
    """
    High-performance OpenRouter provider for online AI reasoning and tool planning.
    Includes circuit-breaker protection so credit exhaustion (402) or auth failures (401)
    fail fast without multi-retry delays, letting local execution proceed unimpeded.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openrouter/free",
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "https://github.com/sanjay/jarvis",
        app_name: str = "JARVIS",
        timeout: float = 25.0,
        circuit_cooldown_seconds: float = 300.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_name = app_name
        self.timeout = timeout
        self._circuit_cooldown_seconds = circuit_cooldown_seconds

        # Circuit breaker state
        self._circuit_open: bool = False
        self._circuit_opened_at: float = 0.0
        self._circuit_reason: str = ""
        self._reasoning_provider_available: bool = True

        # Free-model fallback chain for account credit exhaustion (402) on the primary model.
        # Uses openrouter/free and reliable free-tier reasoning models.
        default_chain = "openrouter/free,nvidia/nemotron-3.5-lightning:free,google/gemma-4-26b-a4b-it:free,meta-llama/llama-3.3-70b-instruct:free"
        chain_env = os.getenv("OPENROUTER_FALLBACK_MODELS") or os.getenv("OPENROUTER_FALLBACK_MODEL") or default_chain
        self.fallback_models: List[str] = [m.strip() for m in chain_env.split(",") if m.strip()]
        self.fallback_model: Optional[str] = self.fallback_models[0] if self.fallback_models else None
        self._primary_credit_limited: bool = False
        self._primary_reprobe_seconds: float = 600.0  # re-probe primary after ~10 min (e.g. after a top-up)
        self._primary_retry_at: float = 0.0
        # Short cooldowns for free fallbacks that are rate-limited / unavailable right now
        self._fallback_cooldown: Dict[str, float] = {}
        self._fallback_cooldown_seconds: float = 60.0

    def _primary_allowed(self) -> bool:
        return (not self._primary_credit_limited) or (time.time() >= self._primary_retry_at)

    def _mark_primary_credit_limited(self) -> None:
        """Remembers that the primary paid model is credit-blocked so later calls
        go straight to the free fallbacks without paying the extra 402 round-trip."""
        self._primary_credit_limited = True
        self._primary_retry_at = time.time() + self._primary_reprobe_seconds

    def _clear_primary_credit_limited(self) -> None:
        self._primary_credit_limited = False
        self._primary_retry_at = 0.0

    def _model_chain(self) -> List[str]:
        """Builds the ordered model chain for one request: primary first (unless
        recently credit-blocked), then every free fallback not in cooldown."""
        chain: List[str] = []
        if self._primary_allowed():
            chain.append(self.model)
        now = time.time()
        for m in self.fallback_models:
            if m not in chain and self._fallback_cooldown.get(m, 0.0) <= now:
                chain.append(m)
        if not chain:
            # Everything is cooling down; try the first fallback (or primary) as last resort
            chain = self.fallback_models[:1] or [self.model]
        return chain

    def _cooldown_fallback(self, model: str, status_code: Optional[int]) -> None:
        """Puts a misbehaving free fallback on cooldown so the next request tries a different free model."""
        seconds = 600.0 if status_code == 402 else self._fallback_cooldown_seconds
        self._fallback_cooldown[model] = time.time() + seconds

    @property
    def name(self) -> str:
        return f"OpenRouter ({self.model})"

    def trip_circuit(self, reason: str, status_code: int = 402, cooldown: Optional[float] = None) -> None:
        """Trips the circuit breaker, preventing further remote LLM attempts during cooldown."""
        self._circuit_open = True
        self._circuit_opened_at = time.time()
        self._circuit_reason = reason
        self._reasoning_provider_available = False
        if cooldown is not None:
            self._circuit_cooldown_seconds = cooldown
        print(f"[AI] OpenRouter unavailable: {reason}")
        print("[AI] Skipping remote reasoning for local task")

    def reset_circuit(self) -> None:
        """Manually resets the circuit breaker to operational state."""
        self._circuit_open = False
        self._circuit_opened_at = 0.0
        self._circuit_reason = ""
        self._reasoning_provider_available = True
        print("[AI] OpenRouter circuit breaker manually reset to available.")

    def is_available(self) -> bool:
        """Returns True if OpenRouter is configured and circuit is healthy."""
        if not (self.api_key and self.api_key.strip()):
            return False

        if self._circuit_open:
            elapsed = time.time() - self._circuit_opened_at
            if elapsed < self._circuit_cooldown_seconds:
                return False
            # Cooldown expired, probe circuit
            print(f"[AI] OpenRouter cooldown expired ({elapsed:.1f}s >= {self._circuit_cooldown_seconds}s). Re-probing availability.")
            self._circuit_open = False
            self._circuit_reason = ""
            self._reasoning_provider_available = True

        return self._reasoning_provider_available

    @property
    def reasoning_provider_available(self) -> bool:
        return self.is_available()

    @reasoning_provider_available.setter
    def reasoning_provider_available(self, val: bool) -> None:
        self._reasoning_provider_available = val
        if not val:
            self.trip_circuit("manual deactivation")
        else:
            self.reset_circuit()

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
        Fails fast without retry on 401/402 errors to protect latency and local routing.
        Uses bounded single retry for transient timeouts, 429 rate limits, and 5xx errors.
        On account credit exhaustion (402) of the primary model, automatically retries
        with the configured free fallback model before failing.
        """
        if not self.is_available():
            if self._circuit_open:
                print(f"[AI] OpenRouter unavailable: {self._circuit_reason}")
                print("[AI] Skipping remote reasoning for local task")
                raise OpenRouterError(f"[AI] OpenRouter unavailable: {self._circuit_reason}", status_code=402 if "credit" in self._circuit_reason else 401)
            raise RuntimeError("OpenRouter API key is not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        messages = self._build_messages(prompt, system_prompt, history)

        def _run(model_name: str) -> str:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            max_attempts = 2  # 1 initial + 1 bounded retry for transient errors
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

                    if resp.status_code != 200:
                        err_text = resp.text[:300].lower()
                        # Fail-fast on 401 / 402 / credit / payment errors
                        if resp.status_code == 402 or "credit" in err_text or "payment" in err_text:
                            print("[OPENROUTER] Request failed: insufficient credits")
                            if model_name == self.model:
                                # Credits exhausted at account level: let the free fallbacks handle it
                                self._mark_primary_credit_limited()
                                raise _CreditExhausted("[AI] OpenRouter unavailable: insufficient credits", status_code=402)
                            raise _CreditExhausted("[AI] OpenRouter unavailable: insufficient credits", status_code=402)
                        elif resp.status_code == 401:
                            self.trip_circuit("unauthorized (invalid API key)", status_code=401)
                            raise OpenRouterError("[AI] OpenRouter unavailable: unauthorized (invalid API key)", status_code=401)
                        elif resp.status_code == 429 or 500 <= resp.status_code < 600:
                            reason = "rate limit exceeded" if resp.status_code == 429 else f"server error (HTTP {resp.status_code})"
                            if attempt < max_attempts:
                                print(f"[OPENROUTER] Transient failure ({reason}). Retrying once with bounded backoff...")
                                time.sleep(1.0)
                                continue
                            raise OpenRouterError(f"[OPENROUTER] Request failed: {reason}", status_code=resp.status_code)
                        else:
                            raise OpenRouterError(f"[OPENROUTER] Request failed: status {resp.status_code}", status_code=resp.status_code)

                    try:
                        data = resp.json()
                    except Exception as je:
                        raise OpenRouterError(f"[OPENROUTER] Invalid response format (JSON decode error): {je}", status_code=502)

                    choices = data.get("choices", [])
                    if not choices:
                        raise OpenRouterError("No choices returned from OpenRouter response.", status_code=502)

                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    return content.strip() if content else ""

                except requests.Timeout:
                    if attempt < max_attempts:
                        print("[OPENROUTER] Request timed out. Retrying once with bounded backoff...")
                        time.sleep(1.0)
                        continue
                    print("[OPENROUTER] Request failed: timeout")
                    raise OpenRouterError("[OPENROUTER] Request failed: timeout", status_code=408)

                except requests.ConnectionError:
                    if attempt < max_attempts:
                        print("[OPENROUTER] Connection failure. Retrying once with bounded backoff...")
                        time.sleep(1.0)
                        continue
                    print("[OPENROUTER] Request failed: connection error")
                    raise OpenRouterError("[OPENROUTER] Request failed: connection error", status_code=503)

                except Exception as e:
                    if isinstance(e, OpenRouterError):
                        raise
                    print(f"[OPENROUTER] Request failed: {e}")
                    raise OpenRouterError(f"[OPENROUTER] Request failed: {e}") from e

            raise OpenRouterError("[OPENROUTER] Max retries exhausted", status_code=500)

        # Primary paid model first (unless recently credit-blocked), then ANY working free model
        last_error: Optional[OpenRouterError] = None
        for model_name in self._model_chain():
            try:
                if model_name != self.model:
                    print(f"[AI] Falling back to free model: {model_name}")
                result = _run(model_name)
                if model_name == self.model:
                    self._clear_primary_credit_limited()
                elif model_name in self.fallback_models:
                    self._fallback_cooldown.pop(model_name, None)
                return result
            except _CreditExhausted as cred_err:
                if model_name in self.fallback_models:
                    self._cooldown_fallback(model_name, 402)
                last_error = cred_err
                continue
            except OpenRouterError as oe:
                if oe.status_code == 401:
                    # Invalid API key: no fallback model can fix the credentials
                    raise
                if model_name in self.fallback_models:
                    self._cooldown_fallback(model_name, oe.status_code)
                last_error = oe
                continue
        if last_error is not None:
            low = str(last_error).lower()
            if "credit" in low or getattr(last_error, "status_code", None) == 402:
                self.trip_circuit("insufficient credits", status_code=402)
            raise last_error
        raise OpenRouterError("[AI] OpenRouter unavailable: insufficient credits", status_code=402)

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
        Fails fast on 401/402 and trips the circuit breaker to enable local fallback.
        """
        if not self.is_available():
            if self._circuit_open:
                print(f"[AI] OpenRouter unavailable: {self._circuit_reason}")
                print("[AI] Skipping remote reasoning for local task")
                raise OpenRouterError(f"[AI] OpenRouter unavailable: {self._circuit_reason}", status_code=402 if "credit" in self._circuit_reason else 401)
            raise RuntimeError("OpenRouter API key is not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()

        full_prompt = goal
        if screen_context:
            full_prompt = f"[Visual / Screen Context]: {screen_context}\n\n[User Goal]: {goal}"

        messages = self._build_messages(full_prompt, system_prompt, history)

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

        def _run(model_name: str) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": max_tokens,
            }
            if openai_tools:
                payload["tools"] = openai_tools
                payload["tool_choice"] = "auto"

            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                    if resp.status_code != 200:
                        err_text = resp.text[:300].lower()
                        if resp.status_code == 402 or "credit" in err_text or "payment" in err_text:
                            print("[OPENROUTER] Request failed: insufficient credits")
                            if model_name == self.model:
                                # Credits exhausted at account level: let the free fallbacks handle it
                                self._mark_primary_credit_limited()
                                raise _CreditExhausted("[AI] OpenRouter unavailable: insufficient credits", status_code=402)
                            raise _CreditExhausted("[AI] OpenRouter unavailable: insufficient credits", status_code=402)
                        elif resp.status_code == 401:
                            self.trip_circuit("unauthorized (invalid API key)", status_code=401)
                            raise OpenRouterError("[AI] OpenRouter unavailable: unauthorized (invalid API key)", status_code=401)
                        elif resp.status_code == 429 or 500 <= resp.status_code < 600:
                            reason = "rate limit exceeded" if resp.status_code == 429 else f"server error (HTTP {resp.status_code})"
                            if attempt < max_attempts:
                                print(f"[OPENROUTER] Transient failure ({reason}). Retrying once with bounded backoff...")
                                time.sleep(1.0)
                                continue
                            raise OpenRouterError(f"[OPENROUTER] Request failed: {reason}", status_code=resp.status_code)
                        else:
                            raise OpenRouterError(f"[OPENROUTER] Request failed: status {resp.status_code}", status_code=resp.status_code)

                    try:
                        data = resp.json()
                    except Exception as je:
                        raise OpenRouterError(f"[OPENROUTER] Invalid response format (JSON decode error): {je}", status_code=502)

                    choices = data.get("choices", [])
                    if not choices:
                        raise OpenRouterError("No choices returned from OpenRouter response.", status_code=502)

                    choice = choices[0]
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

                except requests.Timeout:
                    if attempt < max_attempts:
                        print("[OPENROUTER] Request timed out. Retrying once with bounded backoff...")
                        time.sleep(1.0)
                        continue
                    print("[OPENROUTER] Request failed: timeout")
                    raise OpenRouterError("[OPENROUTER] Request failed: timeout", status_code=408)

                except requests.ConnectionError:
                    if attempt < max_attempts:
                        print("[OPENROUTER] Connection failure. Retrying once with bounded backoff...")
                        time.sleep(1.0)
                        continue
                    print("[OPENROUTER] Request failed: connection error")
                    raise OpenRouterError("[OPENROUTER] Request failed: connection error", status_code=503)

                except Exception as e:
                    if isinstance(e, OpenRouterError):
                        raise
                    print(f"[OPENROUTER] Request failed: {e}")
                    raise OpenRouterError(f"[OPENROUTER] Request failed: {e}") from e

            raise OpenRouterError("[OPENROUTER] Max retries exhausted", status_code=500)

        # Primary paid model first (unless recently credit-blocked), then ANY working free model
        last_error: Optional[OpenRouterError] = None
        for model_name in self._model_chain():
            try:
                if model_name != self.model:
                    print(f"[AI] Falling back to free model: {model_name}")
                result = _run(model_name)
                if model_name == self.model:
                    self._clear_primary_credit_limited()
                elif model_name in self.fallback_models:
                    self._fallback_cooldown.pop(model_name, None)
                return result
            except _CreditExhausted as cred_err:
                if model_name in self.fallback_models:
                    self._cooldown_fallback(model_name, 402)
                last_error = cred_err
                continue
            except OpenRouterError as oe:
                if oe.status_code == 401:
                    # Invalid API key: no fallback model can fix the credentials
                    raise
                if model_name in self.fallback_models:
                    self._cooldown_fallback(model_name, oe.status_code)
                last_error = oe
                continue
        if last_error is not None:
            low = str(last_error).lower()
            if "credit" in low or getattr(last_error, "status_code", None) == 402:
                self.trip_circuit("insufficient credits", status_code=402)
            raise last_error
        raise OpenRouterError("[AI] OpenRouter unavailable: insufficient credits", status_code=402)

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
        Fails fast if circuit is open.
        """
        if not self.is_available():
            if self._circuit_open:
                print(f"[AI] OpenRouter unavailable: {self._circuit_reason}")
                print("[AI] Skipping remote reasoning for local task")
                raise OpenRouterError(f"[AI] OpenRouter unavailable: {self._circuit_reason}", status_code=402 if "credit" in self._circuit_reason else 401)
            raise RuntimeError("OpenRouter API key is not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        messages = self._build_messages(prompt, system_prompt, history)

        def _run_stream(model_name: str) -> Iterator[str]:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout) as resp:
                if resp.status_code != 200:
                    raise OpenRouterError(f"[OPENROUTER] Request failed: status {resp.status_code}", status_code=resp.status_code)

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

        # Primary paid stream first (unless recently credit-blocked), then ANY working free model.
        # Last resort: the non-streaming generate(), which has its own full fallback chain.
        last_error: Optional[Exception] = None
        for model_name in self._model_chain():
            try:
                if model_name != self.model:
                    print(f"[AI] Falling back to free model: {model_name}")
                for token in _run_stream(model_name):
                    yield token
                if model_name == self.model:
                    self._clear_primary_credit_limited()
                elif model_name in self.fallback_models:
                    self._fallback_cooldown.pop(model_name, None)
                return
            except Exception as e:
                last_error = e
                if model_name in self.fallback_models:
                    self._cooldown_fallback(model_name, getattr(e, "status_code", None))
                continue
        yield self.generate(prompt, system_prompt, history, max_tokens, temperature)
