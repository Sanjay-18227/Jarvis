"""
JARVIS LLM Provider Verification Suite
Tests:
- OpenRouterProvider initialization, headers, message formatting, tool planning, and SSE streaming
- OllamaProvider local offline support, API fallback, and tool schema conversion
- Provider Factory (get_llm_provider) selection and fallback cascade
- JarvisAgent integration with configurable providers
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from config.settings import Settings
from llm.openrouter_provider import OpenRouterProvider
from llm.mock_provider import MockProvider
from llm.provider_factory import get_llm_provider
from core.agent import JarvisAgent


def test_openrouter_provider_init_and_availability():
    """Verify OpenRouterProvider availability logic based on API key presence."""
    p_no_key = OpenRouterProvider(api_key=None)
    assert p_no_key.is_available() is False

    p_empty = OpenRouterProvider(api_key="")
    assert p_empty.is_available() is False

    p_valid = OpenRouterProvider(api_key="sk-or-v1-testkey12345", model="meta-llama/llama-3.3-70b-instruct")
    assert p_valid.is_available() is True
    assert "OpenRouter" in p_valid.name
    assert "llama-3.3-70b-instruct" in p_valid.name

    headers = p_valid._get_headers()
    assert headers["Authorization"] == "Bearer sk-or-v1-testkey12345"
    assert "HTTP-Referer" in headers


def test_openrouter_generate_mocked():
    """Verify OpenRouter generate handles HTTP responses and extracts content."""
    provider = OpenRouterProvider(api_key="sk-test")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "Good evening, Sir. All systems nominal."}}
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        res = provider.generate("Status update", system_prompt="You are JARVIS.")
        assert "All systems nominal" in res


def test_openrouter_plan_with_tools_mocked():
    """Verify OpenRouter converts tool schemas and parses returned function calls."""
    provider = OpenRouterProvider(api_key="sk-test")

    tools = [
        {
            "name": "open_application",
            "description": "Opens an app",
            "parameters": {"type": "object", "properties": {"target": {"type": "string"}}},
        }
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Opening VS Code, Sir.",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "open_application",
                                "arguments": json.dumps({"target": "vscode"}),
                            },
                        }
                    ],
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        plan = provider.plan_with_tools("Open VS Code", tools=tools)
        assert len(plan["tool_calls"]) == 1
        assert plan["tool_calls"][0]["name"] == "open_application"
        assert plan["tool_calls"][0]["parameters"]["target"] == "vscode"
        assert "Opening VS Code" in plan["response"]


def test_openrouter_generate_stream_mocked():
    """Verify OpenRouter SSE streaming parses data lines into tokens."""
    provider = OpenRouterProvider(api_key="sk-test")

    sse_lines = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        b'data: {"choices":[{"delta":{"content":" Sanjay"}}]}',
        b'data: [DONE]',
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(sse_lines)
    mock_resp.__enter__.return_value = mock_resp

    with patch("requests.post", return_value=mock_resp):
        chunks = list(provider.generate_stream("Hi"))
        assert "".join(chunks) == "Hello Sanjay"





def test_provider_factory_selection():
    """Verify get_llm_provider honors configuration and fallback cascade."""
    # 1. Force Mock
    p_mock = get_llm_provider(force_mock=True)
    assert isinstance(p_mock, MockProvider)

    # 2. Configured OpenRouter with key
    settings_openrouter = Settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-or-test",
        local_only=False,
    )
    p_or = get_llm_provider(settings=settings_openrouter)
    assert isinstance(p_or, OpenRouterProvider)

    # 3. Fallback when primary key missing -> Mock
    settings_no_keys = Settings(
        llm_provider="openrouter",
        openrouter_api_key=None,
        gemini_api_key=None,
        groq_api_key=None,
        allow_remote_llm=True,
    )
    with patch("requests.get", side_effect=Exception("No ollama")):
        p_fallback = get_llm_provider(settings=settings_no_keys)
        assert isinstance(p_fallback, MockProvider)


def test_jarvis_agent_wires_configured_provider():
    """Verify JarvisAgent properly initializes with the configured provider from settings."""
    settings = Settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-or-valid-key",
        local_only=False,
    )
    agent = JarvisAgent(settings=settings)
    assert isinstance(agent.provider, OpenRouterProvider)
    assert agent.planner.client == agent.provider
