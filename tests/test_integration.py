"""Integration tests — validate real OpenRouter connectivity (requires API key).

Run with:  pytest tests/test_integration.py -m integration
Skip when no key is set (CI-safe).
"""

import os
import pytest

pytestmark = pytest.mark.integration

SKIP_REASON = "OPENROUTER_API_KEY not set"

# Respect user's configured models; fall back to a cheap default if not set
_BACKBONE = os.getenv("LLM_BACKBONE", "or-gpt-5-nano")
_PROMPT_LLM = os.getenv("LLM_PROMPT_AGENT", "or-gpt-5-nano")


@pytest.fixture
def openrouter_key():
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip(SKIP_REASON)
    return key


class TestOpenRouterConnectivity:
    def test_list_models_reachable(self, openrouter_key):
        """Sanity check: can we reach OpenRouter and list models?"""
        import os
        import requests
        USER_AGENT = os.getenv("USER_AGENT") or "AutoML-Agent/1.0 (https://github.com/caiomaz/automl-agent)"
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "User-Agent": USER_AGENT,
            },
            timeout=15,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data

    def test_chat_completion_round_trip(self, openrouter_key):
        """Minimal chat completion via OpenRouter using the configured backbone."""
        from utils import get_client
        from configs import AVAILABLE_LLMs

        client = get_client(_BACKBONE)
        model = AVAILABLE_LLMs[_BACKBONE]["model"]
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: PING"}],
            max_tokens=10,
            temperature=0,
        )
        assert res.choices[0].message.content.strip().upper().startswith("PING")


class TestGetClientIntegration:
    def test_get_client_openrouter_valid(self, openrouter_key):
        """get_client with the configured backbone alias should return a working client."""
        from utils import get_client
        from configs import AVAILABLE_LLMs
        client = get_client(_BACKBONE)
        model = AVAILABLE_LLMs[_BACKBONE]["model"]
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
            temperature=0,
        )
        assert res.choices[0].message.content is not None


class TestPromptAgentIntegration:
    def test_parse_openai_returns_json(self, openrouter_key):
        """PromptAgent.parse_openai should return valid JSON from the configured prompt LLM."""
        from prompt_agent import PromptAgent
        agent = PromptAgent(llm=_PROMPT_LLM)
        result = agent.parse_openai(
            "Build a model to classify banana quality as good or bad "
            "based on size, weight, sweetness.",
            return_json=True,
        )
        assert isinstance(result, dict)
