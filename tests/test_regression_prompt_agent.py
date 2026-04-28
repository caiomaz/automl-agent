"""Regression tests for PromptAgent — freeze parsing contract.

All LLM calls are fully mocked.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PARSED = {
    "user": {"intent": "build", "expertise": "medium"},
    "problem": {
        "area": "machine learning",
        "downstream_task": "tabular_regression",
        "application_domain": "biology",
        "description": "Predict crab age",
        "performance_metrics": [],
        "complexity_metrics": [],
    },
    "dataset": [{"name": "Crab Age Dataset", "modality": "tabular", "source": "user-upload"}],
    "model": [],
    "knowledge": [],
    "service": {},
}


def _mock_completion(content=None, usage=None):
    content = content or json.dumps(_VALID_PARSED)
    usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    msg = MagicMock()
    msg.content = content

    choice = MagicMock()
    choice.message = msg

    usage_obj = MagicMock()
    usage_obj.to_dict.return_value = usage

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage_obj
    return resp


def _mock_client(content=None):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_completion(content)
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _mock_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BACKBONE", "or-glm-5")
    monkeypatch.setenv("LLM_PROMPT_AGENT", "or-glm-5")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPromptAgentInit:
    """Freeze constructor defaults."""

    def test_default_llm(self, _mock_env):
        with patch("prompt_agent.get_client", return_value=_mock_client()):
            from prompt_agent import PromptAgent
            agent = PromptAgent()
            assert agent.llm == "prompt-llm"

    def test_custom_llm(self, _mock_env):
        with patch("prompt_agent.get_client", return_value=_mock_client()):
            from prompt_agent import PromptAgent
            agent = PromptAgent(llm="or-glm-5")
            assert agent.llm == "or-glm-5"

    def test_agent_type(self, _mock_env):
        with patch("prompt_agent.get_client", return_value=_mock_client()):
            from prompt_agent import PromptAgent
            agent = PromptAgent()
            assert agent.agent_type == "prompt"


@pytest.mark.unit
class TestPromptAgentParseOpenAI:
    """Freeze parse_openai contract: returns dict with expected schema keys."""

    def test_returns_dict_with_schema_keys(self, _mock_env):
        mock_client = _mock_client()
        with patch("prompt_agent.get_client", return_value=mock_client):
            from prompt_agent import PromptAgent
            agent = PromptAgent(llm="or-glm-5")
            result = agent.parse_openai(
                "Predict crab age from features",
                return_json=True,
                task="tabular_regression",
            )

        assert isinstance(result, dict)
        # Schema root keys
        for key in ("user", "problem", "dataset", "model", "knowledge", "service"):
            assert key in result, f"Missing schema key: {key}"

    def test_task_enforcement_in_prompt(self, _mock_env):
        """Verify task constraint is injected into the prompt."""
        captured_messages = []

        def _capture_create(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_completion()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _capture_create

        with patch("prompt_agent.get_client", return_value=mock_client):
            from prompt_agent import PromptAgent
            agent = PromptAgent(llm="or-glm-5")
            agent.parse_openai(
                "Classify images",
                return_json=True,
                task="image_classification",
            )

        assert len(captured_messages) > 0
        user_msg = captured_messages[0][-1]["content"]
        assert "image_classification" in user_msg

    def test_response_format_json_object(self, _mock_env):
        """Verify response_format is set to json_object."""
        captured_kwargs = []

        def _capture_create(**kwargs):
            captured_kwargs.append(kwargs)
            return _mock_completion()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _capture_create

        with patch("prompt_agent.get_client", return_value=mock_client):
            from prompt_agent import PromptAgent
            agent = PromptAgent(llm="or-glm-5")
            agent.parse_openai("Test prompt", return_json=True)

        assert len(captured_kwargs) > 0
        assert captured_kwargs[0].get("response_format") == {"type": "json_object"}


@pytest.mark.unit
class TestPromptAgentParseFallback:
    """Freeze parse() regex fallback extraction."""

    def test_extracts_json_from_code_block(self, _mock_env):
        json_in_block = '```json\n' + json.dumps(_VALID_PARSED) + '\n```'
        mock_client = _mock_client(json_in_block)
        with patch("prompt_agent.get_client", return_value=mock_client):
            from prompt_agent import PromptAgent
            agent = PromptAgent(llm="or-glm-5")
            result = agent.parse("Predict crab age", return_json=True)

        assert isinstance(result, dict)
        assert "problem" in result

    def test_returns_raw_string_when_not_json(self, _mock_env):
        mock_client = _mock_client("This is just text, not JSON.")
        with patch("prompt_agent.get_client", return_value=mock_client):
            from prompt_agent import PromptAgent
            agent = PromptAgent(llm="or-glm-5")
            result = agent.parse("Hello", return_json=False)

        assert isinstance(result, str)
        assert "just text" in result
