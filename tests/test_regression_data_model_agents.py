"""Regression tests for DataAgent and ModelAgent — freeze contracts.

All LLM calls and retrievers are fully mocked.
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_completion(content="Mocked agent response", usage=None):
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


def _mock_client():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_completion()
    return client


_USER_REQS = {
    "user": {"intent": "build"},
    "problem": {"downstream_task": "tabular_regression", "description": "Predict crab age"},
    "dataset": [{"name": "Crab Age", "modality": "tabular"}],
    "model": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _mock_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BACKBONE", "or-glm-5")
    monkeypatch.setenv("LLM_PROMPT_AGENT", "or-glm-5")


# ---------------------------------------------------------------------------
# DataAgent Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDataAgentInit:
    """Freeze DataAgent constructor contract."""

    def test_agent_type(self, _mock_env):
        with patch("data_agent.get_client", return_value=_mock_client()):
            from data_agent import DataAgent
            agent = DataAgent(user_requirements=_USER_REQS, llm="or-glm-5")
            assert agent.agent_type == "data"

    def test_accepts_rap_decomp(self, _mock_env):
        with patch("data_agent.get_client", return_value=_mock_client()):
            from data_agent import DataAgent
            agent = DataAgent(user_requirements=_USER_REQS, llm="or-glm-5", rap=False, decomp=False)
            assert agent.rap is False
            assert agent.decomp is False

    def test_money_initially_empty(self, _mock_env):
        with patch("data_agent.get_client", return_value=_mock_client()):
            from data_agent import DataAgent
            agent = DataAgent(user_requirements=_USER_REQS, llm="or-glm-5")
            assert agent.money == {}


@pytest.mark.unit
class TestDataAgentUnderstandPlan:
    """Freeze understand_plan contract: returns text summary."""

    def test_returns_text(self, _mock_env):
        mock_client = _mock_client()
        with patch("data_agent.get_client", return_value=mock_client):
            from data_agent import DataAgent
            agent = DataAgent(user_requirements=_USER_REQS, llm="or-glm-5")
            result = agent.understand_plan("Phase 1: Load data. Phase 2: Preprocess.")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_tracks_decomposition_cost(self, _mock_env):
        mock_client = _mock_client()
        with patch("data_agent.get_client", return_value=mock_client):
            from data_agent import DataAgent
            agent = DataAgent(user_requirements=_USER_REQS, llm="or-glm-5")
            agent.understand_plan("Phase 1: Load data.")

        assert "Data_Plan_Decomposition" in agent.money


@pytest.mark.unit
class TestDataAgentExecutePlan:
    """Freeze execute_plan contract: returns text, tracks cost by pid."""

    def test_returns_text_and_tracks_cost(self, _mock_env):
        mock_client = _mock_client()
        with patch("data_agent.get_client", return_value=mock_client):
            with patch("data_agent.retriever.retrieve_datasets", return_value=[]):
                from data_agent import DataAgent
                agent = DataAgent(user_requirements=_USER_REQS, llm="or-glm-5", decomp=False)
                result = agent.execute_plan("Load csv data", "/data/path", pid=1)

        assert isinstance(result, str)
        assert "Data_Plan_Execution_1" in agent.money


# ---------------------------------------------------------------------------
# ModelAgent Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestModelAgentInit:
    """Freeze ModelAgent constructor contract."""

    def test_agent_type(self, _mock_env):
        with patch("model_agent.get_client", return_value=_mock_client()):
            from model_agent import ModelAgent
            agent = ModelAgent(user_requirements=_USER_REQS, llm="or-glm-5")
            assert agent.agent_type == "model"

    def test_accepts_rap_decomp(self, _mock_env):
        with patch("model_agent.get_client", return_value=_mock_client()):
            from model_agent import ModelAgent
            agent = ModelAgent(user_requirements=_USER_REQS, llm="or-glm-5", rap=False, decomp=False)
            assert agent.rap is False
            assert agent.decomp is False

    def test_money_initially_empty(self, _mock_env):
        with patch("model_agent.get_client", return_value=_mock_client()):
            from model_agent import ModelAgent
            agent = ModelAgent(user_requirements=_USER_REQS, llm="or-glm-5")
            assert agent.money == {}


@pytest.mark.unit
class TestModelAgentUnderstandPlan:
    """Freeze understand_plan contract: accepts plan + data_result, returns text."""

    def test_returns_text(self, _mock_env):
        mock_client = _mock_client()
        with patch("model_agent.get_client", return_value=mock_client):
            from model_agent import ModelAgent
            agent = ModelAgent(user_requirements=_USER_REQS, llm="or-glm-5")
            result = agent.understand_plan("Build XGBoost model", "CSV loaded, 3000 rows")

        assert isinstance(result, str)

    def test_tracks_decomposition_cost(self, _mock_env):
        mock_client = _mock_client()
        with patch("model_agent.get_client", return_value=mock_client):
            from model_agent import ModelAgent
            agent = ModelAgent(user_requirements=_USER_REQS, llm="or-glm-5")
            agent.understand_plan("Build model", "Data ready")

        assert "Model_Plan_Decomposition" in agent.money


@pytest.mark.unit
class TestModelAgentExecutePlan:
    """Freeze execute_plan contract: returns text with model candidates, tracks cost by pid."""

    def test_returns_text_and_tracks_cost(self, _mock_env):
        mock_client = _mock_client()
        with patch("model_agent.get_client", return_value=mock_client):
            with patch("model_agent.retriever.retrieve_models", return_value=[]):
                from model_agent import ModelAgent
                agent = ModelAgent(user_requirements=_USER_REQS, llm="or-glm-5", decomp=False)
                result = agent.execute_plan(
                    k=3,
                    project_plan="Build regression model",
                    data_result="CSV loaded",
                    pid=1,
                )

        assert isinstance(result, str)
        assert "Model_Plan_Execution_1" in agent.money
