"""Regression tests for OperationAgent — freeze construction and execution contracts.

All LLM calls and subprocess executions are fully mocked.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_completion(content="```python\nprint('hello')\n```", usage=None):
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


def _mock_client(content="```python\nprint('hello')\n```"):
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


@pytest.fixture()
def op_agent(_mock_env, tmp_path):
    """Build an OperationAgent pointing at a temporary workspace."""
    with patch("operation_agent.ensure_workspace"):
        with patch("operation_agent.get_client", return_value=_mock_client()):
            from operation_agent import OperationAgent
            agent = OperationAgent(
                user_requirements={"problem": {"downstream_task": "tabular_regression"}},
                llm="or-glm-5",
                code_path="/test_run",
                device=0,
                system_info="Python 3.11, scikit-learn 1.4",
            )
            # Point root_path to tmp_path for safe file writes
            agent.root_path = str(tmp_path)
            return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOperationAgentInit:
    """Freeze construction contract."""

    def test_agent_type(self, op_agent):
        assert op_agent.agent_type == "operation"

    def test_code_path_stored(self, op_agent):
        assert op_agent.code_path == "/test_run"

    def test_system_info_stored(self, op_agent):
        assert "scikit-learn" in op_agent.system_info

    def test_money_initially_empty(self, op_agent):
        assert op_agent.money == {}

    def test_root_path_set(self, op_agent):
        # In production this is EXP_DIR; in test it's tmp_path
        assert op_agent.root_path is not None


@pytest.mark.unit
class TestOperationAgentWorkspaceInjection:
    """Verify that the exec_prompt template includes workspace paths and path-blocking rules."""

    def test_exec_prompt_contains_workspace_dirs(self, _mock_env, tmp_path):
        captured_messages = []

        def _capture_create(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_completion("```python\nprint('ok')\n```")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _capture_create

        with patch("operation_agent.ensure_workspace"):
            with patch("operation_agent.get_client", return_value=mock_client):
                with patch("operation_agent.execute_script", return_value=(0, "Success")):
                    from operation_agent import OperationAgent
                    agent = OperationAgent(
                        user_requirements={"problem": {"downstream_task": "tabular_regression"}},
                        llm="or-glm-5",
                        code_path="/test_ws",
                        system_info="test",
                    )
                    agent.root_path = str(tmp_path)
                    agent.implement_solution("Build a model", n_attempts=1)

        # The exec_prompt (in the user message) should contain path rules
        assert len(captured_messages) > 0
        user_msg = captured_messages[0][-1]["content"]
        assert "NEVER use absolute paths" in user_msg or "CRITICAL" in user_msg
        assert "Datasets" in user_msg
        assert "Trained models" in user_msg
        assert "Experiment outputs" in user_msg


@pytest.mark.unit
class TestOperationAgentRetryLoop:
    """Freeze retry behavior: on failure, retry with error feedback."""

    def test_retry_on_failure_then_success(self, _mock_env, tmp_path):
        call_count = [0]

        def _mock_execute(script, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (1, "ImportError: no module named foo")
            return (0, "Model trained successfully")

        mock_client = _mock_client("```python\nprint('ok')\n```")

        with patch("operation_agent.ensure_workspace"):
            with patch("operation_agent.get_client", return_value=mock_client):
                with patch("operation_agent.execute_script", side_effect=_mock_execute):
                    from operation_agent import OperationAgent
                    agent = OperationAgent(
                        user_requirements={"problem": {"downstream_task": "tabular_regression"}},
                        llm="or-glm-5",
                        code_path="/test_retry",
                        system_info="test",
                    )
                    agent.root_path = str(tmp_path)
                    result = agent.implement_solution("Build a model", n_attempts=5)

        assert result["rcode"] == 0
        assert call_count[0] == 2

    def test_max_attempts_exhausted(self, _mock_env, tmp_path):
        def _always_fail(script, **kwargs):
            return (1, "Error: always fails")

        mock_client = _mock_client("```python\nraise Exception('fail')\n```")

        with patch("operation_agent.ensure_workspace"):
            with patch("operation_agent.get_client", return_value=mock_client):
                with patch("operation_agent.execute_script", side_effect=_always_fail):
                    from operation_agent import OperationAgent
                    agent = OperationAgent(
                        user_requirements={"problem": {"downstream_task": "tabular_regression"}},
                        llm="or-glm-5",
                        code_path="/test_exhaust",
                        system_info="test",
                    )
                    agent.root_path = str(tmp_path)
                    result = agent.implement_solution("Build a model", n_attempts=3)

        assert result["rcode"] != 0
        assert len(result["error_logs"]) > 0


@pytest.mark.unit
class TestOperationAgentCodeExtraction:
    """Freeze code extraction logic from LLM responses."""

    def test_code_extracted_from_python_block(self, _mock_env, tmp_path):
        code_response = '```python\nimport pandas as pd\ndf = pd.read_csv("data.csv")\nprint(df.head())\n```'
        mock_client = _mock_client(code_response)

        with patch("operation_agent.ensure_workspace"):
            with patch("operation_agent.get_client", return_value=mock_client):
                with patch("operation_agent.execute_script", return_value=(0, "Success")):
                    from operation_agent import OperationAgent
                    agent = OperationAgent(
                        user_requirements={"problem": {"downstream_task": "tabular_regression"}},
                        llm="or-glm-5",
                        code_path="/test_extract",
                        system_info="test",
                    )
                    agent.root_path = str(tmp_path)
                    result = agent.implement_solution("Build a model", n_attempts=1)

        assert "pandas" in result["code"]
        assert "read_csv" in result["code"]
