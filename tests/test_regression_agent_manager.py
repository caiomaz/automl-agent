"""Regression tests for AgentManager — freeze current construction and orchestration contracts.

These tests ensure the public interface and behavioral contracts of AgentManager
remain stable across future refactors (Phases 1–10 in TASKS.md).  All LLM calls
and external services are fully mocked.
"""

import re
import time
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# Helper: build a mock OpenAI completion response
# ---------------------------------------------------------------------------

def _mock_completion(content="mocked", usage=None):
    usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    msg = MagicMock()
    msg.content = content
    msg.to_dict.return_value = {"role": "assistant", "content": content}

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _mock_llm_env(monkeypatch):
    """Set up environment so AgentManager can be imported without real keys."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BACKBONE", "or-glm-5")
    monkeypatch.setenv("LLM_PROMPT_AGENT", "or-glm-5")


@pytest.fixture()
def manager(_mock_llm_env):
    """Create a minimal AgentManager with mocked dependencies."""
    with patch("agent_manager.get_client", return_value=_mock_client()):
        with patch("agent_manager.parser") as mock_parser:
            mock_parser.parse.return_value = {"problem": {"downstream_task": "tabular_regression"}}
            mock_parser.parse_openai.return_value = {"problem": {"downstream_task": "tabular_regression"}}
            from agent_manager import AgentManager
            mgr = AgentManager(
                task="tabular_regression",
                n_plans=2,
                n_candidates=3,
                n_revise=1,
                rap=True,
                decomp=True,
                verification=True,
                full_pipeline=True,
                interactive=False,
            )
            return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAgentManagerDefaults:
    """Freeze default constructor values."""

    def test_default_n_plans(self, manager):
        assert manager.n_plans == 2

    def test_default_n_candidates(self, manager):
        assert manager.n_candidates == 3

    def test_default_n_revise(self, manager):
        assert manager.n_revise == 1

    def test_rap_enabled(self, manager):
        assert manager.rap is True

    def test_decomp_enabled(self, manager):
        assert manager.decomp is True

    def test_verification_enabled(self, manager):
        assert manager.verification is True

    def test_full_pipeline_enabled(self, manager):
        assert manager.full_pipeline is True

    def test_interactive_disabled(self, manager):
        assert manager.interactive is False

    def test_initial_state_is_init(self, manager):
        assert manager.state == "INIT"

    def test_empty_plans_list(self, manager):
        assert manager.plans == []

    def test_empty_money_dict(self, manager):
        assert manager.money == {}

    def test_empty_timer_dict(self, manager):
        assert manager.timer == {}

    def test_constraints_default_empty(self, manager):
        assert manager.constraints == {}


@pytest.mark.unit
class TestAgentManagerCodePath:
    """Freeze code_path format: /{uid}_{llm}_p{n}_{flags}"""

    def test_code_path_contains_task(self, manager):
        assert "tabular_regression" in manager.code_path

    def test_code_path_contains_plan_count(self, manager):
        assert "_p2_" in manager.code_path

    def test_code_path_rap_flag(self, manager):
        assert "rap" in manager.code_path

    def test_code_path_decomp_flag(self, manager):
        assert "decomp" in manager.code_path

    def test_code_path_ver_flag(self, manager):
        assert "ver" in manager.code_path

    def test_code_path_full_flag(self, manager):
        assert "full" in manager.code_path

    def test_code_path_pattern(self, manager):
        # Format: /{task}_{timestamp}_{llm}_p{n}_{flags}
        pattern = r"^/tabular_regression_\d+_.*_p2_"
        assert re.match(pattern, manager.code_path), f"code_path '{manager.code_path}' does not match pattern"

    def test_code_path_no_flags_when_disabled(self, _mock_llm_env):
        with patch("agent_manager.get_client", return_value=_mock_client()):
            with patch("agent_manager.parser"):
                from agent_manager import AgentManager
                mgr = AgentManager(
                    task="tabular_regression",
                    n_plans=1,
                    rap=False,
                    decomp=False,
                    verification=False,
                    full_pipeline=False,
                )
                assert "rap" not in mgr.code_path
                assert "decomp" not in mgr.code_path
                assert "ver" not in mgr.code_path
                assert "full" not in mgr.code_path


@pytest.mark.unit
class TestAgentManagerStateMachine:
    """Freeze the expected state keys."""

    def test_possible_states_keys(self):
        from agent_manager import possible_states
        expected = {"INIT", "PLAN", "ACT", "PRE_EXEC", "EXEC", "POST_EXEC", "REV", "RES"}
        assert set(possible_states.keys()) == expected


@pytest.mark.unit
class TestAgentManagerConstraints:
    """Freeze constraints injection into planning prompt."""

    def test_constraints_stored(self, _mock_llm_env):
        constraints = {
            "model": "XGBoost",
            "perf_metric": "F1",
            "perf_value": "0.95",
            "max_train_time": "30 minutes",
        }
        with patch("agent_manager.get_client", return_value=_mock_client()):
            with patch("agent_manager.parser"):
                from agent_manager import AgentManager
                mgr = AgentManager(
                    task="tabular_classification",
                    constraints=constraints,
                )
                assert mgr.constraints == constraints
                assert mgr.constraints["model"] == "XGBoost"

    def test_constraints_injection_in_make_plans(self, _mock_llm_env):
        """Verify that make_plans includes 'Hard constraints' when constraints are set."""
        constraints = {
            "model": "XGBoost",
            "perf_metric": "F1",
            "perf_value": "0.95",
        }
        captured_messages = []

        def _capture_create(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_completion("Plan: use XGBoost")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _capture_create

        with patch("agent_manager.get_client", return_value=mock_client):
            with patch("agent_manager.parser"):
                from agent_manager import AgentManager
                mgr = AgentManager(
                    task="tabular_classification",
                    n_plans=1,
                    constraints=constraints,
                    rap=False,
                )
                mgr.user_requirements = {"problem": {"downstream_task": "tabular_classification"}}
                mgr.req_summary = "Classify data"
                mgr.make_plans()

        # Find the planning call (the one with plan_conditions content)
        found_constraint_block = False
        for msgs in captured_messages:
            for msg in msgs:
                content = msg.get("content", "")
                if "Hard constraints" in content and "XGBoost" in content:
                    found_constraint_block = True
                    assert "F1" in content
                    assert "0.95" in content
                    break
        assert found_constraint_block, "Hard constraints block not found in planning prompt"


@pytest.mark.unit
class TestAgentManagerMoneyTracking:
    """Freeze money tracking key naming pattern."""

    def test_make_plans_money_keys(self, _mock_llm_env):
        mock_client = _mock_client()
        with patch("agent_manager.get_client", return_value=mock_client):
            with patch("agent_manager.parser"):
                from agent_manager import AgentManager
                mgr = AgentManager(
                    task="tabular_regression",
                    n_plans=2,
                    rap=False,
                )
                mgr.user_requirements = {"problem": {"downstream_task": "tabular_regression"}}
                mgr.req_summary = "Predict values"
                mgr.make_plans()

        assert "manager_plan_1" in mgr.money
        assert "manager_plan_2" in mgr.money
        # Verify the value is a dict from usage.to_dict()
        assert isinstance(mgr.money["manager_plan_1"], dict)


@pytest.mark.unit
class TestAgentManagerExecutePlan:
    """Freeze execute_plan contract: calls DataAgent and ModelAgent, returns dict."""

    def test_execute_plan_returns_data_and_model(self, _mock_llm_env):
        mock_client = _mock_client()

        with patch("agent_manager.get_client", return_value=mock_client):
            with patch("agent_manager.parser"):
                with patch("agent_manager.DataAgent") as MockData:
                    with patch("agent_manager.ModelAgent") as MockModel:
                        mock_data_inst = MagicMock()
                        mock_data_inst.execute_plan.return_value = "data result"
                        mock_data_inst.money = {"Data_Plan_Execution_1": {}}
                        MockData.return_value = mock_data_inst

                        mock_model_inst = MagicMock()
                        mock_model_inst.execute_plan.return_value = "model result"
                        mock_model_inst.money = {"Model_Plan_Execution_1": {}}
                        MockModel.return_value = mock_model_inst

                        from agent_manager import AgentManager
                        mgr = AgentManager(
                            task="tabular_regression",
                            n_plans=1,
                            rap=False,
                        )
                        mgr.user_requirements = {"problem": {"downstream_task": "tabular_regression"}}

                        # Mock current_process to avoid multiprocessing dependency
                        with patch("agent_manager.current_process") as mock_proc:
                            mock_proc.return_value._identity = (1,)
                            result = mgr.execute_plan("Test plan")

        assert "data" in result
        assert "model" in result
        assert result["data"] == "data result"
        assert result["model"] == "model result"

    def test_execute_plan_accepts_explicit_job_payload_in_serial_mode(self, _mock_llm_env):
        mock_client = _mock_client()

        with patch("agent_manager.get_client", return_value=mock_client):
            with patch("agent_manager.parser"):
                with patch("agent_manager.DataAgent") as MockData:
                    with patch("agent_manager.ModelAgent") as MockModel:
                        with patch("agent_manager.emit_handoff"):
                            with patch("agent_manager.append_event"):
                                mock_data_inst = MagicMock()
                                mock_data_inst.execute_plan.return_value = "data result"
                                mock_data_inst.money = {"Data_Plan_Execution_7": {}}
                                MockData.return_value = mock_data_inst

                                mock_model_inst = MagicMock()
                                mock_model_inst.execute_plan.return_value = "model result"
                                mock_model_inst.money = {"Model_Plan_Execution_7": {}}
                                MockModel.return_value = mock_model_inst

                                from agent_manager import AgentManager
                                mgr = AgentManager(
                                    task="tabular_regression",
                                    n_plans=1,
                                    rap=False,
                                )
                                mgr.user_requirements = {
                                    "problem": {"downstream_task": "tabular_regression"}
                                }
                                mgr.run_ctx = SimpleNamespace(
                                    run_id="r1",
                                    branch_id="b0",
                                    agent_id="manager",
                                    attempt_id=0,
                                )

                                with patch("agent_manager.current_process") as mock_proc:
                                    mock_proc.return_value._identity = ()
                                    result = mgr.execute_plan(
                                        {
                                            "plan": "Test plan",
                                            "pid": 7,
                                            "branch_id": "r1__b7",
                                        }
                                    )

        assert result == {"data": "data result", "model": "model result"}
        assert MockData.call_args.kwargs["branch_id"] == "r1__b7"
        mock_data_inst.execute_plan.assert_called_once_with("Test plan", mgr.data_path, 7)
        mock_model_inst.execute_plan.assert_called_once_with(
            k=mgr.n_candidates,
            project_plan="Test plan",
            data_result="data result",
            pid=7,
        )
