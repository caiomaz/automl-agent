"""Phase 8 — Manager↔TokenEconomy system tests.

Verify that ``token_economy`` propagates from constraints into the
manager's planning loop and the operation agent's payload, and that
``tokens_saved`` events are emitted when reductions happen.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _fake_run_ctx():
    return SimpleNamespace(
        run_id="r1",
        branch_id="b1",
        agent_id="manager",
        attempt_id=0,
    )


def _mock_completion(content="Plan body"):
    msg = MagicMock()
    msg.content = content
    msg.to_dict.return_value = {"role": "assistant", "content": content}
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.to_dict.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.reasoning_tokens = None
    usage.completion_tokens_details = None
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


@pytest.fixture(autouse=True)
def _llm_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BACKBONE", "or-glm-5")
    monkeypatch.setenv("LLM_PROMPT_AGENT", "or-glm-5")


@pytest.fixture(autouse=True)
def _reset_active_run():
    from utils.run_context import clear_active_run
    clear_active_run()
    yield
    clear_active_run()


@pytest.mark.unit
class TestManagerTokenEconomy:
    def test_token_economy_default_is_off(self):
        with patch("agent_manager.get_client", return_value=MagicMock()):
            with patch("agent_manager.parser"):
                from agent_manager import AgentManager
                mgr = AgentManager(task="tabular_regression", n_plans=1)
                assert mgr.token_economy == "off"

    def test_token_economy_read_from_constraints(self):
        with patch("agent_manager.get_client", return_value=MagicMock()):
            with patch("agent_manager.parser"):
                from agent_manager import AgentManager
                mgr = AgentManager(
                    task="tabular_regression",
                    n_plans=1,
                    constraints={"token_economy": "aggressive"},
                )
                assert mgr.token_economy == "aggressive"

    def test_dynamic_n_plans_reduces_when_aggressive_and_high_confidence(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_completion("Plan A")

        recorded = []

        def _capture(ctx, *, source, saved_tokens, workspace=None, **extra):
            recorded.append({"source": source, "saved_tokens": saved_tokens, **extra})

        with patch("agent_manager.get_client", return_value=client):
            with patch("agent_manager.parser"):
                with patch("utils.token_economy.record_tokens_saved", side_effect=_capture):
                    from agent_manager import AgentManager
                    mgr = AgentManager(
                        task="tabular_classification",
                        n_plans=5,
                        rap=False,
                        decomp=False,
                        verification=False,
                        full_pipeline=False,
                        constraints={"token_economy": "aggressive"},
                    )
                    mgr.run_ctx = _fake_run_ctx()
                    mgr.user_requirements = {
                        "problem": {"downstream_task": "tabular_classification"},
                        "confidence": 0.95,
                    }
                    mgr.req_summary = "Classify rows."
                    mgr.make_plans()

        assert len(mgr.plans) == 3  # 5 - 2 = 3
        dyn = [r for r in recorded if r["source"] == "dynamic_n_plans"]
        assert len(dyn) == 1
        assert dyn[0]["original_n_plans"] == 5
        assert dyn[0]["effective_n_plans"] == 3

    def test_token_economy_off_keeps_full_n_plans(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_completion("Plan A")

        recorded = []

        def _capture(ctx, *, source, saved_tokens, workspace=None, **extra):
            recorded.append({"source": source})

        with patch("agent_manager.get_client", return_value=client):
            with patch("agent_manager.parser"):
                with patch("utils.token_economy.record_tokens_saved", side_effect=_capture):
                    from agent_manager import AgentManager
                    mgr = AgentManager(
                        task="tabular_classification",
                        n_plans=4,
                        rap=False,
                        decomp=False,
                        verification=False,
                        full_pipeline=False,
                        constraints={"token_economy": "off"},
                    )
                    mgr.run_ctx = _fake_run_ctx()
                    mgr.user_requirements = {
                        "problem": {"downstream_task": "tabular_classification"},
                        "confidence": 0.99,
                    }
                    mgr.req_summary = "Classify rows."
                    mgr.make_plans()

        assert len(mgr.plans) == 4
        assert not any(r["source"] == "dynamic_n_plans" for r in recorded)
