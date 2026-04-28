"""Phase 7 — Manager↔Critic system tests.

Verify that the AgentManager actually invokes the Critic Agent at the
two integration points (after the prompt parse and after planning) and
that the policy from constraints is honored.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _fake_run_ctx():
    """A run context stub compatible with append_event serialization.

    All attributes the ledger reads (``run_id``, ``branch_id``,
    ``agent_id``, ``attempt_id``) are plain strings/ints, so JSON
    serialization works.
    """
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
class TestManagerCriticIntegration:
    def test_critic_policy_default_is_warn(self):
        with patch("agent_manager.get_client", return_value=MagicMock()):
            with patch("agent_manager.parser"):
                from agent_manager import AgentManager
                mgr = AgentManager(task="tabular_regression", n_plans=1)
                assert mgr.critic_policy == "warn"

    def test_critic_policy_read_from_constraints(self):
        with patch("agent_manager.get_client", return_value=MagicMock()):
            with patch("agent_manager.parser"):
                from agent_manager import AgentManager
                mgr = AgentManager(
                    task="tabular_regression",
                    n_plans=1,
                    constraints={"critic_policy": "block"},
                )
                assert mgr.critic_policy == "block"

    def test_make_plans_invokes_critic_review_plans(self):
        """Manager wiring: after planning, run_review is called with target='plans'
        and the active critic_policy from constraints. We spy on the
        critic_agent module to avoid filesystem coupling."""
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_completion(
            "Use TensorFlow Keras to build a deep model."
        )

        captured = {}

        def fake_run_review(ctx, *, target, findings, policy, **kw):
            captured["target"] = target
            captured["policy"] = policy
            captured["findings"] = findings
            return {"action": "warn", "review_id": "fake"}

        with patch("agent_manager.get_client", return_value=client):
            with patch("agent_manager.parser"):
                # Patch via critic_agent so the lazy import inside make_plans
                # picks up the spy.
                with patch("critic_agent.run_review", side_effect=fake_run_review):
                    from agent_manager import AgentManager
                    mgr = AgentManager(
                        task="tabular_classification",
                        n_plans=1,
                        rap=False,
                        decomp=False,
                        verification=False,
                        full_pipeline=False,
                        constraints={
                            "framework": "lightgbm",
                            "critic_policy": "warn",
                        },
                    )
                    mgr.run_ctx = _fake_run_ctx()
                    mgr.user_requirements = {
                        "problem": {"downstream_task": "tabular_classification"}
                    }
                    mgr.req_summary = "Classify rows."
                    mgr.make_plans()

        assert captured["target"] == "plans"
        assert captured["policy"] == "warn"
        # The TensorFlow plan against a lightgbm constraint must produce a
        # framework drift finding.
        codes = [f.code for f in captured["findings"]]
        assert "constraint_violation" in codes

    def test_critic_off_skips_review(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_completion(
            "Use TensorFlow Keras to build a deep model."
        )

        called = {"count": 0}

        def fake_run_review(*a, **kw):
            called["count"] += 1
            return {"action": "pass"}

        with patch("agent_manager.get_client", return_value=client):
            with patch("agent_manager.parser"):
                with patch("critic_agent.run_review", side_effect=fake_run_review):
                    from agent_manager import AgentManager
                    mgr = AgentManager(
                        task="tabular_classification",
                        n_plans=1,
                        rap=False,
                        decomp=False,
                        verification=False,
                        full_pipeline=False,
                        constraints={
                            "framework": "lightgbm",
                            "critic_policy": "off",
                        },
                    )
                    mgr.run_ctx = _fake_run_ctx()
                    mgr.user_requirements = {
                        "problem": {"downstream_task": "tabular_classification"}
                    }
                    mgr.req_summary = "Classify rows."
                    mgr.make_plans()

        assert called["count"] == 0
