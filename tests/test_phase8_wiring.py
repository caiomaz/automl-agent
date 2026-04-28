"""Phase 8 wiring tests — stage routing in agent constructors and
PromptAgent parse-cache behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _llm_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BACKBONE", "or-glm-5")
    monkeypatch.setenv("LLM_PROMPT_AGENT", "or-glm-5")
    yield


def _fake_run_ctx():
    return SimpleNamespace(
        run_id="r1",
        branch_id="b1",
        agent_id="prompt",
        attempt_id=0,
    )


@pytest.mark.unit
class TestStageRoutingInAgents:
    def test_prompt_agent_uses_default_when_no_stage_env(self, monkeypatch):
        monkeypatch.delenv("LLM_STAGE_PROMPT_PARSE", raising=False)
        with patch("prompt_agent.get_client", return_value=MagicMock()):
            from prompt_agent import PromptAgent
            agent = PromptAgent(llm="or-glm-5")
            assert agent.llm == "or-glm-5"

    def test_prompt_agent_overrides_alias_via_stage_env(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_PROMPT_PARSE", "or-gpt-5-mini")
        with patch("prompt_agent.get_client", return_value=MagicMock()):
            from prompt_agent import PromptAgent
            agent = PromptAgent(llm="or-glm-5")
            assert agent.llm == "or-gpt-5-mini"

    def test_operation_agent_uses_default_when_no_stage_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_STAGE_CODE_GENERATION", raising=False)
        from operation_agent import OperationAgent
        agent = OperationAgent(
            user_requirements={"problem": {"downstream_task": "tabular_classification"}},
            llm="or-glm-5",
            code_path=str(tmp_path / "code.py"),
        )
        assert agent.llm == "or-glm-5"

    def test_operation_agent_overrides_alias_via_stage_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_STAGE_CODE_GENERATION", "or-gpt-5-mini")
        from operation_agent import OperationAgent
        agent = OperationAgent(
            user_requirements={"problem": {"downstream_task": "tabular_classification"}},
            llm="or-glm-5",
            code_path=str(tmp_path / "code.py"),
        )
        assert agent.llm == "or-gpt-5-mini"


def _mock_parse_response(json_payload):
    msg = MagicMock()
    msg.content = json_payload
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.total_tokens = 42
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


@pytest.mark.unit
class TestPromptAgentParseCache:
    def test_parse_does_not_cache_when_token_economy_off(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_STAGE_PROMPT_PARSE", raising=False)
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_parse_response(
            '{"problem": {"downstream_task": "tabular_classification"}}'
        )
        with patch("prompt_agent.get_client", return_value=client):
            from prompt_agent import PromptAgent
            agent = PromptAgent(llm="or-glm-5")
            agent.client = client

            for _ in range(2):
                agent.parse(
                    "Predict churn from tabular data.",
                    return_json=True,
                    task="tabular_classification",
                    run_ctx=_fake_run_ctx(),
                    workspace=tmp_path,
                    token_economy="off",
                )
            assert client.chat.completions.create.call_count == 2

    def test_parse_cache_hit_skips_llm_call(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_STAGE_PROMPT_PARSE", raising=False)
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_parse_response(
            '{"problem": {"downstream_task": "tabular_classification"}}'
        )
        from utils.run_cache import RunCache as _RealRunCache

        def _factory(ctx, *, workspace=None):
            return _RealRunCache(ctx, workspace=tmp_path)

        with patch("prompt_agent.get_client", return_value=client):
            with patch("utils.run_cache.RunCache", side_effect=_factory):
                from prompt_agent import PromptAgent
                agent = PromptAgent(llm="or-glm-5")
                agent.client = client

                ctx = _fake_run_ctx()
                from utils.workspace import run_exp_dir
                run_exp_dir(ctx.run_id, tmp_path).mkdir(parents=True, exist_ok=True)

                first = agent.parse(
                    "Predict churn from tabular data.",
                    return_json=True,
                    task="tabular_classification",
                    run_ctx=ctx,
                    workspace=tmp_path,
                    token_economy="moderate",
                )
                second = agent.parse(
                    "Predict churn from tabular data.",
                    return_json=True,
                    task="tabular_classification",
                    run_ctx=ctx,
                    workspace=tmp_path,
                    token_economy="moderate",
                )

                assert first == second
                assert client.chat.completions.create.call_count == 1

    def test_parse_cache_miss_when_instruction_differs(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_STAGE_PROMPT_PARSE", raising=False)
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_parse_response(
            '{"problem": {"downstream_task": "tabular_classification"}}'
        )
        from utils.run_cache import RunCache as _RealRunCache

        def _factory(ctx, *, workspace=None):
            return _RealRunCache(ctx, workspace=tmp_path)

        with patch("prompt_agent.get_client", return_value=client):
            with patch("utils.run_cache.RunCache", side_effect=_factory):
                from prompt_agent import PromptAgent
                agent = PromptAgent(llm="or-glm-5")
                agent.client = client

                ctx = _fake_run_ctx()
                from utils.workspace import run_exp_dir
                run_exp_dir(ctx.run_id, tmp_path).mkdir(parents=True, exist_ok=True)

                agent.parse(
                    "Predict churn.",
                    return_json=True, task="tabular_classification",
                    run_ctx=ctx, workspace=tmp_path,
                    token_economy="aggressive",
                )
                agent.parse(
                    "Predict revenue.",
                    return_json=True, task="tabular_regression",
                    run_ctx=ctx, workspace=tmp_path,
                    token_economy="aggressive",
                )
                assert client.chat.completions.create.call_count == 2
