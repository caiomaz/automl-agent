"""Phase 2 tests: tracing spans, structured handoff events, HITL/Critic events,
reasoning trail, prompt agent instrumentation, and cost reconciliation.

Follows the TDD convention used in tests/test_ledger.py and
tests/test_run_lifecycle.py.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_ctx(workspace: Path, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=kwargs.get("run_id") or str(uuid.uuid4()),
        branch_id=kwargs.get("branch_id"),
        agent_id=kwargs.get("agent_id", "test-agent"),
        trace_id=kwargs.get("trace_id"),
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_WORKSPACE_DIR", str(tmp_path))
    # Refresh the cached WORKSPACE_DIR on the modules that already imported it
    import importlib
    import utils.workspace as ws_mod
    importlib.reload(ws_mod)
    import utils.ledger as ledger_mod
    importlib.reload(ledger_mod)
    return tmp_path


# ── 1. Tracing spans ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSpans:
    """utils.tracing.span(...) emits paired span_started / span_ended events."""

    def test_span_emits_start_and_end_events(self, tmp_workspace):
        from utils.tracing import span
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        with span(ctx, "data_retrieval", source="data-agent", workspace=tmp_workspace):
            pass
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        events_path = tmp_workspace / "exp" / "runs" / ctx.run_id / "events.jsonl"
        events = _read_jsonl(events_path)
        names = [e["event"] for e in events]
        assert "span_started" in names
        assert "span_ended" in names

        started = next(e for e in events if e["event"] == "span_started")
        ended = next(e for e in events if e["event"] == "span_ended")
        assert started["span_name"] == "data_retrieval"
        assert started["source"] == "data-agent"
        assert ended["span_name"] == "data_retrieval"
        assert isinstance(ended["elapsed_ms"], (int, float))
        assert ended["elapsed_ms"] >= 0
        assert ended["status"] == "ok"

    def test_span_records_failure_status_on_exception(self, tmp_workspace):
        from utils.tracing import span
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        with pytest.raises(ValueError):
            with span(ctx, "model_search", source="model-agent", workspace=tmp_workspace):
                raise ValueError("boom")
        finalize_run(ctx, status="failed", workspace=tmp_workspace)

        events_path = tmp_workspace / "exp" / "runs" / ctx.run_id / "events.jsonl"
        events = _read_jsonl(events_path)
        ended = next(e for e in events if e["event"] == "span_ended")
        assert ended["status"] == "error"
        assert ended.get("error_type") == "ValueError"


# ── 2. Structured handoff_emitted event ──────────────────────────────────────


@pytest.mark.unit
class TestHandoffEvent:
    """emit_handoff writes BOTH a handoffs.jsonl record AND a structured
    handoff_emitted event in events.jsonl, sharing the same handoff_id."""

    def test_emit_handoff_writes_both_artifacts(self, tmp_workspace):
        from utils.ledger import emit_handoff
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        handoff_id = emit_handoff(
            ctx,
            source_agent_id="manager",
            dest_agent_id="data-agent",
            payload_summary="bind dataset",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        run_dir = tmp_workspace / "exp" / "runs" / ctx.run_id
        handoffs = _read_jsonl(run_dir / "handoffs.jsonl")
        events = _read_jsonl(run_dir / "events.jsonl")

        assert len(handoffs) == 1
        assert handoffs[0]["handoff_id"] == handoff_id
        assert handoffs[0]["source_agent_id"] == "manager"
        assert handoffs[0]["dest_agent_id"] == "data-agent"

        handoff_events = [e for e in events if e["event"] == "handoff_emitted"]
        assert len(handoff_events) == 1
        assert handoff_events[0]["handoff_id"] == handoff_id
        assert handoff_events[0]["source"] == "manager"
        assert handoff_events[0]["destination"] == "data-agent"


# ── 3. HITL / Critic event helpers ───────────────────────────────────────────


@pytest.mark.unit
class TestHitlAndCriticEvents:
    """Helpers for hitl_requested, hitl_resolved, critic_warned, critic_blocked."""

    def test_hitl_requested_and_resolved(self, tmp_workspace):
        from utils.ledger import append_hitl_requested, append_hitl_resolved
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        append_hitl_requested(
            ctx,
            checkpoint="post-plan-decomposition",
            question="Approve plan?",
            workspace=tmp_workspace,
        )
        append_hitl_resolved(
            ctx,
            checkpoint="post-plan-decomposition",
            decision="approved",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        events = _read_jsonl(tmp_workspace / "exp" / "runs" / ctx.run_id / "events.jsonl")
        names = [e["event"] for e in events]
        assert "hitl_requested" in names
        assert "hitl_resolved" in names
        req = next(e for e in events if e["event"] == "hitl_requested")
        res = next(e for e in events if e["event"] == "hitl_resolved")
        assert req["checkpoint"] == "post-plan-decomposition"
        assert req["question"] == "Approve plan?"
        assert res["decision"] == "approved"

    def test_critic_warned_and_blocked(self, tmp_workspace):
        from utils.ledger import append_critic_warned, append_critic_blocked
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        append_critic_warned(
            ctx,
            target="plan_2",
            reason="metric mismatch",
            workspace=tmp_workspace,
        )
        append_critic_blocked(
            ctx,
            target="plan_3",
            reason="forbidden dependency",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        events = _read_jsonl(tmp_workspace / "exp" / "runs" / ctx.run_id / "events.jsonl")
        warned = [e for e in events if e["event"] == "critic_warned"]
        blocked = [e for e in events if e["event"] == "critic_blocked"]
        assert len(warned) == 1 and warned[0]["target"] == "plan_2"
        assert len(blocked) == 1 and blocked[0]["reason"] == "forbidden dependency"


# ── 4. Reasoning trail ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestReasoningTrail:
    """write_reasoning writes a structured reasoning record under
    analyses/reasoning/ and emits an event."""

    def test_write_reasoning_creates_file_and_event(self, tmp_workspace):
        from utils.ledger import write_reasoning
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        path = write_reasoning(
            ctx,
            agent="model-agent",
            label="why-lightgbm",
            content="LightGBM was chosen because of tabular size and time budget.",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        assert path.exists()
        assert path.parent.name == "reasoning"
        assert path.parent.parent.name == "analyses"
        body = path.read_text()
        assert "LightGBM" in body

        events = _read_jsonl(tmp_workspace / "exp" / "runs" / ctx.run_id / "events.jsonl")
        assert any(e["event"] == "reasoning_recorded" for e in events)


# ── 5. PromptAgent instrumentation ───────────────────────────────────────────


@pytest.mark.unit
class TestPromptAgentInstrumentation:
    """When run_ctx is passed, PromptAgent emits agent_started, agent_finished,
    llm_call_completed events and records cost."""

    def test_parse_with_run_ctx_emits_full_lifecycle(self, tmp_workspace, monkeypatch):
        from utils.run_context import prepare_new_run, finalize_run

        # Build a fake OpenAI response object
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        message = SimpleNamespace(content='{"downstream_task": "tabular_classification"}')
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice], usage=usage)

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = response

        # Patch get_client globally so PromptAgent picks up the fake
        monkeypatch.setattr("utils.get_client", lambda *a, **kw: fake_client)
        # Also patch in the prompt_agent module if it imported get_client directly
        import prompt_agent as pa_mod
        monkeypatch.setattr(pa_mod, "get_client", lambda *a, **kw: fake_client)

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )

        agent = pa_mod.PromptAgent(llm="prompt-llm")
        result = agent.parse_openai(
            "Classify customer churn.",
            return_json=True,
            task="tabular_classification",
            run_ctx=ctx,
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        assert result["downstream_task"] == "tabular_classification"

        events = _read_jsonl(tmp_workspace / "exp" / "runs" / ctx.run_id / "events.jsonl")
        names = [e["event"] for e in events]
        assert "agent_started" in names
        assert "agent_finished" in names
        assert "llm_call_completed" in names

        agent_started = next(e for e in events if e["event"] == "agent_started")
        assert agent_started["source"] == "prompt-agent"

        cost_records = _read_jsonl(
            tmp_workspace / "exp" / "runs" / ctx.run_id / "cost_records.jsonl"
        )
        assert len(cost_records) == 1
        assert cost_records[0]["alias"] == "prompt-llm"
        assert cost_records[0]["phase"] == "prompt_parse"
        assert cost_records[0]["prompt_tokens"] == 10


# ── 6. Cost reconciliation across multiple agents ────────────────────────────


@pytest.mark.unit
class TestCostReconciliation:
    """The cost_summary totals must equal the sum of cost_records for any
    multi-agent / multi-call scenario."""

    def test_summary_matches_sum_of_records(self, tmp_workspace):
        from utils.ledger import append_cost_record, write_cost_summary
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        # Simulate 4 LLM calls across 3 agents with 2 different models
        calls = [
            ("openrouter", "or-glm-5", "z-ai/glm-4.6", "prompt_parse", 100, 50, 150),
            ("openrouter", "or-glm-5", "z-ai/glm-4.6", "plan_decomposition", 200, 80, 280),
            ("openrouter", "or-glm-5", "z-ai/glm-4.6", "data_retrieval", 50, 25, 75),
            ("openrouter", "or-deepseek-v3.1", "deepseek/v3.1", "code_synthesis", 400, 200, 600),
        ]
        for prov, alias, slug, phase, p, c, t in calls:
            append_cost_record(
                ctx,
                provider=prov, alias=alias, model_slug=slug, phase=phase,
                prompt_tokens=p, completion_tokens=c, total_tokens=t,
                workspace=tmp_workspace,
            )
        write_cost_summary(ctx, workspace=tmp_workspace)
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        summary = json.loads(
            (tmp_workspace / "exp" / "runs" / ctx.run_id / "cost_summary.json").read_text()
        )
        assert summary["records_count"] == 4
        assert summary["total_prompt_tokens"] == 750
        assert summary["total_completion_tokens"] == 355
        assert summary["total_tokens"] == 1105
        assert set(summary["by_model"].keys()) == {"z-ai/glm-4.6", "deepseek/v3.1"}
        assert summary["by_model"]["z-ai/glm-4.6"]["record_count"] == 3
        assert summary["by_model"]["deepseek/v3.1"]["record_count"] == 1
