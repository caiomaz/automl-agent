"""Unit tests for utils/ledger.py (Phase 2, ADR-007)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from utils.ledger import (
    _append_jsonl,
    _payload_hash,
    _payload_size,
    append_cost_record,
    append_event,
    append_handoff,
    ensure_analyses_dir,
    record_llm_usage,
    write_analysis,
    write_cost_summary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_ctx(run_id: str | None = None, **kwargs) -> SimpleNamespace:
    """Create a minimal RunContext-like object for testing."""
    return SimpleNamespace(
        run_id=run_id or str(uuid.uuid4()),
        branch_id=kwargs.get("branch_id"),
        agent_id=kwargs.get("agent_id"),
        trace_id=kwargs.get("trace_id"),
    )


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ── _payload_size and _payload_hash ──────────────────────────────────────────


@pytest.mark.unit
def test_payload_size_none_returns_none():
    assert _payload_size(None) is None


@pytest.mark.unit
def test_payload_size_ascii():
    assert _payload_size("hello") == 5


@pytest.mark.unit
def test_payload_hash_none_returns_none():
    assert _payload_hash(None) is None


@pytest.mark.unit
def test_payload_hash_is_16_char_hex():
    h = _payload_hash("hello world")
    assert isinstance(h, str)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


@pytest.mark.unit
def test_payload_hash_deterministic():
    assert _payload_hash("abc") == _payload_hash("abc")


# ── _append_jsonl ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_append_jsonl_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "test.jsonl"
    _append_jsonl(p, {"key": "value"})
    assert p.exists()


@pytest.mark.unit
def test_append_jsonl_writes_valid_json(tmp_path):
    p = tmp_path / "test.jsonl"
    _append_jsonl(p, {"x": 1})
    records = _read_jsonl(p)
    assert records == [{"x": 1}]


@pytest.mark.unit
def test_append_jsonl_multiple_records(tmp_path):
    p = tmp_path / "test.jsonl"
    _append_jsonl(p, {"n": 1})
    _append_jsonl(p, {"n": 2})
    records = _read_jsonl(p)
    assert len(records) == 2
    assert records[0]["n"] == 1
    assert records[1]["n"] == 2


# ── append_event ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_append_event_creates_events_jsonl(tmp_path):
    ctx = _make_ctx()
    append_event(ctx, "run_started", workspace=tmp_path)
    events_file = tmp_path / "exp" / "runs" / ctx.run_id / "events.jsonl"
    assert events_file.exists()


@pytest.mark.unit
def test_append_event_has_all_required_fields(tmp_path):
    ctx = _make_ctx(branch_id="b1", agent_id="ag1", trace_id="t1")
    append_event(
        ctx,
        "agent_started",
        source="operation",
        destination="manager",
        payload_summary="test summary",
        workspace=tmp_path,
    )
    evts = _read_jsonl(tmp_path / "exp" / "runs" / ctx.run_id / "events.jsonl")
    assert len(evts) == 1
    rec = evts[0]
    assert rec["event"] == "agent_started"
    assert rec["run_id"] == ctx.run_id
    assert rec["branch_id"] == "b1"
    assert rec["agent_id"] == "ag1"
    assert rec["trace_id"] == "t1"
    assert rec["source"] == "operation"
    assert rec["destination"] == "manager"
    assert rec["payload_summary"] == "test summary"
    assert "timestamp" in rec
    # JSONL schema: all 13 fields present
    for field in (
        "timestamp", "trace_id", "run_id", "branch_id", "agent_id",
        "handoff_id", "event", "source", "destination",
        "payload_summary", "payload_ref", "payload_size", "payload_hash",
    ):
        assert field in rec, f"Missing field: {field}"


@pytest.mark.unit
def test_append_event_computes_payload_size_and_hash(tmp_path):
    ctx = _make_ctx()
    payload = "some text payload"
    append_event(ctx, "artifact_written", payload_text=payload, workspace=tmp_path)
    evts = _read_jsonl(tmp_path / "exp" / "runs" / ctx.run_id / "events.jsonl")
    rec = evts[0]
    assert rec["payload_size"] == len(payload.encode())
    assert rec["payload_hash"] == _payload_hash(payload)


@pytest.mark.unit
def test_append_event_multiple_events_each_valid_json(tmp_path):
    ctx = _make_ctx()
    for name in ["run_started", "agent_started", "llm_call_completed", "run_completed"]:
        append_event(ctx, name, workspace=tmp_path)
    evts = _read_jsonl(tmp_path / "exp" / "runs" / ctx.run_id / "events.jsonl")
    assert len(evts) == 4
    names = [e["event"] for e in evts]
    assert names == ["run_started", "agent_started", "llm_call_completed", "run_completed"]


@pytest.mark.unit
def test_append_event_extra_kwargs_stored(tmp_path):
    ctx = _make_ctx()
    append_event(ctx, "custom_event", custom_field="hello", workspace=tmp_path)
    evts = _read_jsonl(tmp_path / "exp" / "runs" / ctx.run_id / "events.jsonl")
    assert evts[0]["custom_field"] == "hello"


@pytest.mark.unit
def test_append_event_noop_when_ctx_none():
    """Should raise AttributeError — callers must guard with 'if ctx is not None'."""
    # This test confirms the contract: append_event is NOT itself None-safe.
    # Callers use `if run_ctx is not None:` guards.
    with pytest.raises(AttributeError):
        append_event(None, "run_started")


# ── append_handoff ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_append_handoff_creates_handoffs_jsonl(tmp_path):
    ctx = _make_ctx()
    hid = str(uuid.uuid4())
    append_handoff(
        ctx, hid,
        source_agent_id="manager_abc",
        dest_agent_id="data_abc",
        workspace=tmp_path,
    )
    hf = tmp_path / "exp" / "runs" / ctx.run_id / "handoffs.jsonl"
    assert hf.exists()


@pytest.mark.unit
def test_append_handoff_has_required_fields(tmp_path):
    ctx = _make_ctx(branch_id="branch-1")
    hid = "hid-001"
    append_handoff(
        ctx, hid,
        source_agent_id="manager_xyz",
        dest_agent_id="data_xyz",
        direction="emitted",
        payload_summary="plan 1",
        payload_text="full plan text here",
        workspace=tmp_path,
    )
    records = _read_jsonl(tmp_path / "exp" / "runs" / ctx.run_id / "handoffs.jsonl")
    rec = records[0]
    assert rec["handoff_id"] == hid
    assert rec["source_agent_id"] == "manager_xyz"
    assert rec["dest_agent_id"] == "data_xyz"
    assert rec["direction"] == "emitted"
    assert rec["branch_id"] == "branch-1"
    assert rec["payload_summary"] == "plan 1"
    assert rec["payload_size"] is not None


# ── append_cost_record ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_append_cost_record_creates_file(tmp_path):
    ctx = _make_ctx()
    append_cost_record(
        ctx,
        provider="openrouter",
        alias="or-glm-5",
        model_slug="z-ai/glm-5",
        phase="planning",
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        workspace=tmp_path,
    )
    path = tmp_path / "exp" / "runs" / ctx.run_id / "cost_records.jsonl"
    assert path.exists()


@pytest.mark.unit
def test_append_cost_record_schema(tmp_path):
    ctx = _make_ctx(branch_id="b0", agent_id="op-1")
    append_cost_record(
        ctx,
        provider="openai",
        alias="gpt-4o",
        model_slug="gpt-4o",
        phase="synthesis",
        prompt_tokens=500,
        completion_tokens=100,
        total_tokens=600,
        reasoning_tokens=10,
        estimated_cost=0.005,
        workspace=tmp_path,
    )
    records = _read_jsonl(tmp_path / "exp" / "runs" / ctx.run_id / "cost_records.jsonl")
    rec = records[0]
    assert rec["provider"] == "openai"
    assert rec["alias"] == "gpt-4o"
    assert rec["model_slug"] == "gpt-4o"
    assert rec["phase"] == "synthesis"
    assert rec["prompt_tokens"] == 500
    assert rec["completion_tokens"] == 100
    assert rec["total_tokens"] == 600
    assert rec["reasoning_tokens"] == 10
    assert rec["estimated_cost"] == 0.005
    assert rec["branch_id"] == "b0"
    assert rec["agent_id"] == "op-1"


# ── record_llm_usage ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_record_llm_usage_noop_when_ctx_none():
    """Should silently do nothing."""
    record_llm_usage(None, MagicMock(), alias="or-glm-5", model_slug="z-ai/glm-5", phase="planning")


@pytest.mark.unit
def test_record_llm_usage_extracts_tokens_and_writes(tmp_path):
    ctx = _make_ctx()
    usage = SimpleNamespace(
        prompt_tokens=80,
        completion_tokens=120,
        total_tokens=200,
        reasoning_tokens=None,
        completion_tokens_details=None,
    )
    response = SimpleNamespace(usage=usage)
    record_llm_usage(
        ctx, response,
        alias="or-glm-5",
        model_slug="z-ai/glm-5",
        phase="planning",
        workspace=tmp_path,
    )
    records = _read_jsonl(tmp_path / "exp" / "runs" / ctx.run_id / "cost_records.jsonl")
    rec = records[0]
    assert rec["prompt_tokens"] == 80
    assert rec["completion_tokens"] == 120
    assert rec["total_tokens"] == 200
    assert rec["provider"] == "openrouter"


@pytest.mark.unit
def test_record_llm_usage_infers_provider_openai(tmp_path):
    ctx = _make_ctx()
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
        reasoning_tokens=None, completion_tokens_details=None
    )
    record_llm_usage(
        ctx, SimpleNamespace(usage=usage),
        alias="gpt-4o",
        model_slug="gpt-4o",
        phase="verification",
        workspace=tmp_path,
    )
    records = _read_jsonl(tmp_path / "exp" / "runs" / ctx.run_id / "cost_records.jsonl")
    assert records[0]["provider"] == "openai"


@pytest.mark.unit
def test_record_llm_usage_no_usage_attribute_is_noop(tmp_path):
    ctx = _make_ctx()
    record_llm_usage(
        ctx, SimpleNamespace(),  # no .usage attr
        alias="or-glm-5",
        model_slug="z-ai/glm-5",
        phase="planning",
        workspace=tmp_path,
    )
    path = tmp_path / "exp" / "runs" / ctx.run_id / "cost_records.jsonl"
    assert not path.exists()


# ── write_cost_summary ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_write_cost_summary_creates_file(tmp_path):
    ctx = _make_ctx()
    write_cost_summary(ctx, workspace=tmp_path)
    path = tmp_path / "exp" / "runs" / ctx.run_id / "cost_summary.json"
    assert path.exists()


@pytest.mark.unit
def test_write_cost_summary_with_no_records(tmp_path):
    ctx = _make_ctx()
    write_cost_summary(ctx, workspace=tmp_path)
    path = tmp_path / "exp" / "runs" / ctx.run_id / "cost_summary.json"
    data = json.loads(path.read_text())
    assert data["run_id"] == ctx.run_id
    assert data["records_count"] == 0
    assert data["total_tokens"] == 0
    assert data["by_model"] == {}


@pytest.mark.unit
def test_write_cost_summary_totals_match_records(tmp_path):
    ctx = _make_ctx()
    for alias, pt, ct, tt in [
        ("or-glm-5", 100, 200, 300),
        ("or-glm-5", 50, 100, 150),
        ("gpt-4o", 200, 300, 500),
    ]:
        append_cost_record(
            ctx,
            provider="openrouter" if alias.startswith("or-") else "openai",
            alias=alias,
            model_slug=alias,
            phase="planning",
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            workspace=tmp_path,
        )
    write_cost_summary(ctx, workspace=tmp_path)
    path = tmp_path / "exp" / "runs" / ctx.run_id / "cost_summary.json"
    data = json.loads(path.read_text())
    assert data["records_count"] == 3
    assert data["total_prompt_tokens"] == 350
    assert data["total_completion_tokens"] == 600
    assert data["total_tokens"] == 950
    assert "or-glm-5" in data["by_model"]
    assert "gpt-4o" in data["by_model"]
    assert data["by_model"]["or-glm-5"]["total_tokens"] == 450
    assert data["by_model"]["gpt-4o"]["total_tokens"] == 500


@pytest.mark.unit
def test_write_cost_summary_overwritten_on_second_call(tmp_path):
    ctx = _make_ctx()
    write_cost_summary(ctx, workspace=tmp_path)
    append_cost_record(
        ctx,
        provider="openai",
        alias="gpt-4o",
        model_slug="gpt-4o",
        phase="x",
        prompt_tokens=9,
        completion_tokens=1,
        total_tokens=10,
        workspace=tmp_path,
    )
    write_cost_summary(ctx, workspace=tmp_path)
    path = tmp_path / "exp" / "runs" / ctx.run_id / "cost_summary.json"
    data = json.loads(path.read_text())
    assert data["records_count"] == 1
    assert data["total_tokens"] == 10


# ── ensure_analyses_dir ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_ensure_analyses_dir_creates_directory(tmp_path):
    ctx = _make_ctx()
    analyses = ensure_analyses_dir(ctx, workspace=tmp_path)
    assert analyses.is_dir()
    assert analyses.name == "analyses"
    assert analyses.parent == tmp_path / "exp" / "runs" / ctx.run_id


@pytest.mark.unit
def test_ensure_analyses_dir_idempotent(tmp_path):
    ctx = _make_ctx()
    d1 = ensure_analyses_dir(ctx, workspace=tmp_path)
    d2 = ensure_analyses_dir(ctx, workspace=tmp_path)
    assert d1 == d2
    assert d1.is_dir()


# ── write_analysis ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_write_analysis_plain_text(tmp_path):
    ctx = _make_ctx()
    path = write_analysis(ctx, "req_summary", "This is the summary.", workspace=tmp_path)
    assert path.suffix == ".txt"
    assert path.read_text() == "This is the summary."


@pytest.mark.unit
def test_write_analysis_dict_serialized_as_json(tmp_path):
    ctx = _make_ctx()
    payload = {"problem": "tabular_classification", "dataset": "iris"}
    path = write_analysis(ctx, "prompt_parse", payload, workspace=tmp_path)
    assert path.suffix == ".json"
    data = json.loads(path.read_text())
    assert data["problem"] == "tabular_classification"


@pytest.mark.unit
def test_write_analysis_returns_path_in_analyses_dir(tmp_path):
    ctx = _make_ctx()
    path = write_analysis(ctx, "plan_0", "Step 1: do something", workspace=tmp_path)
    assert path.parent.name == "analyses"
    assert path.parent.parent == tmp_path / "exp" / "runs" / ctx.run_id


@pytest.mark.unit
def test_write_analysis_list_serialized_as_json(tmp_path):
    ctx = _make_ctx()
    path = write_analysis(ctx, "candidates", [1, 2, 3], workspace=tmp_path)
    assert json.loads(path.read_text()) == [1, 2, 3]
