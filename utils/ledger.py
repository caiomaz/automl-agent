"""Run-scoped event ledger, handoff ledger, cost records, and analyses.

This module is the single point of control for all structured audit
artifacts created during a run (Phase 2 / ADR-007).

Public API
----------
append_event(ctx, event_name, *, ...)
    Write one record to ``exp/runs/<run_id>/events.jsonl``.

append_handoff(ctx, handoff_id, *, source_agent_id, dest_agent_id, ...)
    Write one record to ``exp/runs/<run_id>/handoffs.jsonl``.

append_cost_record(ctx, *, provider, alias, model_slug, phase, ...)
    Write one cost record to ``exp/runs/<run_id>/cost_records.jsonl``.

record_llm_usage(ctx, response, *, alias, model_slug, phase)
    Convenience wrapper: extract tokens/reasoning from an OpenAI response
    object and call append_cost_record.

write_cost_summary(ctx, ...)
    Read all cost_records.jsonl entries and write cost_summary.json.

write_analysis(ctx, name, content, ...)
    Write a human-readable/JSON analysis file to ``analyses/<name>``.

All functions are pure I/O — no shared state — so they are safe to call
from worker processes. Each call opens the target file, appends one
newline-terminated JSON record, and closes the file immediately (O_APPEND
atomicity on Linux for small records).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.workspace import WORKSPACE_DIR, run_exp_dir


# ── Helpers ───────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_dir(run_id: str, workspace: Path | None) -> Path:
    return run_exp_dir(run_id, workspace)


def _payload_hash(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _payload_size(text: str | None) -> int | None:
    if not text:
        return None
    return len(text.encode())


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL file.  Thread/process-safe for
    records < ~4 KB because O_APPEND writes on Linux are atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


# ── Event ledger ──────────────────────────────────────────────────────────────


def append_event(
    ctx: Any,
    event_name: str,
    *,
    source: str | None = None,
    destination: str | None = None,
    payload_summary: str | None = None,
    payload_ref: str | None = None,
    payload_text: str | None = None,
    handoff_id: str | None = None,
    workspace: Path | None = None,
    **extra: Any,
) -> None:
    """Write one event record to ``events.jsonl``.

    Parameters
    ----------
    ctx:
        Active ``RunContext`` (or any object with ``run_id``, ``branch_id``,
        ``agent_id``).
    event_name:
        One of the canonical event names (e.g. ``"run_started"``).
    payload_text:
        If provided, its size and hash are computed and stored.  The raw
        text is NOT persisted in the ledger to keep records compact.
    extra:
        Any additional key-value pairs are stored under the record.
    """
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    record: dict[str, Any] = {
        "timestamp": _now(),
        "trace_id": getattr(ctx, "trace_id", None),
        "run_id": ctx.run_id,
        "branch_id": getattr(ctx, "branch_id", None),
        "agent_id": getattr(ctx, "agent_id", None),
        "handoff_id": handoff_id,
        "event": event_name,
        "source": source,
        "destination": destination,
        "payload_summary": payload_summary,
        "payload_ref": payload_ref,
        "payload_size": _payload_size(payload_text),
        "payload_hash": _payload_hash(payload_text),
    }
    record.update(extra)
    path = _run_dir(ctx.run_id, ws) / "events.jsonl"
    _append_jsonl(path, record)


# ── Handoff ledger ────────────────────────────────────────────────────────────


def append_handoff(
    ctx: Any,
    handoff_id: str,
    *,
    source_agent_id: str,
    dest_agent_id: str,
    direction: str = "emitted",
    payload_summary: str | None = None,
    payload_text: str | None = None,
    workspace: Path | None = None,
) -> None:
    """Write one handoff record to ``handoffs.jsonl``.

    Parameters
    ----------
    direction:
        ``"emitted"`` or ``"received"`` — written from the perspective of
        the calling agent.
    """
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    record: dict[str, Any] = {
        "timestamp": _now(),
        "run_id": ctx.run_id,
        "branch_id": getattr(ctx, "branch_id", None),
        "handoff_id": handoff_id,
        "source_agent_id": source_agent_id,
        "dest_agent_id": dest_agent_id,
        "direction": direction,
        "payload_summary": payload_summary,
        "payload_size": _payload_size(payload_text),
        "payload_hash": _payload_hash(payload_text),
    }
    path = _run_dir(ctx.run_id, ws) / "handoffs.jsonl"
    _append_jsonl(path, record)


# ── Cost records ──────────────────────────────────────────────────────────────


def append_cost_record(
    ctx: Any,
    *,
    provider: str,
    alias: str,
    model_slug: str,
    phase: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    estimated_cost: float | None = None,
    workspace: Path | None = None,
) -> None:
    """Write one LLM cost record to ``cost_records.jsonl``."""
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    record: dict[str, Any] = {
        "timestamp": _now(),
        "run_id": ctx.run_id,
        "branch_id": getattr(ctx, "branch_id", None),
        "agent_id": getattr(ctx, "agent_id", None),
        "provider": provider,
        "alias": alias,
        "model_slug": model_slug,
        "phase": phase,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "estimated_cost": estimated_cost,
    }
    path = _run_dir(ctx.run_id, ws) / "cost_records.jsonl"
    _append_jsonl(path, record)


def record_llm_usage(
    ctx: Any,
    response: Any,
    *,
    alias: str,
    model_slug: str,
    phase: str,
    workspace: Path | None = None,
) -> None:
    """Extract token usage from an OpenAI response and call append_cost_record.

    Safe to call even when ``ctx`` is None (no-op).
    """
    if ctx is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    # Some providers expose reasoning_tokens as a sub-field
    reasoning_tokens = getattr(usage, "reasoning_tokens", None)
    if reasoning_tokens is None:
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

    # Infer provider from the alias naming convention
    if alias.startswith("or-") or alias.startswith("openrouter"):
        provider = "openrouter"
    elif alias == "prompt-llm":
        provider = "local"
    else:
        provider = "openai"

    append_cost_record(
        ctx,
        provider=provider,
        alias=alias,
        model_slug=model_slug,
        phase=phase,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        workspace=workspace,
    )


# ── Cost summary ──────────────────────────────────────────────────────────────


def write_cost_summary(ctx: Any, workspace: Path | None = None) -> None:
    """Read ``cost_records.jsonl`` and write consolidated ``cost_summary.json``.

    Called by ``finalize_run()`` at the end of a run.
    """
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    run_dir = _run_dir(ctx.run_id, ws)
    records_path = run_dir / "cost_records.jsonl"
    summary_path = run_dir / "cost_summary.json"

    records: list[dict] = []
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    total_prompt = sum(r.get("prompt_tokens") or 0 for r in records)
    total_completion = sum(r.get("completion_tokens") or 0 for r in records)
    total_tokens = sum(r.get("total_tokens") or 0 for r in records)
    total_reasoning = sum(r.get("reasoning_tokens") or 0 for r in records)
    estimated_costs = [r.get("estimated_cost") for r in records]
    total_cost: float | None = (
        sum(c for c in estimated_costs if c is not None) if any(c is not None for c in estimated_costs) else None
    )

    by_model: dict[str, dict] = {}
    for r in records:
        slug = r.get("model_slug") or "unknown"
        if slug not in by_model:
            by_model[slug] = {
                "alias": r.get("alias"),
                "provider": r.get("provider"),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "record_count": 0,
            }
        by_model[slug]["prompt_tokens"] += r.get("prompt_tokens") or 0
        by_model[slug]["completion_tokens"] += r.get("completion_tokens") or 0
        by_model[slug]["total_tokens"] += r.get("total_tokens") or 0
        by_model[slug]["reasoning_tokens"] += r.get("reasoning_tokens") or 0
        by_model[slug]["record_count"] += 1

    summary = {
        "run_id": ctx.run_id,
        "finalized_at": _now(),
        "records_count": len(records),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "total_reasoning_tokens": total_reasoning,
        "total_estimated_cost": total_cost,
        "by_model": by_model,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ── Analyses directory ────────────────────────────────────────────────────────


def ensure_analyses_dir(ctx: Any, workspace: Path | None = None) -> Path:
    """Create and return the ``analyses/`` directory for this run."""
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    analyses = _run_dir(ctx.run_id, ws) / "analyses"
    analyses.mkdir(parents=True, exist_ok=True)
    return analyses


def write_analysis(
    ctx: Any,
    name: str,
    content: str | dict | list,
    workspace: Path | None = None,
) -> Path:
    """Write a named analysis file under ``analyses/``.

    If ``content`` is a dict or list, it is serialised as JSON.
    Otherwise it is written as plain text.

    Returns the path of the written file.
    """
    analyses_dir = ensure_analyses_dir(ctx, workspace)
    if isinstance(content, (dict, list)):
        path = analyses_dir / f"{name}.json"
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        path = analyses_dir / f"{name}.txt"
        path.write_text(str(content), encoding="utf-8")
    return path


# ── Phase 2 helpers ──────────────────────────────────────────────────────────


def emit_handoff(
    ctx: Any,
    *,
    source_agent_id: str,
    dest_agent_id: str,
    payload_summary: str | None = None,
    payload_text: str | None = None,
    handoff_id: str | None = None,
    workspace: Path | None = None,
) -> str:
    """Write a paired handoff record AND a structured ``handoff_emitted`` event.

    Returns the ``handoff_id`` used (auto-generated when not supplied) so
    the caller can correlate later events such as ``handoff_received``.
    """
    if handoff_id is None:
        handoff_id = str(uuid.uuid4())
    append_handoff(
        ctx,
        handoff_id,
        source_agent_id=source_agent_id,
        dest_agent_id=dest_agent_id,
        direction="emitted",
        payload_summary=payload_summary,
        payload_text=payload_text,
        workspace=workspace,
    )
    append_event(
        ctx,
        "handoff_emitted",
        source=source_agent_id,
        destination=dest_agent_id,
        payload_summary=payload_summary,
        payload_text=payload_text,
        handoff_id=handoff_id,
        workspace=workspace,
    )
    return handoff_id


def append_hitl_requested(
    ctx: Any,
    *,
    checkpoint: str,
    question: str,
    workspace: Path | None = None,
    **extra: Any,
) -> None:
    """Record that a Human-in-the-Loop interaction was requested."""
    append_event(
        ctx,
        "hitl_requested",
        workspace=workspace,
        checkpoint=checkpoint,
        question=question,
        **extra,
    )


def append_hitl_resolved(
    ctx: Any,
    *,
    checkpoint: str,
    decision: str,
    workspace: Path | None = None,
    **extra: Any,
) -> None:
    """Record the outcome of a previously requested HITL checkpoint."""
    append_event(
        ctx,
        "hitl_resolved",
        workspace=workspace,
        checkpoint=checkpoint,
        decision=decision,
        **extra,
    )


def append_critic_warned(
    ctx: Any,
    *,
    target: str,
    reason: str,
    workspace: Path | None = None,
    **extra: Any,
) -> None:
    """Record a non-blocking warning emitted by the Critic Agent."""
    append_event(
        ctx,
        "critic_warned",
        workspace=workspace,
        target=target,
        reason=reason,
        **extra,
    )


def append_critic_blocked(
    ctx: Any,
    *,
    target: str,
    reason: str,
    workspace: Path | None = None,
    **extra: Any,
) -> None:
    """Record a blocking decision emitted by the Critic Agent."""
    append_event(
        ctx,
        "critic_blocked",
        workspace=workspace,
        target=target,
        reason=reason,
        **extra,
    )


def write_reasoning(
    ctx: Any,
    *,
    agent: str,
    label: str,
    content: str,
    workspace: Path | None = None,
) -> Path:
    """Persist a reasoning trail entry under ``analyses/reasoning/`` and
    emit a ``reasoning_recorded`` event.

    Each call writes a separate file named ``<agent>__<label>.txt``.
    """
    analyses_dir = ensure_analyses_dir(ctx, workspace)
    reasoning_dir = analyses_dir / "reasoning"
    reasoning_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "_")
    path = reasoning_dir / f"{agent}__{safe_label}.txt"
    path.write_text(str(content), encoding="utf-8")
    append_event(
        ctx,
        "reasoning_recorded",
        source=agent,
        workspace=workspace,
        label=label,
        path=str(path.relative_to(_run_dir(ctx.run_id, Path(workspace) if workspace else WORKSPACE_DIR))),
    )
    return path
