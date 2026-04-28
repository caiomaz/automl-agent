"""Run lifecycle management — RunContext, prepare_new_run, finalize_run.

This module introduces the formal run lifecycle required by Phase 1
(TASKS.md) and ADR-007. It provides:

1. ``RunContext`` — a serializable value object carrying all run-scoped
   identity and configuration.
2. ``prepare_new_run()`` — creates a new ``RunContext``, provisions
   namespaced workspace directories, and writes the initial manifest.
3. ``finalize_run()`` — marks the run as completed/failed/cancelled,
   updates the manifest, and records the end timestamp.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.workspace import WORKSPACE_DIR, ensure_run_workspace, cleanup_workspace
from utils.ledger import append_event, ensure_analyses_dir, write_cost_summary

# ── Valid enumerations ────────────────────────────────────────────────────────

_VALID_STATUSES = frozenset({"created", "running", "completed", "failed", "cancelled"})
_VALID_HITL_LEVELS = frozenset({"off", "standard", "strict"})
_VALID_CLEANUP_MODES = frozenset({"preserve", "archive", "purge"})


# ── Active run registry (process-local) ───────────────────────────────────────


class ActiveRunError(RuntimeError):
    """Raised when ``prepare_new_run`` is called while another run is active."""


_active_run_id: str | None = None


def get_active_run_id() -> str | None:
    """Return the run_id of the currently active run, if any."""
    return _active_run_id


def clear_active_run() -> None:
    """Forget the active run (test/utility hook)."""
    global _active_run_id
    _active_run_id = None


def _set_active_run(run_id: str | None) -> None:
    global _active_run_id
    _active_run_id = run_id


# ── RunContext ────────────────────────────────────────────────────────────────


class RunContext:
    """Serializable run-scoped identity and configuration.

    Parameters
    ----------
    task_type:
        Downstream task type (e.g. ``"tabular_classification"``).
    llm_backbone:
        LLM alias or slug used for planning/synthesis/verification.
    prompt_llm:
        LLM alias used by the Prompt Agent.
    hitl_level:
        Human-in-the-loop checkpoint level: ``"off"``, ``"standard"``,
        or ``"strict"``.
    cleanup_mode:
        Workspace cleanup policy: ``"preserve"``, ``"archive"``, or
        ``"purge"``.
    run_id:
        Override the auto-generated UUIDv4 (useful for deserialization).
    started_at:
        Override the auto-generated ISO timestamp.
    status:
        Initial status. Defaults to ``"created"``.
    """

    __slots__ = (
        "_run_id",
        "_status",
        "branch_id",
        "agent_id",
        "_manager_agent_id",
        "attempt_id",
        "started_at",
        "ended_at",
        "task_type",
        "llm_backbone",
        "prompt_llm",
        "hitl_level",
        "cleanup_mode",
    )

    def __init__(
        self,
        *,
        task_type: str,
        llm_backbone: str,
        prompt_llm: str = "prompt-llm",
        hitl_level: str = "off",
        cleanup_mode: str = "preserve",
        run_id: str | None = None,
        started_at: str | None = None,
        status: str = "created",
        branch_id: str | None = None,
        agent_id: str | None = None,
        attempt_id: int = 0,
        ended_at: str | None = None,
    ) -> None:
        if hitl_level not in _VALID_HITL_LEVELS:
            raise ValueError(
                f"Invalid hitl_level '{hitl_level}'. Must be one of: {sorted(_VALID_HITL_LEVELS)}"
            )
        if cleanup_mode not in _VALID_CLEANUP_MODES:
            raise ValueError(
                f"Invalid cleanup_mode '{cleanup_mode}'. Must be one of: {sorted(_VALID_CLEANUP_MODES)}"
            )

        self._run_id = run_id or str(uuid.uuid4())
        self._status = "created"  # will be set properly via property
        self.branch_id = branch_id
        self.agent_id = agent_id
        self._manager_agent_id = None  # set by AgentManager if run_ctx is passed
        self.attempt_id = attempt_id
        self.started_at = started_at or datetime.now(timezone.utc).isoformat()
        self.ended_at = ended_at
        self.task_type = task_type
        self.llm_backbone = llm_backbone
        self.prompt_llm = prompt_llm
        self.hitl_level = hitl_level
        self.cleanup_mode = cleanup_mode

        # Apply initial status through the validated setter
        if status != "created":
            self.status = status

    # ── run_id (read-only) ────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._run_id

    # ── status (validated) ────────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        if value not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{value}'. Must be one of: {sorted(_VALID_STATUSES)}"
            )
        self._status = value

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "agent_id": self.agent_id,
            "attempt_id": self.attempt_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "task_type": self.task_type,
            "llm_backbone": self.llm_backbone,
            "prompt_llm": self.prompt_llm,
            "hitl_level": self.hitl_level,
            "cleanup_mode": self.cleanup_mode,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunContext:
        return cls(
            run_id=data["run_id"],
            task_type=data["task_type"],
            llm_backbone=data["llm_backbone"],
            prompt_llm=data.get("prompt_llm", "prompt-llm"),
            hitl_level=data.get("hitl_level", "off"),
            cleanup_mode=data.get("cleanup_mode", "preserve"),
            started_at=data.get("started_at"),
            status=data.get("status", "created"),
            branch_id=data.get("branch_id"),
            agent_id=data.get("agent_id"),
            attempt_id=data.get("attempt_id", 0),
            ended_at=data.get("ended_at"),
        )

    def __repr__(self) -> str:
        return (
            f"RunContext(run_id={self.run_id!r}, task_type={self.task_type!r}, "
            f"status={self.status!r})"
        )


# ── Lifecycle helpers ─────────────────────────────────────────────────────────


def _manifest_path(run_id: str, workspace: Path | None = None) -> Path:
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    return ws / "exp" / "runs" / run_id / "run_manifest.json"


def _write_manifest(ctx: RunContext, workspace: Path | None = None) -> None:
    path = _manifest_path(ctx.run_id, workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ctx.to_json())


def prepare_new_run(
    *,
    task_type: str,
    llm_backbone: str,
    prompt_llm: str = "prompt-llm",
    hitl_level: str = "off",
    cleanup_mode: str = "preserve",
    workspace: Path | None = None,
    force: bool = False,
) -> RunContext:
    """Create a new run, provision workspace dirs, and write the manifest.

    If another run is currently active (``prepare_new_run`` was called
    without a matching ``finalize_run``), an ``ActiveRunError`` is raised
    unless ``force=True``.

    The configured ``cleanup_mode`` is applied to existing run subtrees
    *before* the new run directory is created. Cleanup events are written
    to the new run's ledger so the policy decision is auditable.

    Returns a ``RunContext`` with status ``"running"``.
    """
    active = get_active_run_id()
    if active is not None and not force:
        raise ActiveRunError(
            f"Cannot start a new run while run {active!r} is still active. "
            "Call finalize_run() first or pass force=True."
        )

    ctx = RunContext(
        task_type=task_type,
        llm_backbone=llm_backbone,
        prompt_llm=prompt_llm,
        hitl_level=hitl_level,
        cleanup_mode=cleanup_mode,
    )
    ctx.status = "running"

    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR

    # Run cleanup BEFORE provisioning the new run dir, otherwise ``purge``
    # would wipe the freshly-written ledger files. We snapshot the start
    # timestamp first so the (started, completed) pair still reflects the
    # real cleanup window when both events are appended afterwards.
    cleanup_started_at = datetime.now(timezone.utc).isoformat()
    cleanup_info = cleanup_workspace(cleanup_mode, workspace=ws)

    ensure_run_workspace(ctx.run_id, workspace=ws)
    ensure_analyses_dir(ctx, workspace=ws)

    append_event(
        ctx, "run_cleanup_started",
        source="manager",
        payload_summary=f"cleanup mode={cleanup_mode}",
        workspace=ws,
        timestamp=cleanup_started_at,
        mode=cleanup_mode,
    )
    append_event(
        ctx, "run_cleanup_completed",
        source="manager",
        payload_summary=f"affected={len(cleanup_info.get('affected_run_ids', []))}",
        workspace=ws,
        mode=cleanup_mode,
        affected_run_ids=cleanup_info.get("affected_run_ids", []),
        archive_path=cleanup_info.get("archive_path"),
    )

    _write_manifest(ctx, workspace=ws)
    append_event(ctx, "run_started", source="manager", workspace=ws)

    _set_active_run(ctx.run_id)
    return ctx


def finalize_run(
    ctx: RunContext,
    *,
    status: str,
    workspace: Path | None = None,
) -> None:
    """Mark a run as completed, failed, or cancelled.

    Updates the context in place, rewrites the manifest, writes the cost
    summary, and clears the active-run registry so a new run may start.
    """
    ctx.status = status  # validates via setter
    ctx.ended_at = datetime.now(timezone.utc).isoformat()
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    _write_manifest(ctx, workspace=ws)
    _event = "run_completed" if status == "completed" else (
        "run_failed" if status == "failed" else "run_cancelled"
    )
    append_event(ctx, _event, source="manager", workspace=ws)
    write_cost_summary(ctx, workspace=ws)

    if get_active_run_id() == ctx.run_id:
        _set_active_run(None)
