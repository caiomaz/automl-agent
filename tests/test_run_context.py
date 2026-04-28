"""Tests for RunContext lifecycle — Phase 1.

TDD: these tests were written BEFORE the implementation.
They define the expected contracts for RunContext, prepare_new_run,
and finalize_run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_workspace(tmp_path):
    """Create a temporary workspace root for isolation."""
    return tmp_path / "agent_workspace"


# ── RunContext dataclass ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestRunContext:
    """RunContext must be a serializable value object with the right defaults."""

    def test_creation_with_defaults(self):
        from utils.run_context import RunContext
        ctx = RunContext(task_type="tabular_classification", llm_backbone="or-glm-5")
        assert ctx.run_id  # non-empty UUID
        assert ctx.status == "created"
        assert ctx.task_type == "tabular_classification"
        assert ctx.llm_backbone == "or-glm-5"
        assert ctx.prompt_llm == "prompt-llm"
        assert ctx.hitl_level == "off"
        assert ctx.cleanup_mode == "preserve"
        assert ctx.attempt_id == 0
        assert ctx.branch_id is None
        assert ctx.agent_id is None
        assert ctx.started_at  # non-empty ISO timestamp

    def test_run_id_is_uuid4(self):
        import uuid
        from utils.run_context import RunContext
        ctx = RunContext(task_type="image_classification", llm_backbone="or-glm-5")
        parsed = uuid.UUID(ctx.run_id, version=4)
        assert str(parsed) == ctx.run_id

    def test_custom_fields(self):
        from utils.run_context import RunContext
        ctx = RunContext(
            task_type="text_classification",
            llm_backbone="or-claude-sonnet",
            prompt_llm="or-glm-5",
            hitl_level="standard",
            cleanup_mode="archive",
        )
        assert ctx.prompt_llm == "or-glm-5"
        assert ctx.hitl_level == "standard"
        assert ctx.cleanup_mode == "archive"

    def test_to_dict(self):
        from utils.run_context import RunContext
        ctx = RunContext(task_type="tabular_regression", llm_backbone="or-glm-5")
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert d["run_id"] == ctx.run_id
        assert d["task_type"] == "tabular_regression"
        assert d["status"] == "created"

    def test_to_json_roundtrip(self):
        from utils.run_context import RunContext
        ctx = RunContext(task_type="ts_forecasting", llm_backbone="or-glm-5")
        json_str = ctx.to_json()
        parsed = json.loads(json_str)
        assert parsed["run_id"] == ctx.run_id
        assert parsed["task_type"] == "ts_forecasting"

    def test_from_dict(self):
        from utils.run_context import RunContext
        ctx = RunContext(task_type="node_classification", llm_backbone="or-glm-5")
        d = ctx.to_dict()
        restored = RunContext.from_dict(d)
        assert restored.run_id == ctx.run_id
        assert restored.task_type == ctx.task_type
        assert restored.status == ctx.status

    def test_status_transitions(self):
        from utils.run_context import RunContext
        ctx = RunContext(task_type="tabular_classification", llm_backbone="or-glm-5")
        assert ctx.status == "created"
        ctx.status = "running"
        assert ctx.status == "running"
        ctx.status = "completed"
        assert ctx.status == "completed"

    def test_invalid_status_rejected(self):
        from utils.run_context import RunContext
        ctx = RunContext(task_type="tabular_classification", llm_backbone="or-glm-5")
        with pytest.raises(ValueError, match="status"):
            ctx.status = "bogus"

    def test_invalid_hitl_level_rejected(self):
        from utils.run_context import RunContext
        with pytest.raises(ValueError, match="hitl_level"):
            RunContext(task_type="t", llm_backbone="x", hitl_level="extreme")

    def test_invalid_cleanup_mode_rejected(self):
        from utils.run_context import RunContext
        with pytest.raises(ValueError, match="cleanup_mode"):
            RunContext(task_type="t", llm_backbone="x", cleanup_mode="nuke")

    def test_ended_at_initially_none(self):
        from utils.run_context import RunContext
        ctx = RunContext(task_type="t", llm_backbone="x")
        assert ctx.ended_at is None


# ── prepare_new_run ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPrepareNewRun:
    """prepare_new_run must create a RunContext and namespaced workspace dirs."""

    def test_returns_run_context(self, tmp_workspace):
        from utils.run_context import prepare_new_run
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        assert ctx.run_id
        assert ctx.status == "running"

    def test_creates_namespaced_dirs(self, tmp_workspace):
        from utils.run_context import prepare_new_run
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        run_id = ctx.run_id
        assert (tmp_workspace / "datasets" / "runs" / run_id).is_dir()
        assert (tmp_workspace / "exp" / "runs" / run_id).is_dir()
        assert (tmp_workspace / "trained_models" / "runs" / run_id).is_dir()

    def test_creates_cache_dir(self, tmp_workspace):
        from utils.run_context import prepare_new_run
        prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        assert (tmp_workspace / "datasets" / "cache").is_dir()

    def test_preserves_existing_runs(self, tmp_workspace):
        from utils.run_context import prepare_new_run
        ctx1 = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        # Write a marker file in run 1
        marker = tmp_workspace / "exp" / "runs" / ctx1.run_id / "marker.txt"
        marker.write_text("run1")

        ctx2 = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
            force=True,
        )
        # Run 1 marker survives (preserve mode)
        assert marker.exists()
        assert ctx1.run_id != ctx2.run_id

    def test_writes_manifest(self, tmp_workspace):
        from utils.run_context import prepare_new_run
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        manifest = tmp_workspace / "exp" / "runs" / ctx.run_id / "run_manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["run_id"] == ctx.run_id
        assert data["status"] == "running"

    def test_custom_hitl_and_cleanup(self, tmp_workspace):
        from utils.run_context import prepare_new_run
        ctx = prepare_new_run(
            task_type="text_classification",
            llm_backbone="or-glm-5",
            hitl_level="strict",
            cleanup_mode="archive",
            workspace=tmp_workspace,
        )
        assert ctx.hitl_level == "strict"
        assert ctx.cleanup_mode == "archive"

    def test_two_runs_same_task_no_collision(self, tmp_workspace):
        """Acceptance criterion #1: two runs don't collide."""
        from utils.run_context import prepare_new_run
        ctx1 = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        ctx2 = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
            force=True,
        )
        exp1 = tmp_workspace / "exp" / "runs" / ctx1.run_id
        exp2 = tmp_workspace / "exp" / "runs" / ctx2.run_id
        assert exp1 != exp2
        assert exp1.is_dir()
        assert exp2.is_dir()


# ── finalize_run ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFinalizeRun:
    """finalize_run must mark the run as completed/failed/cancelled."""

    def test_marks_completed(self, tmp_workspace):
        from utils.run_context import prepare_new_run, finalize_run
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="completed", workspace=tmp_workspace)
        assert ctx.status == "completed"
        assert ctx.ended_at is not None

    def test_marks_failed(self, tmp_workspace):
        from utils.run_context import prepare_new_run, finalize_run
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="failed", workspace=tmp_workspace)
        assert ctx.status == "failed"

    def test_marks_cancelled(self, tmp_workspace):
        from utils.run_context import prepare_new_run, finalize_run
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="cancelled", workspace=tmp_workspace)
        assert ctx.status == "cancelled"

    def test_updates_manifest(self, tmp_workspace):
        from utils.run_context import prepare_new_run, finalize_run
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="completed", workspace=tmp_workspace)
        manifest = tmp_workspace / "exp" / "runs" / ctx.run_id / "run_manifest.json"
        data = json.loads(manifest.read_text())
        assert data["status"] == "completed"
        assert data["ended_at"] is not None

    def test_rejects_invalid_status(self, tmp_workspace):
        from utils.run_context import prepare_new_run, finalize_run
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        with pytest.raises(ValueError):
            finalize_run(ctx, status="bogus", workspace=tmp_workspace)

    def test_history_preserved_by_default(self, tmp_workspace):
        """Acceptance criterion #2: cleanup doesn't delete history by default."""
        from utils.run_context import prepare_new_run, finalize_run
        ctx1 = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        marker = tmp_workspace / "exp" / "runs" / ctx1.run_id / "result.txt"
        marker.write_text("important")
        finalize_run(ctx1, status="completed", workspace=tmp_workspace)

        # Start a new run — old run's data must survive
        ctx2 = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        assert marker.exists(), "Finalized run data must NOT be deleted"


# ── Workspace path helpers ───────────────────────────────────────────────────


@pytest.mark.unit
class TestWorkspaceRunPaths:
    """Run-namespaced path helpers must produce the expected layout."""

    def test_run_datasets_dir(self):
        from utils.workspace import run_datasets_dir
        p = run_datasets_dir("abc-123")
        assert p.parts[-3:] == ("datasets", "runs", "abc-123")

    def test_run_exp_dir(self):
        from utils.workspace import run_exp_dir
        p = run_exp_dir("abc-123")
        assert p.parts[-3:] == ("exp", "runs", "abc-123")

    def test_run_models_dir(self):
        from utils.workspace import run_models_dir
        p = run_models_dir("abc-123")
        assert p.parts[-3:] == ("trained_models", "runs", "abc-123")

    def test_datasets_cache_dir(self):
        from utils.workspace import datasets_cache_dir
        p = datasets_cache_dir()
        assert p.parts[-2:] == ("datasets", "cache")

    def test_ensure_run_workspace(self, tmp_workspace):
        from utils.workspace import ensure_run_workspace
        ensure_run_workspace("test-run-42", workspace=tmp_workspace)
        assert (tmp_workspace / "datasets" / "runs" / "test-run-42").is_dir()
        assert (tmp_workspace / "exp" / "runs" / "test-run-42").is_dir()
        assert (tmp_workspace / "trained_models" / "runs" / "test-run-42").is_dir()
        assert (tmp_workspace / "datasets" / "cache").is_dir()

    def test_canonical_dirs_still_exist(self):
        """The canonical WORKSPACE_DIR, DATASETS_DIR, EXP_DIR, MODELS_DIR must remain."""
        from utils.workspace import WORKSPACE_DIR, DATASETS_DIR, EXP_DIR, MODELS_DIR
        assert WORKSPACE_DIR.name == "agent_workspace"
        assert DATASETS_DIR == WORKSPACE_DIR / "datasets"
        assert EXP_DIR == WORKSPACE_DIR / "exp"
        assert MODELS_DIR == WORKSPACE_DIR / "trained_models"
