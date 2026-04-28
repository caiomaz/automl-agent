"""Phase 4 tests: CLI lifecycle helpers (signal handling, cancelled status,
post-run summary, listing of past runs).

We test the small pure helpers we add to cli.py / utils.run_context, not the
full interactive UI loop (which requires stdin I/O).
"""

from __future__ import annotations

import json
import signal as _signal
from pathlib import Path

import pytest


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_WORKSPACE_DIR", str(tmp_path))
    import importlib
    import utils.workspace as ws_mod
    importlib.reload(ws_mod)
    import utils.ledger as ledger_mod
    importlib.reload(ledger_mod)
    return tmp_path


# ── 1. CancellationRequested signal helper ───────────────────────────────────


@pytest.mark.unit
class TestCancellation:
    """install_cancellation_handler installs a handler that flips a flag and,
    if installed twice, raises KeyboardInterrupt to allow Ctrl+C-twice escape."""

    def test_handler_sets_flag_on_sigterm(self):
        from utils.cli_lifecycle import (
            install_cancellation_handler,
            cancellation_requested,
            _reset_cancellation_for_tests,
        )
        _reset_cancellation_for_tests()
        token = install_cancellation_handler()
        try:
            assert cancellation_requested() is False
            # Simulate signal delivery by calling the registered handler directly
            token.handler(_signal.SIGTERM, None)
            assert cancellation_requested() is True
        finally:
            token.uninstall()
            _reset_cancellation_for_tests()

    def test_double_signal_raises_keyboard_interrupt(self):
        from utils.cli_lifecycle import (
            install_cancellation_handler,
            _reset_cancellation_for_tests,
        )
        _reset_cancellation_for_tests()
        token = install_cancellation_handler()
        try:
            token.handler(_signal.SIGINT, None)  # first: flag
            with pytest.raises(KeyboardInterrupt):
                token.handler(_signal.SIGINT, None)  # second: escalate
        finally:
            token.uninstall()
            _reset_cancellation_for_tests()


# ── 2. Past-run listing ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestListRuns:
    """list_past_runs reads run_manifest.json from each runs/<id>/ subdirectory
    and returns a list of dicts sorted by start time descending."""

    def test_list_returns_empty_when_no_runs(self, tmp_workspace):
        from utils.cli_lifecycle import list_past_runs
        assert list_past_runs(workspace=tmp_workspace) == []

    def test_list_returns_two_finalized_runs(self, tmp_workspace):
        from utils.cli_lifecycle import list_past_runs
        from utils.run_context import prepare_new_run, finalize_run

        ctx1 = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        finalize_run(ctx1, status="completed", workspace=tmp_workspace)

        ctx2 = prepare_new_run(
            task_type="text_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        finalize_run(ctx2, status="failed", workspace=tmp_workspace)

        runs = list_past_runs(workspace=tmp_workspace)
        assert len(runs) == 2
        run_ids = {r["run_id"] for r in runs}
        assert ctx1.run_id in run_ids and ctx2.run_id in run_ids
        # Each entry must surface essential fields
        for r in runs:
            assert "run_id" in r and "task_type" in r and "status" in r
            assert "started_at" in r


# ── 3. Post-run summary ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestPostRunSummary:
    """build_post_run_summary returns a dict with run_id, status, exp_dir path,
    has_terminal_log, has_cost_summary and event_count."""

    def test_summary_with_minimal_artifacts(self, tmp_workspace):
        from utils.cli_lifecycle import build_post_run_summary
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        summary = build_post_run_summary(ctx.run_id, workspace=tmp_workspace)
        assert summary["run_id"] == ctx.run_id
        assert summary["status"] == "completed"
        assert summary["exp_dir"].endswith(ctx.run_id)
        assert summary["event_count"] >= 1  # at least run_started
        assert summary["has_cost_summary"] is True  # finalize_run writes it


# ── 4. Cancellation flow integration ─────────────────────────────────────────


@pytest.mark.unit
class TestRunCancellationStatus:
    """When a cancellation signal is delivered before finalize_run, the manifest
    must record status='cancelled' (not 'failed')."""

    def test_finalize_with_cancelled_status(self, tmp_workspace):
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        finalize_run(ctx, status="cancelled", workspace=tmp_workspace)
        manifest = json.loads(
            (tmp_workspace / "exp" / "runs" / ctx.run_id / "run_manifest.json").read_text()
        )
        assert manifest["status"] == "cancelled"
