"""Phase 3 tests: BranchScheduler with parallel mode + serial fallback.

The scheduler must:
1. Dispatch a list of jobs (one per branch) preserving the branch_id.
2. Run them in parallel up to a configurable max_concurrency.
3. Fall back to serial execution when an explicit fallback signal is raised
   or when ``mode='serial'`` is forced.
4. Emit ledger events ``scheduler_started``, ``scheduler_fallback_serial``,
   ``scheduler_completed`` so the trace shows what mode actually ran.
5. Aggregate results in the same order as the input job list.
6. Never share mutable state with the worker function.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_WORKSPACE_DIR", str(tmp_path))
    import importlib
    import utils.workspace as ws_mod
    importlib.reload(ws_mod)
    import utils.ledger as ledger_mod
    importlib.reload(ledger_mod)
    return tmp_path


def _square(x: int) -> int:
    return x * x


def _raise_fallback(x: int) -> int:
    from utils.scheduler import SchedulerFallback
    raise SchedulerFallback("provider blocked")


def _flaky_then_pass(x: int) -> int:
    # Used only in serial mode after fallback
    return x + 100


@pytest.mark.unit
class TestBranchScheduler:
    def test_parallel_runs_all_jobs_preserving_order(self, tmp_workspace):
        from utils.scheduler import BranchScheduler
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        sched = BranchScheduler(max_concurrency=3, run_ctx=ctx, workspace=tmp_workspace)
        results = sched.map(_square, [1, 2, 3, 4])
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        assert results == [1, 4, 9, 16]
        events = _read_jsonl(tmp_workspace / "exp" / "runs" / ctx.run_id / "events.jsonl")
        names = [e["event"] for e in events]
        assert "scheduler_started" in names
        assert "scheduler_completed" in names

    def test_serial_mode_runs_inline(self, tmp_workspace):
        from utils.scheduler import BranchScheduler
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        sched = BranchScheduler(mode="serial", run_ctx=ctx, workspace=tmp_workspace)
        results = sched.map(_square, [5, 6])
        finalize_run(ctx, status="completed", workspace=tmp_workspace)
        assert results == [25, 36]

    def test_fallback_to_serial_when_signaled(self, tmp_workspace):
        from utils.scheduler import BranchScheduler
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )

        # The first call raises SchedulerFallback so the scheduler must
        # restart in serial with a different worker function. We model
        # that by passing a "retry_fn" kwarg to .map.
        sched = BranchScheduler(max_concurrency=2, run_ctx=ctx, workspace=tmp_workspace)
        results = sched.map(_raise_fallback, [1, 2, 3], serial_fn=_flaky_then_pass)
        finalize_run(ctx, status="completed", workspace=tmp_workspace)

        assert results == [101, 102, 103]
        events = _read_jsonl(tmp_workspace / "exp" / "runs" / ctx.run_id / "events.jsonl")
        names = [e["event"] for e in events]
        assert "scheduler_fallback_serial" in names
        fallback = next(e for e in events if e["event"] == "scheduler_fallback_serial")
        assert fallback.get("reason")

    def test_max_concurrency_one_means_serial(self, tmp_workspace):
        from utils.scheduler import BranchScheduler
        from utils.run_context import prepare_new_run, finalize_run

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        sched = BranchScheduler(max_concurrency=1, run_ctx=ctx, workspace=tmp_workspace)
        results = sched.map(_square, [2, 3])
        finalize_run(ctx, status="completed", workspace=tmp_workspace)
        assert results == [4, 9]

    def test_invalid_mode_rejected(self, tmp_workspace):
        from utils.scheduler import BranchScheduler
        with pytest.raises(ValueError):
            BranchScheduler(mode="bogus")

    def test_branch_id_passed_to_worker(self, tmp_workspace):
        """Workers must receive their branch_id when the scheduler is given
        per-job branch ids."""
        from utils.scheduler import BranchScheduler
        from utils.run_context import prepare_new_run, finalize_run

        captured: list[str] = []

        def worker(payload):
            captured.append(payload["branch_id"])
            return payload["x"] * 10

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_workspace,
        )
        sched = BranchScheduler(mode="serial", run_ctx=ctx, workspace=tmp_workspace)
        jobs = [
            {"branch_id": "b1", "x": 1},
            {"branch_id": "b2", "x": 2},
        ]
        results = sched.map(worker, jobs)
        finalize_run(ctx, status="completed", workspace=tmp_workspace)
        assert results == [10, 20]
        assert captured == ["b1", "b2"]
