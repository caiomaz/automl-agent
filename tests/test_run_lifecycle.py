"""Phase 1 — Run lifecycle hardening (TDD).

Tests for:
- active-run validation (only one in-flight run at a time, unless forced)
- cleanup policy (preserve / archive / purge)
- cleanup events (run_cleanup_started, run_cleanup_completed)
- dataset provenance recording
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Active-run validation ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestActiveRunValidation:
    def test_first_run_succeeds(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        clear_active_run()
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        assert ctx.status == "running"
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_second_run_without_finalize_raises(self, tmp_path):
        from utils.run_context import (
            prepare_new_run,
            finalize_run,
            clear_active_run,
            ActiveRunError,
        )
        clear_active_run()
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        with pytest.raises(ActiveRunError, match=ctx.run_id):
            prepare_new_run(
                task_type="tabular_regression",
                llm_backbone="or-glm-5",
                workspace=tmp_path,
            )
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_finalize_clears_active_run(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        clear_active_run()
        ctx_a = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        finalize_run(ctx_a, status="completed", workspace=tmp_path)
        # Should now be free to start another
        ctx_b = prepare_new_run(
            task_type="tabular_regression",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        assert ctx_b.run_id != ctx_a.run_id
        finalize_run(ctx_b, status="completed", workspace=tmp_path)

    def test_force_overrides_active_run(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        clear_active_run()
        ctx_a = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        # force=True takes over even though ctx_a was not finalized
        ctx_b = prepare_new_run(
            task_type="tabular_regression",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
            force=True,
        )
        assert ctx_b.run_id != ctx_a.run_id
        finalize_run(ctx_b, status="completed", workspace=tmp_path)


# ── Cleanup policy ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCleanupPolicy:
    def _seed_old_run(self, tmp_path: Path, run_id: str = "old-run-id") -> None:
        for sub in ("datasets", "exp", "trained_models"):
            d = tmp_path / sub / "runs" / run_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "marker.txt").write_text("seed")

    def test_preserve_keeps_old_runs(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        clear_active_run()
        self._seed_old_run(tmp_path)
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            cleanup_mode="preserve",
            workspace=tmp_path,
        )
        assert (tmp_path / "exp" / "runs" / "old-run-id" / "marker.txt").exists()
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_purge_removes_old_runs(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        clear_active_run()
        self._seed_old_run(tmp_path)
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            cleanup_mode="purge",
            workspace=tmp_path,
        )
        # Old run sub-tree must be gone
        assert not (tmp_path / "exp" / "runs" / "old-run-id").exists()
        assert not (tmp_path / "datasets" / "runs" / "old-run-id").exists()
        assert not (tmp_path / "trained_models" / "runs" / "old-run-id").exists()
        # The new run directory must exist
        assert (tmp_path / "exp" / "runs" / ctx.run_id).exists()
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_purge_preserves_dataset_cache(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        clear_active_run()
        cache = tmp_path / "datasets" / "cache" / "remote-blob"
        cache.mkdir(parents=True)
        (cache / "data.csv").write_text("col1,col2\n1,2\n")
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            cleanup_mode="purge",
            workspace=tmp_path,
        )
        assert (cache / "data.csv").exists()
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_archive_moves_old_runs_to_archive_dir(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        clear_active_run()
        self._seed_old_run(tmp_path)
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            cleanup_mode="archive",
            workspace=tmp_path,
        )
        # Old run gone from the live area...
        assert not (tmp_path / "exp" / "runs" / "old-run-id").exists()
        # ...but still discoverable under archive/
        archive_root = tmp_path / "archive"
        assert archive_root.exists()
        archived = list(archive_root.rglob("old-run-id/marker.txt"))
        assert len(archived) >= 1
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_cleanup_emits_started_and_completed_events(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        clear_active_run()
        self._seed_old_run(tmp_path)
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            cleanup_mode="purge",
            workspace=tmp_path,
        )
        events_path = tmp_path / "exp" / "runs" / ctx.run_id / "events.jsonl"
        records = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
        names = [r["event"] for r in records]
        assert "run_cleanup_started" in names
        assert "run_cleanup_completed" in names
        # Cleanup events must precede run_started
        idx_clean = names.index("run_cleanup_completed")
        idx_start = names.index("run_started")
        assert idx_clean < idx_start
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_preserve_mode_still_emits_cleanup_events(self, tmp_path):
        """Cleanup events must be emitted regardless of mode, recording the choice."""
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        clear_active_run()
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            cleanup_mode="preserve",
            workspace=tmp_path,
        )
        events_path = tmp_path / "exp" / "runs" / ctx.run_id / "events.jsonl"
        records = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
        cleanup_evts = [r for r in records if r["event"] == "run_cleanup_started"]
        assert cleanup_evts
        assert cleanup_evts[0].get("mode") == "preserve"
        finalize_run(ctx, status="completed", workspace=tmp_path)


# ── Dataset provenance ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestDatasetProvenance:
    def test_provenance_dataclass_fields(self):
        from utils.provenance import DatasetProvenance
        prov = DatasetProvenance(
            mode="manual-upload",
            source="/local/path/data.csv",
            local_path="/local/path/data.csv",
        )
        assert prov.mode == "manual-upload"
        assert prov.recorded_at  # auto-set
        assert prov.checksum_sha256 is None  # not auto-computed unless asked

    def test_provenance_invalid_mode_rejected(self):
        from utils.provenance import DatasetProvenance
        with pytest.raises(ValueError, match="mode"):
            DatasetProvenance(mode="ftp", source="x", local_path="x")

    def test_compute_checksum(self, tmp_path):
        from utils.provenance import compute_checksum
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b\n1,2\n")
        digest = compute_checksum(f)
        assert isinstance(digest, str)
        assert len(digest) == 64  # full sha256 hex
        # Deterministic
        assert compute_checksum(f) == digest

    def test_record_provenance_writes_analysis_and_event(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        from utils.provenance import DatasetProvenance, record_provenance
        clear_active_run()
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b\n1,2\n")
        prov = DatasetProvenance(
            mode="user-link",
            source="https://example.com/data.csv",
            local_path=str(f),
            checksum_sha256=None,
        )
        record_provenance(ctx, prov, workspace=tmp_path, compute_checksum_now=True)

        # File written under analyses/
        prov_path = tmp_path / "exp" / "runs" / ctx.run_id / "analyses" / "dataset_provenance.json"
        assert prov_path.exists()
        data = json.loads(prov_path.read_text())
        assert isinstance(data, list)
        assert data[0]["mode"] == "user-link"
        assert data[0]["checksum_sha256"]  # was computed

        # Event emitted
        events_path = tmp_path / "exp" / "runs" / ctx.run_id / "events.jsonl"
        names = [json.loads(l)["event"] for l in events_path.read_text().splitlines() if l.strip()]
        assert "dataset_recorded" in names

        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_record_provenance_appends_multiple(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run, clear_active_run
        from utils.provenance import DatasetProvenance, record_provenance
        clear_active_run()
        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        for src in ("a.csv", "b.csv"):
            record_provenance(
                ctx,
                DatasetProvenance(mode="manual-upload", source=src, local_path=src),
                workspace=tmp_path,
            )
        prov_path = tmp_path / "exp" / "runs" / ctx.run_id / "analyses" / "dataset_provenance.json"
        data = json.loads(prov_path.read_text())
        assert len(data) == 2
        assert {d["source"] for d in data} == {"a.csv", "b.csv"}
        finalize_run(ctx, status="completed", workspace=tmp_path)
