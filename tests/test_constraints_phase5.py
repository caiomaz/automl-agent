"""Phase 5 — Granular constraints + analyses persistence (TDD).

Tests for:
- ``utils.constraints.normalize_constraints`` accepting both legacy keys
  (``model``, ``perf_metric``...) and the granular block (``split_policy``,
  ``seed``, ``framework``, ``hitl_policy``, ``cleanup_policy``...).
- ``utils.constraints.persist_constraints`` writing the structured block
  to ``run_manifest.json`` AND ``analyses/constraints.json`` and emitting
  a ``constraints_recorded`` event.
- Schema accepting the new optional ``constraints`` block without
  breaking on legacy payloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_active_run():
    from utils.run_context import clear_active_run
    clear_active_run()
    yield
    clear_active_run()


@pytest.mark.unit
class TestNormalizeConstraints:
    def test_passes_through_granular_keys(self):
        from utils.constraints import normalize_constraints

        out = normalize_constraints(
            {
                "model": "XGBoost",
                "seed": 42,
                "split_policy": "k-fold",
                "framework": "lightgbm",
                "hitl_policy": "standard",
                "cleanup_policy": "archive",
                "token_economy": "aggressive",
            }
        )
        assert out["seed"] == 42
        assert out["split_policy"] == "k-fold"
        assert out["framework"] == "lightgbm"
        assert out["hitl_policy"] == "standard"
        assert out["cleanup_policy"] == "archive"
        assert out["token_economy"] == "aggressive"
        assert out["model"] == "XGBoost"

    def test_legacy_only_payload_is_kept(self):
        from utils.constraints import normalize_constraints

        out = normalize_constraints(
            {"model": "LightGBM", "perf_metric": "RMSLE", "perf_value": "0.15"}
        )
        assert out == {
            "model": "LightGBM",
            "perf_metric": "RMSLE",
            "perf_value": "0.15",
        }

    def test_unknown_keys_are_dropped(self):
        from utils.constraints import normalize_constraints

        out = normalize_constraints({"model": "X", "garbage": True})
        assert "garbage" not in out
        assert out["model"] == "X"

    def test_invalid_hitl_policy_rejected(self):
        from utils.constraints import normalize_constraints

        with pytest.raises(ValueError, match="hitl_policy"):
            normalize_constraints({"hitl_policy": "wat"})

    def test_invalid_cleanup_policy_rejected(self):
        from utils.constraints import normalize_constraints

        with pytest.raises(ValueError, match="cleanup_policy"):
            normalize_constraints({"cleanup_policy": "burn"})


@pytest.mark.unit
class TestPersistConstraints:
    def test_writes_manifest_and_analyses_file(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.constraints import persist_constraints
        from utils.workspace import run_exp_dir

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        persist_constraints(
            ctx,
            {"model": "XGBoost", "seed": 7, "framework": "xgboost"},
            workspace=tmp_path,
        )

        manifest = json.loads(
            (run_exp_dir(ctx.run_id, tmp_path) / "run_manifest.json").read_text()
        )
        assert manifest["constraints"]["seed"] == 7
        assert manifest["constraints"]["framework"] == "xgboost"
        assert manifest["constraints"]["model"] == "XGBoost"

        analyses_file = (
            run_exp_dir(ctx.run_id, tmp_path) / "analyses" / "constraints.json"
        )
        assert analyses_file.exists()
        loaded = json.loads(analyses_file.read_text())
        assert loaded["seed"] == 7

        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_emits_constraints_recorded_event(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.constraints import persist_constraints
        from utils.workspace import run_exp_dir

        ctx = prepare_new_run(
            task_type="tabular_regression",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        persist_constraints(ctx, {"model": "LightGBM"}, workspace=tmp_path)

        events_file = run_exp_dir(ctx.run_id, tmp_path) / "events.jsonl"
        events = [
            json.loads(line) for line in events_file.read_text().splitlines() if line
        ]
        names = [e["event"] for e in events]
        assert "constraints_recorded" in names

        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_persist_empty_constraints_is_noop(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.constraints import persist_constraints
        from utils.workspace import run_exp_dir

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        persist_constraints(ctx, {}, workspace=tmp_path)
        manifest = json.loads(
            (run_exp_dir(ctx.run_id, tmp_path) / "run_manifest.json").read_text()
        )
        # No constraints block written when there's nothing to record.
        assert "constraints" not in manifest or manifest["constraints"] in (
            {},
            None,
        )
        finalize_run(ctx, status="completed", workspace=tmp_path)


@pytest.mark.unit
class TestSchemaAcceptsConstraintsBlock:
    def test_constraints_block_is_optional_in_schema(self):
        import json as _json

        schema = _json.loads(
            Path("prompt_agent/schema.json").read_text(encoding="utf-8")
        )
        # If the granular block is declared, it must be optional (not in
        # top-level "required") so legacy parses keep working.
        if "constraints" in schema.get("properties", {}):
            assert "constraints" not in schema.get("required", [])
        # The block itself must permit the documented granular keys.
        block = schema.get("properties", {}).get("constraints")
        assert block is not None, "Phase 5 schema must declare 'constraints'."
        props = block.get("properties", {})
        for key in (
            "split_policy",
            "seed",
            "framework",
            "hitl_policy",
            "cleanup_policy",
        ):
            assert key in props, f"missing '{key}' in constraints schema"
