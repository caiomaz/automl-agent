"""Phase 6 — Strategic HITL checkpoints (TDD).

Tests for ``utils.hitl.request_checkpoint``:
- Non-interactive mode auto-resolves with the documented default.
- Interactive mode reads from a stdin-like callable.
- Both branches emit ``hitl_requested`` + ``hitl_resolved`` events with
  matching ``checkpoint`` and ``hitl_id``.
- ``hitl_level`` policy gates whether a checkpoint is enforced or
  bypassed.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _reset_active_run():
    from utils.run_context import clear_active_run
    clear_active_run()
    yield
    clear_active_run()


def _events(tmp_path, run_id):
    from utils.workspace import run_exp_dir

    f = run_exp_dir(run_id, tmp_path) / "events.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line]


@pytest.mark.unit
class TestRequestCheckpoint:
    def test_non_interactive_uses_default_decision(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.hitl import request_checkpoint

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            hitl_level="standard",
            workspace=tmp_path,
        )
        result = request_checkpoint(
            ctx,
            checkpoint="before_destructive_cleanup",
            question="Purge prior runs?",
            interactive=False,
            default="approve",
            workspace=tmp_path,
        )
        assert result["decision"] == "approve"
        assert result["mode"] == "auto"
        assert result["checkpoint"] == "before_destructive_cleanup"

        evs = _events(tmp_path, ctx.run_id)
        names = [e["event"] for e in evs]
        assert "hitl_requested" in names
        assert "hitl_resolved" in names
        # both share the same hitl_id
        req = next(e for e in evs if e["event"] == "hitl_requested")
        res = next(e for e in evs if e["event"] == "hitl_resolved")
        assert req["hitl_id"] == res["hitl_id"]
        assert res["decision"] == "approve"
        assert res["mode"] == "auto"

        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_interactive_uses_input_callable(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.hitl import request_checkpoint

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            hitl_level="strict",
            workspace=tmp_path,
        )
        # The user picks "reject" via the injected input callable.
        result = request_checkpoint(
            ctx,
            checkpoint="after_parse",
            question="Continue with this parse?",
            interactive=True,
            default="approve",
            options=("approve", "reject"),
            input_fn=lambda _prompt: "reject",
            workspace=tmp_path,
        )
        assert result["decision"] == "reject"
        assert result["mode"] == "human"

        evs = _events(tmp_path, ctx.run_id)
        res = next(e for e in evs if e["event"] == "hitl_resolved")
        assert res["decision"] == "reject"
        assert res["mode"] == "human"

        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_invalid_input_falls_back_to_default(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.hitl import request_checkpoint

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            hitl_level="standard",
            workspace=tmp_path,
        )
        result = request_checkpoint(
            ctx,
            checkpoint="before_deploy",
            question="Launch web app?",
            interactive=True,
            default="reject",
            options=("approve", "reject"),
            input_fn=lambda _p: "garbage_value",
            workspace=tmp_path,
        )
        assert result["decision"] == "reject"
        assert result["mode"] == "human-fallback"
        finalize_run(ctx, status="completed", workspace=tmp_path)


@pytest.mark.unit
class TestHitlPolicy:
    def test_hitl_off_skips_checkpoint(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.hitl import request_checkpoint

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            hitl_level="off",
            workspace=tmp_path,
        )
        result = request_checkpoint(
            ctx,
            checkpoint="after_parse",
            question="Continue?",
            interactive=True,
            default="approve",
            input_fn=lambda _p: "reject",  # would reject, but should be skipped
            workspace=tmp_path,
        )
        assert result["decision"] == "approve"
        assert result["mode"] == "policy-skipped"

        evs = _events(tmp_path, ctx.run_id)
        # Skipped checkpoints should NOT prompt the user; they still
        # record the policy decision so the run remains auditable.
        res = [e for e in evs if e["event"] == "hitl_resolved"]
        assert len(res) == 1
        assert res[0]["mode"] == "policy-skipped"

        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_strict_enforces_all_known_checkpoints(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.hitl import request_checkpoint, KNOWN_CHECKPOINTS

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            hitl_level="strict",
            workspace=tmp_path,
        )
        for cp in KNOWN_CHECKPOINTS:
            result = request_checkpoint(
                ctx,
                checkpoint=cp,
                question="?",
                interactive=False,
                default="approve",
                workspace=tmp_path,
            )
            # In strict mode, even the auto path runs (it doesn't skip).
            assert result["mode"] == "auto"

        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_standard_only_enforces_safety_critical(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.hitl import request_checkpoint

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            hitl_level="standard",
            workspace=tmp_path,
        )
        # Safety-critical: should NOT be skipped.
        r1 = request_checkpoint(
            ctx,
            checkpoint="before_destructive_cleanup",
            question="?",
            interactive=False,
            default="approve",
            workspace=tmp_path,
        )
        assert r1["mode"] == "auto"

        # Non-safety-critical (e.g. after_parse): SHOULD be skipped under
        # the "standard" policy.
        r2 = request_checkpoint(
            ctx,
            checkpoint="after_parse",
            question="?",
            interactive=False,
            default="approve",
            workspace=tmp_path,
        )
        assert r2["mode"] == "policy-skipped"

        finalize_run(ctx, status="completed", workspace=tmp_path)
