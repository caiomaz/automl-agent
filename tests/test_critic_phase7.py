"""Phase 7 — Critic Agent (TDD).

Tests for ``critic_agent`` covering:

- Findings model (severity + code + target + reason).
- Rule-based checks per target (parse / plans / handoff / instruction
  / execution-result).
- Policy gate (``off``/``warn``/``request_hitl``/``block``) producing
  the right action and the right ledger events.
- Persistence: every review writes ``analyses/critic/<target>__<id>.json``.
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
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line]


# ── Finding model ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFinding:
    def test_finding_serializes(self):
        from critic_agent import Finding

        f = Finding(
            code="constraint_violation",
            severity="error",
            target="plans",
            reason="seed=42 not propagated",
        )
        d = f.to_dict()
        assert d["code"] == "constraint_violation"
        assert d["severity"] == "error"
        assert d["target"] == "plans"
        assert d["reason"] == "seed=42 not propagated"

    def test_invalid_severity_rejected(self):
        from critic_agent import Finding

        with pytest.raises(ValueError, match="severity"):
            Finding(code="x", severity="catastrophic", target="parse", reason="r")


# ── Rule checks ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestParseRules:
    def test_parse_with_unknown_task_flags_inconsistency(self):
        from critic_agent import review_parse

        parse = {"problem": {"downstream_task": ""}}
        findings = review_parse(parse, task_type="tabular_classification")
        codes = [f.code for f in findings]
        assert "task_type_unknown" in codes

    def test_parse_with_mismatched_task_type_flags_inconsistency(self):
        from critic_agent import review_parse

        parse = {"problem": {"downstream_task": "image classification"}}
        findings = review_parse(parse, task_type="tabular_regression")
        codes = [f.code for f in findings]
        assert "task_type_mismatch" in codes

    def test_parse_aligned_returns_no_findings(self):
        from critic_agent import review_parse

        parse = {"problem": {"downstream_task": "tabular regression"}}
        findings = review_parse(parse, task_type="tabular_regression")
        assert findings == []


@pytest.mark.unit
class TestPlansRules:
    def test_plan_missing_constraint_seed_flagged(self):
        from critic_agent import review_plans

        plans = ["Train an XGBoost model with default settings."]
        findings = review_plans(plans, constraints={"seed": 42})
        codes = [f.code for f in findings]
        assert "constraint_violation" in codes

    def test_plan_with_constraint_seed_in_text_passes(self):
        from critic_agent import review_plans

        plans = ["Train an XGBoost model with seed=42 for reproducibility."]
        findings = review_plans(plans, constraints={"seed": 42})
        codes = [f.code for f in findings]
        assert "constraint_violation" not in codes

    def test_plan_referring_to_disallowed_framework_flagged(self):
        from critic_agent import review_plans

        plans = ["Use TensorFlow Keras to build the model."]
        findings = review_plans(plans, constraints={"framework": "lightgbm"})
        codes = [f.code for f in findings]
        assert "constraint_violation" in codes


@pytest.mark.unit
class TestInstructionRules:
    def test_instruction_with_absolute_unsafe_path_flagged(self):
        from critic_agent import review_instruction

        instr = "Save the model to /tmp/project/output.pkl"
        findings = review_instruction(instr, allowed_paths=("agent_workspace/",))
        codes = [f.code for f in findings]
        assert "unsafe_path" in codes

    def test_instruction_using_workspace_path_is_clean(self):
        from critic_agent import review_instruction

        instr = "Save the model to agent_workspace/trained_models/model.pkl"
        findings = review_instruction(instr, allowed_paths=("agent_workspace/",))
        assert findings == []


@pytest.mark.unit
class TestExecutionResultRules:
    def test_zero_metric_flagged(self):
        from critic_agent import review_execution_result

        findings = review_execution_result(
            {"metric": "accuracy", "value": 0.0}, task_type="tabular_classification"
        )
        codes = [f.code for f in findings]
        assert "metric_suspicious" in codes

    def test_normal_metric_clean(self):
        from critic_agent import review_execution_result

        findings = review_execution_result(
            {"metric": "accuracy", "value": 0.93}, task_type="tabular_classification"
        )
        assert findings == []


# ── Policy decisions + persistence ───────────────────────────────────────────


@pytest.mark.unit
class TestPolicyDecisions:
    def test_off_policy_returns_pass_even_with_findings(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from critic_agent import run_review, Finding

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        result = run_review(
            ctx,
            target="parse",
            findings=[Finding("x", "error", "parse", "r")],
            policy="off",
            workspace=tmp_path,
        )
        assert result["action"] == "pass"
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_warn_policy_emits_critic_warned(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from critic_agent import run_review, Finding

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        result = run_review(
            ctx,
            target="plans",
            findings=[Finding("constraint_violation", "warning", "plans", "no seed")],
            policy="warn",
            workspace=tmp_path,
        )
        assert result["action"] == "warn"
        evs = _events(tmp_path, ctx.run_id)
        assert any(e["event"] == "critic_warned" for e in evs)
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_block_policy_with_error_emits_critic_blocked(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from critic_agent import run_review, Finding

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        result = run_review(
            ctx,
            target="instruction",
            findings=[Finding("unsafe_path", "error", "instruction", "/tmp/foo")],
            policy="block",
            workspace=tmp_path,
        )
        assert result["action"] == "block"
        evs = _events(tmp_path, ctx.run_id)
        assert any(e["event"] == "critic_blocked" for e in evs)
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_block_policy_with_only_warnings_falls_back_to_warn(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from critic_agent import run_review, Finding

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        result = run_review(
            ctx,
            target="plans",
            findings=[Finding("c", "warning", "plans", "soft issue")],
            policy="block",
            workspace=tmp_path,
        )
        # No error-severity finding → block downgraded to warn.
        assert result["action"] == "warn"
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_request_hitl_policy_emits_hitl_request(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from critic_agent import run_review, Finding

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            hitl_level="strict",
            workspace=tmp_path,
        )
        result = run_review(
            ctx,
            target="parse",
            findings=[Finding("task_type_mismatch", "error", "parse", "x")],
            policy="request_hitl",
            workspace=tmp_path,
            interactive=False,
            default_decision="approve",
        )
        assert result["action"] == "request_hitl"
        evs = _events(tmp_path, ctx.run_id)
        names = [e["event"] for e in evs]
        assert "critic_warned" in names or "critic_blocked" in names
        assert "hitl_requested" in names
        assert "hitl_resolved" in names
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_no_findings_returns_pass(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from critic_agent import run_review

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        result = run_review(
            ctx, target="parse", findings=[], policy="block", workspace=tmp_path,
        )
        assert result["action"] == "pass"
        finalize_run(ctx, status="completed", workspace=tmp_path)


@pytest.mark.unit
class TestPersistence:
    def test_review_writes_critic_report(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.workspace import run_exp_dir
        from critic_agent import run_review, Finding

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        result = run_review(
            ctx,
            target="plans",
            findings=[Finding("constraint_violation", "warning", "plans", "no seed")],
            policy="warn",
            workspace=tmp_path,
        )
        report_path = (
            run_exp_dir(ctx.run_id, tmp_path)
            / "analyses"
            / "critic"
            / f"plans__{result['review_id']}.json"
        )
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["target"] == "plans"
        assert report["action"] == "warn"
        assert report["policy"] == "warn"
        assert len(report["findings"]) == 1
        finalize_run(ctx, status="completed", workspace=tmp_path)
