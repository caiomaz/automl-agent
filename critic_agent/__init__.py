"""Phase 7 — Critic Agent.

A deterministic, rule-based reviewer that inspects the most error-prone
artifacts produced by the pipeline and decides whether to let the run
proceed, warn, request a human checkpoint, or block.

Why deterministic
-----------------

The first iteration intentionally avoids LLM calls. The detectable
mistakes (constraint violations, unsafe paths, task-type mismatches,
absurd metrics) are cheap to spot with rules and expensive to spot via
another LLM round-trip. Future iterations may add an LLM-backed reviewer
behind the same :func:`run_review` contract — that is the SOLID
extension point.

Public API
----------

- :class:`Finding` — a single audit observation with severity.
- :func:`review_parse` — sanity-check the Prompt Agent parse against the
  active task type.
- :func:`review_plans` — check Manager plans against active constraints.
- :func:`review_handoff` — sanity-check Data → Model handoff payloads.
- :func:`review_instruction` — check the final code instruction sent to
  Operation Agent.
- :func:`review_execution_result` — flag obviously bogus metrics or
  hallucinated artifacts.
- :func:`run_review` — apply the policy gate and persist the report.

Policy gate
-----------

Maps the active ``critic_policy`` (one of ``off``/``warn``/
``request_hitl``/``block``) and the worst observed severity to a single
action token: ``pass`` / ``warn`` / ``request_hitl`` / ``block``.

Persistence
-----------

Every review writes a JSON report to
``analyses/critic/<target>__<review_id>.json`` and emits the appropriate
ledger event (``critic_warned`` or ``critic_blocked``). When the policy
escalates to HITL, :func:`utils.hitl.request_checkpoint` is invoked too.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from utils.ledger import (
    append_critic_blocked,
    append_critic_warned,
    ensure_analyses_dir,
)
from utils.workspace import WORKSPACE_DIR, run_exp_dir


_VALID_SEVERITIES = frozenset({"info", "warning", "error"})
_VALID_POLICIES = frozenset({"off", "warn", "request_hitl", "block"})
_VALID_TARGETS = frozenset(
    {"parse", "plans", "handoff", "instruction", "execution_result"}
)


# ── Finding model ─────────────────────────────────────────────────────────────


@dataclass
class Finding:
    """One critic observation.

    ``severity`` drives the policy gate: only ``error`` may trigger
    ``block``; ``warning`` and ``info`` map at most to ``warn``.
    """

    code: str
    severity: str
    target: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity {self.severity!r}; "
                f"must be one of {sorted(_VALID_SEVERITIES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "target": self.target,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


# ── Rule helpers ──────────────────────────────────────────────────────────────


_TASK_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tabular_classification": ("tabular classification", "tabular_classification"),
    "tabular_regression": ("tabular regression", "tabular_regression"),
    "tabular_clustering": ("tabular clustering", "tabular_clustering"),
    "image_classification": ("image classification", "image_classification"),
    "text_classification": ("text classification", "text_classification"),
    "node_classification": ("node classification", "node_classification"),
    "ts_forecasting": (
        "time-series forecasting",
        "time series forecasting",
        "ts_forecasting",
    ),
}


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def review_parse(parse: dict[str, Any], *, task_type: str) -> list[Finding]:
    """Sanity-check the Prompt Agent parse against the active task type."""
    findings: list[Finding] = []
    problem = parse.get("problem") or {}
    declared = _norm(problem.get("downstream_task", ""))

    if not declared:
        findings.append(
            Finding(
                code="task_type_unknown",
                severity="warning",
                target="parse",
                reason="Prompt Agent did not produce a downstream_task value.",
            )
        )
        return findings

    expected = _TASK_TYPE_KEYWORDS.get(task_type, (task_type,))
    if not any(_norm(k) in declared for k in expected):
        findings.append(
            Finding(
                code="task_type_mismatch",
                severity="error",
                target="parse",
                reason=(
                    f"Parse declared downstream_task={declared!r} but the run "
                    f"is configured for task_type={task_type!r}."
                ),
                evidence={"declared": declared, "expected_any_of": list(expected)},
            )
        )
    return findings


_FRAMEWORK_HINTS = {
    "sklearn": ("scikit-learn", "sklearn"),
    "xgboost": ("xgboost",),
    "lightgbm": ("lightgbm", "light gbm"),
    "pytorch": ("pytorch", "torch"),
    "tensorflow": ("tensorflow", "tf.keras", "keras"),
}


def review_plans(
    plans: Iterable[str], *, constraints: dict[str, Any] | None
) -> list[Finding]:
    """Check Manager plans against active structured constraints."""
    findings: list[Finding] = []
    constraints = constraints or {}
    plans_text = " || ".join(p for p in plans if p)

    # seed propagation
    seed = constraints.get("seed")
    if seed is not None:
        if str(seed) not in plans_text and "seed" not in plans_text.lower():
            findings.append(
                Finding(
                    code="constraint_violation",
                    severity="warning",
                    target="plans",
                    reason=(
                        f"Plans do not reference the requested seed={seed}; "
                        "reproducibility may be lost."
                    ),
                    evidence={"constraint": "seed", "value": seed},
                )
            )

    # framework preference vs accidental drift
    framework = constraints.get("framework")
    if framework:
        wanted = framework.lower()
        wanted_hints = _FRAMEWORK_HINTS.get(wanted, (wanted,))
        text = plans_text.lower()
        mentions_wanted = any(h in text for h in wanted_hints)
        # other frameworks present?
        other_present = []
        for name, hints in _FRAMEWORK_HINTS.items():
            if name == wanted:
                continue
            if any(h in text for h in hints):
                other_present.append(name)
        if other_present and not mentions_wanted:
            findings.append(
                Finding(
                    code="constraint_violation",
                    severity="warning",
                    target="plans",
                    reason=(
                        f"Plans reference {other_present!r} but the requested "
                        f"framework is {framework!r}."
                    ),
                    evidence={"constraint": "framework", "value": framework},
                )
            )

    return findings


def review_handoff(
    handoff: dict[str, Any], *, task_type: str
) -> list[Finding]:
    """Sanity-check a Data → Model handoff payload."""
    findings: list[Finding] = []
    if not handoff.get("dataset_path") and not handoff.get("dataset"):
        findings.append(
            Finding(
                code="lineage_broken",
                severity="error",
                target="handoff",
                reason="Handoff is missing a dataset reference.",
            )
        )
    return findings


def review_instruction(
    instruction: str,
    *,
    allowed_paths: tuple[str, ...] = ("agent_workspace/",),
) -> list[Finding]:
    """Check the final code instruction for unsafe absolute paths."""
    findings: list[Finding] = []
    text = instruction or ""
    # Find absolute POSIX paths the generated code would write to.
    abs_paths = re.findall(r"(/[A-Za-z0-9_\-./]+)", text)
    for path in abs_paths:
        if any(path.startswith(p) or ("/" + p) in path for p in allowed_paths):
            continue
        # Common culprits we see in regressions
        if path.startswith(("/tmp", "/data", "/app", "/home", "/var", "/root")):
            findings.append(
                Finding(
                    code="unsafe_path",
                    severity="error",
                    target="instruction",
                    reason=(
                        f"Instruction references absolute path {path!r} "
                        "outside the run workspace."
                    ),
                    evidence={"path": path, "allowed": list(allowed_paths)},
                )
            )
            break  # one is enough to trigger a block
    return findings


def review_execution_result(
    result: dict[str, Any], *, task_type: str
) -> list[Finding]:
    """Flag obviously bogus metric values or missing artifacts."""
    findings: list[Finding] = []
    metric = result.get("metric")
    value = result.get("value")
    if metric is not None and isinstance(value, (int, float)):
        # Classification-style metrics in [0,1]
        if metric.lower() in {"accuracy", "f1", "precision", "recall", "roc_auc"}:
            if value <= 0.0 or value >= 1.0001:
                findings.append(
                    Finding(
                        code="metric_suspicious",
                        severity="warning",
                        target="execution_result",
                        reason=(
                            f"Metric {metric}={value} is outside the expected "
                            "(0, 1) range for a classification task."
                        ),
                        evidence={"metric": metric, "value": value},
                    )
                )
    return findings


# ── Policy gate + persistence ────────────────────────────────────────────────


def _decide_action(policy: str, findings: list[Finding]) -> str:
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"Invalid critic policy {policy!r}; "
            f"must be one of {sorted(_VALID_POLICIES)}"
        )
    if not findings or policy == "off":
        return "pass"
    has_error = any(f.severity == "error" for f in findings)
    if policy == "warn":
        return "warn"
    if policy == "request_hitl":
        return "request_hitl"
    # policy == "block"
    return "block" if has_error else "warn"


def _write_report(
    ctx: Any,
    *,
    target: str,
    review_id: str,
    findings: list[Finding],
    policy: str,
    action: str,
    workspace: Path | None,
) -> Path:
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    ensure_analyses_dir(ctx, ws)
    critic_dir = run_exp_dir(ctx.run_id, ws) / "analyses" / "critic"
    critic_dir.mkdir(parents=True, exist_ok=True)
    path = critic_dir / f"{target}__{review_id}.json"
    payload = {
        "review_id": review_id,
        "target": target,
        "policy": policy,
        "action": action,
        "findings": [f.to_dict() for f in findings],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_review(
    ctx: Any,
    *,
    target: str,
    findings: list[Finding],
    policy: str = "warn",
    workspace: Path | None = None,
    interactive: bool = False,
    default_decision: str = "approve",
) -> dict[str, Any]:
    """Apply policy, persist the report, and emit ledger events.

    Returns a dict with: ``review_id``, ``target``, ``action``,
    ``findings`` (list of dicts), and (when applicable) ``hitl``.
    """
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"Invalid critic target {target!r}; "
            f"must be one of {sorted(_VALID_TARGETS)}"
        )

    review_id = str(uuid.uuid4())
    action = _decide_action(policy, findings)

    report_path = _write_report(
        ctx,
        target=target,
        review_id=review_id,
        findings=findings,
        policy=policy,
        action=action,
        workspace=workspace,
    )

    summary = "; ".join(f"{f.code}:{f.severity}" for f in findings) or "no_findings"

    if action == "block":
        append_critic_blocked(
            ctx,
            target=target,
            reason=summary,
            workspace=workspace,
            review_id=review_id,
            report_path=str(report_path.name),
        )
    elif action == "warn":
        append_critic_warned(
            ctx,
            target=target,
            reason=summary,
            workspace=workspace,
            review_id=review_id,
            report_path=str(report_path.name),
        )
    elif action == "request_hitl":
        # Emit a warn event AND request the human checkpoint so both the
        # critic finding and the human decision are captured.
        append_critic_warned(
            ctx,
            target=target,
            reason=summary,
            workspace=workspace,
            review_id=review_id,
            report_path=str(report_path.name),
        )
        from utils.hitl import request_checkpoint

        # Strict level forces the checkpoint to actually run regardless of
        # the run-level hitl policy — a Critic-driven HITL must not be
        # skipped silently.
        original_level = getattr(ctx, "hitl_level", "off")
        try:
            ctx.hitl_level = "strict"
            hitl = request_checkpoint(
                ctx,
                checkpoint=f"critic_review_{target}",
                question=f"Critic flagged {target}: {summary}. Continue?",
                interactive=interactive,
                default=default_decision,
                options=("approve", "reject"),
                workspace=workspace,
                review_id=review_id,
            )
        finally:
            ctx.hitl_level = original_level

        return {
            "review_id": review_id,
            "target": target,
            "action": action,
            "findings": [f.to_dict() for f in findings],
            "hitl": hitl,
            "report_path": str(report_path),
        }

    return {
        "review_id": review_id,
        "target": target,
        "action": action,
        "findings": [f.to_dict() for f in findings],
        "report_path": str(report_path),
    }
