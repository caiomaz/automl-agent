"""Phase 6 — Strategic Human-In-The-Loop checkpoints.

Provides :func:`request_checkpoint`, a single entry point that:

1. honors the run-level ``hitl_level`` policy (``off``/``standard``/``strict``),
2. prompts the user when running interactively and the policy enforces
   the checkpoint,
3. auto-approves with the documented default in non-interactive mode,
4. emits ``hitl_requested`` + ``hitl_resolved`` events sharing the same
   ``hitl_id`` for full audit trail.

Known checkpoints (see TASKS.md §6):

- ``after_parse``
- ``after_plans``
- ``before_critic_override``
- ``before_code_generation_high_risk``
- ``before_destructive_cleanup``
- ``before_deploy``
- ``before_final_acceptance_on_conflict``

The "safety-critical" subset (cleanup that destroys files, public deploy)
is enforced even at ``hitl_level="standard"``. Other checkpoints are only
enforced at ``hitl_level="strict"``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from utils.ledger import append_hitl_requested, append_hitl_resolved


KNOWN_CHECKPOINTS: tuple[str, ...] = (
    "after_parse",
    "after_plans",
    "before_critic_override",
    "before_code_generation_high_risk",
    "before_destructive_cleanup",
    "before_deploy",
    "before_final_acceptance_on_conflict",
)

_SAFETY_CRITICAL: frozenset[str] = frozenset(
    {"before_destructive_cleanup", "before_deploy"}
)


def _enforced(hitl_level: str, checkpoint: str) -> bool:
    """Return True when the checkpoint should actually run under the policy."""
    if hitl_level == "strict":
        return True
    if hitl_level == "standard":
        return checkpoint in _SAFETY_CRITICAL
    # "off"
    return False


def request_checkpoint(
    ctx: Any,
    *,
    checkpoint: str,
    question: str,
    interactive: bool,
    default: str,
    options: Iterable[str] = ("approve", "reject"),
    input_fn: Callable[[str], str] | None = None,
    workspace: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Run one HITL checkpoint and return the structured decision.

    Parameters
    ----------
    ctx:
        Active ``RunContext``. ``ctx.hitl_level`` selects the policy.
    checkpoint:
        Logical name (see :data:`KNOWN_CHECKPOINTS`). Custom names are
        allowed but ignored by the policy gate.
    question:
        Human-readable prompt shown to the user / recorded in the ledger.
    interactive:
        Whether the surface (CLI) can prompt the user. Pass ``False``
        for the non-interactive ``cli run`` flow.
    default:
        Decision used when the policy skips the checkpoint, when no
        ``input_fn`` is provided in interactive mode, or when the user
        replies with an invalid token.
    options:
        Accepted decision tokens. ``default`` must be one of them.
    input_fn:
        Callable taking the rendered prompt and returning the user reply.
        Defaults to :func:`builtins.input`.
    workspace:
        Optional workspace override for tests.

    Returns
    -------
    dict with keys: ``checkpoint``, ``decision``, ``mode``, ``hitl_id``.
    ``mode`` is one of:

    - ``"policy-skipped"`` — the run policy bypassed this checkpoint;
    - ``"auto"`` — non-interactive surface, default applied;
    - ``"human"`` — user replied with a valid option;
    - ``"human-fallback"`` — user replied with an invalid token, default
      applied.
    """
    options = tuple(options)
    if default not in options:
        raise ValueError(
            f"default {default!r} must be one of options {options!r}"
        )

    hitl_id = str(uuid.uuid4())
    hitl_level = getattr(ctx, "hitl_level", "off")

    # Always emit the request event so the ledger reflects intent.
    append_hitl_requested(
        ctx,
        checkpoint=checkpoint,
        question=question,
        workspace=workspace,
        hitl_id=hitl_id,
        options=list(options),
        default=default,
        hitl_level=hitl_level,
        **extra,
    )

    # Policy gate
    if not _enforced(hitl_level, checkpoint):
        decision, mode = default, "policy-skipped"
    elif not interactive or input_fn is None and not interactive:
        decision, mode = default, "auto"
    else:
        if not interactive:
            decision, mode = default, "auto"
        else:
            fn = input_fn if input_fn is not None else input
            prompt_text = f"[HITL/{checkpoint}] {question} {list(options)}: "
            try:
                reply = (fn(prompt_text) or "").strip().lower()
            except (EOFError, KeyboardInterrupt):
                reply = ""
            if reply in options:
                decision, mode = reply, "human"
            else:
                decision, mode = default, "human-fallback"

    append_hitl_resolved(
        ctx,
        checkpoint=checkpoint,
        decision=decision,
        workspace=workspace,
        hitl_id=hitl_id,
        mode=mode,
        hitl_level=hitl_level,
    )

    return {
        "checkpoint": checkpoint,
        "decision": decision,
        "mode": mode,
        "hitl_id": hitl_id,
    }
