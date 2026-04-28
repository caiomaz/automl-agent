"""Phase 8 — Token economy helpers.

Three responsibilities, deliberately split (SRP):

1. :func:`normalize_policy` validates and canonicalizes the
   ``token_economy`` constraint value.
2. :func:`truncate_payload` and :func:`summarize_error` shrink long
   blobs before they're embedded in prompts. Both return ``(text,
   info)`` so callers can record how many characters/lines were saved.
3. :func:`dynamic_n_plans` adjusts the planning fan-out based on the
   active policy and an optional confidence signal.
4. :func:`record_tokens_saved` emits a ``tokens_saved`` ledger event so
   savings from cache hits, fallbacks, or dynamic budgets are auditable
   alongside the cost ledger.

The numeric thresholds are intentionally conservative; aggressive mode
is meant to noticeably trim payloads without losing the head/tail
context that makes errors and plans interpretable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.ledger import append_event


_VALID_POLICIES = frozenset({"off", "moderate", "aggressive"})


# ── Policy ────────────────────────────────────────────────────────────────────


def normalize_policy(value: str | None) -> str:
    """Return a validated policy string. ``None`` becomes ``"off"``."""
    if value is None or value == "":
        return "off"
    if value not in _VALID_POLICIES:
        raise ValueError(
            f"Invalid token_economy policy {value!r}; "
            f"must be one of {sorted(_VALID_POLICIES)}"
        )
    return value


# ── Payload truncation ───────────────────────────────────────────────────────


# Per-policy maximum characters for free-form payloads (handoff blobs,
# data/model results, etc.). Aggressive trims roughly to half of moderate.
_PAYLOAD_LIMITS = {"off": None, "moderate": 8000, "aggressive": 4000}


def truncate_payload(text: str, *, policy: str = "off") -> tuple[str, dict[str, Any]]:
    """Truncate a long payload, keeping the head with a clear marker."""
    policy = normalize_policy(policy)
    if text is None:
        return "", {"truncated": False, "saved_chars": 0}
    limit = _PAYLOAD_LIMITS[policy]
    if limit is None or len(text) <= limit:
        return text, {"truncated": False, "saved_chars": 0}
    head = text[: limit - 80]
    saved = len(text) - len(head)
    out = head + f"\n... [truncated {saved} chars; policy={policy}] ..."
    return out, {"truncated": True, "saved_chars": saved, "policy": policy}


# ── Error summarization (head + tail) ────────────────────────────────────────


_ERROR_BUDGET = {
    "off": None,
    "moderate": (40, 60),  # (head_lines, tail_lines)
    "aggressive": (15, 25),
}


def summarize_error(
    stderr: str, *, policy: str = "off"
) -> tuple[str, dict[str, Any]]:
    """Keep the first N + last M lines of an error trace.

    The tail almost always contains the actual exception; the head holds
    useful framework warnings. The middle is usually verbose Python
    tracebacks that the LLM does not need to retry effectively.
    """
    policy = normalize_policy(policy)
    if stderr is None:
        return "", {"truncated": False, "original_lines": 0, "kept_lines": 0}

    budget = _ERROR_BUDGET[policy]
    lines = stderr.splitlines()
    if budget is None or len(lines) <= sum(budget):
        return stderr, {
            "truncated": False,
            "original_lines": len(lines),
            "kept_lines": len(lines),
        }

    head_n, tail_n = budget
    head = lines[:head_n]
    tail = lines[-tail_n:]
    omitted = len(lines) - head_n - tail_n
    out = "\n".join(
        head
        + [f"... [omitted {omitted} lines; policy={policy}] ..."]
        + tail
    )
    return out, {
        "truncated": True,
        "original_lines": len(lines),
        "kept_lines": head_n + tail_n,
        "policy": policy,
    }


# ── Dynamic budget for planning fan-out ──────────────────────────────────────


def dynamic_n_plans(
    *, default: int, policy: str = "off", confidence: float | None = None
) -> int:
    """Reduce ``n_plans`` when the policy permits and confidence is high.

    - ``off``: never reduce.
    - ``moderate``: reduce by 1 when confidence ≥ 0.85.
    - ``aggressive``: reduce by 2 when confidence ≥ 0.85, otherwise
      reduce by 1 when confidence ≥ 0.65.

    The result is always ≥ 1.
    """
    policy = normalize_policy(policy)
    if policy == "off" or default <= 1:
        return max(1, default)

    if confidence is None:
        return default

    reduction = 0
    if policy == "moderate" and confidence >= 0.85:
        reduction = 1
    elif policy == "aggressive":
        if confidence >= 0.85:
            reduction = 2
        elif confidence >= 0.65:
            reduction = 1
    return max(1, default - reduction)


# ── Ledger event ─────────────────────────────────────────────────────────────


def record_tokens_saved(
    ctx: Any,
    *,
    source: str,
    saved_tokens: int,
    workspace: Path | None = None,
    **extra: Any,
) -> None:
    """Emit a ``tokens_saved`` event for auditing.

    ``source`` is a short tag like ``"cache_hit"``, ``"truncation"``,
    ``"dynamic_n_plans"``, or ``"stage_routing_fallback"``.
    """
    if ctx is None:
        return
    append_event(
        ctx,
        "tokens_saved",
        source=source,
        saved_tokens=int(saved_tokens),
        workspace=workspace,
        **extra,
    )
