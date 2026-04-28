"""Phase 8 — Per-stage LLM routing.

Allows each pipeline stage to pick a cheaper or stronger model than the
global ``LLM_BACKBONE`` via ``LLM_STAGE_<NAME>`` environment variables.

Stages are intentionally enumerated (``KNOWN_STAGES``) so unknown names
fail loudly instead of silently falling back to the default. The
function only resolves an alias — translating that alias to a provider
slug remains the responsibility of :mod:`configs`.
"""

from __future__ import annotations

import os


KNOWN_STAGES: frozenset[str] = frozenset(
    {
        "prompt_parse",
        "planning",
        "critic",
        "code_generation",
        "verification",
        "summary",
    }
)


def _env_var(stage: str) -> str:
    return f"LLM_STAGE_{stage.upper()}"


def resolve_stage_alias(stage: str, *, default: str) -> str:
    """Return the LLM alias to use for ``stage``.

    Reads ``LLM_STAGE_<STAGE>`` from the environment if set, otherwise
    falls back to ``default``.
    """
    if stage not in KNOWN_STAGES:
        raise ValueError(
            f"Unknown stage {stage!r}; must be one of {sorted(KNOWN_STAGES)}"
        )
    return os.getenv(_env_var(stage)) or default


def current_routing_map(*, default: str) -> dict[str, str]:
    """Snapshot of the currently effective stage→alias mapping."""
    return {stage: resolve_stage_alias(stage, default=default) for stage in KNOWN_STAGES}
