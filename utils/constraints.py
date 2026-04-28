"""Phase 5 — Granular constraints normalization and persistence.

The CLI and Prompt Agent both produce constraint dictionaries with a
mix of legacy keys (``model``, ``perf_metric``, ``perf_value``,
``max_train_time``, ``max_inference_time``) and a documented set of
granular keys recommended by ``TASKS.md`` (split policy, seed, framework
preference, fairness/explainability, HITL/Critic/cleanup/token-economy
policies).

This module:

1. :data:`KNOWN_CONSTRAINT_KEYS` enumerates the accepted keys.
2. :func:`normalize_constraints` validates/filters incoming dicts.
3. :func:`persist_constraints` merges them into ``run_manifest.json``,
   writes a copy to ``analyses/constraints.json``, and emits a
   ``constraints_recorded`` event so downstream agents can audit which
   constraints were active.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.ledger import append_event, ensure_analyses_dir
from utils.workspace import WORKSPACE_DIR, run_exp_dir


# Documented enums
_VALID_HITL_POLICIES = frozenset({"off", "standard", "strict"})
_VALID_CLEANUP_POLICIES = frozenset({"preserve", "archive", "purge"})
_VALID_CRITIC_POLICIES = frozenset({"off", "warn", "request_hitl", "block"})
_VALID_TOKEN_ECONOMY = frozenset({"off", "moderate", "aggressive"})
_VALID_SPLIT_POLICIES = frozenset(
    {"holdout", "k-fold", "stratified-k-fold", "group-split", "time-split"}
)


KNOWN_CONSTRAINT_KEYS: frozenset[str] = frozenset(
    {
        # Legacy (kept for backward compat with the CLI prompt builder)
        "model",
        "perf_metric",
        "perf_value",
        "max_train_time",
        "max_inference_time",
        # Phase 5 — granular
        "split_policy",
        "seed",
        "max_memory",
        "max_storage",
        "allowed_packages",
        "framework",
        "explainability_required",
        "fairness_required",
        "deploy_required",
        "required_artifacts",
        "concurrency_policy",
        "hitl_policy",
        "critic_policy",
        "cleanup_policy",
        "token_economy",
    }
)


def normalize_constraints(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and filter a constraints dict.

    - Unknown keys are silently dropped (forward compatibility).
    - Enum values raise ``ValueError`` when invalid.
    - Empty input returns an empty dict.
    """
    if not raw:
        return {}

    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in KNOWN_CONSTRAINT_KEYS:
            continue
        out[key] = value

    if "hitl_policy" in out and out["hitl_policy"] not in _VALID_HITL_POLICIES:
        raise ValueError(
            f"Invalid hitl_policy {out['hitl_policy']!r}; "
            f"must be one of {sorted(_VALID_HITL_POLICIES)}"
        )
    if "cleanup_policy" in out and out["cleanup_policy"] not in _VALID_CLEANUP_POLICIES:
        raise ValueError(
            f"Invalid cleanup_policy {out['cleanup_policy']!r}; "
            f"must be one of {sorted(_VALID_CLEANUP_POLICIES)}"
        )
    if "critic_policy" in out and out["critic_policy"] not in _VALID_CRITIC_POLICIES:
        raise ValueError(
            f"Invalid critic_policy {out['critic_policy']!r}; "
            f"must be one of {sorted(_VALID_CRITIC_POLICIES)}"
        )
    if "token_economy" in out and out["token_economy"] not in _VALID_TOKEN_ECONOMY:
        raise ValueError(
            f"Invalid token_economy {out['token_economy']!r}; "
            f"must be one of {sorted(_VALID_TOKEN_ECONOMY)}"
        )
    if "split_policy" in out and out["split_policy"] not in _VALID_SPLIT_POLICIES:
        raise ValueError(
            f"Invalid split_policy {out['split_policy']!r}; "
            f"must be one of {sorted(_VALID_SPLIT_POLICIES)}"
        )

    return out


def persist_constraints(
    ctx: Any,
    raw: dict[str, Any] | None,
    *,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Persist a constraints block for the active run.

    Side effects (only when the normalized dict is non-empty):

    1. merge ``constraints`` into ``run_manifest.json``,
    2. write ``analyses/constraints.json``,
    3. emit a ``constraints_recorded`` event listing the recorded keys.

    Returns the normalized dict (empty when ``raw`` is empty).
    """
    normalized = normalize_constraints(raw)
    if not normalized:
        return normalized

    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    run_dir = run_exp_dir(ctx.run_id, ws)

    # 1) Merge into manifest
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    existing = manifest.get("constraints") or {}
    if isinstance(existing, dict):
        existing.update(normalized)
        manifest["constraints"] = existing
    else:
        manifest["constraints"] = dict(normalized)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 2) Write analyses/constraints.json
    ensure_analyses_dir(ctx, ws)
    analyses_path = run_dir / "analyses" / "constraints.json"
    analyses_path.write_text(
        json.dumps(manifest["constraints"], indent=2), encoding="utf-8"
    )

    # 3) Emit event with the keys recorded (values may be sensitive paths,
    #    so we keep the event payload as the key list).
    append_event(
        ctx,
        "constraints_recorded",
        source="manager",
        workspace=ws,
        keys=sorted(normalized.keys()),
    )

    return normalized
