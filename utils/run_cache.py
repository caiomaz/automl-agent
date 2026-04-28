"""Phase 8 — Run-scoped content-addressed cache.

Reuses identical-input results (parse, retrieval, summaries) within a
single run so we never pay for the same LLM call twice.

The cache is intentionally simple:

- keys are SHA-256 hashes of joined parts (``make_key``);
- values are JSON-serializable Python objects;
- entries are kept in-memory for fast hits and mirrored to
  ``exp/runs/<id>/cache/<key>.json`` for post-run inspection;
- every hit emits a ``tokens_saved`` event with the recorded
  ``tokens_estimate`` so savings are auditable.

This is run-scoped on purpose: cross-run cache sharing would require
content provenance, which is out of scope for Phase 8.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.token_economy import record_tokens_saved
from utils.workspace import WORKSPACE_DIR, run_exp_dir


def make_key(*parts: str) -> str:
    """Deterministic content-addressed key for ``parts``."""
    h = hashlib.sha256()
    for p in parts:
        h.update(b"\x00")
        h.update((p or "").encode("utf-8", errors="replace"))
    return h.hexdigest()


class RunCache:
    """In-memory + on-disk cache scoped to a single run."""

    def __init__(self, ctx: Any, *, workspace: Path | None = None) -> None:
        self._ctx = ctx
        self._ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
        self._mem: dict[str, dict[str, Any]] = {}
        self._dir = run_exp_dir(ctx.run_id, self._ws) / "cache"
        self._dir.mkdir(parents=True, exist_ok=True)
        # Warm in-memory layer from disk so cache survives restarts.
        for f in self._dir.glob("*.json"):
            try:
                self._mem[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue

    def set(self, key: str, value: Any, *, tokens_estimate: int = 0) -> None:
        """Store ``value`` under ``key`` and persist the entry."""
        record = {"value": value, "tokens_estimate": int(tokens_estimate)}
        self._mem[key] = record
        path = self._dir / f"{key}.json"
        path.write_text(json.dumps(record), encoding="utf-8")

    def get(self, key: str, *, stage: str | None = None) -> Any | None:
        """Return the cached value or ``None``. Records savings on hit."""
        record = self._mem.get(key)
        if record is None:
            return None
        record_tokens_saved(
            self._ctx,
            source="cache_hit",
            saved_tokens=record.get("tokens_estimate", 0),
            workspace=self._ws,
            stage=stage or "unknown",
            cache_key=key,
        )
        return record["value"]
