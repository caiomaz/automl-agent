"""Dataset provenance — Phase 1 / ADR-007.

Captures where a dataset came from for each run, the local landing path,
and an optional checksum so the same input can be verified later.

Provenance is appended to ``analyses/dataset_provenance.json`` (a JSON
list) and a ``dataset_recorded`` event is emitted to the run ledger.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.ledger import append_event, ensure_analyses_dir
from utils.workspace import WORKSPACE_DIR

# Allowed dataset entry modes (mirrors docs/06_WORKSPACE_AND_DATASETS.md).
_VALID_MODES = frozenset({"manual-upload", "user-link", "auto-retrieval"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DatasetProvenance:
    """Structured provenance record for a single dataset entry.

    Attributes
    ----------
    mode:
        One of ``"manual-upload"``, ``"user-link"``, ``"auto-retrieval"``.
    source:
        Origin reference — local path, URL, Kaggle competition slug, etc.
    local_path:
        Resolved on-disk location used by downstream agents.
    checksum_sha256:
        Optional SHA-256 of the file content. Use ``compute_checksum``
        before recording, or pass ``compute_checksum_now=True`` to
        ``record_provenance``.
    note:
        Optional free-form context.
    recorded_at:
        ISO-8601 UTC timestamp; auto-set if omitted.
    """

    mode: str
    source: str
    local_path: str
    checksum_sha256: str | None = None
    note: str | None = None
    recorded_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid provenance mode '{self.mode}'. "
                f"Must be one of: {sorted(_VALID_MODES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_checksum(path: Path | str, *, chunk_size: int = 65536) -> str:
    """Return the full SHA-256 hex digest of a file."""
    p = Path(path)
    digest = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def record_provenance(
    ctx: Any,
    provenance: DatasetProvenance,
    *,
    workspace: Path | None = None,
    compute_checksum_now: bool = False,
) -> Path:
    """Append a provenance record for the active run.

    The record is appended to ``analyses/dataset_provenance.json`` (the
    file holds a JSON list) and a ``dataset_recorded`` event is emitted.

    When ``compute_checksum_now=True`` and ``provenance.checksum_sha256``
    is empty, the file at ``provenance.local_path`` is hashed.

    Returns the path to the provenance JSON file.
    """
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR

    if compute_checksum_now and not provenance.checksum_sha256:
        try:
            provenance.checksum_sha256 = compute_checksum(provenance.local_path)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            # Hash is best-effort — don't break the run on missing files.
            provenance.checksum_sha256 = None

    analyses_dir = ensure_analyses_dir(ctx, workspace=ws)
    prov_path = analyses_dir / "dataset_provenance.json"

    existing: list[dict] = []
    if prov_path.exists():
        try:
            loaded = json.loads(prov_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except json.JSONDecodeError:
            existing = []

    existing.append(provenance.to_dict())
    prov_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    append_event(
        ctx, "dataset_recorded",
        source="data",
        payload_summary=f"mode={provenance.mode} source={provenance.source}",
        payload_ref=provenance.local_path,
        workspace=ws,
        mode=provenance.mode,
        checksum_sha256=provenance.checksum_sha256,
    )
    return prov_path
