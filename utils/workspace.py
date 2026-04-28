"""Canonical workspace paths and helpers.

All agents and CLI components must use these constants so the workspace
layout stays consistent regardless of the current working directory.

Standard structure
------------------
agent_workspace/
    datasets/          ← manual uploads + URL-cached downloads
    exp/               ← generated scripts and experiment outputs
    trained_models/    ← saved model checkpoints
"""

import re
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Project root is two levels above this file:  utils/workspace.py → utils/ → project/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE_DIR = _PROJECT_ROOT / "agent_workspace"
DATASETS_DIR  = WORKSPACE_DIR / "datasets"
EXP_DIR       = WORKSPACE_DIR / "exp"
MODELS_DIR    = WORKSPACE_DIR / "trained_models"


def ensure_workspace(workspace: Path | None = None) -> None:
    """Create all standard workspace subdirectories if they don't exist.

    Parameters
    ----------
    workspace:
        Override the default ``WORKSPACE_DIR`` (useful in tests).
        Can be any ``Path``-like value; will be converted with ``Path()``.
    """
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    for sub in ("datasets", "exp", "trained_models"):
        (ws / sub).mkdir(parents=True, exist_ok=True)


# ── Run-namespaced path helpers (Phase 1 / ADR-007) ──────────────────────────


def run_datasets_dir(run_id: str, workspace: Path | None = None) -> Path:
    """Return ``<workspace>/datasets/runs/<run_id>``."""
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    return ws / "datasets" / "runs" / run_id


def run_exp_dir(run_id: str, workspace: Path | None = None) -> Path:
    """Return ``<workspace>/exp/runs/<run_id>``."""
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    return ws / "exp" / "runs" / run_id


def run_models_dir(run_id: str, workspace: Path | None = None) -> Path:
    """Return ``<workspace>/trained_models/runs/<run_id>``."""
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    return ws / "trained_models" / "runs" / run_id


def datasets_cache_dir(workspace: Path | None = None) -> Path:
    """Return ``<workspace>/datasets/cache``."""
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    return ws / "datasets" / "cache"


def ensure_run_workspace(run_id: str, workspace: Path | None = None) -> None:
    """Create all run-namespaced subdirectories and the shared cache dir.

    This does NOT touch the canonical flat dirs — those are managed by
    ``ensure_workspace()``.
    """
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    for sub_fn in (run_datasets_dir, run_exp_dir, run_models_dir):
        sub_fn(run_id, workspace=ws).mkdir(parents=True, exist_ok=True)
    datasets_cache_dir(workspace=ws).mkdir(parents=True, exist_ok=True)


def dataset_path_for_url(
    url: str,
    name: str = "",
    datasets_dir: Path | None = None,
) -> Path:
    """Return a stable, unique local ``Path`` for a URL-sourced dataset.

    The path is deterministic — the same URL always maps to the same
    directory, enabling transparent caching. Different URLs get different
    directories, so concurrent runs never share downloaded data.

    Format: ``<datasets_dir>/<safe_name>_<sha256_prefix_8>/``

    Parameters
    ----------
    url:
        Source URL of the dataset.
    name:
        Human-readable label (e.g. ``"banana_quality"``). Sanitised for
        filesystem safety.
    datasets_dir:
        Directory that contains dataset sub-folders. Defaults to
        ``DATASETS_DIR``. Pass a custom path for testing.
    """
    d = Path(datasets_dir) if datasets_dir is not None else DATASETS_DIR
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    # Keep only word characters and hyphens; collapse repeated underscores
    safe = re.sub(r"[^\w-]", "_", name.strip())
    safe = re.sub(r"_+", "_", safe).strip("_") or "dataset"
    return d / f"{safe}_{url_hash}"


# ── Cleanup policies (Phase 1 / ADR-007) ─────────────────────────────────────


_VALID_CLEANUP_MODES = frozenset({"preserve", "archive", "purge"})


def _live_run_subtrees(workspace: Path) -> list[Path]:
    """Return all existing ``runs/<run_id>`` directories under the live area."""
    out: list[Path] = []
    for sub in ("datasets", "exp", "trained_models"):
        runs_root = workspace / sub / "runs"
        if not runs_root.exists():
            continue
        for child in runs_root.iterdir():
            if child.is_dir():
                out.append(child)
    return out


def cleanup_workspace(
    mode: str,
    *,
    workspace: Path | None = None,
    timestamp: str | None = None,
) -> dict:
    """Apply the configured cleanup policy to the live workspace area.

    The dataset cache (``datasets/cache``) is always preserved — only
    per-run subtrees are touched.

    Parameters
    ----------
    mode:
        One of ``"preserve"``, ``"archive"``, ``"purge"``.
    workspace:
        Override workspace root (useful in tests).
    timestamp:
        Override the archive folder timestamp (useful for determinism in
        tests). Defaults to the current UTC time.

    Returns
    -------
    dict with ``mode``, ``affected_run_ids`` and (for ``archive``) the
    target ``archive_path``. Used by the caller to emit ledger events.
    """
    if mode not in _VALID_CLEANUP_MODES:
        raise ValueError(
            f"Invalid cleanup mode '{mode}'. Must be one of: {sorted(_VALID_CLEANUP_MODES)}"
        )
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    info: dict = {"mode": mode, "affected_run_ids": []}

    if mode == "preserve":
        return info

    subtrees = _live_run_subtrees(ws)
    affected = sorted({p.name for p in subtrees})
    info["affected_run_ids"] = affected
    if not subtrees:
        return info

    if mode == "purge":
        for path in subtrees:
            shutil.rmtree(path, ignore_errors=True)
        return info

    # mode == "archive"
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = ws / "archive" / ts
    info["archive_path"] = str(archive_root)
    for path in subtrees:
        # Mirror the live structure under archive/<ts>/<sub>/runs/<run_id>
        rel = path.relative_to(ws)
        target = archive_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    return info
