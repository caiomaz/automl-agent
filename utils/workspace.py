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
import hashlib
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
