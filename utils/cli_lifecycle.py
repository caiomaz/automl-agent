"""CLI lifecycle helpers — Phase 4.

Provides:

1. :func:`install_cancellation_handler` — install a signal handler for
   SIGINT/SIGTERM that flips a process-local flag on the first signal and
   raises :class:`KeyboardInterrupt` on the second (so users still have an
   "escape hatch" if a single graceful cancel hangs).
2. :func:`cancellation_requested` — read the flag.
3. :func:`list_past_runs` — read all ``run_manifest.json`` files under the
   workspace and return them sorted by start time descending.
4. :func:`build_post_run_summary` — assemble a dict suitable for printing
   a one-screen post-run report.
"""

from __future__ import annotations

import json
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from utils.workspace import WORKSPACE_DIR, run_exp_dir


# ── Cancellation flag (process-local) ────────────────────────────────────────


_cancel_state: dict[str, Any] = {"requested": False, "count": 0}


def cancellation_requested() -> bool:
    return bool(_cancel_state["requested"])


def _reset_cancellation_for_tests() -> None:
    """Reset the module-level cancellation state. Tests only."""
    _cancel_state["requested"] = False
    _cancel_state["count"] = 0


@dataclass
class CancellationToken:
    """Handle returned by :func:`install_cancellation_handler` so callers can
    uninstall the handler when finished."""

    handler: Callable[[int, Any], None]
    previous: dict[int, Any]
    signals: tuple[int, ...]

    def uninstall(self) -> None:
        for sig, prev in self.previous.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                # Outside the main thread, signal.signal raises ValueError.
                pass


def install_cancellation_handler(
    signals: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM),
) -> CancellationToken:
    """Install a process-local cancellation handler.

    On the first signal: flips ``cancellation_requested()`` to ``True`` so
    the run can finalize gracefully.

    On the second signal: raises :class:`KeyboardInterrupt` so a stuck
    workflow can still be killed by the user.
    """

    def _handler(signum: int, frame: Any) -> None:
        _cancel_state["count"] += 1
        if _cancel_state["count"] >= 2:
            raise KeyboardInterrupt(f"second signal {signum} — escalating")
        _cancel_state["requested"] = True

    previous: dict[int, Any] = {}
    for sig in signals:
        try:
            previous[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):
            # signal.signal only works from the main thread of the main
            # interpreter. Skip silently in worker threads.
            continue
    return CancellationToken(handler=_handler, previous=previous, signals=signals)


# ── Past-run listing ─────────────────────────────────────────────────────────


def list_past_runs(*, workspace: Path | None = None) -> list[dict]:
    """Return all run manifests under ``exp/runs/`` sorted by start time desc."""
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    runs_root = ws / "exp" / "runs"
    if not runs_root.is_dir():
        return []
    out: list[dict] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        manifest = run_dir / "run_manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append(data)
    out.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return out


# ── Post-run summary ─────────────────────────────────────────────────────────


def build_post_run_summary(run_id: str, *, workspace: Path | None = None) -> dict:
    """Inspect the artifacts of ``run_id`` and return a summary dict suitable
    for the post-run menu / report."""
    ws = Path(workspace) if workspace is not None else WORKSPACE_DIR
    exp_dir = run_exp_dir(run_id, ws)
    manifest_path = exp_dir / "run_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    events_path = exp_dir / "events.jsonl"
    event_count = 0
    if events_path.is_file():
        event_count = sum(
            1 for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    return {
        "run_id": run_id,
        "status": manifest.get("status"),
        "task_type": manifest.get("task_type"),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "exp_dir": str(exp_dir),
        "has_terminal_log": (exp_dir / "terminal.log").is_file(),
        "has_cost_summary": (exp_dir / "cost_summary.json").is_file(),
        "event_count": event_count,
    }
