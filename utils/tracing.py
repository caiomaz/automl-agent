"""Centralised LangSmith tracing utilities.

All agents import from here so there is a single point of control and a
consistent no-op fallback when LangSmith is not installed or tracing is
disabled.

Usage
-----
::

    from utils.tracing import traceable, set_run_metadata, tracing_context, is_tracing_enabled

    @traceable(name="my_step", run_type="chain", tags=["automl-agent"])
    def my_step(...):
        set_run_metadata(llm="gpt-4o", task="tabular_classification")
        ...
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any

__all__ = [
    "traceable",
    "set_run_metadata",
    "get_current_run_tree",
    "tracing_context",
    "is_tracing_enabled",
    "build_run_tags",
]

_TRACING_ENV_VARS = ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGSMITH_TRACING_V2")


def is_tracing_enabled() -> bool:
    """Return True when any LangSmith tracing env-var is set to 'true'."""
    return any(
        os.getenv(v, "").lower() == "true" for v in _TRACING_ENV_VARS
    )


# ── Try importing real LangSmith symbols ──────────────────────────────────────
try:
    from langsmith import traceable  # type: ignore[import-untyped]
    from langsmith.run_helpers import (  # type: ignore[import-untyped]
        get_current_run_tree,
        set_run_metadata as _set_run_metadata,
        tracing_context as _tracing_context,
    )
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False

    # No-op stubs ──────────────────────────────────────────────────────────────

    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        """No-op decorator when langsmith is not installed."""
        def decorator(fn):
            return fn
        # Support both @traceable and @traceable(...) usage
        return decorator(args[0]) if args and callable(args[0]) else decorator

    def get_current_run_tree():  # type: ignore[misc]
        return None

    def _set_run_metadata(**metadata: Any) -> None:  # type: ignore[misc]
        pass

    @contextmanager
    def _tracing_context(**kwargs: Any):  # type: ignore[misc]
        yield


# ── Public wrappers ───────────────────────────────────────────────────────────

def set_run_metadata(**metadata: Any) -> None:
    """Attach *metadata* key-value pairs to the currently active LangSmith run.

    Safe to call even when tracing is disabled or outside a traced context.
    """
    if not is_tracing_enabled():
        return
    try:
        _set_run_metadata(**metadata)
    except Exception:
        pass  # Never crash production code due to tracing failure


@contextmanager
def tracing_context(**kwargs: Any):
    """Context manager that injects top-level metadata/tags for a run tree.

    Wraps ``langsmith.run_helpers.tracing_context``. Safe no-op when
    LangSmith is unavailable.

    Typical usage::

        with tracing_context(
            project_name="automl-agent",
            tags=["tabular_classification", "rap"],
            metadata={"llm": "minimax/minimax-m2.5", "n_plans": 3},
        ):
            manager.initiate_chat(...)
    """
    if not _LANGSMITH_AVAILABLE or not is_tracing_enabled():
        yield
        return

    try:
        with _tracing_context(**kwargs):
            yield
    except Exception:
        yield  # traceback from tracing must never kill the agent run


def build_run_tags(
    *,
    task: str | None = None,
    llm: str | None = None,
    rap: bool | None = None,
    decomp: bool | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Build a standardised list of tags for a LangSmith run.

    Tags follow the ``key:value`` convention so they are easy to filter
    on the LangSmith dashboard.

    Parameters
    ----------
    task:   downstream task type, e.g. ``"tabular_classification"``
    llm:    LLM alias or slug, e.g. ``"minimax/minimax-m2.5"``
    rap:    whether Retrieval-Augmented Planning is enabled
    decomp: whether plan decomposition is enabled
    extra:  any additional free-form tag strings
    """
    tags: list[str] = ["automl-agent"]
    if task:
        # normalise: "tabular_classification" → "task:tabular_classification"
        safe = re.sub(r"[^a-z0-9_\-]", "-", task.lower())
        tags.append(f"task:{safe}")
    if llm:
        safe = re.sub(r"[^a-z0-9_\-/]", "-", llm.lower())
        tags.append(f"llm:{safe}")
    if rap is not None:
        tags.append("rap:enabled" if rap else "rap:disabled")
    if decomp is not None:
        tags.append("decomp:enabled" if decomp else "decomp:disabled")
    if extra:
        tags.extend(extra)
    return tags
