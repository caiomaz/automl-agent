"""Regression tests – utils.tracing contract.

These tests freeze the public surface and fallback behavior of the
tracing module so that future changes (Phase 2+) do not silently
break existing agent code.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from utils.tracing import (
    build_run_tags,
    is_tracing_enabled,
    set_run_metadata,
    tracing_context,
    traceable,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_tracing_env(monkeypatch):
    """Ensure no tracing env-vars leak between tests."""
    for var in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGSMITH_TRACING_V2"):
        monkeypatch.delenv(var, raising=False)


# ── is_tracing_enabled ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestIsTracingEnabled:
    """The function must return True only when a recognised env-var is 'true'."""

    def test_disabled_by_default(self):
        assert is_tracing_enabled() is False

    @pytest.mark.parametrize("var", [
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
    ])
    def test_enabled_when_env_set(self, monkeypatch, var):
        monkeypatch.setenv(var, "true")
        assert is_tracing_enabled() is True

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "True")
        assert is_tracing_enabled() is True

    def test_random_value_means_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "yes")
        assert is_tracing_enabled() is False


# ── traceable decorator ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestTraceableDecorator:
    """@traceable must be a no-op decorator when LangSmith is absent."""

    def test_bare_decorator(self):
        @traceable
        def my_fn(x):
            return x + 1
        assert my_fn(1) == 2

    def test_parameterised_decorator(self):
        @traceable(name="step", run_type="chain", tags=["t"])
        def my_fn(x):
            return x * 2
        assert my_fn(3) == 6


# ── set_run_metadata ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSetRunMetadata:
    """set_run_metadata must never raise, even outside a traced context."""

    def test_noop_when_disabled(self):
        # Must not raise
        set_run_metadata(llm="gpt-4o", task="tabular_classification")

    def test_noop_when_enabled_but_no_context(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        # Still must not raise (inner exception is swallowed)
        set_run_metadata(llm="gpt-4o", task="tabular_classification")


# ── tracing_context ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTracingContext:
    """tracing_context must work as a no-op context manager."""

    def test_noop_yields(self):
        with tracing_context(project_name="test"):
            pass  # Must not raise

    def test_body_executes(self):
        result = []
        with tracing_context(tags=["a"]):
            result.append(42)
        assert result == [42]


# ── build_run_tags ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildRunTags:
    """build_run_tags must produce deterministic tag lists."""

    def test_minimal(self):
        tags = build_run_tags()
        assert tags == ["automl-agent"]

    def test_full(self):
        tags = build_run_tags(
            task="tabular_classification",
            llm="minimax/minimax-m2.5",
            rap=True,
            decomp=False,
        )
        assert tags == [
            "automl-agent",
            "task:tabular_classification",
            "llm:minimax/minimax-m2-5",
            "rap:enabled",
            "decomp:disabled",
        ]

    def test_rap_disabled(self):
        tags = build_run_tags(rap=False)
        assert "rap:disabled" in tags

    def test_extra_tags_appended(self):
        tags = build_run_tags(extra=["custom-tag", "run:42"])
        assert "custom-tag" in tags
        assert "run:42" in tags
        # First tag is always "automl-agent"
        assert tags[0] == "automl-agent"

    def test_task_sanitisation(self):
        tags = build_run_tags(task="Some Weird/Task!!")
        task_tag = [t for t in tags if t.startswith("task:")][0]
        # All non-alphanumeric/underscore/dash chars replaced with '-'
        assert " " not in task_tag
        assert "!" not in task_tag
        assert "/" not in task_tag

    def test_llm_preserves_slash(self):
        """LLM slugs like 'anthropic/claude-opus-4.6' keep '/' but '.' → '-'."""
        tags = build_run_tags(llm="anthropic/claude-opus-4.6")
        llm_tag = [t for t in tags if t.startswith("llm:")][0]
        assert "anthropic/claude-opus-4-6" in llm_tag
        # Slash is preserved
        assert "/" in llm_tag
