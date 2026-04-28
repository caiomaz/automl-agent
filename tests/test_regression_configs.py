"""Regression tests – configs.py contract.

These tests freeze the public surface of the LLM registry, task
metrics map, and environment-driven overrides so that future
changes do not silently break downstream agent expectations.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
import configs
from configs import (
    AVAILABLE_LLMs,
    TASK_METRICS,
    Configs,
    LLMConfig,
    LLMRegistry,
    OPENROUTER_BASE_URL,
)


# ── TASK_METRICS ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTaskMetrics:
    """TASK_METRICS must contain exactly the 7 canonical task types."""

    EXPECTED_TASKS = {
        "image_classification",
        "text_classification",
        "tabular_classification",
        "tabular_regression",
        "tabular_clustering",
        "node_classification",
        "ts_forecasting",
    }

    def test_exact_keys(self):
        assert set(TASK_METRICS.keys()) == self.EXPECTED_TASKS

    def test_count(self):
        assert len(TASK_METRICS) == 7

    @pytest.mark.parametrize("task,metric", [
        ("image_classification", "accuracy"),
        ("text_classification", "accuracy"),
        ("tabular_classification", "F1"),
        ("tabular_regression", "RMSLE"),
        ("tabular_clustering", "RI"),
        ("node_classification", "accuracy"),
        ("ts_forecasting", "RMSLE"),
    ])
    def test_metric_values(self, task, metric):
        assert TASK_METRICS[task] == metric


# ── LLMConfig ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestLLMConfig:
    """LLMConfig must be a frozen dataclass with the expected fields."""

    def test_frozen(self):
        cfg = LLMConfig(api_key="k", model="m")
        with pytest.raises(AttributeError):
            cfg.api_key = "other"  # type: ignore[misc]

    def test_base_url_optional(self):
        cfg = LLMConfig(api_key="k", model="m")
        assert cfg.base_url is None

    def test_fields(self):
        cfg = LLMConfig(api_key="k", model="m", base_url="http://x")
        assert cfg.api_key == "k"
        assert cfg.model == "m"
        assert cfg.base_url == "http://x"


# ── LLMRegistry ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestLLMRegistry:
    """LLMRegistry must support registration, raw-slug fallback, and dict access."""

    def test_register_and_get(self):
        r = LLMRegistry()
        cfg = LLMConfig(api_key="k", model="m")
        r.register("test", cfg)
        assert r.get("test") is cfg

    def test_raw_slug_auto_creates(self):
        r = LLMRegistry()
        cfg = r.get("anthropic/claude-opus-4.6")
        assert cfg.model == "anthropic/claude-opus-4.6"
        assert cfg.base_url == OPENROUTER_BASE_URL

    def test_unknown_alias_raises(self):
        r = LLMRegistry()
        with pytest.raises(KeyError, match="not registered"):
            r.get("nonexistent")

    def test_contains_registered(self):
        r = LLMRegistry()
        r.register("x", LLMConfig(api_key="k", model="m"))
        assert "x" in r

    def test_contains_raw_slug(self):
        r = LLMRegistry()
        assert "provider/model" in r

    def test_contains_unknown(self):
        r = LLMRegistry()
        assert "unknown" not in r

    def test_dict_access(self):
        r = LLMRegistry()
        r.register("x", LLMConfig(api_key="k", model="m", base_url="http://x"))
        d = r["x"]
        assert d["api_key"] == "k"
        assert d["model"] == "m"
        assert d["base_url"] == "http://x"

    def test_list(self):
        r = LLMRegistry()
        r.register("a", LLMConfig(api_key="k", model="m"))
        r.register("b", LLMConfig(api_key="k", model="m"))
        assert r.list() == ["a", "b"]


# ── Default registry ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDefaultRegistry:
    """The module-level AVAILABLE_LLMs must contain key registrations."""

    def test_prompt_llm_registered(self):
        """The vLLM-based prompt-llm must always be present."""
        cfg = AVAILABLE_LLMs.get("prompt-llm")
        assert cfg.model == "prompt-llama"

    def test_openrouter_aliases_present(self):
        """Spot-check a few OpenRouter aliases."""
        for alias in ("or-glm-5", "or-claude-sonnet", "or-gpt-5", "or-deepseek-r1"):
            assert alias in AVAILABLE_LLMs, f"{alias} missing"

    def test_openrouter_base_url(self):
        cfg = AVAILABLE_LLMs.get("or-glm-5")
        assert cfg.base_url == OPENROUTER_BASE_URL

    def test_legacy_gpt5(self):
        cfg = AVAILABLE_LLMs.get("gpt-5")
        assert cfg.model == "gpt-5"
        assert cfg.base_url is None  # Direct OpenAI, no base_url


# ── Configs dataclass ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestConfigs:
    """Configs must expose the expected env-var-backed fields."""

    def test_frozen(self):
        with pytest.raises(AttributeError):
            configs.configs.OPENAI_KEY = "x"  # type: ignore[misc]

    def test_expected_fields(self):
        expected = {"OPENAI_KEY", "OPENROUTER_KEY", "HF_KEY",
                    "KAGGLE_API_TOKEN", "SEARCHAPI_API_KEY", "TAVILY_API_KEY"}
        actual = {f.name for f in configs.Configs.__dataclass_fields__.values()}
        assert actual == expected


# ── OPENROUTER_BASE_URL ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestOpenRouterBaseURL:
    def test_value(self):
        assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
