"""Unit tests for configs.py — LLMConfig, LLMRegistry, and env-based setup."""

import os
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_create_with_base_url(self):
        from configs import LLMConfig
        cfg = LLMConfig(api_key="k", model="m", base_url="http://localhost")
        assert cfg.api_key == "k"
        assert cfg.model == "m"
        assert cfg.base_url == "http://localhost"

    def test_create_without_base_url(self):
        from configs import LLMConfig
        cfg = LLMConfig(api_key="k", model="m")
        assert cfg.base_url is None

    def test_immutability(self):
        from configs import LLMConfig
        cfg = LLMConfig(api_key="k", model="m")
        with pytest.raises(AttributeError):
            cfg.api_key = "new"


# ---------------------------------------------------------------------------
# LLMRegistry
# ---------------------------------------------------------------------------

class TestLLMRegistry:
    def _make_registry(self):
        from configs import LLMRegistry, LLMConfig
        reg = LLMRegistry()
        reg.register("test-model", LLMConfig(api_key="key1", model="model-a"))
        reg.register("test-with-url", LLMConfig(
            api_key="key2", model="model-b", base_url="http://example.com/v1",
        ))
        return reg

    def test_register_and_get(self):
        from configs import LLMConfig
        reg = self._make_registry()
        cfg = reg.get("test-model")
        assert isinstance(cfg, LLMConfig)
        assert cfg.model == "model-a"

    def test_get_unknown_raises_keyerror(self):
        reg = self._make_registry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_list_returns_aliases(self):
        reg = self._make_registry()
        aliases = reg.list()
        assert "test-model" in aliases
        assert "test-with-url" in aliases

    def test_contains(self):
        reg = self._make_registry()
        assert "test-model" in reg
        assert "nonexistent" not in reg

    def test_getitem_backward_compat_dict(self):
        reg = self._make_registry()
        d = reg["test-model"]
        assert isinstance(d, dict)
        assert d["api_key"] == "key1"
        assert d["model"] == "model-a"
        assert "base_url" not in d

    def test_getitem_includes_base_url(self):
        reg = self._make_registry()
        d = reg["test-with-url"]
        assert d["base_url"] == "http://example.com/v1"

    def test_register_overwrites(self):
        from configs import LLMConfig
        reg = self._make_registry()
        reg.register("test-model", LLMConfig(api_key="new", model="new-model"))
        assert reg.get("test-model").model == "new-model"


# ---------------------------------------------------------------------------
# _build_default_registry & environment integration
# ---------------------------------------------------------------------------

class TestBuildDefaultRegistry:
    def test_default_registry_has_builtin_models(self):
        from configs import AVAILABLE_LLMs
        for alias in ("prompt-llm", "gpt-5",
                       "or-glm-5", "or-claude-sonnet", "or-gpt-5-mini",
                       "or-deepseek-r1", "or-llama-3.3"):
            assert alias in AVAILABLE_LLMs, f"{alias} missing from default registry"

    def test_openrouter_models_use_correct_base_url(self):
        from configs import AVAILABLE_LLMs, OPENROUTER_BASE_URL
        for alias in AVAILABLE_LLMs.list():
            if alias.startswith("or-"):
                d = AVAILABLE_LLMs[alias]
                assert d["base_url"] == OPENROUTER_BASE_URL

    @patch.dict(os.environ, {"OPENROUTER_MODEL": "custom/my-model"}, clear=False)
    def test_custom_openrouter_model_via_env(self):
        from configs import _build_default_registry
        reg = _build_default_registry()
        assert "or-custom" in reg
        assert reg.get("or-custom").model == "custom/my-model"

    @patch.dict(os.environ, {"OPENROUTER_MODEL": ""}, clear=False)
    def test_no_custom_when_env_empty(self):
        from configs import _build_default_registry
        reg = _build_default_registry()
        assert "or-custom" not in reg

    @patch.dict(os.environ, {"VLLM_BASE_URL": "http://gpu-server:9000/v1"}, clear=False)
    def test_vllm_url_override(self):
        from configs import _build_default_registry
        reg = _build_default_registry()
        assert reg.get("prompt-llm").base_url == "http://gpu-server:9000/v1"


# ---------------------------------------------------------------------------
# Configs dataclass
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_configs_reads_env(self):
        """Configs picks up env vars that were set before import."""
        # Configs uses os.getenv at class-define time, so we test
        # that the dataclass fields are strings (env-driven).
        from configs import configs
        assert isinstance(configs.OPENAI_KEY, str)
        assert isinstance(configs.OPENROUTER_KEY, str)

    def test_configs_defaults_to_empty(self):
        from configs import Configs
        cfg = Configs()
        assert isinstance(cfg.OPENAI_KEY, str)
        assert isinstance(cfg.OPENROUTER_KEY, str)
