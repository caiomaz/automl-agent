import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Configs:
    OPENAI_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENROUTER_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    HF_KEY: str = os.getenv("HF_API_KEY", "")
    KAGGLE_API_TOKEN: str = os.getenv("KAGGLE_API_TOKEN", "")
    SEARCHAPI_API_KEY: str = os.getenv("SEARCHAPI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")


configs = Configs()


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str | None = None


class LLMRegistry:
    """Single-responsibility registry for available LLM configurations.

    Supports adding custom OpenRouter models at runtime via register(),
    and overriding the model slug via the OPENROUTER_MODEL env var.
    """

    def __init__(self) -> None:
        self._models: dict[str, LLMConfig] = {}

    def register(self, alias: str, config: LLMConfig) -> None:
        self._models[alias] = config

    def get(self, alias: str) -> LLMConfig:
        if alias in self._models:
            return self._models[alias]
        # Auto-create for raw OpenRouter slugs (e.g. "z-ai/glm-5", "anthropic/claude-opus-4.6")
        # No pre-registration needed — any valid OpenRouter model ID works directly.
        if "/" in alias:
            return LLMConfig(
                api_key=configs.OPENROUTER_KEY,
                model=alias,
                base_url=OPENROUTER_BASE_URL,
            )
        raise KeyError(
            f"LLM '{alias}' not registered. "
            f"You can use a registered alias ({list(self._models.keys())}) "
            f"or any raw OpenRouter slug (e.g. 'anthropic/claude-opus-4.6')."
        )

    def list(self) -> list[str]:
        return list(self._models.keys())

    def __contains__(self, alias: str) -> bool:
        # Raw OpenRouter slugs (provider/model) are always "available"
        return alias in self._models or "/" in alias

    def __getitem__(self, alias: str) -> dict:
        """Backward-compatible dict-style access for AVAILABLE_LLMs[key]."""
        cfg = self.get(alias)
        d: dict = {"api_key": cfg.api_key, "model": cfg.model}
        if cfg.base_url is not None:
            d["base_url"] = cfg.base_url
        return d


def _build_default_registry() -> LLMRegistry:
    registry = LLMRegistry()

    # Local vLLM for PromptAgent (fine-tuned adapter)
    vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    registry.register("prompt-llm", LLMConfig(
        api_key="empty", model="prompt-llama", base_url=vllm_url,
    ))

    # OpenAI direct (legacy, kept for backward compat)
    registry.register("gpt-5", LLMConfig(api_key=configs.OPENAI_KEY, model="gpt-5"))

    # OpenRouter models — current as of 2026
    _or = configs.OPENROUTER_KEY
    _or_models = {
        # Anthropic
        "or-claude-sonnet": "anthropic/claude-sonnet-4.6",
        "or-claude-opus": "anthropic/claude-opus-4.6",
        "or-claude-haiku": "anthropic/claude-haiku-4.5",
        # OpenAI
        "or-gpt-5": "openai/gpt-5",
        "or-gpt-5-mini": "openai/gpt-5-mini",
        "or-gpt-5-nano": "openai/gpt-5-nano",
        # Google
        "or-gemini-flash": "google/gemini-3-flash-preview",
        "or-gemini-pro": "google/gemini-3.1-pro-preview",
        "or-gemini-flash-lite": "google/gemini-2.5-flash-lite",
        # Meta
        "or-llama-3.3": "meta-llama/llama-3.3-70b-instruct",
        # DeepSeek
        "or-deepseek-r1": "deepseek/deepseek-r1",
        # Mistral
        "or-mistral-large": "mistralai/mistral-large",
        "or-mistral-medium": "mistralai/mistral-medium-3.1",
        "or-codestral": "mistralai/codestral-2508",
        # Qwen
        "or-qwen3": "qwen/qwen3-235b-a22b",
        "or-qwen3.5": "qwen/qwen3.5-397b-a17b",
        "or-qwen3.5-plus": "qwen/qwen3.5-plus-02-15",
        "or-qwen3.5-flash": "qwen/qwen3.5-flash-02-23",
        "or-qwen3.6-plus": "qwen/qwen3.6-plus",
        "or-qwen3-coder": "qwen/qwen3-coder",
        # xAI
        "or-grok-4": "x-ai/grok-4",
        "or-grok-4-fast": "x-ai/grok-4.1-fast",
        "or-grok-3-mini": "x-ai/grok-3-mini",
        # Z-AI — GLM (affordable, strong code generation)
        "or-glm-5": "z-ai/glm-5",
        # Moonshot
        "or-kimi-k2": "moonshotai/kimi-k2-thinking",
        # MiniMax
        "or-minimax": "minimax/minimax-m2.5",
        # Perplexity
        "or-sonar": "perplexity/sonar",
    }
    for alias, model_slug in _or_models.items():
        registry.register(alias, LLMConfig(
            api_key=_or, model=model_slug, base_url=OPENROUTER_BASE_URL,
        ))

    # Dynamic override: OPENROUTER_MODEL env var → workflow-wide backbone override.
    # If set, it overrides LLM_BACKBONE for all agents without editing configs.py.
    # Accepts a raw OpenRouter slug (e.g. "qwen/qwen3.5-flash-02-23") OR a
    # registered alias (e.g. "or-qwen3.5-flash") — aliases are resolved to their
    # underlying model slug so the API never receives an internal alias name.
    custom_model = os.getenv("OPENROUTER_MODEL", "")
    if custom_model:
        # Resolve alias → real slug if the value matches a registered entry.
        if custom_model in registry and "/" not in custom_model:
            custom_model = registry[custom_model]["model"]
        registry.register("or-custom", LLMConfig(
            api_key=_or, model=custom_model, base_url=OPENROUTER_BASE_URL,
        ))
        # Propagate as the effective backbone (agent_manager reads _OR_BACKBONE_OVERRIDE first)
        os.environ["LLM_BACKBONE"] = "or-custom"

    return registry


AVAILABLE_LLMs = _build_default_registry()

TASK_METRICS = {
    "image_classification": "accuracy",
    "text_classification": "accuracy",
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "tabular_clustering": "RI",
    "node_classification": "accuracy",
    "ts_forecasting": "RMSLE",
}
